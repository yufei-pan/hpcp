# Disk-level patch images (`-dd -diff` / `-siff` on whole disks)

Date: 2026-10-10  
Status: Approved design (pending implementation plan)  
Builds on: `docs/superpowers/specs/2026-10-10-diff-image-design.md` (phase 1, shipped in 9.65)

## Problem

Phase 1 patches one filesystem: a directory or a single-filesystem image. Golden OS images are whole disks —
an EFI System Partition, often a BIOS-boot partition, a root filesystem, swap — cloned onto nodes with
`hpcp -dd` (optionally resized with `-ddr`). Updating such nodes today means re-cloning the whole disk.
Phase 1 reserved per-partition `section` lines in the manifest for this; nothing uses them yet.

## Goal

1. `hpcp -dd -diff [PATCH.img] NEW_DISK -d OLD_DISK` writes one patch image that turns a disk laid out like
   OLD_DISK into NEW_DISK: a phase-1 change list and payload per filesystem partition, raw payloads for areas
   that cannot be patched file by file, and the partition / filesystem identity changes.
2. `hpcp -siff PATCH.img -d TARGET_DISK` applies it to a node disk that is either offline (unmounted block
   device or disk-image file) or live (the node's running OS disk), checking everything up front and refusing
   what cannot be done safely on a live disk.

Primary workflow (decided): golden-image rollout — diff the new golden disk against the previous golden disk
image, apply the patch to many node disks, live or offline.

Success criterion: applying `-dd -diff NEW OLD` to an offline copy of OLD yields a disk whose filesystem
partitions match NEW path for path (phase-1 states), whose raw areas hash equal to NEW's, and whose identity
fields (GPT unique GUID / name / attributes, disk GUID, filesystem UUID / label) equal NEW's.

## Non-goals

- MBR (msdos) partition tables — dd mode is GPT-only (`sgdisk`).
- Layout changes: added, removed, re-typed or resized partitions. These require a re-clone.
- Looking inside LVM, md, dm-crypt or unknown filesystems: such partitions are raw areas only.
- Patch chains (backup-style series); one patch moves one base to one new state.
- Changing mkfs parameters (block size, features) on existing filesystems — reported as warnings only.

## Decisions

| Topic | Decision |
|---|---|
| Use case | Golden-image rollout; targets may be live (mounted) or offline. |
| Packaging | One patch image (phase-1 container, `-dis` splittable) with per-partition sections, `raw/` payloads and `disk.tsv`. Rejected: a bundle of phase-1 patches (several artifacts), a GPT disk image as the patch (wasteful, needs a partition table just to carry data). |
| Layout matching | Pair partitions by number. Require the same partition count, type GUIDs and filesystem types. Sizes may differ (except raw areas). Filesystem UUIDs and labels on the target may differ from the base (warning); per-file checks guard contents. |
| Raw areas | BIOS-boot partitions, unknown-filesystem partitions and the MBR boot code (bytes 0–439 of sector 0) are compared by sha256; when they differ the patch carries the new raw bytes. Swap content is never carried — only its UUID / label. |
| Identity changes | Applied when safe: offline, filesystem UUID / label via the filesystem's tools and GPT unique GUID / name / attributes and the disk GUID via `sgdisk`. On a live disk, an identity change to a partition that is in use refuses the whole patch up front. mkfs-parameter changes are warnings only. |

## CLI

```
hpcp -dd -diff [PATCH.img]  NEW_DISK  -d OLD_DISK  [-dis SIZE] [-fh] [-hs N] [-e PAT] [-j N]
hpcp -siff PATCH.img [-siff PATCH_1.img ...]  -d TARGET_DISK  [-fa] [-j N]
```

- `-dd -diff`: NEW_DISK is the single positional source (block device or disk-image file, as `-dd` takes
  today); OLD_DISK is the single `-d`. `-di` is not used in dd mode. `-diff`'s optional value and the
  auto-name rule are phase 1's.
- `-siff` reads the patch's manifest: `layout disk` selects disk apply; `-dd` is accepted but not required on
  apply. A disk patch with a non-disk target (directory, single-filesystem image) and a phase-1 patch with a
  disk target are rejected with `Invalid arguments: …`.
- Validation (phase-1 `_validate_diff_args` extended): `-dd -diff` needs exactly one positional and one `-d`;
  `-dd` with `-siff` is now allowed; every other phase-1 incompatibility stays.

## Formats

All phase-1 format rules (escaping, entry states, `changes.tsv`) hold unchanged.

### `manifest` (format_version 2)

```
format	hpcp-diff
format_version	2
layout	disk
patch_id	…
volume	1/2
created / hpcp_version / new / base / hash_algo / hash_size / full_hash / xattrs   (as phase 1)
section	part1	part1	part1.changes.tsv
section	part3	part3	part3.changes.tsv
raw	part2	raw/part2.img	<old sha256>	<new sha256>	<size>
raw	mbr	raw/mbr-bootcode.bin	<old sha256>	<new sha256>	440
disk	disk.tsv
added / modified / deleted   (totals over all sections)
payload_bytes / raw_bytes
```

- `read_manifest` accepts `raw` as a repeatable key (like `section`) and `disk` as a single key.
- Section names and raw names must be `part<N>` (N a positive integer) or `mbr` (raw only); payload dirs equal
  the section name; raw files are bare names under `raw/`; `disk` names a bare file under `.hpcp_diff/`.
- hpcp 9.65 rejects these manifests (`format_version` 2 is newer than it supports). Phase-1 manifests
  (version 1, layouts `root` / `store`) remain valid and keep their behaviour.

### `disk.tsv`

Escaped key/value rows: `kind<TAB>number<TAB>field<TAB>old<TAB>new`, first line `#hpcp-disk<TAB>1`.

| Row | Fields |
|---|---|
| `disk - …` | `guid`, `sector_size` |
| `part N …` | `kind` (`fs` / `swap` / `raw`), `type_guid`, `unique_guid`, `name`, `attrs`, `size`, `fs_type`, `fs_uuid`, `fs_label`, `fs_params` (sorted `k=v;…`) |

Field roles:

- Must match on the target (compared with `old`): `kind`, `type_guid`, `fs_type`, the partition numbers and
  count.
- Applied when `old != new`: `unique_guid`, `name`, `attrs`, `fs_uuid`, `fs_label`, disk `guid`.
- Informational: `size` (target may differ, except raw areas which must equal it), `fs_params` (warning when
  `old != new`), `sector_size` (must match).

### Partition kinds (decided at generation)

- `fs`: both sides mounted successfully (reusing the live mount when the partition is already mounted).
- `swap`: `fs_type` is `swap`.
- `raw`: everything else (BIOS boot, unknown filesystem, LVM PV, md member, dm-crypt).

### Raw areas

- sha256 over the whole partition (or bytes 0–439 of sector 0 for `mbr`), streamed in chunks, independent of
  xxhash.
- A raw entry exists only when old and new hashes differ. Its payload is the full new content.
- A raw payload must fit in one patch volume; otherwise generation fails with `Diff image error: …` naming
  the needed `-dis`.

## Generating a disk patch (`process_get_disk_diff_image`)

1. Attach NEW_DISK and OLD_DISK read-only (read-only loop devices for image files). Read both layouts with
   `read_disk_layout` (a fresh read; `get_partition_infos` is `lru_cache`d).
2. `compare_disk_layouts(old, new)`: refuse with `Disk layout mismatch: …` when partition count, numbers,
   type GUIDs, filesystem types or sector size differ.
3. For each partition:
   - `fs`: mount both read-only (phase-1 options: `nouuid,norecovery` for xfs, `noload` for ext3/4), or use an
     existing mount whose filesystem root is `/`; build root-layout maps; `compute_changes` → that section's
     rows; `_validate_patch_rows`.
   - `swap`: nothing but identity.
   - `raw`: hash both; if different and sizes equal, add a raw entry; if different and sizes differ, refuse
     with `Disk layout mismatch: …`.
4. MBR boot code: hash bytes 0–439 on both sides; add a `mbr` raw entry when different.
5. Identity rows for every partition and the disk; mkfs-parameter differences are printed as warnings.
6. Nothing differs → print "No differences", create no image, exit 0.
7. Claim the planned volume names, create volumes sized from all sections' allocation estimate plus raw
   bytes, build each section into `part<N>/` on the volumes (phase-1 `build_patch_tree` with the section as
   root), copy raw payloads into `raw/`, write every `part<N>.changes.tsv` and `disk.tsv` on every volume,
   manifests last.

## Applying a disk patch (`apply_disk_patch`)

### Target state

- Image file target: refuse with `Disk in use: …` if it is already attached and mounted elsewhere
  (`losetup -j` + mountinfo); otherwise attach read-write — offline.
- Block-device target: per partition, determine in-use state from `/proc/self/mountinfo`, `/proc/swaps` and
  `/sys/class/block/<part>/holders`.

### Up-front checks (nothing changes before all pass)

1. Load and cross-check volumes (phase 1), read every section's rows, raw entries and `disk.tsv`.
2. Read the target layout; apply the matching rules; raw areas need exactly the recorded size.
3. Per `fs` section: use the live mount whose filesystem root is `/` (a read-only mount is a conflict), or
   mount read-write on a temp directory; run phase-1 `check_patch_target`.
4. Raw areas: target hash equal to `new` → skip; equal to `old` → apply; otherwise conflict.
5. Identity fields: target value equal to `new` → skip; otherwise apply (a target value different from both
   is a warning, not a conflict — node identities may legitimately differ).
6. Refusals not overridable by `-fa`:
   - `Disk in use: …` (live disk): a raw write to an in-use partition; an identity change (filesystem UUID /
     label, GPT unique GUID / name / attributes) on an in-use partition. The disk GUID is not a partition and
     is changed even on a live disk (it only rewrites the GPT headers).
   - `Disk layout mismatch: …` (any target): an `fs` section whose target partition cannot be mounted
     read-write and has no usable live mount.
7. Filesystem types without an identity setter that need a UUID / label change: `Identity change error: …`,
   overridable by `-fa` (then a warning).
8. All conflicts from all sections, raw areas included, are printed together and resolved with one prompt
   (`Apply anyway? [y/N]`, default abort; non-tty aborts; `-fa` forces) — phase 1's rules.
9. Phase-1 payload pre-check and outside-symlink pre-flight for every section; raw payload files must hash to
   their recorded `new` value.

### Order of changes

1. Filesystem sections — root filesystem first, ESP (`type_guid` C12A7328-F81F-11D2-BA4B-00A0C93EC93B) last —
   with phase-1 apply semantics (atomic replacement, 0700 new directories, containment re-checks). Temp mounts
   are unmounted when their section finishes.
2. Raw partition areas: open the partition device, write in chunks, `fsync`, re-hash.
3. MBR boot code: write exactly bytes 0–439 of sector 0, `fsync`, re-hash.
4. Identity changes on partitions that are not in use: filesystem UUID / label with `_fs_identity_commands`;
   GPT unique GUID / name / attributes and the disk GUID with `sgdisk`.
5. Verify: every section's rows (phase 1), every raw hash, every identity value re-read with `blkid` /
   `sgdisk`. Mismatches are recorded with `eprint` (non-zero exit).
6. Summary per section plus totals.

A re-run after an interruption skips paths, raw areas and identity values already in their new state.

## Code units (all in `hpcp.py`)

| Unit | Responsibility |
|---|---|
| `read_disk_layout(device)` | fresh GPT + filesystem identity read → `{'disk': {…}, 'parts': {N: {…}}}` |
| `compare_disk_layouts(old, new)` | refusal reasons + identity rows |
| `write_disk_tsv` / `read_disk_tsv` | `disk.tsv` |
| `_hash_area(path, offset=0, length=None)` | streaming sha256 |
| `_read_mbr_bootcode(disk)` | the 440 boot-code bytes |
| `_live_mounts(device, mountinfo_path='/proc/self/mountinfo')` | mounts of a device's partitions with their filesystem root |
| `_partition_in_use(part, mountinfo_path, swaps_path, sys_block_path)` | mounted / active swap / holders |
| `_fs_identity_commands(fs_type, device, uuid=None, label=None)` | argv lists to set a filesystem UUID and/or label; UUID setters extracted from `write_partition_info` (which then calls it), label setters added (`e2label`/`tune2fs -L`, `xfs_admin -L`, `btrfs filesystem label`, `fatlabel`, `ntfslabel`, `exfatlabel`, `swaplabel -L`, …); returns `None` for an unsupported type |
| `_gpt_identity_commands(disk, number, unique_guid=None, name=None, attrs=None)` | `sgdisk` argv lists |
| `process_get_disk_diff_image(...)` | generation wrapper |
| `apply_disk_patch(volume_roots, target_disk, force=False, …)` | apply wrapper |

Targeted refactors of phase-1 code:

- `load_patch_volumes` accepts format_version 2 and returns per-section rows plus raw / disk entries.
- `apply_patch_tree` splits into prepare-section (checks + pre-flights), one shared conflict resolution, and
  apply-section + verify; single-filesystem apply becomes the one-section case.
- `write_patch_metadata` writes several change lists, `disk.tsv` and raw entries, manifests last.
- `process_load_diff_image` dispatches on `layout`.
- `_validate_diff_args` gains the `-dd` rules above.

New error categories: `Disk layout mismatch` 248, `Raw area error` 249, `Identity change error` 250,
`Disk in use` 251.

## Testing

No root required:

- `disk.tsv` and manifest v2 round trips; validation of `raw` / `section` / `disk` names.
- `compare_disk_layouts`: count, numbers, type GUID, filesystem type, sector size, raw size refusals; other
  size differences allowed.
- `_fs_identity_commands` / `_gpt_identity_commands` argv for each supported type; `None` for unsupported.
- `_live_mounts` / `_partition_in_use` from fixture mountinfo, swaps and holders files.
- Raw-area check / write / verify with plain files standing in for partition devices.
- A two-section apply on directories: one shared conflict prompt, sections applied in order.
- Phase-1 tests stay green, including single-filesystem patches through the refactored apply path.

Root + loop:

- Fixture: a GPT image built with `sgdisk`: p1 ESP (vfat), p2 BIOS boot (raw), p3 root (ext4), p4 swap. Copy
  it as OLD; on NEW change files on p1 and p3, bytes in p2, p3's label and the MBR boot code.
- `-dd -diff NEW OLD` → patch; apply to an offline copy of OLD → filesystem states, raw hashes, identity
  values and layout all equal NEW; a re-run skips everything.
- Layout mismatch (an extra partition on the target) → `Disk layout mismatch`, nothing changed.
- Live simulation: mount the target's p3 before applying → the section goes through that mount; a label
  change on the mounted p3 → refused up front with `Disk in use`, nothing changed.

## Release and docs

- `-diff` / `-siff` help text mentions `-dd`; README gains a "Disk patches" subsection under the patch-image
  note.
- Version 9.66; `cas-toolbox` pin rides the next Monday roll-up.

## Known limitations

- GPT only; layout changes need a re-clone.
- Raw areas are all-or-nothing and must fit in one patch volume.
- On a live disk, identity changes to in-use partitions and raw writes to in-use partitions are refused;
  apply such patches offline.
- The phase-1 residual race (nothing else may modify the target while a patch is applied) applies per
  partition.
