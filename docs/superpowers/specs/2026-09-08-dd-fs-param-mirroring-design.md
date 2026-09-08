# Mirror source filesystem parameters in dd mode

Date: 2026-09-08  
Status: Approved design (pending implementation plan)

## Problem

In dd mode (`-dd` / `--disk_dump`), `hpcp` recreates each source partition's filesystem on the
destination and then copies files into it. The recreation uses **whatever the building host's
`mkfs` defaults are**. Only three properties are carried over from the source: `fs_type`,
`fs_label`, and `fs_uuid`.

Everything else — block size, inode size, cluster size, reserved-block policy, and the entire
feature set — is silently replaced by the local defaults. A disk cloned on a modern host comes
back with a filesystem the original system may not be able to mount or boot.

### Verified

A source image with three deliberately non-default filesystems, copied with
`python3 hpcp.py -dd src.img dest.img`:

| Partition | Source | hpcp destination |
|---|---|---|
| p1 (ESP) | **FAT32**, 512 B clusters | **FAT16** (`SEC_TYPE=msdos`), 8192 B clusters |
| p2 (ext4) | 1024 B blocks, 128 B inodes, `-m 0`, no `64bit` / `metadata_csum` / `dir_index` | 4096 B blocks, 256 B inodes, 5 % reserved (25 MiB), **added `64bit`, `metadata_csum`, `metadata_csum_seed`, `dir_index`** |
| p3 (xfs) | `isize=1024`, dir block 8192, `reflink=0` | `isize=512`, dir block 4096, **`reflink=1`** (`ro_compat` `0xb`→`0xf`, `incompat` `0x2b`→`0x2f`) |

Consequences:

- **Unbootable / unmountable clones.** `metadata_csum` and `64bit` on ext4, or xfs v5 features,
  are not understood by the older GRUB or kernel the source disk was built for.
- **Silent capacity loss.** A `/boot` built with `-m 0` gains a 5 % reserve; a nearly-full source
  partition can run out of space partway through the copy.
- **Broken firmware boot.** An EFI System Partition recreated as FAT16 is not readable by firmware
  that requires FAT32.

## Root cause

- `get_partition_details()` (hpcp.py:810) collects only `fs_type`, `fs_uuid`, and `fs_label` from
  `blkid -o export`. Geometry and features are never gathered.
- `write_partition_info()` (hpcp.py:926) builds each `mkfs` command from that same three-field set,
  so there is nothing else it *could* apply.

### Secondary bug: the FAT width branches are unreachable

`write_partition_info()` branches on `fs_type in ('fat32', 'fat16', 'fat12', ...)` and passes
`-F 16` / `-F 12` accordingly (hpcp.py:1019-1031). `blkid -o export` reports `TYPE=vfat` for every
FAT width, so `fs_type` is never `fat32`/`fat16`/`fat12` in practice and those branches are dead
code. `mkfs.vfat` is left to auto-select the width from the partition size, which is how a FAT32
ESP becomes FAT16.

The width is available from `blkid -p -o export` as `VERSION=FAT32`, which `hpcp` does not query.

## Goal

Recreate each destination filesystem with the source's geometry and feature set, so a clone is
mountable and bootable everywhere the source was, while remaining safe when the destination
partition is resized with `-ddr`.

## Non-goals

- Changing the file-copy phase, partition table handling, or `-ddr` resize semantics.
- Mirroring runtime state (mount counts, check intervals, dirty/needs-recovery flags).
- Creating filesystems `hpcp` already refuses to create (`zfs`, `cramfs`, `iso9660`).
- Mirroring btrfs data/metadata *profiles* (`single`/`dup`/`raid*`), which require the source to be
  mounted and belong to a later change.
- Making the copy fail when a source parameter cannot be reproduced — see Error handling.

## Approach

**Probe the source filesystem, translate the result into `mkfs` arguments.**

Two alternatives were rejected:

1. *Post-`mkfs` tuning* (`tune2fs -O`, `xfs_admin`) — block size, inode size, and cluster size
   cannot be changed after creation, which is precisely where the worst breakage is.
2. *Raw `dd` of each partition when sizes match* — defeats hpcp's sparse, file-level, parallel copy,
   and still leaves resized partitions broken.

The chosen mechanism was validated before this design was written:

- **ext**: `-b`/`-I`/`-m`/`-i` plus `-O` with an explicit `^feature` for every absent feature
  reproduced the source feature list exactly. A plain `-O list` is **not** sufficient — `mke2fs`
  merges it with the `mke2fs.conf` defaults, so absent features must be negated explicitly.
- **xfs**: `-b size= -s size= -i size=,sparse=,projid32bit=,nrext64=,maxpct= -n size= -m crc=,finobt=,rmapbt=,reflink=,bigtime=,inobtcount=`
  reproduced `blocksize`, `sectsize`, `inodesize`, `dirblklog`, `features_ro_compat`, and
  `features_incompat` exactly.
- **vfat**: `-F 32 -s 1 -S 512` preserved FAT32 and cluster size on a partition where `mkfs.vfat`
  would otherwise pick FAT16.
- **btrfs**: `btrfs inspect-internal dump-super` exposes `nodesize`, `sectorsize`, `csum_type`, and
  a decoded incompat flag list.

## Architecture

### Data model

`get_partition_details()` gains one key, `fs_params`. It already creates the loop device and
resolves `target_partition` for its `blkid` call, so the probe runs in that same block before the
loop is detached — no extra attach/detach cycle.

`fs_params` is an **opaque per-filesystem dict**. Only the builder registered for the same
`fs_type` interprets it, so each probe/builder pair is independently testable and there is no
shared schema to keep in sync across 14 filesystems.

### Registries

Two module-level dicts keyed by `fs_type`, in the style of the existing `_FS_FIX_COMMANDS`
(hpcp.py:706):

```python
_FS_PARAM_PROBES   = {'ext4': _probe_ext, 'ext3': _probe_ext, 'xfs': _probe_xfs, ...}
_FS_MKFS_BUILDERS  = {'ext4': _build_ext,  'ext3': _build_ext,  'xfs': _build_xfs,  ...}
```

- `probe_fs_params(device, fs_type) -> dict` — dispatches through `_FS_PARAM_PROBES`; returns `{}`
  on **any** failure (tool missing, parse error, exception). Probing must never break a copy.
- `build_mkfs_params(fs_type, fs_params) -> list[str]` — dispatches through `_FS_MKFS_BUILDERS`;
  returns `[]` for an unknown type or empty params.

### Integration

In `write_partition_info()`, one lookup at the top of the `if fs_type:` block:

```python
fs_params  = partition_infos[partition_name].get('fs_params') or {}
param_args = build_mkfs_params(fs_type, fs_params) if MIRROR_FS_PARAMS else []
```

then `command.extend(param_args)` immediately before each branch's existing
`command.append(target_partition)`. Roughly 14 single-line insertions; no branch body is
restructured.

The FAT width now arrives through `param_args` as `-F 32`, which fixes the ESP case regardless of
what `blkid` reports as `TYPE`. The `fat32`/`fat16`/`fat12` keys stay in the branch condition for
callers that pass them explicitly, but no longer carry the width decision.

## Scaling: size-dependent parameters are never copied verbatim

This is what keeps `-ddr` resize safe. Absolute counts from the source would be wrong — or
outright invalid — on a resized destination.

| Parameter | Handling |
|---|---|
| ext inode count | **Not** mirrored. Mirror bytes-per-inode via `-i`, computed as `block_count * block_size / inode_count`, which scales with the new size. |
| ext reserved blocks | **Not** mirrored as a count. Mirror the percentage via `-m`, computed as `reserved / block_count * 100`. |
| ext block count | Never passed; `mkfs` derives it from the partition. |
| xfs log size, agcount | **Not** mirrored — both scale with filesystem size. |
| xfs / ext / btrfs geometry and features | Size-independent; mirrored directly. |
| vfat cluster size, FAT width | Mirrored, but may be invalid on a much smaller target; the fallback in Error handling covers this. |

## Coverage

Every filesystem type already listed in `_FS_FIX_COMMANDS` gets a registry entry. Where the probe
tool is absent or its output cannot be parsed, the entry degrades to today's behavior plus one
warning line.

| Filesystem | Probe | Mirrored parameters |
|---|---|---|
| ext2 / ext3 / ext4 | `dumpe2fs -h` | `-b` block size, `-I` inode size, `-m` reserved %, `-i` bytes-per-inode, `-O` full feature delta |
| xfs | `xfs_info` | `-b size`, `-s size`, `-i size,sparse,projid32bit,nrext64,maxpct`, `-n size,ftype`, `-m crc,finobt,rmapbt,reflink,bigtime,inobtcount` |
| btrfs | `btrfs inspect-internal dump-super` | `-n` nodesize, `-s` sectorsize, `--csum`, `-O` features (flag-name mapping table) |
| vfat / fat / fat12 / fat16 / fat32 / msdos | `blkid -p -o export` + `fsck.fat -nv` | `-F` 12/16/32, `-S` bytes per sector, `-s` sectors per cluster, `-f` FAT count, `-R` reserved sectors |
| ntfs | `ntfsinfo -m` | `-c` cluster size, `-s` sector size |
| exfat | `dump.exfat` | `-c` cluster size, `-b` boundary alignment |
| f2fs | `dump.f2fs` | `-w` sector size, `-O` feature list |
| udf | `udfinfo` | `--blocksize`, `--udfrev`, `--media-type` |
| reiserfs | `debugreiserfs` | `-b` block size |
| jfs | `jfs_tune -l` | block size where reported (jfs is effectively fixed at 4096; likely a no-op entry) |
| hfs / hfsplus | `fsck.hfsplus -n` | `-b` block size where reported |
| minix | superblock magic | `-1` / `-2` / `-3` version, `-i` inode count |
| bfs, ufs, swap | — | `{}` today; entries reserved so the dispatch is total |
| zfs, cramfs, iso9660 | — | Unchanged; `hpcp` already declines to create these |

### ext feature handling

- The feature universe is **the source's own feature list ∪ a curated set of commonly default-on
  features** (`64bit`, `metadata_csum`, `metadata_csum_seed`, `dir_index`, `orphan_file`,
  `fast_commit`, `has_journal`, `extent`, `flex_bg`, `huge_file`, `dir_nlink`, `extra_isize`,
  `sparse_super`, `large_file`, `ext_attr`, `resize_inode`, `filetype`, `casefold`, `project`,
  `quota`, `verity`, `encrypt`, `bigalloc`, `inline_data`, `ea_inode`, `mmp`, `stable_inodes`,
  `uninit_bg`, `sparse_super2`, `meta_bg`).
- Features present in the source are emitted plain; curated features absent from the source are
  emitted as `^feature`. Names unknown to the local `mke2fs` therefore appear only when the source
  actually has them, and the fallback covers that case.
- Runtime-state flags are filtered out and never passed to `mke2fs`:
  `needs_recovery`, `orphan_present`, `has_snapshot`, `journal_dev`, `shared_blocks`.

## CLI

```text
-nfp, --no_fs_param_mirror
    Do not mirror source filesystem parameters (geometry, features) in dd mode.
    Create destination filesystems with mkfs defaults, preserving only label and UUID.
```

`action='store_true'`, default off (mirroring **on** by default). The value is assigned to a module
global `MIRROR_FS_PARAMS` in `main`, following the existing `RANDOM_DESTINATION_SELECTION` /
`NO_CREATE_DIR` pattern rather than threading a parameter through `hpcp()` →
`create_dd_dest_part_table()` → `create_partition_table()` → `write_partition_info()`.

## Error handling

Mirrored parameters can be legitimately invalid on a resized destination — 1024-byte blocks on a
filesystem grown past 16 TiB, FAT32 on a partition too small for it, a cluster size the new size
does not permit. The copy must not die for that reason.

1. Run `mkfs` **with** the mirrored arguments.
2. On a non-zero return code, `eprint` a warning naming the filesystem, the partition, and the
   rejected arguments.
3. Re-run the identical command **without** the mirrored arguments — today's behavior.

`run_command_in_multicmd_with_path_check()` returns output lines and discards the return code, so
this needs a small `_run_mkfs_with_fallback(command, param_args, fs_type, target_partition)` helper
built on multiCMD's `return_object=True`, the same pattern `detach_loop_device()` (hpcp.py:3568)
already uses.

Probe failures are separate and quieter: `probe_fs_params()` returns `{}`, `build_mkfs_params()`
returns `[]`, and creation proceeds exactly as it does today with a single informational line.

## Testing

Two tiers, because not every filesystem can be verified on a given host.

**Tier 1 — parser unit tests (`tests/test_dd_fs_params.py`).** Recorded tool output is fed to each
probe function and each builder as fixtures. Pure functions, no root, no loop devices; covers all
listed filesystems including those whose tools are not installed. Also asserts:

- the ext feature delta emits `^` for absent curated features and filters runtime-state flags,
- bytes-per-inode and reserved-percentage arithmetic,
- that `build_mkfs_params()` returns `[]` for `{}` and for unknown types,
- `-nfp` parses and defaults to off.

**Tier 2 — live round-trip integration test.** Build a small multi-partition image with
non-default filesystems, run dd mode against it, and assert the destination's probed parameters
equal the source's. Skipped unless running as root with the required tools present
(`pytest.mark.skipif`).

Honest coverage split:

- **Round-trip verified:** ext2/3/4, xfs, btrfs, vfat — and ntfs, exfat, f2fs where their tools are
  installed.
- **Fixture-tested only, not round-trip verified:** jfs, reiserfs, udf, hfs/hfsplus, minix, bfs,
  ufs. Their probes and builders are written against documented tool output formats. This
  limitation is stated here deliberately rather than implied away.

## Success criteria

- A dd-mode clone of the verification image reproduces the source's FAT width and cluster size,
  ext block/inode size, feature set and reserved percentage, and xfs geometry and feature flags.
- `-ddr`-resized partitions still create successfully, with scaled (not copied) inode ratio and
  reserved percentage.
- A filesystem whose mirrored parameters `mkfs` rejects still gets created with defaults, with a
  warning, and the copy completes.
- A missing probe tool produces no failure and no behavior change versus today.
- `-nfp` restores today's behavior exactly.
- Non-dd copy paths are untouched.

## Files touched

- `hpcp.py` — probes, builders, registries, `fs_params` key, `write_partition_info()` integration,
  `_run_mkfs_with_fallback()`, the `-nfp` flag, new binaries added to `_binCalled`, version bump to
  `9.59`.
- `tests/test_dd_fs_params.py` — new.
- `README.md` — document `-nfp` and the mirroring behavior.
