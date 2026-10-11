# Patch apply: post-delete repair and case handling (group A of the 9.67 follow-ups)

## Problem

hpcp patch apply (single-filesystem `-siff` patches and the sections of disk patches) classifies every
row against the target *before* anything changes (`check_patch_target`): an A / M row whose target
already holds the new state is skipped. `_apply_rows` then runs the deletes and type-change removals
and writes only the rows that were not skipped. A skipped row is never looked at again, so a delete
that removes the entry a skipped write depends on loses it silently:

- a case-only rename (`D EFI/BOOT/x` + `A EFI/Boot/x`) on a case-insensitive target (FAT, exFAT,
  HFS+, case-insensitive NFS servers, ext4/f2fs casefold directories, ZFS insensitive datasets);
- a symlinked directory alias inside the target;
- a forced (`-fa`) type change whose `rmtree` removes an entry reached through an alias.

9.67 closed these with refusals: filesystem-type allow/deny lists, a case-only-rename refusal that
fails closed on unknown types, an inode-identity refusal, an old-patch case-collision refusal, an
extra verify of skipped rows on case-insensitive sections, and case-insensitive name pairing at
generation (`str.casefold()`). The release reviews showed the approach keeps leaking or over-refusing:
unforceable refusals on case-sensitive filesystems it does not know (gfs2, ocfs2, glusterfs, virtiofs,
9p…), dependence on inode numbers being stable (not true on vfat, exFAT, FUSE, CIFS), `casefold()`
merging names FAT keeps apart (`ß` / `ss`), aliases the key-prefix scoping misses.

Related deferred items in the same area: single-filesystem patches onto FAT targets still compare
mode and owner (which FAT takes from mount options); FAT mtimes depend on the kernel timezone or
`tz=` / `time_offset=` mount options; replacing a file on FAT drops its hidden / system / read-only
attributes; case-sensitive HFSX reports the same type (`hfsplus`) as case-insensitive HFS+; unforceable
pre-flight refusals run after the "Apply anyway?" prompt; `_path_mounts` can pick a mount that a later
mount on a parent hides.

## Goal

A patch applies correctly on any filesystem, including case-only renames, without refusals that
`-fa` cannot override and without depending on filesystem-type lists or inode stability for
correctness. Released as hpcp 9.68.

## Decisions (made with the user)

1. Add a post-delete repair step and drop the 9.67 refusals it makes redundant.
2. Generation stops pairing names case-insensitively; a case-only rename is an ordinary D + A pair
   and apply leaves the node with the golden's exact spelling.
3. Content-only sections tolerate mtime differences shaped like a timezone shift.

Defaults (presented, not objected to): keep a replaced FAT file's attributes; single-filesystem
patches onto content-only filesystems (including a mount below the target) are content-only;
single-filesystem generation is content-only when either side is on a content-only filesystem; HFSX
is told apart by its volume signature; `_path_mounts` matches by device number; the remaining
case-insensitive comparison is per character (no `casefold()`); unforceable pre-flight refusals run
before the conflict prompt.

## Design

### Part 1 — the apply engine

1. **Repair step** (in `_apply_rows`). `_apply_rows` also receives the A / M rows that were skipped
   as already done. After the delete loop and the type-change removal loop, if at least one entry was
   actually removed, it lstats the target of every skipped A / M row (no hashing). A row whose target
   is now missing, or whose type no longer matches the row's new state (file / directory / symlink),
   joins the write set as an ordinary, unforced row: directories are created before files as today,
   files come from the payload, symlinks are recreated. Repaired rows are verified like every written
   row (`_verify_rows`). The section summary gains `restored N` (N = repaired rows; 0 when none).
2. **Removed**: `_refuse_case_only_renames` and `_CASE_SENSITIVE_TYPES`; `_same_entry_pair` and
   `_caseless` (only `_same_entry_pair` uses it; `_names_removed_first` stays, the outside-symlink
   check uses it too); the old-patch `_case_collision` refusal in `apply_disk_patch`; the extra
   verify of skipped rows on case-insensitive sections in `_apply_section`. Docstrings and the README
   that describe them are updated.
3. **Prompt order**: `_preflight_section` (outside-symlink, missing / unsafe payload refusals) runs
   before `_resolve_section_conflicts`, in `apply_patch_tree` and `apply_disk_patch`, so the user is
   never asked to confirm an apply that a non-forceable refusal then aborts.
4. **Contradictory patches stay loud**: on an aliased target where a delete and a write disagree
   (`D x/f` + `A y/f` with `y -> x`), the write lands and the delete's verify fails; the apply exits
   non-zero. Not refused up front.

### Part 2 — case handling, content-only targets, FAT details

1. **Generation uses exact names.** `_partition_changes` no longer folds names; `_fold_case_keys`,
   `_case_folded_names` and `_case_collision` are removed. A case-only rename in any section is an
   ordinary D + A pair whose payload is carried.
2. **Planning on a known case-insensitive target** (FAT family, exFAT, HFS+ that is not HFSX):
   `check_patch_target` compares names with `_fold_name(name)` — each character upper-cased only when
   `c.upper()` is a single character, otherwise kept (so `ß` never becomes `SS`). The fold applies to
   (a) the D-directory child check (`_unlisted_children`, as today), (b) the "already replaced" lookup
   for D rows (`added_by_name`), (c) `removed_names` for A / M rows (`_under_removed`). A case rename,
   with or without a content change, is therefore planned as delete + write (no `-fa`), and a re-run
   after a successful apply skips both rows. On a case-insensitive target hpcp does not recognise, the
   repair step keeps the files; a case rename with a content change there is a conflict (needs `-fa`).
3. **Which rows are case-insensitive / content-only.**
   - Disk sections: from the partition's `fs_type` in `disk.tsv` (as today).
   - Single-filesystem apply: from `/proc/self/mountinfo`. `_path_mounts` picks, among the entries
     whose mount point is the path or an ancestor, the one whose device number matches
     `os.stat(path).st_dev` (falling back to the longest mount point when none matches), so a mount
     hidden by a later mount on a parent is not chosen. Rows under a mount below the target (e.g.
     `boot/efi/` when applying to `/`) get that mount's classification: the patch is split into one
     sub-plan per mount scope (the target's own scope plus each mount below it that holds rows), each
     with its own `metadata` / case flags, applied through the existing multi-section machinery with a
     single shared conflict prompt and a single summary.
4. **Single-filesystem generation** (`process_get_diff_image`) compares content only when either side
   sits on a content-only filesystem (FAT family, exFAT, NTFS): images probed with `_probe_fs_type`,
   directories classified through mountinfo.
5. **FAT timestamps.** `states_equal(..., metadata=False)` also treats two file states as equal when
   size and hash match and the mtimes differ by a whole multiple of 15 minutes within ±24 hours
   (a timezone-shaped offset). Any other mtime difference still counts.
6. **FAT attributes.** When a file on a FAT-family target is replaced, its existing attribute byte is
   read (`FAT_IOCTL_GET_ATTRIBUTES`) before the rename and the hidden / system / read-only / archive
   bits are set on the replacement afterwards (`FAT_IOCTL_SET_ATTRIBUTES`). Best effort: a failure
   prints a warning, never fails the row. New files get the filesystem's defaults. (Linux exposes no
   such ioctl for exFAT; exFAT is unchanged.)
7. **HFSX.** An `hfsplus` filesystem is case-sensitive when its volume header signature (2 bytes at
   offset 1024 of the device) is `HX`; `H+` or an unreadable header keeps it case-insensitive. Disk
   sections read the partition device; single-filesystem apply reads the mount's source device.
8. **Docs.** README "Patch Images" / "Disk patches" notes and the affected docstrings describe: case
   renames propagate; content-only rules (what is compared and written); the timezone tolerance; FAT
   attributes kept on replace; the repair step and its `restored` count.

## Compatibility

- Patch formats are unchanged (manifest v1 / v2, `changes.tsv`, `disk.tsv`).
- Patches made by 9.67 (case-only renames folded away) and by 9.65 / 9.66 apply with the new code.
- Behaviour visible to users: case-only renames apply instead of being refused; `restored N` in the
  summary; single-filesystem patches onto FAT targets compare content only; the prompt no longer
  precedes non-forceable refusals.

## Testing

Real-filesystem tests (root + loop where needed; scratch only under /tmp; probes `set -euo pipefail`):

- vfat case rename, single-filesystem and disk patch: names only, and names + content change (no
  `-fa`); target ends with the golden's spelling and content, `failed 0`; a re-run skips every row.
- Repair on an unrecognised case-insensitive target: a vfat image whose mountinfo type lookup is
  monkeypatched to report `nfs4` — files are restored, summary shows `restored N`.
- Symlink alias inside the target (`y -> x`): skipped row lost to a delete is restored; contradictory
  `D x/f` + `A y/f` fails at verify, non-zero exit.
- Forced type change (`-fa`) over an aliased skipped row: the row is restored.
- ESP mounted below the target (`-d /`-style test tree with a vfat mount under it, umask differing
  from the generator): content-only, no mode conflicts.
- `_path_mounts` with a hidden mount (entries `[/x/esp vfat, /x tmpfs]`): picks tmpfs.
- Timezone tolerance: a 2 h offset with equal size + hash is equal; a 7 min offset is not.
- FAT attributes: a hidden + system file keeps both bits after replacement.
- HFSX (`mkfs.hfsplus -s`) is case-sensitive; plain HFS+ is case-insensitive.
- Prompt order: a patch with conflicts and an outside-symlink refusal raises without prompting.
- Existing suites stay green; removed-code tests are deleted or rewritten to the new behaviour.

## Out of scope (later rounds)

Group B (disk-write hardening: O_EXCL / flock, cache-bypassing read-back, refusals before hashing,
cciss in the remaining helpers, non-ASCII labels, GPT names with apostrophes, duplicate probes) and
group C (`-e` anchoring for `-dd -diff`, input images in single-filesystem patches, payload re-check
cost, `/x/out` test guard). The README casefold sentence is rewritten here, by Part 2 item 8.
