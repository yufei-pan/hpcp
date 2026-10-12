# Disk-write safety and FAT full hashes (group B1 of the 9.67 follow-ups)

## Problem

Group A (9.68, `docs/superpowers/specs/2026-10-11-apply-repair-and-case-handling-design.md`) made patch apply
repair and verify instead of refusing. These gaps remain where a node's data can be destroyed, or overwritten
without a report, or where a check cannot see what it claims to check:

- A forced (`-fa`) removal of a directory runs `shutil.rmtree` (`_apply_rows`, `remove`). That covers a D row on a
  non-empty directory, and a directory replaced by a file or symlink. `rmtree` follows into any filesystem mounted
  below the directory, bind mounts included, and deletes its contents.
- A skipped A / M row (the target already holds its new state) can be overwritten by another row's write that
  reaches the same entry through a symlinked directory (`A y/f` skipped, `M m/x/f` written, `y -> m/x`). Verify
  covers only written and restored rows, so the apply ends with `failed 0`. 9.67 had the same gap. Group A closed
  only the case-only-name form of it (`_apply_section`, the fold-equal sibling check).
- In disk apply, `_partition_in_use` reads this process's mount table, so it misses mounts in another mount
  namespace and mounts hidden under another mount. Partitions are not held while raw areas and identity values are
  written, so something can mount them in between. The whole disk is not locked, so udev can re-read the partition
  table in the middle of the write phase.
- The read-back after each raw write (`_hash_area`) reads through the page cache, so it cannot detect a write that
  never reached the device.
- The timezone tolerance (group A) trusts a hash only when it covers the whole file. Patches hash the last 64 KiB
  by default (`-hs`), so on a node whose FAT ESP is read under another timezone, every changed ESP file over
  64 KiB (shim, GRUB, kernels, initrds) is a conflict.

## Goal

Close these gaps without new unforceable over-refusals, and keep group A's rule that a node's data is never lost
or altered while the apply ends `failed 0`. Released as hpcp 9.69. 9.68 is pushed but not on PyPI; whether to
publish it separately first is the user's call at release time.

## Decisions (made with the user)

1. Full hashing applies to FAT (vfat / msdos) only, not exFAT or NTFS. NTFS stores UTC times, and exFAT records
   the UTC offset, so neither shows the timezone shift; both can be large data partitions. The timezone tolerance
   is narrowed to FAT too. (This revises an earlier "FAT, exFAT and NTFS" choice, after a side note pointed out the
   cost on 100+ GB NTFS partitions.)
2. A forced directory removal deletes around mounts: everything on the directory's own filesystem is removed,
   no mount is ever entered, and a row that cannot complete because of a mount fails loudly while the rest of the
   patch applies. The alternative, refusing the whole apply up front, was declined.
3. Design sections 1–3 below were presented and approved as written here.

## Design

### Part 1 — the apply engine

1. **Delete around mounts.** Forced removals of a real directory (`_apply_rows` `remove` for rows in
   `forced_rows`) no longer call `shutil.rmtree`. A new helper walks the tree bottom-up and removes files,
   symlinks and empty directories on the directory's own filesystem:
   - Before the walk it reads the mount table once (`_mountinfo_entries`). It never enters, and never removes, a
     directory that is the mount point of any entry (visible or hidden, bind mounts included). This applies to the
     starting directory too: if the row's own directory is a mount point, nothing under it is touched.
   - As a backstop it never enters a directory whose `st_dev` differs from the starting directory's. This catches
     a mount made after the table was read.
   - Symlinks are unlinked, never followed. Errors on single entries are collected; the walk continues.
   - When a mount point (or an error) leaves the directory in place, the row is not complete. Verify reports it
     (`Patch verify error: D opt/app/: ... still holds the mount point opt/app/data`, or the type-change row's
     copy error and verify failure), the exit is non-zero, and the rest of the patch applies. As in group A, a
     removal that ran at all counts as a removal for the repair step.
   - Planning warns before anything changes. Each forced row whose tree holds a mount point is listed with the
     conflicts, before the "Apply anyway?" prompt, or before the first change when `-fa` skips the prompt. Example:
     `Note: forced removal of opt/app/ stops at the mount at opt/app/data; that row will fail verify`.
   - Non-forced D directory rows keep `os.rmdir`.
2. **Skipped rows overwritten through an alias.**
   - **Recording:** `check_patch_target` already lstats the target of every row it plans to skip. It records each
     skipped A / M row's identity: `(st_dev, st_ino, file type)`.
   - **Re-check in the section:** in `_apply_section`, when anything was written, removed or restored, every
     skipped A / M row is lstat'ed again. Those whose identity changed, or that are now missing, are verified with
     `_verify_rows` under the plan's settings. So is every skipped directory row (`d:` new state) whatever its
     identity, because a directory changed in place keeps its inode. A mismatch is a `Patch verify error` counted
     in `failed`.
   - **Replaces:** this replaces group A's fold-equal sibling selection (I1), which it covers. Its vfat test
     (`EFI/X` written, `EFI/x` skipped) must still fail loudly.
   - **Across sections:** the cross-section final pass (`_verify_across_sections`) also re-checks the skipped rows
     of every plan that ran: identity changed, missing, or a directory.
   - **Cost:** one lstat per skipped row, plus a metadata check per skipped directory, only when something was
     written, removed or restored.

### Part 2 — the disk write phase (`apply_disk_patch`)

"Raw/identity partitions" below means the partitions that will get a raw-area write or an identity change
(filesystem UUID or label via `_fs_identity_commands`).

1. **Up-front exclusivity check.** During planning, before `_resolve_section_conflicts` (the prompt), hpcp opens
   each raw/identity partition with `O_RDONLY | O_EXCL | O_CLOEXEC` and closes it at once. If the open fails with
   `EBUSY`, hpcp refuses, and nothing is applied:
   `Disk in use: partition N is mounted, active swap or held (LVM, md, dm), in this or another mount namespace; nothing applied`.
   This cannot be forced. The existing `_partition_in_use` refusal stays first, for its clearer messages
   ("mounted at /x"). The whole disk is not opened exclusively: its other partitions may be in legitimate use, and
   hpcp itself mounts the file sections later.
2. **Holding the disk during writes.** After the sections are released and the cross-section passes ran, and
   before the first raw write:
   - **Lock:** hpcp opens the whole-disk device read-only and takes `flock(LOCK_EX)` (systemd's block-device
     locking convention; udev postpones events for a locked disk). It retries for up to 10 s while udev holds a
     shared lock. If it still cannot lock, it stops: raw areas, boot code and identity changes are not applied,
     reported like a failed section.
   - **Holds:** hpcp opens every raw/identity partition `O_EXCL` and keeps those opens through the raw writes and
     identity commands. If a hold cannot be taken (something took the partition since the up-front check), it
     stops the same way.
   - **Writes:** raw writes (`_write_area`) and identity commands run while the holds are kept. A probe on
     2026-10-11 (loop disk, kernel 7.0) showed these all succeed while the partition is held and the disk is
     locked: `tune2fs -U`, `e2label`, `xfs_admin -U / -L`, `fatlabel` (serial and label), and `sgdisk -c / -u` on
     the whole disk. `mount` of a held partition fails. The plan probes every other identity tool hpcp runs that is
     installed (`btrfstune`, `btrfs filesystem label`, `exfatlabel`, `swaplabel`, `udflabel`, `jfs_tune`,
     `reiserfstune`). The tools the probe finds failing under a hold go in a fixed list in the code; those run
     with that partition's hold dropped for that one command. There is no runtime retry.
   - **Release:** in a `finally`, so every error path releases them. Holds are closed first. Then hpcp asks the
     kernel to re-read the partition table (`BLKRRPART` on the locked disk descriptor; when that fails because a
     partition is in use, `partx -u`). Then the lock is released and the descriptor closed, then
     `_settle_partitions` waits for udev. The kernel cannot re-read the table while partitions are held, which is
     why the holds go first.
3. **A read-back that reaches the disk.**
   - `_hash_area` gains a direct mode: `O_DIRECT` reads into a page-aligned buffer (`mmap`), aligned to the
     device's logical block size (`BLKSSZGET`; 4096 when unknown). The requested range is widened to block
     boundaries and the extra bytes are dropped, so the 440-byte MBR boot code is read as its enclosing block.
   - It is used for the read-back right after each raw write and for the final raw verify.
   - Where `O_DIRECT` is refused (`EINVAL` at open or read), hpcp flushes the device's buffer cache (`BLKFLSBUF`,
     then `posix_fadvise(POSIX_FADV_DONTNEED)`) and reads normally.
   - Planning-time hashes of the node's raw areas and payload hashes stay buffered.

### Part 3 — full hashes for FAT

1. **Generation.** A file row is hashed over the whole file, whatever `-hs` says, when the file sits on FAT on
   either side:
   - **Disk patches:** every file in a section whose partition `fs_type` is `vfat`, `msdos` or `fat`, on the golden
     or the base.
   - **Single-filesystem patches:** every file whose path, in the new tree or the base tree, lies on a FAT mount.
     That includes mounts below the source roots (the golden's `/boot/efi` when diffing `/`), classified through
     the mount table (`_mount_scopes`), and image roots that hpcp mounted. This closes group A's deferred item
     "generation classifies only the two roots" for FAT.
   - **Everything else:** the tail hash, or a full hash with `-fh` as today.
2. **Format.**
   - In a patch whose manifest has `full_hash` 0, a full-file hash is written as `F` followed by the hex digest
     (`f:size:mtime:mode:uid:gid:F<hex>:xattr`). Digests are lowercase hex, so the marker cannot collide with one.
     A row's old and new states are hashed the same way.
   - In a `-fh` patch every hash is already full, and no marker is written.
   - A patch holding at least one marked state is written with `format_version` 3, whatever its layout.
     `DIFF_FORMAT_VERSION` becomes 3, and the reader accepts 1–3. hpcp 9.68 and older refuse a version-3 patch up
     front ("newer than supported"). Patches without marked states keep version 1 (single filesystem) or 2
     (disk), readable by older hpcp.
   - Patches from 9.65 to 9.68 apply exactly as before.
3. **Apply.**
   - **Hashing:** for a row whose recorded old or new state carries the marker, the node's file is hashed in full
     and its state marked the same way. That covers planning (`check_patch_target`), the payload check and verify,
     so states compare like with like. The node's own classification does not matter.
   - **Timezone tolerance:** `states_equal` applies it only on FAT sections and scopes. The caller passes whether
     the section or scope is FAT; exFAT and NTFS lose the tolerance. It applies only when the hash covers the
     whole file: a marked state, a `-fh` patch, or a file no bigger than the patch's hash size.
4. **Remaining limit (documented).** If neither source tree had the files on FAT (a golden built as a plain
   directory tree), their rows carry tail hashes. On a node whose ESP is FAT, the tolerance then covers only files
   up to the hash size; bigger timezone-shifted files are conflicts.

### Docs and version

- README "Patch Images" / "Disk patches" notes describe:
  - full hashing of FAT files;
  - the timezone tolerance narrowed to FAT;
  - forced removal stopping at mounts, and its warning;
  - the exclusive holds and disk lock during disk writes, and the new "Disk in use" refusal;
  - format version 3 and what older hpcp does with it.
- Docstrings of the changed functions are updated.
- `version = '9.69'`, `COMMIT_DATE`, `test_version_bumped`.

## Compatibility

- Old patches (format 1 / 2, unmarked hashes) apply unchanged.
- New patches without FAT files are format 1 / 2 and still apply on 9.65–9.68.
- New patches with FAT files are format 3: 9.68 and older refuse them with "newer than supported".
- Behaviour users will see:
  - `-fa` no longer deletes into mounted filesystems; such rows fail verify, after a warning;
  - a raw/identity partition in use in another namespace is now refused up front;
  - FAT files that differ only by a timezone offset are skipped regardless of size;
  - exFAT / NTFS no longer get the timezone tolerance (they never needed it).

## Testing

Real-filesystem tests run as root with loop devices and mounts. Scratch files go only under the test's `tmp_path`.
Nothing is left mounted or attached.

**Probe safety rule:** any ad-hoc root probe goes in a script file run with `bash <file>`. In this harness,
`set -euo pipefail` typed at the top of a shell command is ignored; it works only in a script file or a `bash -c`
child. Every path variable gets a `: "${VAR:?}"` guard and a prefix check (`case "$D" in /tmp/claude-0/*)`).

- **Delete around mounts:**
  - a tmpfs mounted under a directory that a `-fa` D row removes, with a file inside: the tmpfs file survives, the
    row fails verify, other rows apply, and the warning precedes the prompt;
  - the same with a bind mount of a directory of the same filesystem;
  - a forced directory-to-file type change over a directory holding a mount: the mount survives and the row fails;
  - the row's own directory is a mount point: nothing under it is removed.
- **Alias overwrite:**
  - `A y/f` skipped and `M m/x/f` written with `y -> m/x`, in one scope and across two scopes: `failed 1` both
    ways;
  - a skipped directory whose mode another row changes through an alias: `failed 1`;
  - an ordinary apply with many skipped rows: `failed 0`;
  - group A's vfat case-sibling test still fails loudly.
- **Exclusivity:**
  - a raw/identity partition mounted in this namespace, mounted only inside another namespace (a held
    `unshare -m` child), held `O_EXCL` by another process, or active swap: refused up front, nothing written;
  - a hold that fails at write time (simulated): raw areas, boot code and identity changes are not applied, and
    it is reported like a failed section;
  - each installed identity tool sets its value while the partition is held;
  - while hpcp is in the write phase, `flock -n` on the disk from another process fails (via a test hook);
  - holds and the lock are released on error paths.
- **Read-back:**
  - an area is written through the loop partition, then the backing image's bytes at that offset are changed
    directly; the direct read sees the change;
  - the 440-byte boot code on a 512- and a 4096-byte logical-block loop device (`losetup --sector-size 4096`);
  - the fallback when `O_DIRECT` is refused (simulated `EINVAL`).
- **Full hashes:**
  - a disk patch whose FAT file is bigger than the hash size has marked states and format 3;
  - a node copy changed only in its head is a conflict;
  - a timezone-shifted identical copy is skipped;
  - exFAT / NTFS rows are unmarked and get no tolerance;
  - single-filesystem generation with a vfat mount below the source root marks the rows under it;
  - a patch without FAT files stays format 1 / 2;
  - a format-3 patch is refused by a reader whose `DIFF_FORMAT_VERSION` is 2 (constant patched in the test);
  - 9.65–9.68 patches in the existing tests still apply.
- Existing suites stay green.

## Review focus

1. **No new silent path.** Can any apply end `failed 0` while a node entry was lost or altered? Group A's lesson
   applies: a multi-scope or multi-section apply must end exactly like the same patch applied as one scope.
2. **The delete walk.** It must never enter a mount (bind mounts, mounts made mid-walk, the starting directory
   itself) and never follow a symlink.
3. **Holds and lock.** They are released on every path, never taken on a partition hpcp itself has mounted at
   that moment, and never deadlock with udev.
4. **Format 3.** It is written only when a marked state exists. Marked and unmarked hashes are never compared with
   each other.
5. **Direct reads.** The alignment arithmetic holds at area edges (MBR, partition ends, 4Kn devices).

## Out of scope (later rounds)

- **B2:** non-ASCII labels mangled by `blkid -o export`; GPT names with apostrophes or spaces truncated; cciss names
  in the remaining helpers; refusals that come only after hashing; duplicate blkid / udev settle / stat passes.
- **C:**
  - `-e` anchoring at partition roots for `-dd -diff`;
  - input images in single-filesystem patches;
  - payload re-check cost;
  - the `/x/out` test guard;
  - group A's loud or rare leftovers: an automounted ESP classified as autofs, FAT ioctl numbers on
    ppc / mips / sparc, a written row removed by a later scope's delete reported but not restored, inode reuse
    after a double replacement, and the per-task minors in the group A ledger.
