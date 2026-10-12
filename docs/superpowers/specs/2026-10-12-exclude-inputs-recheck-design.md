# Exclude anchoring, inputs kept out of patches, cheaper payload re-check (round C1 of the 9.67 follow-ups)

## Problem

Patch generation (`-diff` and `-dd -diff`) has four problems left over from 9.67:

- **`-e` patterns on mounted sources.**
  - When the new side is a filesystem hpcp mounts or reads whole, patterns are matched against the temporary path
    hpcp mounts it at (`/tmp/tmpXXXX/var/log/...`). This covers `-diff -si IMG` and every filesystem section of
    `-dd -diff`.
  - So `-e '/var/log/*'` silently excludes nothing, and only `*/var/log/*` works. The README says so for
    `-dd -diff` only.
  - Worse, `-e '*/tmp/*'` matches the temporary mount path itself and excludes the whole filesystem.
  - A live partition in `-dd -diff` is bind-mounted at a temporary path too (`_mount_partition`), so it has the same
    problem.
- **Inputs inside a listed tree.**
  - `-dd -diff` already leaves its input disk-image files out of the listings (9.67). Single-filesystem `-diff`
    does not.
  - `hpcp -diff / -di /root/golden-v1.img` makes the base image an added file, installed on every node.
  - `hpcp -diff / -d /mnt/oldroot` adds the whole old tree under `mnt/oldroot/`.
  - A new directory inside the base directory gives delete rows for it.
- **Payload re-check cost.** After the payload is copied into the patch, `_check_payload` hashes every copy again and
  compares it with its row. The copy is hashed as the row was. With `-fh`, that is a second full read of the whole
  payload.
- **Housekeeping.**
  - Stubbed generation tests in `tests/test_disk_patch.py` pass `/x/new`, `/x/old` and `/x/out` as paths. When a
    stub regressed once, a test created and mounted `/x/out` on the real filesystem.
  - Leftovers from B2:
    - the old-kernel btrfs warning tells the user to wait until "the source is detached";
    - `_lsblk_disk_rows` raises `IndexError` on empty `lsblk` output;
    - a raw-size loop in `process_get_disk_diff_image` is dead code;
    - the `ID_FS_UUID` fallback in `get_partition_details` is untested;
    - README line 129 overstates when the raw-size refusal comes.

## Goal

- An `-e` pattern written as a path inside the filesystem works for every mounted source, the same way whether
  that source is live or not.
- No input of the run ends up in its own patch.
- Generation reads the payload a second time only for files that may have changed.
- The housekeeping items are closed.

This ships as hpcp 9.71. 9.68–9.71 are not on PyPI; publishing needs the user's go.

## Decisions (made with the user)

1. Round C is split. This spec is C1: patch generation plus the B2 leftovers. Round C2 gets its own spec later:
   - md array partitions in `-dd`;
   - the rare apply cases from groups A and B1;
   - path components above a forced removal's top.
2. A `/`-anchored `-e` pattern on a mounted source is matched against the path inside that filesystem. The other
   options were the node's mount point from `/etc/fstab`, or a warning only.
3. Payload re-check: read-only sources are skipped; live sources get a stat check. The other options were skipping
   read-only sources only, or no change.
4. Design sections 1–3 were presented and approved as written here.

## Design

### Part 1 — `-e` matched inside each mounted filesystem

1. **Which sources.** A source counts as mounted when it is one of these:
   - the new side of `-diff -si IMG`;
   - each filesystem section of `-dd -diff`, whether hpcp mounted the partition or it is live.

   For these, every `-e` and `-x` pattern is matched against `/` + the entry's path relative to that filesystem's
   root. The root itself is `/`.
2. **Patterns.**
   - `-e '/var/log/*'` excludes `var/log/...` on that filesystem. `-e '/EFI/Linux/*'` works on an ESP section.
   - `*/var/log/*` keeps matching, because `*` also matches an empty prefix.
   - `-e '*/tmp/*'` excludes only `/tmp/...` inside the filesystem.
   - A separate `/var` partition holds `log/...` at its root, so there the pattern is `/log/*`.
3. **Base side.** The base side keeps honouring the new side's patterns. It is matched against the same `/`-rooted
   path, as today it is matched against the new side's path.
4. **Directory sources are unchanged.** For `-diff DIR` (root or store layout), patterns are matched against real
   paths, as in copy mode.
5. **Excludes hpcp adds itself keep working.** This run's mount points (`_mount_point_excludes`) and input files are
   still excluded. They are expressed in the same `/`-rooted terms.
6. **No walk of excluded directories.** The new side skips an excluded directory without walking into it, as it
   does today with physical paths, so listing time does not regress. Both listings match through the same
   physical-to-logical path mapping, so the base side, which today walks excluded directories and filters them
   afterwards, skips them too.
7. **Compatibility.** `*/…` patterns still match inside a mounted filesystem. Two kinds of pattern change, both
   written against hpcp's temporary mount path (`-diff` first shipped in 9.65, so only 9.65–9.70 are affected):
   - patterns that named the path under it, such as `/tmp/*/var/log/*` or `/tmp/tmp*/…`, matched in 9.65–9.70 and now
     match nothing: write them as `/…` paths inside the filesystem (`/var/log/*`);
   - a pattern that matched the temporary path itself (`*/tmp/*`, `/tmp/*`) and so excluded the whole filesystem now
     matches only inside it.

   Patches are unchanged: `-e` affects only which rows exist.
   (Corrected after the design was approved. The approved text said a live `-dd -diff` partition was matched at its
   live mount point, so `/boot/efi/...` patterns would stop matching. `_mount_partition` bind-mounts a live partition
   at a temporary path, so it never was.)

### Part 2 — inputs kept out of the listings

1. **What counts as an input.** For single-filesystem `-diff`, the inputs are:
   - the `-si` image file;
   - the `-di` image file;
   - the base directory (`-d`);
   - the new directory (the source).
2. **The rule.** Each input whose real path lies strictly inside a directory side (the new or the base directory)
   is left out of both listings. It is excluded the same way this run's temporary mount points are: as a pattern for
   that path relative to the tree, in the terms Part 1 uses. An image side's mount cannot contain a host path, so it
   needs no check.
3. **Cases this covers:**
   - a `-di` image inside the new directory;
   - a `-si` image inside the base directory;
   - the base directory inside the new directory;
   - the new directory inside the base directory.
4. **`-dd -diff` is unchanged.** It already does this for its disk-image files on live partitions. Its excludes move
   to the Part 1 terms with everything else.
5. **Messages and docs.** Nothing new is printed. The README says that inputs inside a listed tree are left out.

### Part 3 — payload re-check only where needed

`_check_payload` keeps its "incomplete" check for every A/M file row: a copy with the recorded size must be on a
volume, or the source must have resized. The hash comparison of the copy with the row, today run for every row,
runs only for the rows below.

1. **Read-only sources: never.** The source file lies on a filesystem whose superblock is read-only (`ro` in
   mountinfo's super options, the field after the source). The filesystem is found from real paths, not through
   `_mount_scopes`: it is the visible mount at the file's own real path (a bind mount of one file), else the deepest
   visible mount holding the real path of its directory. This covers:
   - `-si` images;
   - partitions `-dd -diff` mounts itself;
   - a golden the user mounted read-only.

   A read-only bind mount of a writable filesystem does not count. `_mountinfo_entries` gains the super options.
   A read-only row skips the window and filesystem-type tests but still compares the scan stat with a fresh `lstat`
   (hashing only when it moved), which catches a read-only mount replaced at the same path, a directory swapped for a
   symlink into one, and a remount read-only after a write. A read-only row with no scan stat is not hashed again.
2. **Live sources: only suspects.**
   - The scan's `lstat` of each new-side file already comes before the file is read for its hash
     (`get_entry_state`). The scan keeps that stat's device, inode, size, `mtime_ns` and `ctime_ns`.
   - After the payload copy, each source is `lstat`ed again.
   - The copy is hashed and compared as today when any one of these holds:
     - any of the five values differ;
     - the scan-time `ctime_ns` is no earlier than the racy window before the scan began, which includes a ctime in
       the future. The window is twice the writeback horizon, `2 * (vm.dirty_expire_centisecs +
       vm.dirty_writeback_centisecs)` read from `/proc/sys/vm` when the check runs: at least 2 seconds, 120 seconds
       when either file cannot be read, and no ctime is trusted at all (every live copy is re-hashed) when
       `dirty_writeback_centisecs` is 0.
     - the source's filesystem is not one whose ctime hpcp trusts. The trusted types are `ext2`, `ext3`, `ext4`,
       `xfs`, `btrfs` and `f2fs`. FAT, exFAT, NTFS, FUSE, NFS, CIFS, tmpfs and all others are compared as today.
       The filesystem is the visible mount holding the real path of the source's parent directory, not the one
       holding the spelled path (a store-layout key can be spelled through a symlink).
       (Corrected during implementation: a store through a shared writable mmap to a page that is already dirty does
       not move ctime until writeback cleans the page, and tmpfs never updates it for mmap stores. A fixed 2 second
       window and tmpfs in the trusted list would have trusted a changed copy.)
     - the scan stat is missing for the row.
   - Otherwise the copy is trusted without being read.
   - "The scan began" is a wall-clock time taken before the new side's states are computed.
3. **Errors are unchanged.** A copy that differs stops generation with the same "changed while the patch was built"
   error, before any manifest is written.
4. **Scope.** This applies to single-filesystem patches and to every section of `-dd -diff`; both go through
   `_build_payload`.

### Part 4 — housekeeping

1. **`/x/out`.** The stubbed generation tests (`_fake_generation` and its callers) take their disk and output paths
   from `tmp_path`.
2. **btrfs warning.** It now reads: `… gets a new btrfs UUID; set {uuid} with btrfstune -U once no attached device
   carries {uuid}`.
3. **Empty lsblk output.** `_lsblk_disk_rows` raises
   `RuntimeError('Disk error: lsblk printed nothing for {disk}')` when there are no rows.
4. **Dead loop.** The up-front raw-size loop in `process_get_disk_diff_image` is removed.
   - `compare_disk_layouts` already refuses that case before anything is hashed.
   - The size check inside `add_raw` stays, for partitions that fall back to raw.
5. **UUID fallback test.** A unit test covers `get_partition_details` taking `ID_FS_UUID` when `ID_FS_UUID_ENC` is
   absent.
6. **README line 129.** The early "raw area and its size changed" refusal covers partitions that are raw by type. A
   filesystem partition that falls back to raw because it cannot be mounted is checked once that is found.
7. **Dropped from the B2 list:**
   - the "Task 3 test nits", whose details were not recorded;
   - renaming `read_disk_layout`'s `fs_params` parameter.

### Version and docs

- Version 9.71, with `COMMIT_DATE` set to the day the work is finished.
- The README describes:
  - `-e` on mounted sources, replacing the `-dd -diff` "start them with `*/`" sentence;
  - inputs left out;
  - when the payload is read again.

## Compatibility

- Patch formats are unchanged. Patches made by 9.65–9.70 apply as before.
- `-e` on mounted sources:
  - `*/…` patterns behave as before;
  - `/`-anchored patterns that used to match nothing now match inside the filesystem;
  - patterns written against the temporary mount path (`/tmp/*/var/log/*`, `/tmp/tmp*/…`) matched in 9.65–9.70 and
    now match nothing;
  - a pattern that matched hpcp's temporary mount path itself (`*/tmp/*`, `/tmp/*`), and so excluded the whole
    filesystem, now matches only inside it.
- Generation from read-only sources no longer reads the payload twice. From live sources it reads again only the
  files listed in Part 3.

## Testing

Real-filesystem tests run as root with loop devices. Scratch files go only under `tmp_path`, and nothing is left
mounted or attached.

**Probe safety rule:** any ad-hoc root probe goes in a script file run with `bash <file>`. `set -euo pipefail` typed
at the top of a shell command is ignored in this harness. Every path variable gets a `: "${VAR:?}"` guard and a
`/tmp/claude-0/` prefix check, and devices are only `/dev/loop*` backed by scratch images.

- **`-e`:**
  - A `-si` ext4 image with `var/log/a.log` and `tmp/b`:
    - `-e '/var/log/*'` leaves out `var/log/a.log`;
    - `-e '*/var/log/*'` does too;
    - `-e '*/tmp/*'` leaves out only `tmp/b`.
  - A `-dd -diff` disk with an ext4 root and a vfat ESP: `/var/log/*` applies to the root section, and
    `/EFI/x/*` to the ESP section.
  - A directory source still matches real paths.
  - The base side honours the patterns, so no delete rows appear for excluded base files.
  - A `-x` file behaves like `-e`.
  - A spy on the listing shows an excluded directory is not walked, on either side.
- **Inputs.** None of these show up in the rows:
  - a `-di` image inside the new directory;
  - a `-si` image inside the base directory;
  - the base directory inside the new directory;
  - the new directory inside the base directory.
- **Re-check.** Spies on the hash function used by `_check_payload` show:
  - a read-only loop-mounted `-si` source: zero re-hashes;
  - an unchanged live file whose ctime is older than the window (tmpfs and the other untrusted types are re-hashed): zero;
  - a live file rewritten after the scan with the same size, and its mtime put back with `os.utime`: caught as
    changed;
  - a file within the racy window: re-hashed;
  - a file on a filesystem type outside the trusted list (mountinfo stubbed): re-hashed.
- **Housekeeping:** a unit test for each of the items 2–5 above.
- **Compatibility:** the existing suite stays green.

## Review focus

1. **Stat before hash.** The scan-time stat is always the one taken before the file's content is read. A write
   during the scan's hash read must show up as a changed ctime after the copy.
2. **Read-only detection.** Only a read-only superblock skips the re-check. A read-only bind mount, a mount-level
   `ro` flag alone, or a missing mountinfo entry never does.
3. **Exclude parity.** For mounted sources, the new and base listings exclude exactly the same logical paths. A
   pattern that excludes an entry on one side never leaves a delete or add row for it on the other.
4. **No regressions for directory sources.** `-diff DIR` and copy-mode `-e` match exactly as before.
5. **Inputs.** Input exclusion uses real paths. A symlinked input path or symlinked tree root is still found inside
   the tree.

## Out of scope (later rounds)

Round C2:
- partitions of a partitioned md array (`md127pN`) matched in `-dd`;
- a row written by one section and removed by a later section's delete, reported but not restored;
- inode reuse after a double replacement;
- path components above a forced removal's top resolved by path;
- an automounted (autofs) ESP classified as autofs;
- FAT attribute ioctl numbers on ppc, mips and sparc.

Not planned: a `-diff /` of a live system walks into `/proc`, `/sys` and every other mount unless they are
excluded. That is existing behaviour.
