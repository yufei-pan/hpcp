# hpcp 9.75: -rme leaves filesystems mounted below the destination alone

## Goal

This was a follow-up noted in 9.73. `remove_extra_dirs` never entered a filesystem mounted below the destination, but `remove_extra_files` listed into it. Files there that the source lacks were shown as extra and deleted. The user asked for it to be fixed.

## Rule

As with `rsync -x`: a filesystem below the destination is left alone, with everything in it, files included. That covers:
- a mount, or a bind mount of the same filesystem, as listed in `/proc/self/mountinfo` (`os.path.ismount` misses a bind mount);
- another device, such as a btrfs subvolume. 9.74's directory pass already treated another device as a boundary.

The destination's own filesystem is cleaned even when the destination is a mount point itself, such as a `-di` image or a disk.

## Design (after the review of the first version)

- **`_filesystem_boundary(listed_root, real_root)`** gives the visible mountinfo mount points strictly below the destination's real path, with hidden mounts left out (`_hidden_mounts`), plus that path's device. mountinfo is read once per `-rme` call.
- **`-rme`'s destination listing** (`get_file_list(boundary=...)`, serial and parallel) stops at the boundary, as `rsync -x` does. It skips:
  - those mount points, including bind-mounted files;
  - a directory on another device, from one `fstatat` per directory entry.

  So a hung NFS mount or an unreadable FUSE mount below the destination is never entered, and the y/n prompt lists nothing on another filesystem. The boundary listing takes no file sizes, which `-rme` never used, so it saves a `stat` per file. The boundary is part of the listing cache key.
- **`remove_extra_dirs`** prunes its walk with the same boundary: one `lstat` per directory, where `os.path.ismount` did two.
- **Backstop:** `_remove_listed_files(one_filesystem=True)` (`-rme` files and directories) checks the device of each directory it opens, and the fallback without descriptors does the same.
  - The root's device comes from the identity check's `fstat`.
  - A directory on another device is closed and not entered, with the note `Other filesystem left`, rc 0. That covers a mount made after the listing.
  - In the rmdir pass, EBUSY means the directory is now a mount point: it is kept with a note, not reported as an error.

## Review of the first version (local /code-review)

Fixed, with tests:
- **The listing went into mounted filesystems and filtered them out afterwards**, so a hung NFS mount stalled `-rme` and an unreadable FUSE mount failed the run. The listing now stops at the boundary.
- **A bind-mounted file** (`/etc/hosts` in a container) was offered for deletion and failed with EBUSY.
- **btrfs subvolumes**, which are not in mountinfo, were shown in the prompt and then refused at deletion.
- **Hidden mounts** shielded the visible directories at their paths.
- **A directory mounted on after the walk** failed `rmdir` with EBUSY, reported as an error.
- **The recursive mount lookup**, which could hit Python's recursion limit, is gone.
- **The fallback without descriptors** had no backstop.
- **Cleanup:** the root `fstat` is no longer duplicated; the mount detection is shared; mountinfo is parsed once; the cursor closes a foreign descriptor at once.

Not changed: the shared listing (`get_file_list_serial`) recurses once per directory level, so a tree about 1,000 levels deep raises `RecursionError` in a copy too. This predates 9.75; the deep-tree test uses 600 levels.

## Re-review of 127f141..HEAD (user: "re-review then release")

Fixed, with tests:
- **Relative destinations (reproduced).** The listing names entries as the destination was given (`dst/share` for `dst/`), but the boundary's mount points were absolute. Below a relative destination, a bind mount of the same filesystem was therefore listed, and its files removed. The boundary now spells the visible mount points as the listing does, computed once, so each check is a single set lookup with no string built per entry.
- **No `stat` into a mount point.** `remove_extra_dirs` looks a directory up among the mount points before any `lstat` of it; an `lstat` of a mount point reaches the mounted filesystem, and a dead NFS server would hang.
- **No device boundary on Windows**, where `DirEntry.stat` reports `st_dev` 0.
- **The cursor closes a directory whose `fstat` failed.** `_open_source_root` again returns only a descriptor.
- **The no-descriptor fallback checks every directory above the entry**, as the cursor does.
- **An rmdir EBUSY is treated as a kept mount point only when the directory is one now** (`_is_mount_point`); any other EBUSY is still an error.
- **Hidden mounts are worked out once per call.**

Deferred:
- **An NFSv4 submount (crossmnt or referral) that is not mounted yet** has its parent's device and is not in mountinfo, so it passes the boundary until the listing enters it. Its entries can be offered at the prompt; the backstop then keeps them, with a note, and nothing is removed. Catching it before the prompt needs a descriptor-based listing.

## Performance

Final measurements after the re-review, FUSE, median of 5 alternating runs, 9.74 against 9.75:
- whole `-rme` call, serial: 0.991 s against 0.869 s;
- whole `-rme` call, 32 workers: 0.509 s against 0.291 s;
- deletion phase alone, serial: 0.625 s against 0.739 s.

The deletion phase got slower, but the whole call got faster. 9.74's listing `stat`ed every file, and that warmed the lookup cache its unlinks then used. The boundary listing takes no sizes, so each file's first touch is its unlink. A 9.75 copy whose listing still takes sizes deletes in 0.627 s, level with 9.74. The `stat` per file saved in the listing costs more than the lookup moved into the deletion; on Lustre a file `stat` also asks the storage servers for the size.

### Earlier measurements

Whole `-rme` call (listing, prompt and deletion), 20,000 files in 900 directories on FUSE, median of 7 alternating runs, 9.74 against 9.75:
- serial: 1.012 s against 0.895 s;
- 32 workers: 0.513 s against 0.293 s.

The boundary listing's per-file size `stat` is gone, and its one `fstatat` per directory is cheaper than that. The deletion phase alone is equal to 9.74.

## Testing

`tests/test_rme_mounts.py` uses real tmpfs and bind mounts, always unmounted afterwards (verified after every run: no `hpcp-test` mount left). It covers:
- mounted files and directories kept;
- the same through `hpcp() -rme`, serial and parallel;
- a destination that is itself a mount point still cleaned;
- an unreadable mount never listed;
- another device that mountinfo does not show (a stand-in for a btrfs subvolume) kept out of the prompt;
- a bind-mounted file;
- a hidden mount;
- a directory mounted on after the walk;
- a 600-level tree with a mount;
- the backstop, both with descriptors and in the fallback.
