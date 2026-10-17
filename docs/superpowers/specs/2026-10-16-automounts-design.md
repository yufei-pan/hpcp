# hpcp 9.77: -rme and filesystems mounted on entering a directory

## Goal

9.75 deferred one case. A filesystem that is mounted only when its directory is entered has its parent's device and no mountinfo entry until then: an NFSv4 crossmnt or referral submount, or an autofs or systemd automount. Such a directory passed the `-rme` boundary until the listing entered it, so its files reached the y/n prompt. The deletion backstop then kept them, with a note. The user asked for it to be fixed.

## Design

- **`_entered_on_another_device(path, device)`** opens the directory with `O_DIRECTORY` and `O_NOFOLLOW`. Opening it is what mounts such a filesystem. It then `fstat`s the open descriptor and compares the device.
- **The `-rme` destination listing** (`get_file_list(boundary=...)`, serial and parallel) enters every directory below its root that way and reads it through that descriptor, using `scandir` on the descriptor. A directory on another device once open is left alone, unread.
  - The check by mount-point name (a string lookup) still comes first, so a mounted, possibly dead, filesystem below the destination is never opened.
  - The per-entry `fstatat` device check made before entering is replaced by the check after opening.
- **`remove_extra_dirs`' walk** (`_walk(top, device)`) enters each directory below the top the same way.
- **Where `scandir` cannot take a descriptor** (Windows, which has no device boundary anyway), the 9.76 behaviour is kept.

## Cost

A whole serial `-rme` over 20,000 files in 900 directories makes 28,621 server-side metadata calls in 9.76 and the same 28,621 in 9.77. 900 `fstatat` calls on a path became 900 `fstat` calls on the open directory; the only additions are a local `dup` and `close` per directory, because `scandir` on a descriptor duplicates it.

FUSE, median of 5 alternating runs, 9.76 against 9.77:
- serial: 0.870 s against 0.878 s;
- 32 workers: 0.301 s against 0.307 s.

Both are within noise.

## Testing

`tests/test_rme_mounts.py` simulates the untriggered automount with a real tmpfs. Its mountinfo entry is hidden, and the scandir entry, or the check before entering, sees the parent's device. That fools every check made before entering a directory. The tests cover:
- the y/n prompt lists nothing from it, and no backstop note is needed;
- the destination listing, serial and parallel, stops at it;
- `remove_extra_dirs` does not walk it.

## Review

A local review of `main..1504353` gave nine findings. They are fixed, each with a test that failed first:

1. **`remove_extra_dirs` decided about a directory in its parent's loop**, before the walk entered it. An untriggered mount point was therefore offered itself ("Deleting extra directory") and given an `rmdir`. EBUSY kept a mounted one; an expired one would have left the stub to the `rmdir`. Now:
   - the walk reports a directory it finds on another device once open (`_walk(..., foreign=)`), and that takes it back out;
   - the "Deleting extra directory" lines are printed once the walks are done;
   - the parent's loop no longer `lstat`s for the device where `scandir` takes a descriptor (the walk checks it), and in that mode the walk no longer `lstat`s for links (the `O_NOFOLLOW` open refuses one).
2. **Roots were pinned with `lstat` / `stat`, which do not mount such a filesystem.** A destination that is itself mounted on entering it had the stub's device, so every directory below it looked foreign. Its pin did not match the directory the removal opens ("Remove skipped"). The same was true of a moved source's pin ("replaced since the copy") and of `_rmtree`'s. `_directory_identity` takes a directory's identity as opened; it `lstat`s first, so a file or link root costs one `lstat`, as before.
3. **`-pfl`:** the parent's `recurse=False` listing put each subdirectory in `folders` unchecked. A folder whose own listing does not list it as a directory is now taken out, as in a serial listing.
4. and 9. **An `lstat` per directory before the open.** The open now comes first. An `lstat` only classifies a directory the open refused (ELOOP, ENOTDIR: a link or a file since), which is listed as such, as in 9.76. A directory that cannot be opened is a `Listing error`, as in 9.76.
5. **A double device check in `remove_extra_dirs`**, gone with 1.
6. and 7. **One helper:** `_enter_directory` (a descriptor, or None) with `_BELOW_FLAGS`, which now shares `_DIRECTORY_FLAGS`.
8. **The fallback without `scandir` on a descriptor** is now tested.
