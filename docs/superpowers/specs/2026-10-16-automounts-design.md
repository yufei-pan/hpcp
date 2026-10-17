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
