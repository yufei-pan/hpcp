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

## Python 3.6 to 3.9

`setup.py` promises Python 3.6 or newer, but 9.74 to 9.76 had been tested only on 3.10 and newer. The whole suite now runs on 3.6.15 (the official build), 3.7.17, 3.8, 3.9 and 3.10 to 3.14. The runs found:

- **`-rm -pfl -F` raised a TypeError on 3.6.** It passed `ProcessPoolExecutor` an `initializer`, which 3.6 lacks. On 3.6 the workers inherit the shared rate count instead: they fork at the first submit.
- **The 9.77 checks did not run on 3.6.** Its `scandir` takes no descriptor, so the listing and the walk fell back to 9.76's checks made before entering. Every directory is now entered on every Python. One held open is read through `_scan_target`:
  - the descriptor itself;
  - else `/proc/self/fd/N` (Linux), which is the open directory itself, looked up without its path;
  - else its path, while it still names the held directory.
- **On 3.6, `_dir_entries` stat'ed every entry** (since 9.74; a round trip each on Lustre). `/proc/self/fd/N` gives `d_type`.
- **`get_args`' fallback parsed `sys.argv`, not the `args` it was given.** It runs where `parse_intermixed_args` is missing (3.6).
- **`stdout.reconfigure` is 3.7+.** On 3.6 a name with non-UTF-8 bytes ended the run at its first print. A genuine `TextIOWrapper` stdout is rewrapped with `backslashreplace` instead.

Test-only changes:
- a `scan_mode` fixture runs the descriptor-read tests all three ways;
- on 3.6, the tests' own `subprocess.run` calls get `capture_output` and `text` translated;
- the two `mp_context` tests skip below 3.7;
- the disk-lock checks wait out udev's brief shared lock (a flake seen once in six runs).

## Re-review

A review of `1504353..499c981` gave seven findings. Six are fixed, each with a test that failed first:

1. **A directory on another device that the walk could not open stayed extra** (EACCES, EIO, a mount that failed). An empty btrfs subvolume, which is not a mount point, would have been given an `rmdir`. One not opened is now checked by its `lstat`, as in 9.76; this costs nothing when the open succeeds.
2. **The 3.6 stdout rewrap also replaced a tee or logging proxy** without `reconfigure` on 3.7+. Now only a genuine `TextIOWrapper` is rewrapped.
3. **The directories holding a mount point were offered too, then kept as non-empty.** A mount point, by name or found on entering, now keeps the directories above it out of the extras.
4. **`-pfl` dropped a folder whose listing task crashed.** It stays, as in 9.76.
5. **`remove_extra_dirs` still `lstat`ed each directory for a link.** In enter mode the walk puts a symlink among the filenames by its `d_type`.
6. **Without descriptor `scandir` or `/proc`**, a directory swapped for a symlink after the open was read through the link. The path is now read only while it names the held directory.

One was not changed: `_parent_below`'s path fallback compares an `lstat` with the opened pin. That fallback runs only where descriptor walks are missing (Windows). There a directory cannot be opened, so the pin is the `lstat` too.

## Cost (after the review)

FUSE, median of 5 alternating runs, 9.76 against this branch:
- a whole serial `-rme`: 0.956 s against 0.955 s;
- 32 workers: 0.297 s against 0.299 s;
- the `remove_extra_dirs` walk: 0.053 s against 0.052 s.

The walk now makes one `lstat` fewer per directory than that run measured.
