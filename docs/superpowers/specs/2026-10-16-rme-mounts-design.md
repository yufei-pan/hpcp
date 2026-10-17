# hpcp 9.75: -rme leaves filesystems mounted below the destination alone

## Goal

This was a follow-up noted in 9.73. `remove_extra_dirs` never entered a filesystem mounted below the destination, but `remove_extra_files` listed into it. Files there that the source lacks were shown as extra and deleted. The user asked for it to be fixed.

## Rule

As with `rsync -x`: a filesystem below the destination is left alone, with everything in it, files included. That covers:
- a mount, or a bind mount of the same filesystem, as listed in `/proc/self/mountinfo` (`os.path.ismount` misses a bind mount);
- another device, such as a btrfs subvolume. 9.74's directory pass already treated another device as a boundary.

The destination's own filesystem is cleaned even when the destination is a mount point itself, such as a `-di` image or a disk.

## Design

- **`_mount_points_below(real_root)`:** the mountinfo mount points strictly below the destination's real path. It is read once per destination.
- **`_below_a_mount(listed_root, real_root, mount_points)`:** a check over each extra's directories, looked up once each.
  - `remove_extra_files` drops extras below a mount, in both the normal and the legacy branch, so the y/n list no longer shows them.
  - With no mounts below the destination, the check is a constant `False`, so nothing extra is paid.
- **`remove_extra_dirs`:** skips mountinfo mount points as well as `os.path.ismount` ones, which adds bind mounts.
- **Backstop:** `_remove_listed_files(one_filesystem=True)`, used by `-rme` for files and directories, `fstat`s each directory it opens.
  - A directory on another device than its root is not entered.
  - This covers a mount mountinfo did not list, a btrfs subvolume, or a mount made after the listing.
  - It is reported with the note `Other filesystem left`, rc 0.

## Performance

Serial `-rme`, 20,000 files in 900 directories on FUSE, 11 alternating runs: median 0.630 s for 9.74 against 0.628 s for 9.75. With 32 workers, 0.144 s against 0.145 s. The backstop's one `fstat` per directory is within noise.

## Testing

`tests/test_rme_mounts.py` uses real tmpfs and bind mounts, which are always unmounted afterwards. It covers:
- files and directories in a mounted filesystem kept;
- the same through `hpcp() -rme`, serial and parallel;
- a destination that is itself a mount point still cleaned;
- the cursor backstop, serial and in pool jobs.
