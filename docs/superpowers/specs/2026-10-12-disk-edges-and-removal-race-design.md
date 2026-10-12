# Disk edge cases, the forced-removal race, and C1 polish (round C2 of the 9.67 follow-ups)

## Problem

The remaining 9.67 follow-ups, checked against main 406a2e4 (hpcp 9.71):

- **md array partitions in `-dd`.** `_lsblk_disk_rows` keeps every `lsblk -l` row of TYPE `part` after the disk row.
  For a disk that is a member of a partitioned md array, `lsblk` also lists the array (`md127`, TYPE `raid1`) and the
  array's partitions (`md127p1`, TYPE `part`). So `-dd` takes `md127p1` as one of the disk's own partitions.
- **btrfs root on a subvolume in `-dd -diff`.** `_live_mount` recognises only a mount of the whole filesystem
  (mountinfo root `/`).
  - Fedora's default layout mounts `/` from subvolume `root`. Its root partition is not seen as live, so hpcp mounts
    the partition itself, which shows the default subvolume.
  - The exclusion of this run's temp directories and input disk images under the live root then never applies. On a
    live Fedora golden, `/root/golden-v1.img` can land in the patch, and so can the empty directories of this run's
    temp mounts.
- **autofs ESP.** systemd mounts the ESP on demand (an autofs mount at `/efi` or `/boot`, with the vfat mount stacked
  on it once something enters it). Until something enters it, mountinfo shows only `autofs`.
  - A single-filesystem apply to `/` plans that scope as a POSIX filesystem, so FAT files get mode and owner
    conflicts and fail visibly.
  - Generation from a live tree is not affected: `process_get_diff_image` reads mountinfo only after its listing has
    walked into the automount.
- **FAT attribute ioctls.** `_FAT_IOCTL_GET_ATTRIBUTES` (`0x80047210`) and `_FAT_IOCTL_SET_ATTRIBUTES` (`0x40047211`)
  are the generic encodings. alpha, mips, powerpc and sparc encode the direction bits differently, so there a
  replaced FAT file silently loses its read-only, hidden and system attributes.
- **The forced-removal race.**
  - A `-fa` removal (`_remove_tree_around_mounts`) walks the tree below its top by directory descriptors. But it finds
    the top by calling `os.path.realpath` on the parent at call time and opening the top by its full path.
  - The apply's containment check (`_inside_root`) runs earlier. A local user who swaps a directory above the top for
    a symlink in between makes the removal run outside the target. The `(dev, ino)` comparison does not help, because
    the swap comes before the `lstat` it compares with.
  - This runs as root, on trees other users may write to.
- **C1 polish.**
  - `-e` in file-list compare mode (`process_compare_file_list`) still matches the temporary mount path of a `-si`
    image source.
  - The payload re-check's filesystem lookup (`_rows_to_recheck`'s holder) scans every visible mount per directory,
    which costs O(directories × mounts).
  - `-e /`, a pattern that matches the filesystem root itself, excludes a whole `/` base but never a temp-mounted
    image root.
  - Parked documentation errors from C1:
    - the C1 spec says `-diff` first shipped in 9.67, but it shipped in 9.65;
    - the README says a copy is "never" read again on a read-only filesystem, but it is when its stat moved;
    - no test pins that `compute_changes` collects scan stats only for new-side paths.

## Goal

- `-dd` sees only a disk's own partitions.
- `-dd -diff` keeps this run's paths and inputs out of the patch on btrfs subvolume roots, and on bind mounts.
- Applies and generation see an automounted ESP as the FAT filesystem it is.
- FAT attributes survive on every architecture.
- A forced removal can never act outside the target.
- The C1 leftovers are closed.

This ships as hpcp 9.72. 9.68–9.72 are not on PyPI; publishing needs the user's go.

## Decisions (made with the user)

1. **Scope.** Disk handling, polish, and the forced-removal race. Two rare apply cases become documented known limitations:
   - a cross-section delete removing a written row;
   - inode reuse after a double replacement.
2. **Race fix scope.** Forced removals only. Ordinary writes keep the documented race. The other options were every
   write, rename and delete, or documentation only.
3. **Design sections 1–3** were presented and approved as written here.
4. **Full re-check with `-fh`** (added after the spec was approved). A side review pointed out that since 9.71 a payload
   copy that came out damaged, with the right size, ships when its source is read-only or unchanged. Watching mtime
   cannot catch it, because the copy's mtime is stamped from the source after the data is written. The user chose to
   hash every copy again when `-fh` is set, rather than hashing during the copy or adding an `O_DIRECT` read-back.

## Design

### Part 1 — disk handling

1. **A disk's own partitions.**
   - `_lsblk_disk_rows` asks `lsblk -p -nbl` also for `KNAME` and `PKNAME`.
   - A row is a partition only when its TYPE is `part` and its `PKNAME` equals the disk row's `KNAME`.
   - That drops `md127p1` (its parent is `md127`) and any other device stacked on the disk's partitions.
   - It keeps the partitions of dm disks (`/dev/dm-1` with parent `/dev/dm-0`) and of cciss disks.
   - `NAME` stays the path hpcp uses for a partition, as in B2.
   - The callers (`get_partitions`, `get_largest_partition`, `get_target_partition`) are unchanged apart from the extra
     columns.
2. **Live mounts that show part of a filesystem.**
   - In `-dd -diff`, a partition is still listed through hpcp's own mount, on both sides and on apply. That is a bind of
     a whole-filesystem live mount when one exists, otherwise a fresh read-only mount. So patch paths are unchanged.
   - What changes is how this run's temp mount points and input disk images under a live mount are excluded. For every
     mountinfo entry of the partition's device, whatever its mountinfo root R (a btrfs subvolume, or a bind of a
     directory), hpcp maps the path:
     - a path P below that entry's mount point M is, in the filesystem's own terms, `R + '/' + relpath(P, M)`;
     - if that lies below the root R0 that hpcp's own mount shows (the mountinfo root of hpcp's mount), the path
       `'/' + relpath(it, R0)` is excluded, rooted as in C1.
   - The existing whole-filesystem case is the special case R = R0 = `/`.
   - On a Fedora golden this excludes `/root/golden-v1.img` at `root/root/golden-v1.img` when hpcp's mount shows the
     top level, and at `root/golden-v1.img` when it shows subvolume `root`.
3. **Automounts.**
   - Before an apply plans its scopes, hpcp triggers each visible autofs mount at or below the target that the patch
     writes into: its mount point is the path of some row, or an ancestor of one.
   - Other automounts are left alone, so an apply to `/` never wakes `/net` or NFS home directories.
   - The trigger is a lookup through the mount point (`os.stat(os.path.join(mount_point, '.'))`), because a plain
     `stat` of the mount point does not trigger an automount. Then hpcp reads mountinfo again.
   - Generation needs no trigger, since its listing enters the automount before mountinfo is read. A test pins this
     order.
   - The stacked filesystem (vfat) is then the visible one.
   - A scope that is still autofs afterwards (no medium, a failed mount) makes the apply refuse that filesystem with a
     message naming the automount.
   - A side effect: the ESP gets mounted, which the apply needs anyway, and systemd unmounts it after its idle timeout.
4. **FAT ioctl numbers.**
   - `_FAT_IOCTL_GET_ATTRIBUTES` and `_FAT_IOCTL_SET_ATTRIBUTES` come from a small table keyed by `platform.machine()`.
   - The generic encoding is GET `0x80047210`, SET `0x40047211`.
   - alpha, mips* and ppc* / powerpc*, sparc* use GET `0x40047210`, SET `0x80047211`.

### Part 2 — the forced-removal race

1. `_remove_tree_around_mounts` takes the target root.
   - It opens the target root once (`O_RDONLY | O_DIRECTORY`, by the real path the apply already resolved).
   - From there it walks the components of the top's parent's real path, which the containment check resolved. Each
     component is opened relative to the previous descriptor with `O_NOFOLLOW | O_DIRECTORY`.
2. The top is handled entirely relative to its parent's descriptor: `lstat`ed with `os.stat(name, dir_fd=…,
   follow_symlinks=False)`, opened with `O_NOFOLLOW`, compared with that `lstat`, and finally removed with
   `os.rmdir(name, dir_fd=…)`. Never by path.
3. Mount checks for the top and its parent use the descriptors: `fstat` for `st_dev`, and the mountinfo set as today.
4. **Refusal.** A component that became a symlink or stopped being a directory, or a top that changed, refuses the
   removal. It returns an error and the row fails verify. Nothing outside the target is opened for writing or
   removed.
5. **Symlinked directories inside the target** keep working, because the walk follows the real path the containment
   check already resolved.
6. **Platforms without the descriptor functions** keep today's by-path fallback. Its documented race is unchanged.
7. **Ordinary writes, renames and deletes** keep the documented race (decision 2).

### Part 3 — C1 polish

1. **Compare mode.**
   - `process_compare_file_list` lists a `-si` image source with rooted matching: `-e` / `-x` patterns match `/` plus
     the path inside the image (`_rooted_excludes`).
   - When the base is a `-di` image as well, both sides are listed with `build_rooted_maps`.
   - Directory sources are unchanged.
2. **Lookup cost.**
   - `_rows_to_recheck` finds each row's filesystem with a dict of visible mount points (real paths, trailing separators
     stripped).
   - For each source it checks the file's own real path, then walks up the parents of its directory's real path.
   - The cost grows with path depth, not with the number of mounts. The result is the same as today: the deepest
     visible mount, with a mount on the file itself first.
3. **`-e /`.** A user exclude pattern that excludes the compared tree's root (`-e /`, `-e '*/'`) is refused in
   generation (`_refuse_root_exclude`, before anything is mounted or listed) with "Invalid arguments: exclude pattern
   ... matches the root of ...". For a mounted source (`-si` image, `-dd -diff` section) a pattern matches the root when
   `fnmatch('/', pattern)`; for a directory source when the walker's own test excludes its real path. Blank `-e` values
   and blank or whitespace-only lines in an `-x` file are ignored (`format_exclude`), since they used to become `*/`.
   (Corrected after review: a '/' base was not excluded whole, because the base side is post-filtered, so a
   root-matching pattern half-excluded the tree and could write a patch deleting everything; a blank `-x` line produced
   such a pattern. The earlier rule that `-e /` empties both sides is dropped.)
4. **Scan-stat pin.** A test shows that `compute_changes(scan=…)` asks `_states_for_paths` for stats of new-side paths
   only. The C1 final review found this unpinned.
5. **Full re-check with `-fh`.**
   - When the patch's rows are hashed in full (`-fh`, the manifest's `full_hash` 1), `_check_payload` hashes every A/M
     file copy again and compares it with its row, whatever its source, as up to 9.70.
   - Without `-fh`, C1's rules stand (`_rows_to_recheck`). The cost of `-fh` generation is again one more read of the
     payload, mostly from the page cache right after the copy.
   - The README says so next to the re-check sentence.

### Part 4 — docs

1. **C1 spec** (`docs/superpowers/specs/2026-10-12-exclude-inputs-recheck-design.md`): `-diff` first shipped in 9.65
   (`-si` images mounted under `/tmp` since then). Fix the version ranges in Part 1 item 7, in Compatibility, and in
   "Patches made by 9.67–9.70 apply as before".
2. **README re-check sentence:** "never on a read-only filesystem" becomes "on a read-only filesystem only when the
   file's stat moved".
3. **README behaviour notes:**
   - automounted ESPs are mounted for planning;
   - on btrfs subvolume roots, this run's own paths are excluded through every live mount;
   - `-dd` ignores devices stacked on a disk's partitions (md, LVM, dm).
4. **README known limitations** (decision 1):
   - A file written by one filesystem's section and then removed, through a symlinked directory, by a later section's
     delete is reported by verify (failed), not written again.
   - An entry replaced twice by contradictory rows of one patch can reuse an inode number, which verify then cannot
     tell apart.

### Version

9.72, with `COMMIT_DATE` set to the day the work is finished.

## Compatibility

- Patch formats are unchanged.
- `-dd` results are unchanged for disks without stacked partitions.
- Patches made by 9.65–9.71 apply as before.
- An autofs scope that 9.71 refused, or mis-planned, as POSIX is now planned as what is mounted on it (a direct map is
  triggered; an indirect map is refused with a message saying so).
- A root-matching exclude pattern is now refused in generation; blank `-x` lines are ignored.

## Testing

Real-device tests run as root with loop devices. Scratch files go only under `tmp_path`, and nothing is left mounted,
attached or assembled.

**Probe safety rule:** any ad-hoc root probe goes in a script file run with `bash <file>`. `set -euo pipefail` typed
at the top of a shell command is ignored in this harness. Every path variable gets a `: "${VAR:?}"` guard and a
`/tmp/claude-0/` prefix check. Devices are only `/dev/loop*` backed by scratch images, or md/dm devices built on them
and stopped or removed in a trap.

- **md partitions:**
  - Unit tests with fake `lsblk` output:
    - an md member disk, whose `md127p1` is dropped;
    - a dm disk and a cciss disk, whose own partitions are kept;
    - a disk with an LVM holder, which is dropped.
  - The md output is captured once from a real probe (two loop disks, `mdadm --create` RAID1, `sgdisk` partitions on
    the array, stopped and detached in a trap). There is no permanent md test, because md arrays are host-global.
  - A real loop-disk test checks that plain partitions still come back.
- **Live-mount mapping:**
  - Unit tests with mountinfo doubles:
    - subvolume `/root` at `/` while hpcp's mount shows the top level;
    - the same while hpcp's mount shows subvolume `/root`;
    - a bind of a subdirectory;
    - the whole-filesystem case unchanged.
  - When `mkfs.btrfs` exists, a real test builds a GPT loop disk with a btrfs partition, mounts its subvolume `root`
    live, puts an input disk image under it, runs `-dd -diff`, and checks that the image (and this run's temp
    directories) are absent from the patch.
- **Automounts:**
  - A unit test with a mountinfo double whose autofs entry gains a stacked vfat entry once triggered: the scope is
    planned as FAT.
  - A trigger that leaves only autofs refuses with the message.
  - An autofs mount below the target that no row touches is not triggered.
  - Generation from a tree with an automount reads mountinfo after its listing (a pin on the order).
  - A real `systemd-mount --automount` probe runs once during implementation, if available. It is not a permanent test.
- **FAT ioctls:** a unit test per architecture family, with `platform.machine()` faked.
- **Forced-removal race:**
  - A test hook swaps the top's parent for a symlink to an outside directory between the containment check and the
    removal. The outside directory and its files survive, and an error is reported.
  - The existing removal tests stay green: mounts below, a symlinked top refused, deep trees, a symlinked directory
    inside the target.
- **Polish:**
  - Compare mode with a `-si` source and `-e '/var/log/*'` (fake image mounts).
  - A lookup test with many fake mounts, showing the per-directory work no longer scales with them, and that results
    are unchanged (file mount first, then the deepest).
  - A root-matching exclude pattern (`/`, `*/`) is refused for `-si`, directory sources and `-dd -diff` before anything
    is listed or attached; blank `-x` lines are ignored.
  - The scan-stat pin.
  - With `-fh`, every copy is hashed again even from a read-only source, and a copy damaged after the copy (same size)
    stops generation before any manifest is written.
- **Compatibility:** the full suite stays green.

## Review focus

1. **Partition filtering.** No real partition is lost: dm, cciss, nvme, mmcblk and loop disks; a disk whose lsblk
   `KNAME` differs from its `NAME`.
2. **The mapping.** It never excludes a path the user's tree really has at that place: the mapping is exact, and only
   this run's own paths and inputs are excluded.
3. **Automount triggering.** It triggers only autofs entries the patch writes into, never unrelated ones such as
   network automounts, and a failed trigger leads to the refusal.
4. **The descriptor walk.** It opens nothing outside the target for removal, closes every descriptor on every path,
   and keeps symlinked directories inside the target working.
5. **Rooted compare mode.** Directory sources match exactly as before.

## Out of scope

Known limitations, documented and not fixed:
- cross-section delete of a written row;
- inode reuse after a double replacement;
- the race on ordinary writes, renames and deletes.
