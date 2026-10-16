# hpcp
A simple script that can issue multiple `cp -af` commands simultaneously on a local system.

Optimized for use in HPC scenarios and featuring auto-tuning for files-per-process.

Includes an adaptive progress bar for copying tasks from multiCMD.

Tested on a Lustre filesystem with 1.5 PB capacity running on 180 HDDs. Compared to using `tar`, **hpcp** reduced the time for tarball/image release from over 8 hours to under 10 minutes.

## Development status

Basic functionality (parallel copy) should be stable.

Imaging functionality (source/destination as `.img` files) includes differential patch images (`-diff` / `-siff`, see below). Imaging is only available on Linux—similar to `tar`, but uses disk images.

Block-image functionality is in **beta**. Only available on Linux. Possible use case: cloning a currently running OS without mounting `/` as read-only.

hpcp.bat available on github: simple old tk based GUI intended for basic windows functionality.

## Important Implementation Detail

By default, **hpcp** only checks:
1. The file’s relative path/name is identical.
2. The file mtime is identical.
3. The last `-hs --hash_size` bytes (defaults to `65536`) are identical.

Although in most cases these checks should confirm that both files are identical, in certain scenarios (like bit rot), corrupted files might not be detected. If you need to verify file integrity rather than perform a quick sync, it is recommended to use the `-fh --full_hash` option.

A file or directory that cannot be read, listed or written is reported and the rest of the copy goes on; hpcp then exits with a non-zero code (`Copy failed` 190, `Directory sync error` 193, `Listing error` 209, ...). As with `cp -a`, a non-root copy keeps the group when the user belongs to it but cannot keep another user's ownership, and then does not keep set-user-ID / set-group-ID bits it could not keep the owner / group for.

Symlinks are recreated as symlinks. Without `-nlt --no_link_tracking`, a relative symlink whose target is missing at the destination (it points outside the copied tree, say `../shared/lib`) gets its target copied to the same relative place, which can be outside the destination directory, so the link resolves there too. With several destinations a relative link goes to the destination that holds its target. A link whose path is taken by a real file or directory in the destination is not created and is reported.

Setting `-hs --hash_size` to `0` disables hash checks entirely. This can be helpful on HDDs, as they usually have suboptimal seek performance. However, HDDs are also more prone to bit rot. If the operator can accept that risk, it is possible to rely solely on mtime checks for file comparison by setting `hash_size` to `0`. (Though on a single HDD, the standard `cp` command is already well-optimized.)

## Installation

```bash
pipx install hpcp
```
or
```bash
pip install hpcp
```

After installation, **hpcp** is available as `hpcp`. You can check its version and libraries via:
```bash
hpcp -V
```

It is recommended to install via **pip**, but **hpcp** can also function as a single script using Python’s default libraries.

**Note**:  
- Using `pip` will optionally install the hashing library [xxhash](https://github.com/Cyan4973/xxHash), which can reduce CPU usage for partial hashing and increase performance when using `-fh --full_hash`.  
- `pip` also installs [multiCMD](https://github.com/yufei-pan/multiCMD), used to issue commands and provide helper functions. If it is not available, `hpcp.py` will use its built-in multiCMD interface, which is more limited, has lower performance, and may have issues with files containing spaces. Please install **multiCMD** if possible.

## Disk Imaging Feature Note

Only available on Linux currently!

`-dd --disk_dump` mode differs from the standard Linux `dd` program. **hpcp** will try to mount the block device/image file to a temporary directory and perform a file-based copy to an identically-created image file specified with `-di --dest_image`. This functionality is implemented crudely and is still an **alpha** feature. It works on basic partition types (it does not work with LVM) with GPT partition tables and has been proven able to clone live running system disks to disk images, which can then be booted without issues. A disk's partitions are the ones whose parent is the disk itself: devices stacked on them (md arrays, LVM, dm-crypt) and the partitions of such devices (a partitioned md array built on the disk) are never taken for the disk's own.  
The created disk image can be resized using the `-ddr --dd_resize` option to the desired size. (This feature is provided so that you can shrink the raw size of the resulting image and provides some shrink capability for XFS.)  
For partitions that **hpcp** cannot create a separate unique mount point, **hpcp** will fall back to using the Linux program `dd` to clone the drive. Note that this can be risky and can lead to broken filesystems if the drive is actively being written to. (However, since you generally cannot mount that partition on the current OS, the real-world scenarios for this remain limited.)

### Filesystem parameter mirroring

In `-dd` mode `hpcp` reads each source filesystem's creation parameters and recreates the
destination with matching geometry and features, instead of whatever the local `mkfs` defaults
happen to be. Coverage varies by filesystem: ext2/3/4 (block size, inode size, inode ratio, reserved
percentage, full feature set), xfs (block/sector/inode/directory geometry and v5 feature flags),
btrfs (nodesize, sectorsize, checksum type, features), FAT (width, sector and cluster size, FAT
count, reserved sectors), ntfs and exfat (cluster and sector size), f2fs (geometry and features),
udf and reiserfs (block size and format), hfs+ (allocation block size), and minix (filesystem
version and filename length).

Without it, a clone built on a modern host can be unmountable or unbootable on the system it came
from — for example a FAT32 EFI System Partition recreated as FAT16, or an ext4 `/boot` that gains
`metadata_csum` and `64bit` that its bootloader does not understand.

Size-dependent values are scaled rather than copied, so `-ddr` resizes stay valid: the ext inode
*ratio* and reserved *percentage* are mirrored, never absolute counts. If `mkfs` rejects a mirrored
parameter set, `hpcp` warns and retries with defaults rather than failing the copy. Use `-nfp` to
turn mirroring off entirely.

## Patch Images Feature Note

Only available on Linux (loop devices and mounts, so usually root).

`-diff [PATCH.img]` compares a new tree (a directory, or a single-filesystem image with `-si`) against a base (`-d DIR` or `-di IMG`) and writes a patch image. The patch is an ordinary xfs/ext4 image, split with `-dis` like `-di` output. It holds every added or modified file **whole** at its real path, plus `.hpcp_diff/manifest` and `.hpcp_diff/changes.tsv`, which list every added, modified and deleted path with its before and after state. A moved file appears as one delete plus one add. A file counts as modified when its content, size, mtime, mode, owner, ACLs, file capabilities or user xattrs change; a directory when its mode, owner, ACLs or user xattrs change; a symlink when its target changes. Other xattrs (such as `trusted.*`, `security.selinux` or filesystem-internal ones) are ignored. Without a value the patch is named `DIFF_<src>_TO_<base>_<time>.img`. A file that changes between the comparison and the payload copy (on a live source) stops generation before any manifest is written; exclude it with `-e` or generate from an offline copy. For a directory source, `-e` and `-x` patterns are matched against its real paths, as when copying; for a `-si` image they are matched against `/` plus the path inside the image's filesystem, so `-e '/var/log/*'` leaves out what is under `var/log` there (and `*/tmp/*` only a `tmp/` inside it). Patterns written against hpcp's temporary mount path (for example `/tmp/*/var/log/*`) no longer match; write them as `/…` paths inside the filesystem. A pattern that would exclude the compared tree's root (such as `-e /`) is refused, and blank lines in an `-x` file are ignored. This run's input images (`-si`, `-di`) and the base or new directory are left out of the patch when they lie inside a directory being compared, so `hpcp -diff / -di /root/golden-v1.img` does not install the old image on every node. A payload copy is read again to check it only when its source may have changed since the comparison: on a read-only filesystem (a `-si` image, or a golden mounted read-only) only when the file's stat moved, and on a live one only when the file's inode, size, mtime or change time moved, it was changed shortly before the comparison (within twice the kernel's dirty-page writeback interval, about 70 seconds with default settings, since a write through a memory mapping may leave the change time alone until then; every copy when periodic writeback is off), or it lies on a filesystem whose change time hpcp does not rely on (anything but ext2/3/4, xfs, btrfs and f2fs). With `-fh` (and on FAT, whose files are always hashed in full) every copy is read again, whatever its source, so a copy that came out wrong is caught before the patch exists.

A `-di` image store keeps files at their absolute source path. When the base is such a store, a directory is compared only with its own subtree inside the store, and the patch uses the same layout. So `hpcp -siff patch.img -di store.img` updates the store in place, and `hpcp -siff patch.img -d /` updates the live system.

`-siff PATCH.img` applies a patch to `-d DIR`, `-di IMG` or a single path. Repeat it for every `-dis` volume. It first checks every path the patch touches. If the target differs from what the patch expects, it lists the conflicts and asks `Apply anyway? [y/N]`; the default, and any non-interactive run, aborts. `-fa --force_apply` applies anyway and prints each conflict as a warning. Re-running a patch is safe because already-updated paths are skipped, and every touched path is verified afterwards.

Files and symlinks are replaced atomically: the new version is written next to the old one and renamed over it, so running processes keep the old file and an interrupted apply leaves each file either old or new. A directory the patch deletes is removed only if it is empty once the patch's own deletions are done; whole trees are removed only for conflicts forced with `-fa`, and that removal never enters a mounted filesystem (bind mounts included): a directory holding a mount point is left in place with the mount, that row fails verify, and a note names the mount before the prompt. Paths are followed through symlinked directories that stay inside the target; if any path the patch deletes or writes resolves outside the target through a symlink, nothing is applied, even with `-fa`.

A single-filesystem patch pairs names by exact case, so a case-only rename (`EFI/BOOT` to `EFI/Boot`) is a delete plus an add, and the target ends with the new spelling. Rows are planned per filesystem: the target's own and each one mounted below it (for example the ESP at `/boot/efi` when applying to `/`). An automount (direct map) the patch writes into (an ESP that systemd mounts on demand, shown as autofs until something enters it) is mounted first, so it is planned as what it holds; one that cannot be mounted stops the apply before anything changes, and so does an indirect automount map (only its keys mount; leave it out of the patch with `-e`). A file written by one filesystem's rows and then removed, through a symlinked directory, by a later filesystem's delete is reported by verify (failed), not written again; and an entry replaced twice by contradictory rows of one patch can in principle reuse an inode number, which verify then cannot tell apart (both need patches that contradict themselves). On FAT, exFAT and HFS/HFS+ (not case-sensitive HFSX) names are matched case-insensitively, so such a rename applies, with or without a content change, and a re-run skips it. On FAT, exFAT and NTFS rows are compared and written by content only, since modes and owners there come from mount options. On FAT, which stores local times, a file whose mtime differs only by a timezone offset (a whole number of 15 minutes, at most 24 hours) with the same content counts as unchanged; files on FAT on either side when the patch is made (including a mount below the source root, such as `/boot/efi`) are hashed in full for this, whatever `-hs` says (a golden whose ESP was a plain directory is hashed by its tail, so there the tolerance covers files up to the `-hs` size only). Such a patch (for example a `-diff` of `/` with a FAT ESP mounted below it) is format version 3, which hpcp 9.68 and older refuse. A replaced FAT file keeps its read-only, hidden and system attributes. A write that lands on an entry another row found already done (through a symlinked directory, or a name differing only by case) is reported by verify. The filesystems are applied in turn, the target's own first; if one fails, the ones after it are not applied. When a delete removes an entry whose add or update was skipped as already done (a case-insensitive filesystem hpcp does not recognise, or a symlinked directory alias), the apply writes it again and counts it as `restored`.

An apply that is killed partway leaves directories it created private (mode 0700, owned by root) until the final metadata pass runs. Re-running the same patch then reports those directories as conflicts; re-run with `-fa` to finish.

`changes.tsv` is also what `-dfl` writes:

```
#hpcp-changes	1
#op	path	old	new
D	gone.txt	f:<size>:<mtime_ns>:<mode>:<uid>:<gid>:<hash>:<xattr>	-
A	newdir/	-	d:<mode>:<uid>:<gid>:<xattr>
M	link	l:old-target	l:new-target
```

Paths escape `\`, tab, newline and carriage return as `\\`, `\t`, `\n`, `\r`.

Limitations:

- Hard links are not preserved; applying a patch breaks them.
- Special files (FIFOs, sockets, devices) are ignored.
- Even a small patch is an image of roughly 50 MiB.
- Generating and applying a patch need root (loop devices, mkfs, mount).
- Nothing else may modify the target while a patch is applied.

### Disk patches (`-dd`)

`hpcp -dd -diff [PATCH.img] NEW_DISK -d OLD_DISK` patches whole GPT disks — for rolling a new golden image out to nodes cloned from the previous one. Both disks are block devices or disk-image files and are only read; MBR (msdos) disks are refused. The patch holds one section per filesystem partition (the same change list and payload as a single-filesystem patch), raw copies of areas that cannot be patched file by file when they changed (BIOS-boot and unknown-filesystem partitions, the MBR boot code), and the identity values that changed (filesystem UUID / label, GPT partition GUID / name / attributes, disk GUID). Swap contents are never carried. `-e` patterns are matched against `/` plus the path inside each partition's filesystem: `-e '/var/log/*'` leaves out what is under `var/log` on the root partition and `-e '/EFI/Linux/*'` what is under that directory on the ESP (a separate `/var` partition holds `log/` at its root, so there the pattern is `/log/*`); `*/…` patterns apply on every partition. This run's temporary directories and its disk-image files are left out wherever a live mount of a partition shows them: the whole filesystem, a btrfs subvolume mounted as the root (Fedora's `root`), or a bound directory.

`hpcp -siff PATCH.img -d TARGET_DISK` applies it to a node disk — a block device (a symlink such as `/dev/disk/by-id/...` is fine) or a disk-image file — offline or live:

- The target must have the base's layout: the same partition numbers, type GUIDs and filesystem types (sizes may differ, except raw areas). Layout changes need a re-clone.
- On a live disk, mounted partitions are patched through their live mount. Raw writes and identity changes to a partition that is mounted, active swap or held by LVM/md are refused before anything changes — apply such patches offline. An in-use raw area is not read, so it is refused even if it already holds the new content; an offline apply skips it then.
- FAT, exFAT and NTFS sections (the ESP) are compared and applied by content only: their permissions and owners come from mount options, so differences there are ignored. On FAT sections, files are hashed in full and a file whose mtime differs only by a timezone offset (a whole number of 15 minutes, at most 24 hours) with the same content counts as unchanged, since FAT stores local times (exFAT and NTFS record UTC times and get no such tolerance). A replaced FAT file keeps its read-only, hidden and system attributes. Names are generated by exact case, so a case-only rename (`EFI/BOOT` to `EFI/Boot`) is carried and the node ends with the golden's spelling; FAT, exFAT and HFS/HFS+ sections match names case-insensitively when applying (case-sensitive HFSX is detected). HFS and HFS+ sections compare owners and modes too (hfsplus keeps them on disk).
- Filesystem sections are applied first (ESP last), then raw areas, the MBR boot code, and identity changes; everything is verified afterwards. A section that fails stops the apply before any later section, raw area, boot code or identity change (so does a failed restore or final verify across sections), and a raw area that cannot be written, or that does not hold the new content when read back right after its write, stops it before any later raw area, the boot code and identity changes (boot code that does so stops the identity changes): fix the cause and re-run, and the re-run skips what is already done. Apply prints one line per section and a total.
- Every partition the patch carries a raw area for (even one already done) or whose UUID or label it sets must open exclusively (`O_EXCL`) before anything changes; one that is mounted, active swap or held by LVM/md/dm — in this or another mount namespace — is refused (`Disk in use`). During the raw writes and identity changes hpcp locks the whole disk (`flock`, so udev leaves it alone) and holds those partitions exclusively; raw areas are read back past the page cache.
- A node's filesystem UUID or label that differs from the golden base is warned about and replaced only when the patch changes that value; otherwise the node keeps its own.
- Filesystem labels, GPT partition names and exFAT serials are read and written exactly: non-ASCII characters, spaces, `=` and apostrophes included (GPT names keep leading and trailing spaces too; libblkid drops a filesystem label's trailing spaces); `-dd` copies keep them the same way. Disk patches made by hpcp 9.67–9.69 for disks with such labels or names recorded them mangled or cut short, so regenerate those patches.
- Every refusal that needs no hash (a partition in use, an identity change that cannot be made, a missing raw payload, a section that cannot be mounted) comes before any raw area is read, and so does, when generating, a size change of a partition that is raw by its type (one that falls back to raw because it cannot be mounted is checked once that is found); a raw area too large for `-dis` is refused right after the raw areas are hashed when they alone force more than one volume (otherwise once the filesystems' payload is known).
- An image file that is already attached to a loop device is refused (`Disk in use`); apply to that loop device or detach it first.
- A patch whose FAT files carry full-hash marks is written as format version 3 (patches made with `-fh` or `-hs 0` carry none); hpcp 9.68 and older refuse it ("newer than supported"). Patches without them keep format 1 or 2.

## Remove Extra Feature Note

`-rme --remove_extra`: Especially when combined with `-rf`, **PLEASE PAY CLOSE ATTENTION TO YOUR TARGET DIRECTORY!**  
`--remove_extra` will remove **all** files that are not in the source path. When you are copying a file into a folder, you almost certainly do not want to use this!

It runs only after a copy that reported no errors: if a source path could not be listed, its copy in the destination would look extra. Otherwise hpcp leaves the destination as it is and exits with `Remove skipped` (212). With `-di` the image holds each source at its absolute path, and `-rme` compares against that layout: it mirrors the **whole image** to this run's sources, so another source copied into the same image on an earlier run is removed too. The destination's copy of an excluded source path (`-e`, `-x`, absolute or not) is never removed, and a directory that still holds one is kept, and so are link targets that link tracking placed in the destination. Directories of a filesystem mounted below the destination are left alone; files there are still compared and can be removed.

## Remove Feature Note

`-rm --remove` can remove files in bulk. This might be helpful on distributed file systems like Lustre, as it only gathers the file list once and performs bulk deletion rather than the default recursive deletion in the Linux `rm` program.

With a destination, `-rm` copies first and then removes the sources: a move. The sources are removed only if the copy reported no errors (a file that failed to copy, an unreadable directory, a link that could not be created); otherwise they are left in place and hpcp exits with `Remove skipped` (212).

`-rf --remove_force` implies `--remove`. **Use with care!** This skips the interactive check requiring user confirmation before removing. If **hpcp** did not generate the correct file list from the specified source paths, hopefully you have fast enough reflexes to press `Ctrl + C` repeatedly to stop all parallel deletion processes if you realize a mistake.

`-b --batch`: Using `-b` with `-rm` will gather the file list for all `source_paths` first, then issue the remove command. This can be helpful because **hpcp** will tune its `-f --files_per_job` parameter accordingly for each task, and running one large remove job might be faster than running many small ones. This is especially useful when working with glob patterns like `*`.

```bash
$ hpcp -h
usage: hpcp.py [-h] [-s] [-j MAX_WORKERS] [-b | -nb] [-v] [-do] [-nds] [-fh]
               [-hs HASH_SIZE] [-fpj FILES_PER_JOB] [-sfl SOURCE_FILE_LIST]
               [-fl TARGET_FILE_LIST] [-cfl] [-dfl [DIFF_FILE_LIST]] [-tdfl]
               [-nhfl] [-rm] [-rf] [-rme] [-rwloff] [-e EXCLUDE]
               [-x EXCLUDE_FILE] [-nlt] [-V] [-pfl] [-si SRC_IMAGE]
               [-siff LOAD_DIFF_IMAGE] [-d DEST_PATH] [-rds] [-di DEST_IMAGE]
               [-dis DEST_IMAGE_SIZE] [-diff [GET_DIFF_IMAGE]] [-fa] [-dd]
               [-ddr DD_RESIZE] [-nfp] [-L RATE_LIMIT] [-F FILE_RATE_LIMIT]
               [-tfs TARGET_FILE_SYSTEM] [-ncd] [-co]
               [-ctl COMMAND_TIMEOUT_LIMIT] [-enes]
               [src_path ...]

Copy files from source to destination

positional arguments:
  src_path              Source Path

options:
  -h, --help            show this help message and exit
  -s, --single_thread   Use serial processing
  -j, -m, -t, --max_workers MAX_WORKERS
                        Max workers for parallel processing. Default is 1 *
                        CPU count. Use negative numbers to indicate {n} * CPU
                        count, 0 means 1/2 CPU count.
  -b, --batch           Batch mode, process all files in one go
  -nb, --no_batch, --sequential
                        Do not use batch mode
  -v, --verbose         Verbose output
  -do, --directory_only
                        Only copy directory structure
  -nds, --no_directory_sync
                        Do not sync directory metadata, useful for verfication
  -fh, --full_hash      Checks the full hash of files
  -hs, --hash_size HASH_SIZE
                        Hash size in bytes, default is 65536. This means hpcp
                        will only check the last 64 KiB (about 1 page in a
                        SSD) of the file.
  -fpj, --files_per_job FILES_PER_JOB
                        Base number of files per job, will be adjusted
                        dynamically. Default is 1
  -sfl, -lfl, --source_file_list SOURCE_FILE_LIST
                        Load source file list from file, or use "-" to read
                        from stdin. Will treat it raw meaning do not expand
                        files / folders. files are seperated using newline. If
                        --compare_file_list is specified, it will be used as
                        source for compare
  -fl, -tfl, --target_file_list TARGET_FILE_LIST
                        Specify the file_list file to store list of files in
                        src_path to. If --compare_file_list is specified, it
                        will be used as targets for compare
  -cfl, --compare_file_list
                        Only compare file list. Use --file_list to specify a
                        existing file list or specify the dest_path to compare
                        src_path with. When not using with file_list, will
                        compare hash.
  -dfl, --diff_file_list [DIFF_FILE_LIST]
                        Implies --compare_file_list, specify a file name to
                        store the diff file list to or omit the value to auto-
                        determine.
  -tdfl, --tar_diff_file_list
                        Generate a tar compatible diff file list. ( update /
                        new files only )
  -nhfl, --no_hash_file_list
                        Do not append hash to file list
  -rm, --remove         Remove all files and folders specified in src_path
  -rf, --remove_force   Remove all files without prompt
  -rme, --remove_extra  Remove all files and folders in dest_path that are not
                        in src_path
  -rwloff, --do_not_remove_files_while_listing
                        Do not remove files while listing. This is useful if
                        you want to remove files after listing.
  -e, --exclude EXCLUDE
                        Exclude source files matching the pattern
  -x, --exclude_file EXCLUDE_FILE
                        Exclude source files matching the pattern in the file
  -nlt, --no_link_tracking
                        Do not copy files that symlinks point to.
  -V, --version         show program's version number and exit
  -pfl, --parallel_file_listing
                        Use parallel processing for file listing
  -si, --src_image SRC_IMAGE
                        Source Image, mount the image and copy the files from
                        it.
  -siff, --load_diff_image LOAD_DIFF_IMAGE
                        Apply a patch image made by -diff to the target (-d /
                        -di or a single path). Repeat for each -dis volume.
                        Every path the patch touches is checked first;
                        conflicts prompt before being overridden. A disk patch
                        (made with -dd -diff) takes the target disk with -d.
  -d, -C, --dest_path DEST_PATH
                        Destination Path
  -rds, --random_dest_selection
                        Randomly select destination path from the list of
                        destination paths instead of filling round robin. Can
                        speed up transfer if dests are on different devices.
                        Warning: can cause unable to fit in big files as dests
                        are filled up by smaller files.
  -di, --dest_image DEST_IMAGE
                        Base name for destination Image, create a image file
                        and copy the files into it.
  -dis, --dest_image_size DEST_IMAGE_SIZE
                        Destination Image Size, specify the size of the
                        destination image to split into. Default is 0 (No
                        split). Example: {10TiB} or {1G}
  -diff, --get_diff_image [GET_DIFF_IMAGE]
                        Generate a patch image that updates the base (-d /
                        -di) to the source (src_path / -si): added and
                        modified files in full plus a change list under
                        .hpcp_diff/. Optionally name the output; default
                        DIFF_<src>_TO_<base>_<time>.img. Honors -dis, -fh, -hs
                        and -e. With -dd, compares whole GPT disks: the new
                        disk as the path and the old disk with -d.
  -fa, --force_apply    With -siff, apply even when the target differs from
                        the state the patch expects. Conflicts are still
                        printed as warnings.
  -dd, --disk_dump      Disk to Disk mirror, use this if you are backuping /
                        deploying an OS from / to a disk. Require 1 source,
                        can be 1 src_path or 1 -si src_image, require 1 -di
                        dest_image. Note: will only actually use dd if unable
                        to mount / create a partition.
  -ddr, --dd_resize DD_RESIZE
                        Resize the destination image to the specified size
                        with -dd. Applies to biggest partiton first. Specify
                        multiple -ddr to resize subsequent sized partitions.
                        Example: {100GiB} or {200G}
  -nfp, --no_fs_param_mirror
                        Do not mirror source filesystem parameters (geometry,
                        features) in -dd mode. Create destination filesystems
                        with mkfs defaults, preserving only label and UUID.
  -L, -rl, --rate_limit RATE_LIMIT
                        Approximate a rate limit the copy speed in
                        bytes/second. Example: 10M for 10 MB/s, 1Gi for 1
                        GiB/s. Default is 0: no rate limit.
  -F, -frl, --file_rate_limit FILE_RATE_LIMIT
                        Approximate a rate limit the copy speed in
                        files/second. Example: 10K for 10000 files/s, 1Ki for
                        1024 files/s. Default is 0: no rate limit.
  -tfs, --target_file_system TARGET_FILE_SYSTEM
                        Specify the target file system type. Will abort if the
                        target file system type does not match. Example: ext4,
                        xfs, f2fs, ntfs, fat32, exfat. Default is None: do not
                        check target file system type.
  -ncd, --no_create_dir
                        Ignore any destination folder that does not already
                        exist. ( Will still copy if dest is a file )
  -co, --content_only   Content-only copy: do not sync mode, owner, or
                        timestamps on files or directories. Still create
                        missing destination directories using filesystem-
                        default permissions (so parent ACL/setgid/setuid
                        inherit).
  -ctl, --command_timeout_limit COMMAND_TIMEOUT_LIMIT
                        Set the command timeout limit in seconds for external
                        commands ( ex. cp / dd ). Default is 0: no timeout.
  -enes, --exit_not_enough_space
                        Exit if there is not enough space on the destination
                        instead of continuing (Note: Default is continue as in
                        compressed fs copy can be down even if source is
                        bigger than free space).
```
