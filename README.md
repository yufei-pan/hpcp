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

## Upgrading to 9.81 from 9.80

- **Lists keep their tree (layout change):** a path from a list (`-sfl FILE`, `-sfl -`, or stdin when no source is given) now lands at `<dest>/<path as given>`, normalised, without a leading `/`, `./` or any `..`, as `rsync --files-from` and `cp --parents` do. 9.80 put every entry flat in the destination, so `src/a/x.txt` and `src/b/x.txt` overwrote each other. Scripts that rely on the flat layout have to change: `cd` into the directory that holds the files, or pass the paths as arguments, which keep their layout. See [Copying from a list](#copying-from-a-list).
- **Same-name clashes are errors (exit 213):** when a file named directly (an argument: `hpcp a/x b/x -d out`, or a list entry) and another source land on one destination file, nothing is overwritten silently any more. The first in order stays, the rest of the run is copied, the clash is named, and hpcp exits 213, as `cp` refuses such a copy. It counts even when the contents are the same. A file named directly against a file from a directory is a clash too. A move (`-rm`) or `-rme` that meets a clash removes nothing.
- **A file against a directory is a clash too (exit 213), in 9.80 as well:** when a file lands where a directory is (one already in the destination, one that another source makes, or one a listed path implies: `d/x` with `d/x/y`), the directory stays, the file is not copied and its source is not removed, and the rest of the run is copied. 9.80 put the file inside the directory (`out/x/x`), and `hpcp a/ -d out -rme -rf` then removed it as extra and removed the source. A directory onto a file (or onto a link to one) leaves that file as it is, with its mode and owner, and is reported the same way (9.80 changed the file's mode, and as root its owner, then failed with exit 193). A symlink source that is a directory is as before. A symlink onto a directory that is not its own name, named directly, lands inside it like a file (`hpcp lf d` gives `d/lf`, exit 0; 9.80 exited 162), and that changes two old results: `hpcp lf -d lnd`, with `lnd` a link to a directory, used to replace the link `lnd` and now gives `d/lf`; `hpcp a/x -d la`, with `la` a link to the file's own directory, used to replace the link and now exits 0 and changes nothing. A link to a directory given with `-rme` is still refused (exit 212). A file named directly onto an existing directory still lands inside it (`hpcp a/x d` gives `d/x`, in every mode, as in 9.80); only a `d/x` that is itself a directory is the clash.
- **A file that would land on a source file is refused (exit 213), 9.80 as well:** hpcp works out, on real paths and file by file, where each file lands, and refuses the whole run before anything changes when a file would land on a source file (its own too, in a move), unless, in a copy, that file belongs to a directory that lands on itself, which reads and writes nothing. It names both paths. `cp` rewrites a file in place, so another `cp` reading it gets nothing or a part, and a move then removes the path that held the other content: 9.80 could leave an empty file, both contents gone, the sources removed, exit 0. Examples, each with a file at both places: `hpcp p/q/ -d p/` with `p/q/x` and `p/q/q/x` (one source over its own tree), `hpcp d1 d2/d1/ -d d2/`, `hpcp a out/a/s -d out/`, and `printf '../../c/r\nr\n' | hpcp -d .. -rf` run in `w/D/c`. A link changes nothing, and neither does a bind mount of the destination or another case of its name on a casefolded filesystem: `hpcp d1 l2/d1/ -d d2/`, with `l2` a link to `d2`, is the same run, and so is a landing that a link in the destination sends onto a source (`hpcp src/ -d out/` with `out/e -> ../src/d`, or a `cp` that would write through `out/x -> ../a/x`). What runs: a landing on a path where no source has a file, inside a source's tree too (`hpcp p/q/ -d p/` with only `p/q/q/only`); a directory that lands on itself, which another source updates as it would any destination (`hpcp */conf -d merged/`, every run; `hpcp a/ out/sub -d out/`). A file named directly that lands on itself is not given up that way: `hpcp x y/x -d ./` is refused. Only what is really copied counts: a path left out by `-e` and a special file land nowhere. `-do` and `-di` do not look. A move onto itself or into itself, as a whole, is exit 241 as before.
- **A landing in a directory that answers nothing is refused (exit 213), 9.80 ran:** `chmod 000 out; hpcp a/ -d out/ -rf` went on in 9.80, because the run's own directory sync changed `out`'s mode. With neither read nor search permission nothing can say whether a link in `out` sends a file onto a source, and going on lost data, so hpcp names the directory and the fix (`chmod u+rx out`). One permission is enough to answer: a listing (0600) or a question by name for each landing (0300), and those run as they did. A link they find is followed, and a link that leads through a place the run cannot search is refused too. A directory no file lands in stops nothing: a source with a locked subdirectory repeats 9.80's exit on every run.
- **A failed copy leaves the destination's old file, 9.80 deleted it:** when `cp` fails, only what it changed goes, so a source that cannot be opened no longer costs the good copy a backup already holds. A part that `cp` wrote still goes, also after the last retry, where 9.80 left it.
- **A source that holds the destination leaves it out, 9.80 made a copy of it:** `hpcp . -d out/` and `hpcp .. -d out/`, like `find . | hpcp -d out`, now copy everything but `out/` itself, and one line says so. So does a directory named without a slash that is the destination: `hpcp * -d backup/` with `backup` in the working directory leaves `backup` out, on every run. Its files then count as any file of the destination: another source that lands on one overwrites it. 9.80 copied the destination into itself (`out/out/...`, one level deeper each run), reading files that other sources were writing, so such a copy could come out empty or cut short, exit 0. `-rme` then checks nothing and says so: all that is in the destination belongs to that source. Moved, a source that holds the destination is still refused (exit 241); one that is the destination is left out (exit 0).
- **`..` lands in the destination, as `.` does:** `hpcp .. -d out/` put the files in `out/..`, so they went next to `out/` and `out/` stayed empty. A landing never leaves its destination now: the contents of the directory land in `out/`, as with `hpcp . -d out/`, `hpcp ../ -d out/` and `hpcp .. -d out`, which always did that.
- **Overlays still work:** directories that merge (`hpcp base/ overlay/ -d out`) are not a clash: the later source's file wins, exit 0, as in 9.80. The winner is now the same in every mode (9.80's batch copies could race), and one line says how many files a later source replaced. In a move (`-rm`) nothing is removed when a later source replaced a file, since the earlier source holds that file's only copy: the destination has the later files, all sources stay, and hpcp exits 212. A source named twice (`hpcp a/ b/ a/ -d out`) wins its files back, as a sequential copy does.
- **List entries inside the destination are skipped:** an entry whose path lies inside a destination (or is it) is not copied or moved, and counts as any file of the destination: another entry that lands on it overwrites it. One line says how many. `find . -name '*.log' | hpcp -d archive/ -rf` can run every night: the second run no longer moves `archive/app/app.log` into `archive/archive/...`. With no `-d` the destination is the working directory, so `find src | hpcp` skips everything and exits 0. A directory that holds the destination (`cd src && find . | hpcp -d out`) is copied without the destination; moved, it is refused (exit 241), as a move onto itself is. A relative entry is copied, not skipped, only when it and the working directory share a directory strictly inside every destination that holds the entry (by each spelling of its path: as given, and through its links) and its landing is neither the entry nor below it: `cd t/overlay; find etc -type f | hpcp -d ..` lands them at `t/etc/...`. Everything else stays and a second run finds the same: absolute entries, the destination itself and what is directly in it (`cd t/a/b; printf '../../a\n' | hpcp -d ..`), entries that would land on themselves (`-d .`). One case copies once: `find ../../a -type f` from `t/a/b` also lists `t/a/b/g`, which parts from the destination at `b`, so it lands at `t/a/a/b/g`; the second run skips that copy too.
- **Known limitations and deferrals** (see [docs/limitations.md](docs/limitations.md)): a symlink in the destination at a directory's path is written through, with the target's mode and owner changed (deferred to 9.82); with `-nds`, a directory source meeting a file in the destination exits 70 with a `File exists` traceback instead of a clean exit 213, the file untouched (deferred to 9.82); positional `-rme` judges the whole destination, not only where the source lands: `hpcp b -d out/ -rme` also removes `out/keep` (deferred to 9.82; see the `-rme` note below); `-rme` can take a source inside the destination for extra: `hpcp out/sub/ b/ -d out/ -rme`, answered y, deletes `out/sub/x` (deferred to 9.82); a move of a link's directory and a path through the link can end in exit 192, nothing lost (deferred to 9.82); the check before a copy holds a list's jobs in memory and scans existing destination directories one by one (deferred to 9.82); a stored list used under another spelling can match by chance, and then `-fl STORED -rme` offers the other files for deletion (known limitation).
- **`-rme` with a list judges only below its directories; with only named files or links it checks nothing:** with a list, only what the list's directories cover is cleaned; a list of files cleans nothing and says so, and everything else in the destination stays. 9.80 took everything in the destination that was not on the list for extra. To mirror a directory, list the directory itself (`printf 'src\n' | hpcp -d out -rme`): listing only its entries leaves what sits directly in `out/src` alone. Arguments judge the whole destination as soon as one source is a directory: it is checked against everything the sources copy, so `hpcp b -d out/ -rme` with `b` a directory removes `out/keep` (after the prompt), as in 9.80; limiting that to where the source lands is deferred to 9.82 ([docs/limitations.md](docs/limitations.md)). When every source is a named file or a link (`hpcp a/x -d a -rme`, `hpcp lf -d . -rme`), the files are copied, `-rme` says it checked nothing and nothing else is touched (9.80 took the rest of the destination for extra after a prompt, or exited 212). A file with a directory (`hpcp a/x e/ -d d -rme`) or a directory alone cleans as before; `hpcp a/x d/ -rme` no longer cleans `d/`.
- **Prompts with the list on stdin:** when stdin held the list, `-rme`, `-rm` and a move ask on `/dev/tty`, so `find ... | hpcp -d out -rme` works at a terminal. With no terminal they refuse (exit 147), as before. When stdin was not the list nothing changes.
- **List parsing:** lists are read as bytes, so any file name works under any locale. Data with a NUL byte is NUL-separated and exact (`find -print0`, `fd -0`), with no flag. Otherwise lines split on newline only, a trailing CR is dropped, and a line with surrounding whitespace is the exact name when that exists, else the stripped one.
- **Rerunning a FIFO source no longer hangs (9.80 as well):** `hpcp a/p -d out/` or a list holding a FIFO, run a second time, waited forever, because the check for an identical file opened the FIFO. Only regular files are compared by content now; nodes of one kind for the same device (two FIFOs, two `/dev/null`) count as identical without being opened.
- **Dangling symlinks** given as sources are copied as links (`-rm` removes them); 9.80 reported them as not found.
- **`-fl` from a list** stores names with the list's tree. `-fl STORED -rme` with a source from a list deletes nothing: give the source as an argument.
- **`-fl STORED -rme` fixes (9.80 as well):** `-fl STORED -rme` only compares the tree with the stored list and removes extras: it copies nothing and ignores `-d`, in 9.80 too. `hpcp src -fl S` then `hpcp src -fl S -rme` deleted every file in `src`, since the stored lines carry `:digest` (or `:target` for a link) and no name matched. A line now also means the name before each `:` in it, so a name that holds a `:` is kept too. A stored list is never split at NUL bytes (only `-sfl` lists are), a missing one removes nothing (exit 208), one that shares no name with the tree (empty, `/dev/null`, made for `s/` and used for `s`, or the reverse) is nothing known too: it removes nothing, asks nothing, says `Remove skipped` and exits 212, and a name with a line break is never removed: a stored list cannot tell it from two lines.

## Copying from a list

`hpcp` reads source paths from `-sfl FILE`, from `-sfl -`, or from stdin when no source is given and stdin is not a terminal, so it takes the output of `find` or `fd`:

```bash
find src -type f -name '*.log' | hpcp -d out   # out/src/.../x.log
cd src && find . -type f | hpcp -d ../out      # out/.../x.log: cd to leave src out of the destination
find src -type f -print0 | hpcp -d out         # NUL-separated: any file name is safe, no flag needed
fd -0 -e log . src | hpcp -d out
```

- Each path lands at `<dest>/<path as given>`, normalised, without a leading `/`, and `..` never leaves the destination: `../x/a` lands at `<dest>/x/a`. Directories the path implies that this run creates take their sources' mode and owner, so a private one stays private (`-nds` skips that); one that exists already (a directory, a symlink, even a mount point) is left as it is. A directory that is itself on the list copies its metadata onto an existing one.
- A list always means the destination is a directory, even for one entry.
- A directory in a list copies recursively. An entry below a listed directory is already covered, so plain `find src | hpcp -d out` gives one complete tree, with nothing copied twice.
- An entry inside the destination is not copied or moved (see [Upgrading to 9.81](#upgrading-to-981-from-980)); neither is the destination inside a listed directory.
- Arguments keep their own layout (`hpcp src -d out` copies the contents of `src` to `out`). Given both, each source follows its own rule.
- `-rme` with a list cleans only below the directories on the list: `printf 'src\n' | hpcp -d out -rme` mirrors `src` to `out/src/...` and leaves the rest of `out` alone. A directory skipped for lying inside the destination is not one of them. A list naming no directories checks nothing, and says so (so does `hpcp a/x -d out -rme` with only named files or links). It asks first, on `/dev/tty` when stdin holds the list; with no terminal it refuses (exit 147).
- `-nb` with a list starts a worker pool per source, about 23 ms each, so keep the default batching for big lists. See [docs/limitations.md](docs/limitations.md).

## Upgrading to 9.80 from 9.79

- **`<TOOL>_PATH` warnings:** every `<TOOL>_PATH` variable (`CP_PATH`, `MOUNT_PATH`, `DD_PATH`, ...) that does not name an executable is now named once at the start of every run (`Warning: CP_PATH=... is not an executable file; using cp from PATH.`). 9.79 named one only when it started that tool itself, so a bad `CP_PATH` (cp runs in the worker processes) was never named. A variable that other software sets for its own use prints that line too; it is harmless, and hpcp uses the tool found on PATH.
- **Removals:** a removal (plain `-rm`, a move, `-rme`, a forced patch apply: `-siff` with `-fa`) now holds one directory descriptor per level of the tree, and hpcp raises its soft open-files limit to the hard one at start; the tools it runs inherit the higher limit. A tree deeper than the hard limit (`ulimit -Hn`: 524288 on systemd hosts, 4096 on EL7) stops with "Too many open files" errors and leaves its deep part. On macOS the soft limit (256) stays, so trees deeper than about 250 levels stop there. A forced apply already held one descriptor per level in 9.79 and stopped at the soft limit (often 1024), so it now removes deeper trees than before. See [docs/limitations.md](docs/limitations.md).
- **multiCMD 1.56** is required (`pip` installs it). Without it, or with an older one, hpcp uses its built-in copy, which is the full multiCMD 1.56.
- **Correction to the 9.77 and 9.78 notes:** they wrongly named Ubuntu 22.04. Only e2fsprogs 1.45.x-1.46.2 (EL8, Debian 11, Ubuntu 20.04) left the empty root-owned file `/hpcp-ext-feature-probe`; 1.46.5 (Ubuntu 22.04) and 1.47 don't, and EL9 is untested. The file is safe to remove by hand.

## Important Implementation Detail

By default, **hpcp** only checks:
1. The file’s relative path/name is identical.
2. The file mtime is identical.
3. The last `-hs --hash_size` bytes (defaults to `65536`) are identical.

Although in most cases these checks should confirm that both files are identical, in certain scenarios (like bit rot), corrupted files might not be detected. If you need to verify file integrity rather than perform a quick sync, it is recommended to use the `-fh --full_hash` option.

A file or directory that cannot be read, listed or written is reported and the rest of the copy goes on; hpcp then exits with a non-zero code (`Copy failed` 190, `Directory sync error` 193, `Listing error` 209, ...). FIFOs, sockets and device nodes are not copied: each is noted (`Special file skipped`, no error). As with `cp -a`, a non-root copy keeps the group when the user belongs to it but cannot keep another user's ownership, and then does not keep a set-user-ID / set-group-ID bit (on files or directories) for an owner / group it could not keep.

Symlinks are recreated as symlinks, as `cp -a` does: a link whose target is outside the copied tree (say `../shared/lib`) is copied as it is and does not resolve at the destination unless the target is already there. With `-lt --link_tracking`, a relative symlink whose target is missing at the destination gets its target copied to the same relative place, which can be outside the destination directory (and can be a whole tree: `data -> ../../scratch`), so the link resolves there too. `-nlt --no_link_tracking` (the default) overrides `-lt`, and `-lt` does not apply with `-dd` (each partition's links are copied as they are). A new `-di` image is sized for the sources alone, not for what link tracking adds: a tracked tree that does not fit is reported as copy errors. With several destinations a relative link goes to the destination that holds its target. A link whose path is taken by a real file or directory in the destination is not created and is reported.

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
- `pip` also installs [multiCMD](https://github.com/yufei-pan/multiCMD), used to issue commands and provide helper functions. If it is not available, or older than the version hpcp requires, `hpcp.py` uses its built-in copy, which is the full multiCMD.

## Disk Imaging Feature Note

Only available on Linux currently!

`-dd --disk_dump` mode differs from the standard Linux `dd` program. **hpcp** will try to mount the block device/image file to a temporary directory and perform a file-based copy to an identically-created image file specified with `-di --dest_image`. This functionality is implemented crudely and is still an **alpha** feature. It works on basic partition types (it does not work with LVM) with GPT partition tables and has been proven able to clone live running system disks to disk images, which can then be booted without issues. A disk's partitions are the ones whose parent is the disk itself: devices stacked on them (md arrays, LVM, dm-crypt) and the partitions of such devices (a partitioned md array built on the disk) are never taken for the disk's own. A source partition that is not mounted is mounted read-only (an xfs with `nouuid`, so a clone of a disk whose xfs is mounted on this host can be read); the kernel still replays a dirty journal or log when the source device is writable, so the copy is of the recovered filesystem. A partition that is mounted (a live system disk) is mounted the way it is live, read-write when it is, since the kernel refuses a mount that would change that.  
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

## Several Destinations

Several `-d` are spill-over, not copies: each file goes to one destination, the first with room (`-rds`: a random one), and an identical file at any destination is skipped. A file there that differs, even only in its mtime after a plain `cp`, is not skipped, and a different file at a later destination stays as a second version.

## Remove Extra Feature Note

`-rme --remove_extra`: Especially when combined with `-rf`, **PLEASE PAY CLOSE ATTENTION TO YOUR TARGET DIRECTORY!**  
`--remove_extra` will remove **all** files that are not in the source path. When every source is a named file or a link (`hpcp a/x -d out/ -rme`), or a source is or holds the destination (`hpcp . -d out/ -rme`), it checks nothing, says so, and removes nothing. With arguments, as soon as one source is a directory, the whole destination is judged against everything the sources copy, so check what is in it before you confirm. With a list, only what the list's directories cover is judged (see [Copying from a list](#copying-from-a-list)). A source inside the destination can be taken for extra (`hpcp out/sub/ b/ -d out/ -rme`, deferred to 9.82: [docs/limitations.md](docs/limitations.md)).

It runs only after a copy that reported no errors: if a source path could not be listed, its copy in the destination would look extra. Otherwise hpcp leaves the destination as it is and exits with `Remove skipped` (212). With `-di` the image holds each source at its absolute path, and `-rme` compares against that layout: it mirrors the **whole image** to this run's sources, so another source copied into the same image on an earlier run is removed too. The destination's copy of an excluded path (`-e`, `-x`, absolute or not, a link's excluded target included) is never removed, and a directory that still holds one is kept. With `-lt`, link targets that link tracking placed in the destination are kept too; without it they are not part of the copy and are removed like any other extra file, unless excluded. A filesystem mounted below the destination (a mount or a bind mount, or another device such as a btrfs subvolume, as with `rsync -x`) is left alone with everything in it: its files are not listed as extra and its directories are not removed. That includes one mounted only when its directory is entered (an automount, an NFSv4 submount): each directory is checked once open, before anything in it is read. A destination, or a moved source, that is itself such a mount is taken as mounted. The destination's own filesystem is cleaned even when the destination is a mount point itself (a `-di` image, a disk). A filesystem's own `lost+found` (directly at its root, as in a `-di` image or an ext4 destination) and what fsck put in it are never removed.

## Remove Feature Note

`-rm --remove` can remove files in bulk. This might be helpful on distributed file systems like Lustre: without a destination it walks each tree once, by directory descriptors (one open and one listing per directory, no stat per file), and with `-pfl` empties the directories in parallel jobs; `-rwloff` lists the trees first and then removes the listed files in parallel jobs. Every removal (plain `-rm`, a move, `-rme`) goes by directory descriptors from each source or destination, recorded before the confirmation prompt, without following symlinks: a source or destination replaced, or a directory in it swapped for a symlink, while hpcp runs (a tree other users can write to, with hpcp running as root) is not removed through, and is reported. Excluded paths stay, with the directories that hold them. As with `rm --preserve-root`, `hpcp -rm /` (or a bind mount of `/`) is refused (`Invalid arguments`, 241).

A removal (a forced patch apply, `-siff` with `-fa`, too) holds one directory descriptor per level of the tree, so a tree deeper than the open-files limit stops with "Too many open files" errors and leaves its deep part; see [docs/limitations.md](docs/limitations.md).

With a destination, `-rm` copies first and then removes the sources: a move. A move that lands a source on itself (`hpcp -rm /data/x -d /data/`, `hpcp -rm a/ b/ -d a/`) or whose destination lies inside it is refused before anything is copied (`Invalid arguments`, 241). A directory named without a slash that is itself the destination is left out, and the rest moves: `hpcp a b -d b/ -rf` moves `a` into `b/`, leaves `b` and exits 0. The sources are removed only if the copy reported no errors (a file that failed to copy, an unreadable directory, a path longer than 4096 characters, a link that could not be created); otherwise they are left in place and hpcp exits with `Remove skipped` (212). A move removes only the files and links the copy listed, then each source directory that is empty after that. Anything else stays where it is, with the directories that hold it: a file written to a source after the copy listed its directory, a special file (a FIFO, socket or device node, which hpcp does not copy), an excluded path. What stays, excluded paths aside, is named in a `Move left entries` note (exit code 0). A file changed after it was copied is still removed, as with `mv`, so stop writing to a source before moving it. The removal goes by directory descriptors from each source root, without following symlinks: a source root, or a directory in it, that was replaced (by a symlink, say) while the copy ran is not removed from, and is reported. Without a destination, `-rm` removes the whole source trees. If none of the given destinations can be used (a `-d` glob that matches nothing), nothing is copied or removed (`No valid destination`, 202); a `-di` image that does not exist yet is created first.

`-rf --remove_force` implies `--remove`. **Use with care!** This skips the interactive check requiring user confirmation before removing. If **hpcp** did not generate the correct file list from the specified source paths, hopefully you have fast enough reflexes to press `Ctrl + C` repeatedly to stop all parallel deletion processes if you realize a mistake.

`-b --batch`: Using `-b` with `-rm` will gather the file list for all `source_paths` first, then issue the remove command. This can be helpful because **hpcp** will tune its `-f --files_per_job` parameter accordingly for each task, and running one large remove job might be faster than running many small ones. This is especially useful when working with glob patterns like `*`.

```bash
$ hpcp -h
usage: hpcp.py [-h] [-s] [-j MAX_WORKERS] [-b | -nb] [-v] [-do] [-nds] [-fh]
               [-hs HASH_SIZE] [-fpj FILES_PER_JOB] [-sfl SOURCE_FILE_LIST]
               [-fl TARGET_FILE_LIST] [-cfl] [-dfl [DIFF_FILE_LIST]] [-tdfl]
               [-nhfl] [-rm] [-rf] [-rme] [-rwloff] [-e EXCLUDE]
               [-x EXCLUDE_FILE] [-lt] [-nlt] [-V] [-pfl] [-si SRC_IMAGE]
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
                        Load source paths from a file, or use "-" to read them
                        from stdin (read by default when no source is given
                        and stdin is not a terminal: find src -type f | hpcp
                        -d out). Each path lands in the destination as given,
                        minus a leading / and any .. (cd src && find . | hpcp
                        -d ../out to leave src out); a listed folder copies
                        recursively. Paths are not expanded and are separated
                        by newlines, or by NUL bytes when the data holds one
                        (find -print0, fd -0). If --compare_file_list is
                        specified, it will be used as source for compare
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
                        With -rm and no destination: list the trees first,
                        then remove the listed files in parallel jobs
                        (default: remove while walking the trees).
  -e, --exclude EXCLUDE
                        Exclude source files matching the pattern
  -x, --exclude_file EXCLUDE_FILE
                        Exclude source files matching the pattern in the file
  -lt, --link_tracking  Also copy the target of a relative symlink when the
                        destination does not have it, to the same place
                        relative to the link (this can be outside the
                        destination). Default: off, symlinks are copied as
                        they are.
  -nlt, --no_link_tracking
                        Do not copy the files that symlinks point to (the
                        default). Overrides -lt.
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
                        Destination Path. Repeat it to spill over: each file
                        goes to one destination, the first with room, not to
                        all of them.
  -rds, --random_dest_selection
                        Randomly select destination path from the list of
                        destination paths instead of the first with room. Can
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
                        timestamps on files or directories. A destination
                        file with other hard links is replaced, not written
                        through. A FIFO, socket or device node is made as cp
                        -a makes it, never read. Still create
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
