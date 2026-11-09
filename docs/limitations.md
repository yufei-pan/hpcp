# Limitations

Approved known limitations and deferrals. Reviewers: read this first, and don't raise these again.

### Removal of trees deeper than the open-files limit
- **Kind:** known limitation.
- **What the operator sees:**
  - A removal (plain `-rm`, a move, `-rme`, a forced patch apply: `-siff` with `-fa`) of a tree deeper than the hard open-files limit (`ulimit -Hn`: 524288 on systemd hosts, 4096 on EL7) stops with "Too many open files" errors and leaves its deep part. Nothing outside the tree is touched.
  - On macOS the soft limit (256) stays, so trees deeper than about 250 levels stop there, where 9.79 removed them.
  - The copy, a move and `-rme` list nothing past 4096 characters, about 2,048 levels, so only plain `-rm` and a forced apply can reach a deeper tree on Linux.
  - 9.79's forced apply already held one descriptor per level, with no budget, and stopped at the soft limit (often 1024), so 9.80 removes deeper trees there than 9.79 did.
- **Why:** holding one descriptor per level and raising the soft limit replaced about 90 lines of descriptor budgeting and reopen checks.
- **Revisit when:** someone hits it, or macOS needs deep removals.
- **Approved:** 2026-10-27, the user chose "One per level (Recommended)".

### `-nb` with a list
- **Kind:** known limitation.
- **What the operator sees:** with `-nb` a list starts a worker pool per source, about 23 ms each: 2000 files took 46 s, which projects to about 38 minutes for 100,000, against 8.7 s batched. Library callers of `hpcp()` have the same cost, because their default is `batch=False`.
- **Why:** the operator asked for no batching, and the default CLI batches and is fast.
- **Revisit when:** someone needs `-nb` on big lists.
- **Approved:** 2026-11-05, the user chose "Known limitation (Recommended)".

### False Added/Deleted pair on compare with `-nhfl`
- **Kind:** known limitation.
- **What the operator sees:** comparing against a stored list without hashes, a name with surrounding whitespace shows Added `s/a/x ` and Deleted `s/a/x`. 9.80 does the same; with default hashes there is no false pair.
- **Why:** rare, old and read-only, and reading stored lists exactly would lose forgiving hand-edited lists (an existing test pins the stripping).
- **Revisit when:** someone hits it.
- **Approved:** 2026-11-05, the user chose "Known limitation (Recommended)".

### A stored list used under another spelling can match by chance
- **Kind:** known limitation.
- **What the operator sees:** after `hpcp s -fl S`, `hpcp s/ -fl S -rme` with s/a and s/s/a present shares one name with the list (`s/a` against `a`), so the matches-nothing rule does not fire, and the prompt offers the other files for deletion.
- **Why:** telling a chance match from a real one needs the list to record how it was made, a format change.
- **Revisit when:** someone hits it.
- **Approved:** 2026-11-05, the user chose "Known limitation (Recommended)".

### A rerun scans the destination directories one by one
- **Kind:** deferred.
- **What the operator sees:** a copy onto an existing tree reads each directory it lands in, on one worker, before the first file is copied: about 1 us per file on a local disk, longer on network storage.
- **Why:** correct and bounded as it is; a parallel scan is its own change.
- **Measured (9.85, 2026-11-08):** 9.85 has no serial scan (`_link_targets` is gone; the directory checks run in the workers), but a rerun is not faster for it: 100,000 files in 10,000 directories, nothing to copy, NFSv4.1 loopback `actimeo=0` on EL7 (2 vCPUs): 9.84 1764, 2301, 2212 s, 9.85 1871, 1878, 3044 s. On a local disk (Docker, a loaded host): 9.84 9.3 to 19.8 s, 9.85 6.5 to 15.7 s. The time goes to the per-file checks over NFS (about 60 to 80 GETATTRs per file rewritten, 9.84 as 9.85), not to a scan.
- **Plan (9.86):** cut the per-file checks that ask again what is known. One is new in 9.85: copy_file's look-only pass through the door calls `_landing_of(dest)` only for its path, an lstat per destination per file, a round trip each on NFS `actimeo=0`; the path needs no lstat there.
- **Revisit when:** 9.86, with the copy check's second pass (stat only the entries met twice).

### Rerunning an overlapping form is refused
- **Kind:** known limitation.
- **What the operator sees:** `hpcp p/q/ -d p/` and `hpcp a out/a/s -d out/` copy on the first run and exit 213 on every run after it, because the first run's output has become a source file. Nothing is lost.
- **Why:** narrowing it needs a content check at refusal time.
- **Revisit when:** someone hits it.
- **Approved:** 2026-11-06, the user chose "Known limitation (Recommended)".

### A bind of a deep subdirectory inside a source tree is not caught
- **Kind:** known limitation.
- **What the operator sees:** the landing check looks only at directories it walks. With a deep subdirectory of a source tree bind-mounted so that it is reached from inside the destination, a file can land on itself, with possible loss. Binds of the destination itself and other spellings of its name are caught.
- **Why:** catching it needs an lstat per directory on every run, for a layout that takes root to make.
- **Revisit when:** someone hits it.
- **Approved:** 2026-11-06, the user chose "Known limitation (Recommended)".

### Several destinations and a pair one of them takes for one name
- **Kind:** known limitation.
- **What the operator sees:** with several `-d`, two names that one destination takes for one name (`a/X` and `b/x`, the second `-d` folding case) are a clash, exit 213, though both would have fitted on another destination that keeps them apart. The same holds for a file and a directory of another source (`a/X` and `b/x/`). It is a false 213 for that pair, never a lost file.
- **Why:** files spill from one destination to the next as space runs out, and `-rds` picks at random, so which destination gets each file is not known before the copy.
- **Revisit when:** someone hits it.
- **Approved:** 2026-11-06, the user chose "Known limitation (Recommended)". Widened to a file and a directory of another source: 2026-11-07, the user chose "Keep it (Recommended)".

### A probe left behind by SIGKILL
- **Kind:** known limitation.
- **What the operator sees:** hpcp killed with SIGKILL in the moment it asks a directory about a name leaves a hidden `.hpcp-case-probe-PID-uuid` directory in the directory it was asking: a destination directory, or, for a move that renames a file within one directory (`hpcp out/ONE -d out/OTHER -rf`), that source directory, empty or holding a small file for each spelling of the name it was asking about; where the probe asks each made name alone (a spelling the directory will not look up, as on vfat), a second such directory can be left beside it. No data is affected. SIGINT, SIGTERM and SIGHUP wait for the probe to go, and left nothing in 30 kills each.
- **Why:** SIGKILL cannot be handled, and the only way to ask a directory about two names is to make them.
- **Revisit when:** an operator reports finding one.
- **Approved:** 2026-11-06, the user chose "Known limitation (Recommended)".

### A temporary file left by SIGKILL
- **Kind:** known limitation.
- **What the operator sees:** hpcp killed with SIGKILL (or a power cut) while it rewrites a file leaves a hidden `.hpcp_tmp_` file (16 hex digits after the prefix) beside it, in the destination; the file it was rewriting is untouched, its old copy whole. A rerun leaves the temporary file there; `-rme` removes it as an extra. SIGINT, SIGTERM and SIGHUP remove it (rule 9).
- **Why:** SIGKILL cannot be handled, and a rewrite that keeps the old copy whole needs the new one under another name until it is.
- **Revisit when:** an operator reports finding one.
- **Approved:** 2026-11-08, the user chose "Approve (Recommended)".

### A growing rewrite near a full disk or at quota keeps the old copy, exit 190
- **Kind:** known limitation.
- **What the operator sees:** a rewrite whose source is larger than the old copy, where the temporary file beside it runs out of room (`ENOSPC`, `EDQUOT`) part way, exits 190 with the old copy kept as it was (`Copy failed: ... leaving DEST as it is: ... growing it in place could cost the old copy`), where 9.84 wrote it in place and succeeded when the growth fitted. A rewrite that does not grow, or one where the free space is at least the source's size, is written in place as before.
- **Why:** written in place, a source that grows past the room left would truncate the old copy first and then fail, losing both; the partial temporary file shows the room is short.
- **Revisit when:** a portable space reservation (`fallocate` of the temporary file to the source's size) proves out on EL7's filesystems, so the room is known before anything is written.
- **Approved:** 2026-11-08, the user chose "Exit 190, keep old (Recommended)".

### `-co` writes in place where the old file's metadata cannot be carried
- **Kind:** known limitation.
- **What the operator sees:** with `-co`, a rewrite of a file with one name whose owner, group or an xattr cannot be given to a new file (another user's file for a non-root runner, a group the runner cannot set, an xattr hpcp cannot carry such as `trusted.*`) is written in place after the source has read whole, as 9.84 did. A failure on the destination's side part way (a full disk, an I/O error, a kill) then leaves that file torn, which the beside route would not. A file with other names still gets a new file, owned by the runner.
- **Why:** a new file there would change its owner, group or xattrs, which `-co` keeps; reading the source first rules out the commonest loss, a source that fails on read.
- **Revisit when:** an operator needs `-co` rewrites of other users' files to survive a destination failure.
- **Approved:** 2026-11-08, the user chose "Approve (Recommended)".

### A directory that cannot be asked about a new pair
- **Kind:** known limitation.
- **What the operator sees:** two names that fold alike (`X` and `x`), at least one of them new, landing in a directory where nothing can be made, such as an owned mode 555 directory, are refused with 213: `may be one name, and the destination cannot be asked: nothing can be made in DIR to ask it (Permission denied): make it writable and run again` (a read-only mount or a full disk names its own fix). On a case-sensitive filesystem the refusal is false: 9.82 opened an owned directory and landed both. No file is lost. Where nothing can be made, the directory's listing answers: both names in it are two names, so a rerun with both names there passes.
- **Why:** it takes that exact mix, and opening the directory for the probe means another chmod and restore in the path that writes to destinations.
- **Revisit when:** someone hits it.
- **Approved:** 2026-11-06, the user chose "Known limitation (Recommended)".

### A legacy name read as UTF-8 or ASCII, against its other case
- **Kind:** deferred.
- **What the operator sees:** on a vfat or exFAT mounted with a codepage `iocharset` (`iso8859-1`, `iso8859-9`, `cp1251`, `koi8-r`, `euc-jp`; not `utf8`), two names that the mount takes for one are not asked about before the copy when the net reads one of them as UTF-8 or ASCII, which happens in four ways:
  - one name's bytes happen to be valid UTF-8 and the other's are not: `C3 A9 74` (Latin-1 `Ã©t`, UTF-8 `ét`) against `E3 A9 74`, which a cp1251 mount reads as `Г©t` and `г©t`;
  - both names' bytes are valid UTF-8, but not as each other's case: KOI8-R `бё` and `бЁ` (`C2 A3` and `C2 B3`), CP1251 `EF B8 B8` and `EF A8 B8`; CP1251 alone has 880 such two-byte pairs;
  - an ASCII name against a legacy one: `istanbul` against ISO 8859-9 `İstanbul` (`DD` for `İ`, which that table makes the other case of `i`);
  - two cases of different lengths: EUC-JP `ς` is three bytes (`8F A6 F8`) and `Σ` two (`A6 B2`), and no rule over single bytes can join names of different lengths.
  So the pair is found only after the copy, by the backstop: `hpcp a/ b/ -d out/` with one in `a/` and the other in `b/` ends in exit 213, `Destination clash: copied SRC, but LANDING is one file with OTHER (from OTHER_SRC)`, and the landing holds whichever source was written last, which in a parallel copy can differ from run to run; every rerun copies both again, and with `-rf` both sources stay. A pair caught before the copy is refused there instead: the first source lands and the second is not written, the same on every run. Checked on vfat mounted `iocharset=cp1251`, `koi8-r` and `iso8859-9` (`utf8=0`) on EL8; the EUC-JP pair on kernel exFAT is not checked yet (it needs VM 901). Only for such pairs, and only on such a mount: names that are not UTF-8 as a whole fold byte by byte and meet.
- **Why:** a name that is valid UTF-8 is read as UTF-8, which it is on every other mount, and the mount's own table, not UTF-8, says what its other case is. Joining every valid name with every name of the same length would ask about most trees pair by pair again.
- **Plan:** catch the pairs before the copy, so that neither content is written over: fold every name byte by byte on a mount with a codepage `iocharset` (read from the mount options), or join byte-level keys across all names.
- **Revisit when:** 9.85.

### Directories that differ only in case on vfat mounted `iocharset=utf8`
- **Kind:** deferred (9.86).
- **What the operator sees:** on a vfat mounted with `iocharset=utf8` given explicitly, `hpcp a/ b/ -d out/` with `a/D/f` and `b/d/g` exits 193 (`Directory sync error: .../b/d to .../out/d: [Errno 2] No such file or directory`, then `Copy failed ... No more destination paths to try`), and `g` is not copied; with `-rf` it exits 212 and keeps both sources. A single source does it too: `out/D` from an earlier run, and a source that now spells it `d`, exit 193 and nothing below it copied (checked on EL7 by the 9.85 designer). Nothing is lost, and 9.82 does the same. Checked on VM 901; vfat with its default charset, `utf8=1` or `iocharset=iso8859-1`, kernel exFAT, NTFS (`ignore_case`) and ext4 casefold merge the two, exit 0. Since 9.85 the directory door says so: `mkdir` meets `EEXIST` twice where `lstat` finds nothing, and the 193 names the mount option to use instead.
- **Why:** vfat mounted that way looks a name up in its own case only (`out/d` is not found) but will not make it either, as `out/D` holds the name; the copy names the directory as the source spells it. A loud failure with nothing lost, on a mount option the kernel's vfat documentation advises against.
- **Plan:** in `_dest_directory`, when `mkdir` says `EEXIST` and `lstat` says `ENOENT`, list the parent and take the one entry whose fold matches as the spelling for that directory (cached), so the folder sync, the files and the links below it land there; the move's backstop asks the same helper for the spelling. More than one entry, or none, stays a 193 that says why.
- **Revisit when:** 9.86.
- **Noted:** 2026-11-08, the user's standing rule that deferrals are noted, not asked.

### A user who can write the destination can race a check by path
- **Kind:** deferred (9.86).
- **What the operator sees:** every write below a destination goes by path, so a user who can write a destination directory can swap a directory there for a link between hpcp's look (`_dest_directory`) and `cp`'s open, and a root run then writes where the link leads. The static plant (a link left there before the run) is replaced since 9.85; only the race in that moment remains.
- **Why:** closing it needs each landing's directory opened by descriptors from the destination root with `O_NOFOLLOW`, as the removal does (`_descend`), and `cp` run in that directory: its own change.
- **Plan:** open each landing directory from the root by descriptors (`O_NOFOLLOW` at each step), `fchdir` to it in the worker and give `cp` relative names.
- **Revisit when:** 9.86.
- **Noted:** 2026-11-08, the user's standing rule that deferrals are noted, not asked.

### `-co` reruns rewrite every unchanged file
- **Kind:** deferred (9.86).
- **What the operator sees:** a second `hpcp -co a/ -d out/` with nothing changed copies every file again ("Total files copied" counts them all, and each landing's mtime moves on each run). The cost is the whole copy on every rerun.
- **Why:** `-co` carries no times, so the landing's mtime is that of the last copy and the skip check (size and mtime) never matches.
- **Plan:** under `-co`, where the size matches and only the time differs, compare the full contents (`_same_contents`, as the move's backstop does) and skip a file that reads the same. Not `is_file_identical`'s hash: with the default `-hs 65536` it reads only the last 64 KiB, and a same-size change before that would be skipped on every rerun.
- **Revisit when:** 9.86.
- **Noted:** 2026-11-08, the user's standing rule that deferrals are noted, not asked.

### A 0555 source directory, first copied by a non-root user, exits 190
- **Kind:** deferred (9.86).
- **What the operator sees:** `hpcp s/ -d out/` as a non-root user, with `s/d` mode 0555 holding files: exit 190, `out/d` made with mode 0555 and empty, and a rerun fails the same way (checked by the 9.85 designer). Root is not affected.
- **Why:** the folder sync sets each directory's mode before its files are copied into it.
- **Plan:** set a directory's metadata (mode, then times) after the files below it, as `cp -a` and rsync do, keeping the directory writable for the runner meanwhile.
- **Revisit when:** 9.86.
- **Noted:** 2026-11-08, the user's standing rule that deferrals are noted, not asked.

### Hints quote names as Python escapes, not for the shell
- **Kind:** deferred (9.86).
- **What the operator sees:** with a name the locale cannot show (any non-ASCII name with LANG unset, a non-UTF-8 one under a UTF-8 locale), the hints that give a command to run (the replaced-link Note, the `-ncd` refusal's `-d` form) print it with Python's escapes, such as `hpcp W/s/d\udcc3\udca9/ -d W/o/d\udcc3\udca9/`, which a shell cannot run as pasted.
- **Why:** the hints print the path as Python holds it; messages do the same, which is fine for reading.
- **Plan:** shell-quote the names in a hint as coreutils does (`$'d\303\251'`); messages keep the `\udcXX` forms.
- **Revisit when:** 9.86.
- **Noted:** 2026-11-08, the user's standing rule that deferrals are noted, not asked.

### With LANG unset, a `-siff` apply stops at the first non-ASCII name
- **Kind:** deferred (9.86).
- **What the operator sees:** with LANG unset (Python 3.6 on EL7, as under cron), the `-siff` patch apply reads the names in its change list, manifest and disk TSV (`read_changes`, `read_manifest`, `read_disk_tsv`, through `_load_patch`) as UTF-8 text: `new\xc3\xa9` reads back as `'new\xe9'`, which matches no name `os.listdir` gives (`'new\udcc3\udca9'`), and the apply stops with a UnicodeEncodeError on the first such name. A patch made with LANG unset applies whole under `LANG=en_US.UTF-8`, and one made under it fails applied with LANG unset (checked in the EL7 image, 9.84 and 9.85 alike). Making the patch (`-diff`) and `-dfl` lists are not affected: hpcp writes those correctly under any locale and never reads `-dfl` back. The readers are unchanged since they came in (2026-10-10), so 9.83 has it too (not run).
- **Workaround:** run the `-siff` apply with `LANG=en_US.UTF-8`; the locale must be installed (`locale -a` lists `en_US.utf8`), or Python 3.6 falls back to ASCII and nothing changes.
- **Why:** they open the file as UTF-8 text, where `_read_list_entries` reads bytes and decodes each name with `os.fsdecode`.
- **Plan:** read those files in binary and `os.fsdecode` each name field, as `_read_list_entries` does, with a LANG-unset test.
- **Revisit when:** 9.86.
- **Noted:** 2026-11-08, the user's standing rule that deferrals are noted, not asked.

### A stop right after a rewrite in place reserves room can remove another user's old copy
- **Kind:** deferred (9.86).
- **What the operator sees:** a non-root run rewriting in place a sparse old copy that another user owns but the runner may write (group-writable, or an ACL): the room for the source's data is reserved first (`posix_fallocate`), which moves the old copy's mtime, and only the owner (or root) may set it back. A stop (Ctrl-C, SIGTERM) landing after the reservation and before the first write then finds the old copy's time changed, takes it for a torn copy and removes it, though it still reads as it did; a reservation that falls short (exit 190) keeps the old copy but with a new mtime.
- **Why:** `_cp_in_place` tells a copy it wrote from the old one by inode, size and mtime, and the time cannot be put back without owning the file. The window is the reservation's own few calls; a sparse old copy of another user's, in a directory that refuses new names or on a nearly full disk, is rare.
- **Plan:** remove a failed in-place copy only once its first byte was written (the rewrite reports how far it got), not by its times; with the stop backstop of 9.86.
- **Revisit when:** 9.86.
- **Noted:** 2026-11-09, from the second opinion on 9.85's last review; the main session's decision, not asked (a deferral).

### A rewrite in place on a copy-on-write filesystem or a reflinked XFS file near a full disk can lose the old copy
- **Kind:** deferred (9.86).
- **What the operator sees:** a rewrite written in place (a 0555 directory, a sticky directory, a bind-mounted file, or a temporary file beside it that ran out of room) on a nearly full disk:
  - on btrfs, ZFS or bcachefs it goes to `cp`, as in 9.84: `cp` truncates the old copy first (or clones a source on the same filesystem), so another writer, such as another worker's temporary file, can take the freed room, and the old copy is torn and removed, exit 190, as in 9.84. Written over instead, the old copy needed room for all of it: on Debian 12 btrfs (48 MiB rewritten with 4 or 24 MiB free, found in the 9.85 last review) the overwrite tore it in all 19 direct runs and in 5 of 30 runs of hpcp 3dee92d (exit 190, the old copy removed), where `cp` wrote it whole every time (16 runs of `cp` alone, 16 of hpcp 9.84);
  - for an XFS file whose extents a reflink shares (a `cp --reflink` copy, or one an earlier `cp` of coreutils 9.0 or later cloned from a source on the same filesystem), the overwrite can run out of room part way: the old copy is torn (and removed where its directory lets it go), exit 190. As in 9.84 with coreutils before 9.0 (EL8: both tore); with coreutils 9.0 or later and the source on the same filesystem, 9.84's `cp` cloned it and needed no room (Debian 12, coreutils 9.1).
- **Why:** there an overwrite allocates new blocks, so writing over the old copy needs room as a new file does; the room reserved first (`posix_fallocate`) covers only the source's data landing in the old copy's holes, and on these filesystems it does not cover an overwrite. Telling a shared XFS extent needs FIEMAP.
- **Plan:** on such a filesystem or file, write in place only when the free room holds the source's data, otherwise keep the old copy (exit 190); for a source on the same filesystem, clone it (`FICLONE`), which needs no room.
- **Revisit when:** 9.86.
- **Noted:** 2026-11-09, found building 9.85's in-place rewrite; amended the same day after the last review sent btrfs, ZFS and bcachefs to `cp`; the main session's decision, not asked (a deferral).

### Each rewrite makes a new inode, with no way to write in place instead
- **Kind:** deferred (9.86).
- **What the operator sees:** a rewrite goes to a new file renamed over the old one, so what lives on the old inode is not carried: a per-file Lustre or GPFS layout (`lfs setstripe` on the file) is replaced by the directory's default, and readers holding the file open on other NFS clients may get `ESTALE`. There is no option to write in place instead.
- **Why:** the new inode is what keeps the old copy whole when a source fails on read, the disk fills or the run stops (9.85's core).
- **Plan:** carry a Lustre layout onto the temporary file (`lfs getstripe` / `setstripe` before the copy, where `lfs` is there), and add an opt-out that writes in place, for operators who weigh those costs over the safety.
- **Revisit when:** 9.86.
- **Noted:** 2026-11-08, the user chose "Plan it for 9.86 (Recommended)".

### `-rme` keeps an extra directory named like one beside a source
- **Kind:** deferred (9.86).
- **What the operator sees:** `hpcp a/ b -d out/ -rme`, run where a directory `zzz` exists beside `b` (in the working directory), keeps an extra `out/zzz` that no source has, exit 0 (checked in Docker at 9.85). A directory is judged extra by whether its path below the destination exists below each source's parent (`dirname(src)`), and `b`'s parent is the working directory, which holds `zzz`. Only a kept extra, never a loss: what a source is or holds is guarded by identity.
- **Why:** found in 9.84's review; files are judged by where the copy put them (`_landings`), directories still by a path below each source's parent.
- **Plan:** judge a directory as `remove_extra_files` judges a file: by the landings of the directory sources' listed folders, with the identity guard as it is.
- **Revisit when:** 9.86.
- **Noted:** 2026-11-08, the user's standing rule that deferrals are noted, not asked.

### The copy check stats each landing by name
- **Kind:** deferred (9.86).
- **What the operator sees:** the check at the end of a copy, and the move's before its removal, `lstat`s each landing on its own: on a cached local disk about 0.3 s per 100,000 files idle, and more under load (a cached 100,000-file rerun at load 110 took 1.32-1.73 s in 9.84 and 1.81-2.01 s in 9.85), and on NFS mounted `actimeo=0` a round trip each, about 0.3 ms a file in 4 threads.
- **Why:** correct as it is; reading each landing directory once is its own change.
- **Plan:** read each landing directory once (`scandir`), take each name's file from the listing where the listing gives one, and `lstat` only the names missing from it (an 8.3 alias is not listed under its short name) or where the listing's numbers cannot be trusted (vfat makes up a number for an uncached inode, FUSE exFAT one for each spelling). And when two fingerprints meet, the second pass now stats every landing again (about 0.45 s per 100,000 files, paid by any merge whose sources share a file, or a source named twice): keep each landing's entry index beside its fingerprint, so it stats only the entries met twice.
- **Revisit when:** 9.86.
- **Noted:** 2026-11-08, the user's standing rule that deferrals are noted, not asked.

### A move's directories are not fsynced
- **Kind:** deferred (9.86).
- **What the operator sees:** in a move across filesystems, each file is fsynced by its worker and each link's directory once after the last link, but a directory is fsynced only through those. After a power cut within the journal interval (ext4 5 s, XFS up to 30 s), an empty directory the move made, or the mode, owner, xattrs or times of a directory with no file or link made after it on that filesystem, can be missing at the destination while the source is already gone. On ext4 and XFS the fsync of a file or link also commits every earlier change on its filesystem (the directories made before it), so only the last directories of a tree with nothing after them are open: a tree of only empty directories, or the last sources of an `-nb` run that hold only directories. A file's content and name are never at risk by this.
- **Why:** fsyncing each directory as the folder sync makes it costs 2.6 to 7.8 ms a directory, found in the 9.85 re-review (2,040 empty directories moved in 11.8 to 16.2 s on ext4 and 5.4 to 6.6 s on XFS, on EL7 and EL8, against about 0.4 to 0.6 s). Fsynced once each after all are made it is nearly free (1,000 directories and their 40 parents: 0.02 s on ext4 and 0.07 to 0.23 s on XFS, EL7). The cheap way needs plumbing through `sync_directories_*_batch`, `copy_scheduler` and `process_copy`.
- **Plan (9.86):** the folder sync returns the destination directories it made or set in a move, as it returns links; after `create_sym_links` the parent fsyncs them and the link directories together, once each, and a failure is a Copy failed that stops the removal.
- **Revisit when:** 9.86.
- **Noted:** 2026-11-09, found in the 9.85 re-review; the main session's decision, not asked (a deferral).

### On macOS a move's fsync does not flush the drive's cache
- **Kind:** deferred (9.86).
- **What the operator sees:** `os.fsync` on macOS returns once the data is in the drive's own write cache (Apple documents `fcntl(F_FULLFSYNC)` for the flush), so after a move a power cut can lose both the copy and the already removed source. Read from Apple's documentation; hpcp 9.85 has not been run on a Mac and the flush has not been measured.
- **Why:** `F_FULLFSYNC` is slow on a Mac (it waits for the drive), so adding it blind could cost a move a lot, and there is no Mac here to measure.
- **Plan (9.86):** use `fcntl(fd, F_FULLFSYNC)` where `fcntl` has it, measured on a Mac before it ships.
- **Revisit when:** 9.86, or when a Mac run is possible.
- **Noted:** 2026-11-09, found in the 9.85 re-review; the main session's decision, not asked (a deferral).

### A move's fsync waits are serial in each worker
- **Kind:** deferred (9.86).
- **What the operator sees:** a move of many small files onto a slow disk takes about 1.6 to 1.9 times as long as without the sync (10,000 files of 4 KiB on an EL7 VM: 79 to 92 s on ext4 and 72 to 82 s on XFS, against 47 to 54 s and 43 to 47 s), because each worker waits for its own fsync before it copies the next file. A higher `-j` hides the wait: 2,000 files of 4 KiB with the default (one worker per CPU: 2 on the EL7 VM, 4 on the EL8 VM) against `-j 16`: 18.6 to 22.9 s against 10.3 to 11.7 s (EL7 ext4), 17.9 to 21.2 s against 10.1 to 11.7 s (EL7 XFS), 11.9 to 14.0 s against 9.3 to 11.8 s (EL8 ext4), 14.2 to 15.8 s against 8.2 to 10.5 s (EL8 XFS), 3 runs each. That the gain is overlapped waits and not CPU is an inference: the VMs have 2 and 4 vCPUs.
- **Why:** more fsyncs in flight are batched by the filesystem into fewer journal commits, but a worker has one fsync in flight at a time; overlapping them (a few fsync threads per worker, or the parent's loop) is its own change.
- **Plan (9.86):** overlap a worker's fsyncs, measured against `-j 16` on the same trees; until then the README says to raise `-j`.
- **Revisit when:** 9.86.
- **Noted:** 2026-11-09, found in the 9.85 re-review; the main session's decision, not asked (a deferral).

### Worker error lines glue onto the progress bar in a log
- **Kind:** deferred (9.86).
- **What the operator sees:** with output redirected to a file, an error a worker prints (`Copy failed: ...`) can land after the progress bar on the bar's line, so `grep '^Copy failed'` misses it; `grep 'Copy failed'` finds it. Errors the main process prints start their own line (fixed in 9.85). Checked in 9.85 in Docker (Python 3.14 and 3.6): 12 files that cannot be read gave 9 to 11 lines that start with `Copy failed` in each of 6 runs, and the re-review counted the same on 13dcfa8, a commit earlier in 9.85's work, so it is not new in this round (9.84 itself was not run). Nothing is lost: the error is also in the end summary and the exit code is right.
- **Why:** not traced yet. The bar ends its line with a carriage return and no newline, and a worker's error is written from another process, where the parent's flush-first (`eprint`) cannot order it; that is a reading, not a finding.
- **Plan (9.86):** have a worker's error text go back to the parent with the task's results and print there, or end the bar's line before an error is printed.
- **Revisit when:** 9.86.
- **Noted:** 2026-11-09, found in the 9.85 re-review; the main session's decision, not asked (a deferral).

### A stopped or broken run can still hang, rarely
- **Kind:** deferred (9.86).
- **What the operator sees:** on Python 3.6 (EL7, EL8) and 3.14, a run stopped early with Ctrl-C or SIGTERM (`kill`, `timeout`, a systemd stop), or one whose copy worker dies, can wait for good instead of ending. A second Ctrl-C does nothing there (9.85's parent ignores later stops); Ctrl-\ (SIGQUIT) or `kill -9` ends it, and like SIGKILL leaves a `.hpcp_tmp_` file where a rewrite was under way (its `cp`, in a process group of its own, runs on to its end). Measured with the reviewer's probe (early stops, 20 directories of 20 small files, `-j 8`, across EL7, EL8 and Docker 3.6 and 3.14): 9 hangs in 3300 runs with 9.85, against 22 of 3300 at the commit before the last fixes; EL8 early Ctrl-C alone, 4 of 300. 9.84 hung about as often (EL8: 3 of 150 Ctrl-Cs, 4 of 150 SIGTERMs). Sources stay: a stopped or broken copy removes nothing.
- **Why:** the waits are in Python's process pool, not in hpcp's own code. A stop raised inside CPython's own lock handling strands a lock: the parent inside `concurrent.futures.wait`'s `_AcquireFutures` (verified by injection), or a worker on the pool's call-queue lock (inferred from the hang dumps, not injected). A stop that lands between the two writes of a result over 16 KiB also hangs (verified by injection).
- **Plan (9.86):** kill workers still alive after a grace period, bound the parent's join and end with `os._exit`; consider polling the stop flag in the pool loops instead of raising in the parent; let a second Ctrl-C end a stuck parent.
- **Revisit when:** 9.86.
- **Noted:** 2026-11-09, found in the 9.85 re-reviews; the main session's decision, not asked (a deferral).

### A stop whose KeyboardInterrupt Python drops in the parent is not acted on
- **Kind:** deferred (9.86).
- **What the operator sees:** a stop whose `KeyboardInterrupt` Python drops in the parent (it was raised inside a weakref callback or a `__del__`, where every Python reports it as "Exception ignored" and goes on) is not acted on there. While a pool runs, its workers still end on the pass-on, the pool reads as broken, and the run ends with `Copy process crashed: BrokenProcessPool`, exit 191 for a Ctrl-C instead of 130 (a SIGTERM still ends by its signal, 143); seen once in 200 natural runs on Python 3.14. Outside a pool, nothing breaks: the run goes on to its end, a move removing its sources and `-rme` its extras, and a Ctrl-C exits 0; a second Ctrl-C is ignored as a later stop (injected on 3.6 and 3.14: a Ctrl-C dropped while the sources were checked, a second 0.2 s later, and the move ran whole, exit 0). Nothing is lost: what is removed was copied, fsynced and checked first.
- **Why:** the parent acts on a stop only through the `KeyboardInterrupt` that `_stop` raises, and ignores later stops so that a second Ctrl-C cannot cut the cleanup halfway. Python 3.6 to 3.14 all drop a `KeyboardInterrupt` raised in a weakref callback or a `__del__` (checked: 3.6, 3.12, 3.14); such callbacks run in the parent's main thread as a pool's executor, queues and threads are freed (from reading Python's code, not traced in hpcp: `concurrent.futures`' `weakref_cb`, multiprocessing's `Finalize`, `threading`'s `WeakSet`), so the window is at least the end of each pool phase. Why only 3.14 showed it is not traced.
- **Plan (9.86):** the parent checks its own stop (`_STOPPED_PID == os.getpid()`) where it can act on it, and raises `KeyboardInterrupt` there: each pass of its pool loops, and before each phase, above all before a removal. That also covers the broken pool (exit 130 or the signal, not 191).
- **Revisit when:** 9.86.
- **Noted:** 2026-11-09, found in the 9.85 last review round; the main session's decision, not asked (a deferral).

### On Python 3.14 an early Ctrl-C often prints a traceback
- **Kind:** deferred (9.86).
- **What the operator sees:** on Python 3.14, about one early Ctrl-C in three prints a traceback (a `KeyboardInterrupt`) before the run ends with 130. As in 9.84. Nothing is lost.
- **Why:** under 3.14's default forkserver a worker starts from the fork server, with Python's own SIGINT handler (the server puts back the handlers it started with), and hpcp's handlers go in only at the worker's first task (`_run_task`): the after-fork setup that 9.85's `main()` registers reaches only workers forked from the parent. A Ctrl-C landing before that prints Python's traceback; and a second stop landing in an exit path does the same.
- **Plan (9.86):** set each worker up before its loop on Python 3.7+ with the pool's `initializer=` (as `-pfl` already does for its rate share), which reaches forkserver and spawn workers too; and the exit paths end on the first stop (as `_end_by` now does in a worker). Ignoring the stops until the handlers are in place cannot be done from the parent under forkserver (the server, not the parent, gives a worker its dispositions), and a stop ignored at a worker's start would stay ignored (`_install_stops` keeps what a process started ignoring).
- **Revisit when:** 9.86.
- **Noted:** 2026-11-09, found in the 9.85 last review round; the main session's decision, not asked (a deferral).

### A destination given through `link/..` also makes an empty directory where the link leads
- **Kind:** deferred (9.86).
- **What the operator sees:** `hpcp a/ -d l/../out/`, with `l` a link to `deep/inner`, copies into `out/` (the path is folded as written) and also leaves an empty `deep/out/`, made by following the link. 9.84 does the same. Nothing is lost.
- **Why:** one step makes the destination's directory from the path as given, where the kernel follows `l` before `..`, while the copy uses the folded path. Not traced to the line yet.
- **Plan (9.86):** fold `-d` once, at the start, and use only the folded path.
- **Revisit when:** 9.86.
- **Noted:** 2026-11-09, found in the 9.85 re-review (checked in Docker on 9.85 and 9.84); the main session's decision, not asked (a deferral).
