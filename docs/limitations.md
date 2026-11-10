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
- **Plan (9.87):** cut the per-file checks that ask again what is known. One is new in 9.85: copy_file's look-only pass through the door calls `_landing_of(dest)` only for its path, an lstat per destination per file, a round trip each on NFS `actimeo=0`; the path needs no lstat there.
- **Revisit when:** 9.87, with the copy check's second pass (stat only the entries met twice).

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

### A temporary file left by SIGKILL or Ctrl-\ (SIGQUIT)
- **Kind:** known limitation.
- **What the operator sees:** hpcp killed with SIGKILL, ended with Ctrl-\ (SIGQUIT), or cut by a power cut while it rewrites a file on the beside route (every rewrite with `-wb`; by default a file with other names, one that cannot be opened for writing, or a FIFO, socket, device node or link at the landing; and a `-siff` apply's files) leaves a hidden `.hpcp_tmp_` file (16 hex digits after the prefix) beside it, in the destination; the file it was rewriting is untouched, its old copy whole. A default rewrite in place leaves no such file (see "A stop during a rewrite costs the old copy"). A rerun leaves the temporary file there; `-rme` removes it as an extra. SIGINT, SIGTERM and SIGHUP remove it (rule 9).
- **Why:** SIGKILL cannot be handled, and a rewrite that keeps the old copy whole needs the new one under another name until it is. Ctrl-\ is left without a handler on purpose: it is the way out of a run that is stuck (see "A stopped run waits for a worker whose file operation never returns"), so it must not wait for a cleanup.
- **Revisit when:** an operator reports finding one.
- **Approved:** 2026-11-08, the user chose "Approve (Recommended)"; widened to Ctrl-\ (SIGQUIT) 2026-11-09, the user chose "Approve (Recommended)" ("Ctrl-\ stays an instant exit"). Narrowed 2026-11-09 (9.86) to the beside route, which is no longer every rewrite.

### Windows: nothing is fsynced
- **Kind:** known limitation.
- **What the operator sees:** on Windows hpcp fsyncs no copy, so a move there can lose both the copy (still in memory) and the source (its removal already on disk) to a power cut, as with 9.84.
- **Why:** Windows' fsync wants a descriptor open for writing, and there is no Windows machine here to build and test it on.
- **Revisit when:** a Windows machine is available, or an operator runs hpcp moves on Windows.
- **Approved:** 2026-11-09, the user chose "Approve (Recommended)".

### A move's files and links in another user's unreadable directory are not fsynced
- **Kind:** known limitation.
- **What the operator sees:** a non-root move that writes files or links into a directory another user owns and the runner may not read (a write-only drop box) does not fsync that directory. A power cut soon after the move can lose those names while their sources are already gone. Every other directory a move writes into is fsynced once, before any source is removed.
- **Why:** a directory is fsynced through a descriptor, and one the runner may not read cannot be opened (an `O_PATH` descriptor refuses `fsync` and `syncfs`). The alternative, `syncfs` of the whole filesystem, can take seconds on a busy host, for a rare case.
- **Revisit when:** an operator reports moving into such a directory.
- **Approved:** 2026-11-09, the user chose "Approve (Recommended)" (links). Widened to files 2026-11-10: the user chose "Widen the KL to files". The user did not ask for it explicitly; the files part remains of entry 12, which the 9.85 review found.

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
- **Revisit when:** 9.87.

### Directories that differ only in case on vfat mounted `iocharset=utf8`
- **Kind:** deferred (9.87).
- **What the operator sees:** on a vfat mounted with `iocharset=utf8` given explicitly, `hpcp a/ b/ -d out/` with `a/D/f` and `b/d/g` exits 193 (`Directory sync error: .../b/d to .../out/d: [Errno 2] No such file or directory`, then `Copy failed ... No more destination paths to try`), and `g` is not copied; with `-rf` it exits 212 and keeps both sources. A single source does it too: `out/D` from an earlier run, and a source that now spells it `d`, exit 193 and nothing below it copied (checked on EL7 by the 9.85 designer). Nothing is lost, and 9.82 does the same. Checked on VM 901; vfat with its default charset, `utf8=1` or `iocharset=iso8859-1`, kernel exFAT, NTFS (`ignore_case`) and ext4 casefold merge the two, exit 0. Since 9.85 the directory door says so: `mkdir` meets `EEXIST` twice where `lstat` finds nothing, and the 193 names the mount option to use instead.
- **Why:** vfat mounted that way looks a name up in its own case only (`out/d` is not found) but will not make it either, as `out/D` holds the name; the copy names the directory as the source spells it. A loud failure with nothing lost, on a mount option the kernel's vfat documentation advises against.
- **Plan:** in `_dest_directory`, when `mkdir` says `EEXIST` and `lstat` says `ENOENT`, list the parent and take the one entry whose fold matches as the spelling for that directory (cached), so the folder sync, the files and the links below it land there; the move's backstop asks the same helper for the spelling. More than one entry, or none, stays a 193 that says why.
- **Revisit when:** 9.87.
- **Noted:** 2026-11-08, the user's standing rule that deferrals are noted, not asked.

### A user who can write the destination can race a check by path
- **Kind:** deferred (9.87).
- **What the operator sees:** a user who can write a destination directory can swap a directory on the way to a landing for a link between hpcp's look (`_dest_directory`) and `cp`'s open, and a root run then writes where the link leads; the same holds for a first copy's name (nothing there at the look, a link there when `cp` opens it), and for a landing removed between hpcp's look and its open, which is then a first copy. The landing of a rewrite no longer races on Linux since 9.86: a rewrite in place goes through the file hpcp holds open (`/proc/PID/fd/N`), and the beside route renames a new file over the name, never writing through it. Off Linux a rewrite in place still gives `cp` the name (see "Off Linux, a rewrite in place can be swapped for a link mid-run"). The static plant (a link left there before the run) is replaced since 9.85; only the race in that moment remains. The rerun's skip branch (a file identical to its source is left as it is) sets that landing's owner and mode by name (`_chown_like_cp`, `os.chmod`), and both follow a link, so a landing swapped for one between hpcp's look and those calls passes them on to what it leads to (9.84, 9.85 and 9.86 alike; found by the 9.86 review). On the beside route a plain (not `-co`) rewrite removes stale xattrs from the new copy, and fsyncs it, by its temporary name: a second name of another file put there at that moment (the kernel lets a user link only a file they own or may read and write, `fs.protected_hardlinks`) loses the `user.*` xattrs, ACL and file capability that the source lacks (9.85 on every rewrite, 9.86 on the beside route; found by the 9.86 final review).
- **Why:** closing it needs each landing's directory opened by descriptors from the destination root with `O_NOFOLLOW`, as the removal does (`_descend`), and `cp` run in that directory: its own change.
- **Plan:** open each landing directory from the root by descriptors (`O_NOFOLLOW` at each step), `fchdir` to it in the worker and give `cp` relative names; make a first copy's file in hpcp (`O_CREAT|O_EXCL|O_NOFOLLOW`) and hand it to `cp` as a rewrite in place is handed, with `-co`'s mode set to the source's mode less the umask. Do the skip branch's owner and mode through one `O_PATH|O_NOFOLLOW` descriptor checked against the look's inode, as `_carry_over` does since 9.86 for `-co` on the beside route. Remove the beside route's stale xattrs through that same descriptor.
- **Revisit when:** 9.87.
- **Noted:** 2026-11-08, the user's standing rule that deferrals are noted, not asked; the skip branch's metadata, the landing removed before the open and the beside route's stale xattrs added 2026-11-09 (found by the 9.86 reviews, the main session's decision, not asked).

### `-co` reruns rewrite every unchanged file
- **Kind:** deferred (9.87).
- **What the operator sees:** a second `hpcp -co a/ -d out/` with nothing changed copies every file again ("Total files copied" counts them all, and each landing's mtime moves on each run). The cost is the whole copy on every rerun.
- **Why:** `-co` carries no times, so the landing's mtime is that of the last copy and the skip check (size and mtime) never matches.
- **Plan:** under `-co`, where the size matches and only the time differs, compare the full contents (`_same_contents`, as the move's backstop does) and skip a file that reads the same. Not `is_file_identical`'s hash: with the default `-hs 65536` it reads only the last 64 KiB, and a same-size change before that would be skipped on every rerun.
- **Revisit when:** 9.87.
- **Noted:** 2026-11-08, the user's standing rule that deferrals are noted, not asked.

### Hints quote names as Python escapes, not for the shell
- **Kind:** deferred (9.87).
- **What the operator sees:** with a name the locale cannot show (any non-ASCII name with LANG unset, a non-UTF-8 one under a UTF-8 locale), the hints that give a command to run (the replaced-link Note, the `-ncd` refusal's `-d` form) print it with Python's escapes, such as `hpcp W/s/d\udcc3\udca9/ -d W/o/d\udcc3\udca9/`, which a shell cannot run as pasted.
- **Why:** the hints print the path as Python holds it; messages do the same, which is fine for reading.
- **Plan:** shell-quote the names in a hint as coreutils does (`$'d\303\251'`); messages keep the `\udcXX` forms.
- **Revisit when:** 9.87.
- **Noted:** 2026-11-08, the user's standing rule that deferrals are noted, not asked.

### With LANG unset, a `-siff` apply stops at the first non-ASCII name
- **Kind:** deferred (9.87).
- **What the operator sees:** with LANG unset (Python 3.6 on EL7, as under cron), the `-siff` patch apply reads the names in its change list, manifest and disk TSV (`read_changes`, `read_manifest`, `read_disk_tsv`, through `_load_patch`) as UTF-8 text: `new\xc3\xa9` reads back as `'new\xe9'`, which matches no name `os.listdir` gives (`'new\udcc3\udca9'`), and the apply stops with a UnicodeEncodeError on the first such name. A patch made with LANG unset applies whole under `LANG=en_US.UTF-8`, and one made under it fails applied with LANG unset (checked in the EL7 image, 9.84 and 9.85 alike). Making the patch (`-diff`) and `-dfl` lists are not affected: hpcp writes those correctly under any locale and never reads `-dfl` back. The readers are unchanged since they came in (2026-10-10), so 9.83 has it too (not run).
- **Workaround:** run the `-siff` apply with `LANG=en_US.UTF-8`; the locale must be installed (`locale -a` lists `en_US.utf8`), or Python 3.6 falls back to ASCII and nothing changes.
- **Why:** they open the file as UTF-8 text, where `_read_list_entries` reads bytes and decodes each name with `os.fsdecode`.
- **Plan:** read those files in binary and `os.fsdecode` each name field, as `_read_list_entries` does, with a LANG-unset test.
- **Revisit when:** 9.87.
- **Noted:** 2026-11-08, the user's standing rule that deferrals are noted, not asked.

### `-rme` keeps an extra directory named like one beside a source
- **Kind:** deferred (9.87).
- **What the operator sees:** `hpcp a/ b -d out/ -rme`, run where a directory `zzz` exists beside `b` (in the working directory), keeps an extra `out/zzz` that no source has, exit 0 (checked in Docker at 9.85). A directory is judged extra by whether its path below the destination exists below each source's parent (`dirname(src)`), and `b`'s parent is the working directory, which holds `zzz`. Only a kept extra, never a loss: what a source is or holds is guarded by identity.
- **Why:** found in 9.84's review; files are judged by where the copy put them (`_landings`), directories still by a path below each source's parent.
- **Plan:** judge a directory as `remove_extra_files` judges a file: by the landings of the directory sources' listed folders, with the identity guard as it is.
- **Revisit when:** 9.87.
- **Noted:** 2026-11-08, the user's standing rule that deferrals are noted, not asked.

### The copy check stats each landing by name
- **Kind:** deferred (9.87).
- **What the operator sees:** the check at the end of a copy, and the move's before its removal, `lstat`s each landing on its own: on a cached local disk about 0.3 s per 100,000 files idle, and more under load (a cached 100,000-file rerun at load 110 took 1.32-1.73 s in 9.84 and 1.81-2.01 s in 9.85), and on NFS mounted `actimeo=0` a round trip each, about 0.3 ms a file in 4 threads.
- **Why:** correct as it is; reading each landing directory once is its own change.
- **Plan:** read each landing directory once (`scandir`), take each name's file from the listing where the listing gives one, and `lstat` only the names missing from it (an 8.3 alias is not listed under its short name) or where the listing's numbers cannot be trusted (vfat makes up a number for an uncached inode, FUSE exFAT one for each spelling). And when two fingerprints meet, the second pass now stats every landing again (about 0.45 s per 100,000 files, paid by any merge whose sources share a file, or a source named twice): keep each landing's entry index beside its fingerprint, so it stats only the entries met twice.
- **Revisit when:** 9.87.
- **Noted:** 2026-11-08, the user's standing rule that deferrals are noted, not asked.

### A move's fsync waits are serial in each worker
- **Kind:** deferred (9.87).
- **What the operator sees:** a move of many small files onto a slow disk takes about 1.6 to 1.9 times as long as without the sync (10,000 files of 4 KiB on an EL7 VM: 79 to 92 s on ext4 and 72 to 82 s on XFS, against 47 to 54 s and 43 to 47 s), because each worker waits for its own fsync before it copies the next file. A higher `-j` hides the wait: 2,000 files of 4 KiB with the default (one worker per CPU: 2 on the EL7 VM, 4 on the EL8 VM) against `-j 16`: 18.6 to 22.9 s against 10.3 to 11.7 s (EL7 ext4), 17.9 to 21.2 s against 10.1 to 11.7 s (EL7 XFS), 11.9 to 14.0 s against 9.3 to 11.8 s (EL8 ext4), 14.2 to 15.8 s against 8.2 to 10.5 s (EL8 XFS), 3 runs each. That the gain is overlapped waits and not CPU is an inference: the VMs have 2 and 4 vCPUs.
- **Why:** more fsyncs in flight are batched by the filesystem into fewer journal commits, but a worker has one fsync in flight at a time; overlapping them (a few fsync threads per worker, or the parent's loop) is its own change.
- **Plan (9.87):** overlap a worker's fsyncs, measured against `-j 16` on the same trees; until then the README says to raise `-j`.
- **Revisit when:** 9.87.
- **Noted:** 2026-11-09, found in the 9.85 re-review; the main session's decision, not asked (a deferral).

### Worker error lines glue onto the progress bar in a log
- **Kind:** deferred (9.87).
- **What the operator sees:** with output redirected to a file, an error a worker prints (`Copy failed: ...`) can land after the progress bar on the bar's line, so `grep '^Copy failed'` misses it; `grep 'Copy failed'` finds it. Errors the main process prints start their own line (fixed in 9.85). Checked in 9.85 in Docker (Python 3.14 and 3.6): 12 files that cannot be read gave 9 to 11 lines that start with `Copy failed` in each of 6 runs, and the re-review counted the same on 13dcfa8, a commit earlier in 9.85's work, so it is not new in this round (9.84 itself was not run). Nothing is lost: the error is also in the end summary and the exit code is right.
- **Why:** not traced yet. The bar ends its line with a carriage return and no newline, and a worker's error is written from another process, where the parent's flush-first (`eprint`) cannot order it; that is a reading, not a finding.
- **Plan (9.87):** have a worker's error text go back to the parent with the task's results and print there, or end the bar's line before an error is printed.
- **Revisit when:** 9.87.
- **Noted:** 2026-11-09, found in the 9.85 re-review; the main session's decision, not asked (a deferral).

### A stopped run waits for a worker whose file operation never returns
- **Kind:** known limitation.
- **What the operator sees:** after Ctrl-C, SIGTERM or SIGHUP, a worker inside a write or fsync that never returns holds the run (for example a dead hard-mounted NFS server, or a disk in an uninterruptible wait). hpcp prints `Waiting for N worker(s) still finishing a file (pids ...): Ctrl-\ ends hpcp now, which may leave a temporary file or a torn file there`. Ctrl-\ ends it at once, and can leave that worker's temporary file or a torn file. Nothing is removed from the sources.
- **Why:** killing a worker in a task would leave what its cleanup removes, which the approved limitations promise a stop removes. An uninterruptible wait cannot be killed at all.
- **Revisit when:** an operator reports one.
- **Approved:** 2026-11-10, the user chose "Approve as KL (Recommended)". The user did not ask for it explicitly; it is what remains of the hang the 9.85 review found.

### A destination given through `link/..` also makes an empty directory where the link leads
- **Kind:** deferred (9.87).
- **What the operator sees:** `hpcp a/ -d l/../out/`, with `l` a link to `deep/inner`, copies into `out/` (the path is folded as written) and also leaves an empty `deep/out/`, made by following the link. 9.84 does the same. Nothing is lost.
- **Why:** one step makes the destination's directory from the path as given, where the kernel follows `l` before `..`, while the copy uses the folded path. Not traced to the line yet.
- **Plan (9.87):** fold `-d` once, at the start, and use only the folded path.
- **Revisit when:** 9.87.
- **Noted:** 2026-11-09, found in the 9.85 re-review (checked in Docker on 9.85 and 9.84); the main session's decision, not asked (a deferral).

### A source that fails part way through a rewrite costs the old copy
- **Kind:** known limitation.
- **What the operator sees:** by default a rewrite writes the file in place; when the source fails on read part way (a bad sector, an NFS soft timeout), the copy exits 190 with the source kept, and the old copy, truncated first, is removed (or left torn where its directory keeps it). The destination then has no copy of that file until a rerun. As in 9.84; 9.85 kept the old copy.
- **Why:** writing in place keeps the inode and costs no second copy; `-rbw` (read the source first) or `-wb` keep the old copy when asked.
- **Revisit when:** an operator loses a last good copy this way.
- **Approved:** 2026-11-09, the user chose "Approve (Recommended)"; their words: "even if we overwritten a good file with a bad one, that is exactly what the user requested so it should be fine?"

### A stop during a rewrite costs the old copy
- **Kind:** known limitation.
- **What the operator sees:** Ctrl-C, SIGTERM or SIGHUP during a rewrite in place stops `cp` and removes the torn file; SIGKILL or Ctrl-\ (SIGQUIT) can leave it torn at its name, and a rerun rewrites it (its time differs from the source's). The source is kept, and a stopped move removes nothing. As in 9.84; 9.85 kept the old copy.
- **Why:** a write in place truncates first; `-wb` keeps the old copy until the new one is whole.
- **Revisit when:** an operator needs stops to keep old copies without `-wb`.
- **Approved:** 2026-11-09, the user chose "Approve (Recommended)".

### Another program reading a file while it is rewritten in place sees it empty or partly written
- **Kind:** known limitation.
- **What the operator sees:** while hpcp rewrites a file in place (the default since 9.86), `cp` truncates it and then writes it, so another program reading that file meanwhile sees it empty or partly written, as with plain `cp` and in 9.84. 9.85 wrote beside and renamed, so readers saw the whole old or the whole new file.
- **Why:** it comes with writing in place, which the user chose so the inode stays (a layout, `chattr` flags, a project id, NFS readers); `-wb` gives readers the whole-file view when asked.
- **Revisit when:** an operator needs readers to see whole files without `-wb`.
- **Approved:** 2026-11-09, the user picked "Approve (Recommended)". The user did not ask for it explicitly; the 9.86 final review found it undocumented.

### While a file is rewritten in place, anyone who could read the old file can read the new content
- **Kind:** known limitation.
- **What the operator sees:** a rewrite in place keeps the old file's mode, owner and SELinux label until `cp` ends, so during the copy anyone who could read the old file can open it and read the new content: an existing 0644 file overwritten from a 0600 source is readable by other users while it is written (checked in Docker with nobody as the other user). A program that already had the old file open keeps reading the new content for good. As in 9.84, plain `cp` and `rsync --inplace`; 9.85 wrote a new file beside and renamed it, so neither happened.
- **Why:** it comes with writing in place, which the user chose so the inode stays. `-wb` avoids all of it.
- **Revisit when:** an operator needs a more private source to stay unreadable during a rewrite without `-wb`.
- **Approved:** 2026-11-09, the user picked "KL, as 9.84" (a known limitation with no narrowing planned, `-wb` the way out), over narrowing it before release (giving the held file the tighter mode, the source's owner and label before `cp` writes) or in 9.87. The user did not ask for it explicitly; the 9.86 re-review found it.

### A power cut during or soon after a rewrite outside a move can leave the file partial or empty
- **Kind:** known limitation.
- **What the operator sees:** after a power cut, a file rewritten in place by a copy (not a move) can be partial, or empty on XFS, as in 9.84. A move is not affected: each landing is fsynced before any source goes. 9.85 fsynced each rewrite before its rename.
- **Why:** an fsync per rewrite costs about 1 to 1.5 ms a file on EL7, for a copy whose source is still there; `-wb` fsyncs each new copy.
- **Revisit when:** an operator needs crash-safe plain copies without `-wb`.
- **Approved:** 2026-11-09, the user chose "Approve (Recommended)".

### A rewrite on a nearly full disk can lose the old copy
- **Kind:** known limitation.
- **What the operator sees:** `cp` frees the old copy before it writes, so when another writer (often another hpcp worker) takes that room, or the new file outgrows it, the rewrite fails: exit 190, the old copy lost, the source kept. On ext2/3/4, XFS and tmpfs a `Space warning` comes first, except on an XFS with the source on the same filesystem (`cp` may clone it; the `cp` 8.x of EL7 and EL8 never does, so there that case gets none). Measured on VM 901 (Debian 12, coreutils 9.1, a 48 MiB rewrite with 4 to 5 MiB free, a stand-in writer, 10 runs): XFS reflink=0, a dense old copy: 9.84 lost it 10 of 10, 9.85 (write-over) finished 10 of 10, and 9.86 is 9.84's case; XFS reflink=1 with the source on the same filesystem: all finish (`cp` clones).
- **Why:** writing over the old copy without truncating it was the only way to close this; the user did not pick that option ("I like refuse doomed, warn"), then chose warn and try ("we also try and fail?").
- **Revisit when:** an operator loses an old copy this way after the warning.
- **Approved:** 2026-11-09, the user chose "Approve (Recommended)".

### The beside route never falls back to writing in place
- **Kind:** known limitation.
- **What the operator sees:** a rewrite that goes beside (always with `-wb`; by default a file with other names, one that cannot be opened for writing, one replaced by a FIFO, socket or device node source, or a FIFO, socket, device node or link at the landing) and whose new copy cannot be made, filled, written to disk or renamed over the old one keeps what is there: exit 190 for a file, with the cause named (`... leaving DEST as it is: it has other names, which a write in place would change; the new copy beside it failed: ...`), and 213 for a FIFO, socket, device or link. That covers a 0555 directory, a sticky directory holding another user's file, a bind-mounted file, a disk without room for a second copy, and `-co` on a file whose owner, group or an xattr cannot be given to a new file. 9.85 wrote such files in place.
- **Why:** the fallback is what made 9.85 fragile; going beside means a write in place could corrupt, or the operator asked for the old copy to stay whole.
- **Revisit when:** an operator needs such rewrites in directories that take no new name.
- **Approved:** 2026-11-09, the user chose "Approve (Recommended)".

### With -wb, a rewrite does not keep a per-file layout, chattr flags or the project id
- **Kind:** known limitation.
- **What the operator sees:** with `-wb` (and on the default's beside cases), the new file takes its directory's default Lustre or GPFS layout, not the one set on the old file (`lfs setstripe`), and no `chattr` flags or project id of the old file. A default in-place rewrite keeps all of them.
- **Why:** the new inode is what `-wb` asks for; carrying a layout needs `lfs getstripe`/`setstripe` per file, on a filesystem not testable here.
- **Revisit when:** a Lustre or GPFS operator needs `-wb` with per-file layouts.
- **Approved:** 2026-11-09, the user chose "Known limitation (Recommended)", replacing "Plan it for 9.86 (Recommended)" of 2026-11-08, since the default now keeps the inode.

### Off Linux, a rewrite in place can be swapped for a link mid-run
- **Kind:** known limitation.
- **What the operator sees:** off Linux (macOS, the BSDs), a rewrite in place gives `cp` the landing's name, as 9.84 did, so a user who can write the destination directory can swap the file for a link between hpcp's open and `cp`'s, and a root run then writes where the link leads. The same holds on Linux where `cp` cannot open the worker's `/proc/PID/fd/N` (asked once per worker by doing it): a `cp` that is not GNU `cp` (BusyBox's), a `/proc` of another PID namespace, a worker that `/proc` refuses. On Linux with GNU `cp` the held file is written, and a link put at the name is never written through. Use `-wb` there for a root run into a tree others can write: it renames a new file over the name and never writes through it. A directory on the way can still be swapped, everywhere (see "A user who can write the destination can race a check by path").
- **Why:** only Linux's `/proc` lets `cp` open a file another process holds; elsewhere `cp` would have to inherit the descriptor, which multiCMD's runner does not pass, and there is no macOS or BSD machine here to test either on.
- **Revisit when:** hpcp gets a macOS or BSD test machine, or an operator reports it.
- **Approved:** 2026-11-09, the user picked 'Known limitation (Recommended)'.

### `-enes` skips a rerun onto a nearly full destination though nothing needs copying
- **Kind:** deferred (9.87).
- **What the operator sees:** `-enes` sums each source's allocated size against the destination's free space before it knows which files are already identical, so a rerun onto a nearly full destination skips the source (`Not enough space: free space on [...] too small, skipping copy SRC`, exit 228) though nothing needs copying (checked in Docker on 9.86: a 3 MB file copied onto a 5 MiB tmpfs, then rerun with `-enes`: 228). Without `-enes` the rerun exits 0.
- **Why:** the check runs on the job's listed size, before the skip checks; it is as old as `-enes`.
- **Plan (9.87):** count only what the copy will write (files not already identical, less the room their old copies free).
- **Revisit when:** 9.87.
- **Noted:** 2026-11-09, found writing the 9.86 plan; the main session's decision, not asked (a deferral).
