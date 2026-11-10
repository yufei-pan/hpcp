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

### A directory on the way to a landing can be swapped for a link mid-run
- **Kind:** deferred (9.88).
- **What the operator sees:** a user who can write a directory below a destination can swap a directory on the way to a landing for a link. The window is between hpcp's look at it (`_dest_directory`, once per worker in a run) and the next write below it. A root run then writes the rest of that directory's files where the link leads. The folder sync's metadata, the directory finish and the links made there follow it too. On Linux and macOS a landing's own name no longer races since 9.87: a first copy, a rewrite in place and the beside route's new file are written through a descriptor hpcp holds, and an identical file's metadata is set through a descriptor checked against hpcp's look. Elsewhere see "Where cp cannot write a file hpcp holds, a landing can be swapped for a link mid-run".
- **Why:** closing it needs every write below a destination to go through descriptors walked from the destination with O_NOFOLLOW at each step, the size of 9.86's redesign. It is built apart from 9.87's other work so that each is measured and reviewed on its own.
- **Plan (9.88):** for each landing, walk from the named destination one level at a time by `O_PATH | O_NOFOLLOW | O_DIRECTORY` opens (`O_RDONLY` where there is no `O_PATH`). A link, FIFO or node met on the way gets the door's rules, relative to the held directory. Every name below a destination is made through the held directory (`dir_fd=`, or `/proc/self/fd/D/NAME`). `cp` gets the held file as its own inherited descriptor. Measured against 9.87, local and on NFS, with the bars in the 9.87 plan (W1).
- **Revisit when:** 9.88, started as soon as 9.87 ships.
- **Noted:** 2026-11-10, the main session's decision (a deferral, not asked).

### The copy check stats each landing by name
- **Kind:** deferred (9.87).
- **What the operator sees:** the check at the end of a copy, and the move's before its removal, `lstat`s each landing by its path: on a cached local disk about 0.3 s per 100,000 files idle, and more under load (a cached 100,000-file rerun at load 110 took 1.32-1.73 s in 9.84 and 1.81-2.01 s in 9.85), and on NFS mounted `actimeo=0` a round trip for each part of each path: 900,000 GETATTRs for 100,000 landings, 21.5 to 114.8 s in 4 threads on an EL8 VM (the host busy). Since 9.87 a second look, where two landings may be one file, is taken only for the entries whose fingerprint met another's.
- **Why:** correct as it is. Reading each landing directory once (`scandir`) gives no size, time or link count, and made-up inode numbers on vfat and FUSE exFAT. An `lstat` relative to each landing directory, opened once (9.87's first build), cut NFS to 360,000 GETATTRs and 9.2 to 53.8 s, but was slower on a local disk (0.66 to 1.03 s per 100,000 landings against 9.86's 0.57 to 0.66 s, Docker at load 336), so it missed the plan's bar (faster on NFS, not slower locally) and was dropped.
- **Plan:** none yet; the main session asks the user (the 9.87 plan: a missed bar is a question of a known limitation). One option measured here: the directory-relative look on network filesystems only.
- **Revisit when:** 9.87, with the user's answer.
- **Noted:** 2026-11-08, the user's standing rule that deferrals are noted, not asked; measured and narrowed 2026-11-10 (9.87's R3).
### On Python 3.6 an early stop can rarely hang the run in a pool's submit
- **Kind:** deferred (9.88).
- **What the operator sees:** on Python 3.6 (EL7, EL8), a Ctrl-C or SIGTERM in the first second of a copy, while its pool takes its first jobs, can leave hpcp waiting for good: every worker has ended, and a second Ctrl-C prints nothing (not even `Stopping: cleaning up first`) and SIGTERM does nothing; Ctrl-\ or `kill -9` ends it. About 3 in 2,000 early Ctrl-Cs in the 9.87 review's probe (9.86: 4 in 716). Other Pythons are not affected.
- **Why:** 3.6's `ProcessPoolExecutor.submit` wakes its manager thread through the result queue, under that queue's write lock, which the workers share. A worker that the same Ctrl-C ended can leave that lock held (its stop raised in `SemLock.__enter__`, after it took the semaphore and before `with` armed its exit), and the parent then waits on it with its stops blocked (`_pool.submit`'s mask, which keeps a 3.6 worker's start closed to stops). The review tried letting the parent's stop raise inside 3.6's `submit` (it cut multiprocessing's own bookkeeping and printed new tracebacks) and raising only from the lock wait (it still hung): each made something else worse.
- **Plan (9.88):** hpcp owns the 3.6 wakeup: on 3.6 only, its own `submit` takes the result queue's write lock with a timed acquire in a loop, the stops staying blocked as now, and each pass reads `signal.sigpending()`; a pending stop raises before anything is taken.
- **Revisit when:** 9.88.
- **Noted:** 2026-11-10, found by the 9.87 group 1 review (F1); not a regression (9.86 hung more often); the main session's decision, not asked (a deferral).

### A stopped run waits for a worker whose file operation never returns
- **Kind:** known limitation.
- **What the operator sees:** after Ctrl-C, SIGTERM or SIGHUP, a worker inside a write or fsync that never returns holds the run (for example a dead hard-mounted NFS server, or a disk in an uninterruptible wait). hpcp prints `Waiting for N worker(s) still finishing a file (pids ...): Ctrl-\ ends hpcp now, which may leave a temporary file or a torn file there`. Ctrl-\ ends it at once, and can leave that worker's temporary file or a torn file. Nothing is removed from the sources.
- **Why:** killing a worker in a task would leave what its cleanup removes, which the approved limitations promise a stop removes. An uninterruptible wait cannot be killed at all.
- **Revisit when:** an operator reports one.
- **Approved:** 2026-11-10, the user chose "Approve as KL (Recommended)". The user did not ask for it explicitly; it is what remains of the hang the 9.85 review found.

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

### Where cp cannot write a file hpcp holds, a landing can be swapped for a link mid-run
- **Kind:** known limitation.
- **What the operator sees:** on the BSDs and other systems that are neither Linux nor macOS, a rewrite in place, a first copy and the beside route's new file give `cp` the landing's name, as 9.84 did. A user who can write the destination directory can then swap the landing for a link between hpcp's look and `cp`'s open, and a root run writes where the link leads. The same holds on Linux where `cp` cannot write the held file through its own `/proc/self/fd/N` (hpcp asks once per worker, by doing it):
  - a `cp` other than GNU `cp`, such as BusyBox's or uutils' (uutils' `cp` 0.10 writes the held file but sets no mode or times through it, so it is not asked);
  - no `/proc`, or one of a PID namespace that cannot see `cp`;
  - no writable temporary directory for that question;
  - for one file: a plain first copy, or the beside route's new file, of a source whose owner and group a non-root run cannot give, in a directory whose default ACL gave the new file entries of its own. `cp -a` narrows the ACL of a file it did not make before it gives it an owner, so it would drop them; `cp` makes that file by name, as before, and keeps them.

  BusyBox's `cp` writes data only into a file it creates itself, never through a link. But it sets owner, mode and times by name after closing the file, so a link put at the name in that moment gets them. On Linux with GNU `cp`, and on macOS, the held file is written, and a link put at the name is never written through. Use `-wb` for a root run into a tree others can write: it renames a new file over the name, and never writes data through it. Directories on the way: see "A directory on the way to a landing can be swapped for a link mid-run".
- **Why:**
  - The BSDs show no `/dev/fd/N` past 2 unless fdescfs is mounted (fdescfs(5)), and FreeBSD's `cp -f` unlinks the destination first (freebsd-src `bin/cp/utils.c`). There is no BSD machine here to test a route on.
  - BusyBox's `cp` makes every destination anew with `O_EXCL` (BusyBox `libbb/copy_file.c`; checked in Alpine: the probe gives no route).
  - A `cp` that cannot see itself in `/proc` cannot open its own descriptor by name.
- **Revisit when:** hpcp gets a BSD test machine, or an operator reports it.
- **Approved:** 2026-11-09, the user picked "Known limitation (Recommended)" (the off-Linux entry). Narrowed 2026-11-10: the user chose "Fix at source for macOS", whose option said "The BSDs can't use it and stay a KL". The user did not ask for it explicitly; it remains of a deferral the 9.85 review found.
