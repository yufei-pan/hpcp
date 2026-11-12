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

### `-nb` with merged sources moves a shared directory's ctime on reruns
- **Kind:** known limitation.
- **What the operator sees:** with `-nb` (each source in turn), a directory that several merged sources share (`hpcp a/ b/ -d out/ -nb`) is started and finished once per source. Where their modes or times differ, every rerun sets them again, and the directory's change time moves each run; the later source's mode and times are what it keeps. 9.87 does the same. Batch mode (the default) syncs such a directory once, from the later source.
- **Why:** fixing it needs the later sources known before the first is synced, which `-nb` exists to avoid.
- **Revisit when:** `-nb` gains a pre-scan of the sources.
- **Approved:** 2026-11-11, the user chose "Approve as KL (Recommended)". The user did not ask for it explicitly; the 9.88 A–C review found it.

### A Ctrl-C while Python itself is starting is dropped
- **Kind:** known limitation.
- **What the operator sees:** a Ctrl-C that lands in the first milliseconds after hpcp is started, while Python is still starting (inside pip's launcher script, before hpcp's first line runs), is dropped: the copy starts, and a second Ctrl-C stops it as usual. In the 9.88 build's probes it happened 2 times in 12,000 early stops. 9.87 does the same.
- **Why:** no code of hpcp's runs yet at that point, so hpcp has nothing to hold the signal with.
- **Revisit when:** Python changes how it starts, or an operator reports it.
- **Approved:** 2026-11-11, the user chose "Approve as KL (Recommended)". The user did not ask for it explicitly; the 9.88 build's import probe found it.

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
- **Kind:** known limitation.
- **What the operator sees:** on a vfat or exFAT mounted with a codepage `iocharset` (`iso8859-1`, `iso8859-9`, `cp1251`, `koi8-r`, `euc-jp`; not `utf8`), two names that the mount takes for one are not asked about before the copy when the net reads one of them as UTF-8 or ASCII, which happens in four ways:
  - one name's bytes happen to be valid UTF-8 and the other's are not: `C3 A9 74` (Latin-1 `Ã©t`, UTF-8 `ét`) against `E3 A9 74`, which a cp1251 mount reads as `Г©t` and `г©t`;
  - both names' bytes are valid UTF-8, but not as each other's case: KOI8-R `бё` and `бЁ` (`C2 A3` and `C2 B3`), CP1251 `EF B8 B8` and `EF A8 B8`; CP1251 alone has 880 such two-byte pairs;
  - an ASCII name against a legacy one: `istanbul` against ISO 8859-9 `İstanbul` (`DD` for `İ`, which that table makes the other case of `i`);
  - two cases of different lengths: EUC-JP `ς` is three bytes (`8F A6 F8`) and `Σ` two (`A6 B2`), and no rule over single bytes can join names of different lengths.
  So the pair is found only after the copy, by the backstop: `hpcp a/ b/ -d out/` with one in `a/` and the other in `b/` ends in exit 213, `Destination clash: copied SRC, but LANDING is one file with OTHER (from OTHER_SRC)`, and the landing holds whichever source was written last, which in a parallel copy can differ from run to run; every rerun copies both again, and with `-rf` both sources stay. A pair caught before the copy is refused there instead: the first source lands and the second is not written, the same on every run. Checked on vfat mounted `iocharset=cp1251`, `koi8-r` and `iso8859-9` (`utf8=0`) on EL8; the EUC-JP pair on kernel exFAT is not checked yet (it needs a kernel with exFAT, which EL8's 4.18 lacks, such as a Debian 12 VM's 6.1). Only for such pairs, and only on such a mount: names that are not UTF-8 as a whole fold byte by byte and meet.
- **Why:** a name that is valid UTF-8 is read as UTF-8, which it is on every other mount, and the mount's own table, not UTF-8, says what its other case is. Joining every valid name with every name of the same length would ask about most trees pair by pair again.
- **Revisit when:** an operator hits it.
- **Approved:** 2026-11-10, the user chose "Approve both as KLs (Recommended)". The user did not ask for it explicitly; the 9.83 review found it.

### Directories that differ only in case on vfat mounted `iocharset=utf8`
- **Kind:** known limitation.
- **What the operator sees:** on a vfat mounted with `iocharset=utf8` given explicitly, `hpcp a/ b/ -d out/` with `a/D/f` and `b/d/g` exits 193 (`Directory sync error: .../b/d to .../out/d: [Errno 2] No such file or directory`, then `Copy failed ... No more destination paths to try`), and `g` is not copied; with `-rf` it exits 212 and keeps both sources. A single source does it too: `out/D` from an earlier run, and a source that now spells it `d`, exit 193 and nothing below it copied (checked on EL7 by the 9.85 designer). Nothing is lost, and 9.82 does the same. Checked on a Debian 12 VM (kernel 6.1); vfat with its default charset, `utf8=1` or `iocharset=iso8859-1`, kernel exFAT, NTFS (`ignore_case`) and ext4 casefold merge the two, exit 0. Since 9.85 the directory door says so: `mkdir` meets `EEXIST` twice where `lstat` finds nothing, and the 193 names the mount option to use instead.
- **Why:** vfat mounted that way looks a name up in its own case only (`out/d` is not found) but will not make it either, as `out/D` holds the name; the copy names the directory as the source spells it. A loud failure with nothing lost, on a mount option the kernel's vfat documentation advises against.
- **Revisit when:** an operator needs that mount option.
- **Approved:** 2026-11-10, the user chose "Approve both as KLs (Recommended)". The user did not ask for it explicitly; the 9.84 review found it.

### On Python 3.6 and 3.7 an early Ctrl-C can print an `Exception ignored ... KeyboardInterrupt` block
- **Kind:** known limitation.
- **What the operator sees:** on Python 3.6 and 3.7 (EL7, EL8), an early Ctrl-C (or SIGTERM or SIGHUP, which hpcp stops the same way) now and then prints Python's `Exception ignored in: <...>` line, a traceback ending in hpcp's `_stop` at `raise KeyboardInterrupt`, and `KeyboardInterrupt:`. The run still stops as it should (130 for Ctrl-C, its cleanup done) and nothing is lost. In the 9.87 reviews' probes, 9.87 printed it in 1 of 2,100 early stops and 0 of 1,200, against 11 of 1,200 for 9.86 (about 1 in 2,000 against about 1 in 100). Python 3.8 and later never print it.
- **Why:** Python before 3.8 has no `sys.unraisablehook`, and the block is printed by the interpreter: a raise in code Python runs as an object goes (a weakref callback, a `__del__`, a generator the collector closes) is printed and dropped. hpcp records the stop before it raises, so its next check acts on it. Since 9.87 a stop that lands in the weak containers' callbacks, multiprocessing's finalizers or a `__del__` is only recorded, never raised, but a generator the collector closes cannot be told from one being iterated, and filtering stderr would be a part that can fail in new ways.
- **Revisit when:** hpcp drops Python 3.6 and 3.7, or an operator reports it.
- **Approved:** 2026-11-10, the user chose "KL below Python 3.8 (Recommended)". The user did not ask for it explicitly; the 9.87 group 1 review found it (F8).

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
- **What the operator sees:** `cp` frees the old copy before it writes, so when another writer (often another hpcp worker) takes that room, or the new file outgrows it, the rewrite fails: exit 190, the old copy lost, the source kept. On ext2/3/4, XFS and tmpfs a `Space warning` comes first, except on an XFS with the source on the same filesystem (`cp` may clone it; the `cp` 8.x of EL7 and EL8 never does, so there that case gets none). Measured on a Debian 12 VM (kernel 6.1, coreutils 9.1, a 48 MiB rewrite with 4 to 5 MiB free, a stand-in writer, 10 runs): XFS reflink=0, a dense old copy: 9.84 lost it 10 of 10, 9.85 (write-over) finished 10 of 10, and 9.86 is 9.84's case; XFS reflink=1 with the source on the same filesystem: all finish (`cp` clones).
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
- **What the operator sees:** on the BSDs and other systems that are neither Linux nor macOS, a rewrite in place, a first copy and the beside route's new file give `cp` the landing's name, as 9.84 did. A user who can write the destination directory can then swap the landing for a link between hpcp's look and `cp`'s open, and a root run writes where the link leads. The same holds on Linux where `cp` cannot write the held file through its own `/proc/self/fd/N` (hpcp asks once a run, by doing it, in the parent, which hands the workers the answer), or is not given it:
  - a `cp` other than GNU `cp`, such as BusyBox's or uutils' (uutils' `cp` 0.10 writes the held file but sets no mode or times through it, so it is not asked);
  - no `/proc`, or one of a PID namespace that cannot see `cp`;
  - no writable temporary directory for that question;
  - for one file: a plain first copy, or the beside route's new file, of a source whose owner and group a non-root run cannot give, in a directory whose default ACL gave the new file entries of its own. `cp -a` narrows the ACL of a file it did not make before it gives it an owner, so it would drop them; `cp` makes that file by name, as before, and keeps them;
  - for one file: a root run's plain first copy, or the beside route's new file, of a file of another user or group (an owner or group other than the new file's own: root, and the directory's group where it is setgid, otherwise root's), in a directory whose default ACL gave the new file entries of its own. Keeping them would mean giving the file its owner before `cp` writes it, and then the data is charged to that owner's or group's quota while root writes it (XFS enforces an owner's and a group's quota on a root write, not on root's chown): in an earlier 9.87 build, which did that, a user at the hard limit got `Disk quota exceeded`, exit 190, where 9.86 completed; so did a root file of a group at its hard limit (`root:G`, XFS with `grpquota`, a 4 MiB file against a 1 MiB limit: 9.87 exit 190, 9.86 exit 0). `cp` makes that file by name, as in 9.86, and gives it its owner once it has written it. A root run's file with the owner and group the new file has (root:root, or root and the group of a setgid directory) is given nothing first and keeps the held route.

  BusyBox's `cp` writes data only into a file it creates itself, never through a link. But it sets owner, mode and times by name after closing the file, so a link put at the name in that moment gets them. Otherwise, on Linux with GNU `cp` and on macOS, the held file is written, and a link put at the name is never written through. Use `-wb` for a root run into a tree others can write: it renames a new file over the name, and never writes data through it. Directories on the way: since 9.88 hpcp holds them by descriptor. Where `cp` makes a landing itself on Linux with a `cp` that writes a held file (the default-ACL cases above), it gets the landing's name below the held directory (`/proc/self/fd/D/NAME`), so they cannot lead it elsewhere. Where `cp` cannot write a held file (the BSDs, a `cp` other than GNU `cp`, no usable `/proc` or temporary directory), it gets the full path, and a directory on the way swapped in that moment leads it too.
- **Why:**
  - The BSDs show no `/dev/fd/N` past 2 unless fdescfs is mounted (fdescfs(5)), and FreeBSD's `cp -f` unlinks the destination first (freebsd-src `bin/cp/utils.c`). There is no BSD machine here to test a route on.
  - BusyBox's `cp` makes every destination anew with `O_EXCL` (BusyBox `libbb/copy_file.c`; checked in Alpine: the probe gives no route).
  - A `cp` that cannot see itself in `/proc` cannot open its own descriptor by name.
  - In a default-ACL directory the held route keeps the entries only by giving the file its owner before `cp` writes it (coreutils 9.7; 8.22 and 8.32 keep none by any route), which a non-root run cannot do for another user's file, and which charges a root run's write to the owner's or group's quota (9.87's final review, F3).
- **Revisit when:** hpcp gets a BSD test machine, or an operator reports it.
- **Approved:** 2026-11-09, the user picked "Known limitation (Recommended)" (the off-Linux entry). Narrowed 2026-11-10: the user chose "Fix at source for macOS", whose option said "The BSDs can't use it and stay a KL". The user did not ask for it explicitly; it remains of a deferral the 9.85 review found. Widened 2026-11-10 to the Linux cases where 9.87 gives `cp` the name, exactly as 9.86 did (a non-GNU `cp` such as BusyBox's or uutils'; no usable `/proc` or temporary directory; a non-root first copy into a default-ACL directory; a root first copy of another user's file into a default-ACL directory): the user chose "Add them to the KL (Recommended)". The group case (a root first copy of a file of another group, the same fallback): 2026-11-10, the user chose "Widen to groups (Recommended)".

### On macOS the copy check looks at each landing by path, on a network mount too
- **Kind:** deferred (9.89).
- **What the operator sees:** on macOS the copy check (and a move's) looks at each landing by its path, whatever the destination's filesystem: on an NFS, SMB or AFP mount each part of each path can cost a round trip, as in 9.87. On Linux a destination on a network filesystem is checked by directory (each directory opened once, its names looked at relative to it).
- **Why:** macOS has no mount table hpcp reads. An earlier 9.88 build took every macOS destination by directory: on a local APFS disk that made the check 2.3 times as slow for 10,000 directories of one file each (each open and close about 20 µs) and a rerun 12 % slower (the 9.88 A–C review, F3), and the gain on a network mount was not measured.
- **Plan (9.89):** measure the check on a macOS NFS and SMB mount, by path and by directory; where by directory wins, ask each destination's filesystem type (`statfs`'s `f_fstypename`) and check network mounts by directory.
- **Revisit when:** 9.89.
- **Noted:** 2026-11-11, the main session's call on the review's F3 (a deferral, not asked).

### On macOS a node copy and BSD flags, and on Linux without /proc an identical node's mode, go by name below the held directory
- **Kind:** known limitation.
- **What the operator sees:** in a root run on macOS into a tree another user can write, that user can, by swapping a directory on the way for a link at the right microsecond, have the run make a copied FIFO or socket where the link leads (a device node too, if the source holds one; a user's own tree cannot). On macOS the same swap can also give an identical file's BSD flags to another file of that name there. On Linux with /proc not mounted, it can give an identical FIFO's or device node's mode to such a file. Regular files, directories and links go through the directory hpcp holds (9.88). What `cp` gets where it cannot write a file hpcp holds is in "Where cp cannot write a file hpcp holds, a landing can be swapped for a link mid-run".
- **Why:** hpcp would have to make FIFOs, sockets and nodes itself instead of `cp -a`, plus a ctypes `fchflags`: new code for node files in root macOS runs into other users' trees.
- **Revisit when:** an operator runs root copies on macOS into trees others can write, or Python gains `fchflags`.
- **Approved:** 2026-11-10, the user chose "Approve as KL (Recommended)". The user did not ask for it explicitly; the 9.88 plan found it.

### `-siff`'s patch apply writes below its target by path
- **Kind:** deferred (9.89).
- **What the operator sees:** a root `-siff` apply into a target a user can write: a directory the patch writes into, swapped for a link after hpcp's check (`_unsafe_parent`) and before the write, has the write go where the link leads.
- **Why:** the patch apply is a flow of its own, built before 9.88's descriptor door; it checks paths, which leaves the window the copy closed in 9.88.
- **Plan (9.89):** the patch apply's writes through the copy's door (`_dest_directory`), relative to what it holds.
- **Revisit when:** 9.89.
- **Noted:** 2026-11-10, the 9.88 plan; the main session's decision, not asked (a deferral).

### A directory whose mode shuts its owner out (0607): a rerun over it can fail below it, and a move that leaves its source directory sets the mode all the same
- **Kind:** deferred (9.89).
- **What the operator sees:**
  - **A rerun can fail below such a directory (P1).** A non-root copy or move gives a directory of mode 0607 (no search for its owner) that mode at its end. A later non-root copy or move of the same tree syncs that directory's subdirectories, and one synced before the directory's own start has opened it to its owner again fails: `Directory sync error: SRC/d/sub to OUT/d/sub: [Errno 13] Permission denied: 'OUT/d/sub'`, exit 193 (with `-rf` the sources stay; a retry may pass). Since 9.88 a job's folders are synced parents first, so it takes a parallel run and such a directory with more than one pool job's folders below it (64): with 200 subdirectories, 9.87 failed all 12 reruns on Python 3.6 and 3.14, 9.88 some on 3.6 and none on 3.14; with one subdirectory 9.87 failed some reruns, by hash order, and 9.88 none.
  - **A move that leaves a source directory sets the exact mode anyway (N5).** A non-root move sets such a directory's exact mode after the removal step when the copy reported no errors, also where its source directory stays: a file replaced by a later merged source (`Remove skipped ... replaced by a later source`, exit 212), `Remove directory failed`, or `Move left entries`. Those messages ask for a rerun, and the rerun meets the first case. After a copy's errors, or a stop, only a directory whose source directory is gone gets its exact mode; the others keep owner read and search (0707) for the rerun. The first case left the directory 0607 in 9.87 too (and failed the move with `cannot be written to disk`).
- **Why:** the folder sync reaches a directory below its parent, and only the parent's own start gives the parent owner search; pool jobs run side by side. The late pass keeps 0707 "for the rerun" only after a copy's errors (`removal_ended`, and a test of the source directory) because of the first case: a rerun over a 0607 directory can fail. Not worse than 9.87 in either.
- **Plan (9.89):** the first case at its source: where the door meets EACCES at a directory the runner owns that lacks owner search, give it owner search first, as `_start_directory` does for a rerun's own directory (the finish sets its exact mode again); or start each pool job's parents before it. Then one rule for the late pass: the exact mode for every record on every path, with no `removal_ended` and no test of the source directory (the README's "until a rerun" goes). Not the other way round: setting the exact mode only where the source directory is gone, which the final review proposed for the second case, leaves 0707 for good after a move that succeeded with an excluded file (the source directory stays, the run exits 0, and `out/d` stays 0707 where 9.88 gives 0607; the design check ran it).
- **Revisit when:** 9.89.
- **Noted:** 2026-11-11, P1 found by the 9.88 A–C review, N5 by the final review (the design check changed its fix); the main session's decision, not asked (a deferral).

### macOS: a rerun carries no changed xattr or ACL
- **Kind:** deferred (9.89).
- **What the operator sees:** on macOS a first copy gets its source's xattrs and ACL from `cp -a`, but a rerun that finds a landing identical does not carry an xattr or ACL the source changed since: Python has no `os.listxattr` there, and hpcp reads no ACL. The same in 9.87.
- **Why:** Linux's xattr calls are in Python's `os` module; macOS's are not, and ACLs need `acl_get_fd`/`acl_set_fd`.
- **Plan (9.89):** read and set them on macOS through libc (ctypes: `listxattr`, `getxattr`, `setxattr`, `removexattr` with `XATTR_NOFOLLOW`, and the ACL calls), through the descriptor the rerun already holds.
- **Revisit when:** 9.89.
- **Noted:** 2026-11-11, found by the 9.88 A–C review's macOS run; the main session's decision, not asked (a deferral).

### macOS: a named socket in a source ends in a misleading 213
- **Kind:** deferred (9.89).
- **What the operator sees:** macOS's `cp` does not copy a named socket (`cp: ... is a socket (not copied)`), and hpcp then reports the socket as a failed copy (`Copy failed: ...`, naming its landing), exit 213; with `-rf` the sources stay. On Linux `cp -a` makes the socket node and the run exits 0. The same in 9.87.
- **Why:** hpcp leaves every node to `cp -a`, and macOS's `cp` skips sockets.
- **Plan (9.89):** make the socket node at the landing as Linux's `cp -a` does (an `AF_UNIX` socket bound there, relative to the held directory), so a socket copies on macOS too.
- **Revisit when:** 9.89.
- **Noted:** 2026-11-11, found by the 9.88 A–C review's macOS run; the main session's decision, not asked (a deferral).

### On Linux without /proc an identical node's xattrs and a vfat file's attribute bits go by path
- **Kind:** deferred (9.89).
- **What the operator sees:** on Linux with /proc not mounted, a root run into a tree another user can write: a directory on the way swapped for a link at the right microsecond, after hpcp opened it and before it settles an identical FIFO, socket or device node at a landing, leads that node's xattrs (listed, removed and set by the landing's full path) to the file of that name where the link leads; and, under `-co`, the same for the FAT attribute bits (`_set_fat_attributes`) of a rewritten file on a vfat. Regular files, directories and links, and an identical node's mode, are not affected (the last is in "On macOS a node copy and BSD flags, and on Linux without /proc an identical node's mode, go by name below the held directory"). With /proc they go through the held descriptor.
- **Why:** these calls take a path (`_io`): with /proc the held directory's `/proc/self/fd/N/NAME` is one that cannot lead elsewhere, and without /proc there is no such name, so the logical path is used. Reaching a node's xattrs through a descriptor means opening a FIFO or device node, which needs care (a FIFO opened for reading waits for a writer without `O_NONBLOCK`), and the FAT attribute bits are an ioctl by path in `_set_fat_attributes`.
- **Plan (9.89):** set them through the held descriptor: the node's xattrs through a descriptor opened `O_NONBLOCK` and `O_NOFOLLOW` (or `O_PATH` where the filesystem takes xattrs through one), and the FAT attribute bits through the file's descriptor.
- **Revisit when:** 9.89.
- **Noted:** 2026-11-11, the 9.88 W1 review's docs gap D1; the main session's decision, not asked (a deferral).

### The door goes by path below half the open-files limit, after running out of descriptors, and in a directory the runner may not read (read route)
- **Kind:** deferred (9.89).
- **What the operator sees:** three places where the door (the directories a run holds by descriptor, 9.88) goes by path, as 9.87 did everywhere, so a directory swapped for a link there, in a root run into a tree another user can write (a non-root run for the third), leads the writes below it elsewhere:
  - the levels deeper than half the soft open-files limit (the other half stays for what a copy opens): 128 levels on macOS, whose soft limit stays 256; on Linux hpcp raises the limit to the hard one at import (524,288 on systemd hosts, 4,096 on EL7), so the budget, half of it, is deeper than the 4,096 characters a listed path holds, except on EL7, where it is 2,048 levels;
  - a process that has run out of descriptors (EMFILE or ENFILE);
  - a directory the runner may not read, where the door holds read descriptors (macOS, Linux without /proc; with /proc an `O_PATH` open needs only search): the walk goes on by path from it.
  (Windows goes by path everywhere: its own entry below.)
  The copy itself is whole in each, and what is written there is what 9.87 wrote.
- **Why:** the budget keeps the descriptors a copy needs beside the held ones (its file, `cp`'s pipes, a pool's), and a read descriptor cannot be had on a directory the runner may not read.
- **Plan (9.89):** raise the soft `RLIMIT_NOFILE` toward the hard limit at start (macOS), and hold such directories with `O_PATH` descriptors on Linux where /proc is there but a read is refused. Each goes by path as 9.87 did until then, so none is worse.
- **Revisit when:** 9.89.
- **Noted:** 2026-11-11, the 9.88 W1 review's docs gap D2; the main session's decision, not asked (a deferral).

### On Python 3.6 and 3.7 an early stop with forkserver or spawn can print a worker's EOFError
- **Kind:** deferred (9.89).
- **What the operator sees:** on Python 3.6 (3.7 not measured) with the forkserver or spawn start method (EL7's and EL8's Python when a program sets one; hpcp's default on Linux is fork there), an early Ctrl-C now and then prints a worker's `Process Process-2:` traceback ending in `EOFError: Ran out of input`, from `concurrent/futures/process.py`'s `call_queue.get`. The run still ends 130 with its cleanup done and nothing lost. 9.87 printed it in 2 of 3,800 early stops, 9.88 in 0 of 3,800 (and 2 of 3,200 in the build's own probes): about 1 in 1,600 to 2,000 early stops. The early-stop test accepts that block, and only that block (`_OLD_EOF`).
- **Why:** a stop's `SystemExit` lands in an idle worker between a queued job's length and its payload (`Connection._recv_bytes` reads the 4-byte length and then the payload in two calls), leaving the payload in the pipe; the next worker reads its first bytes as a negative length and unpickles nothing. hpcp raises it in an idle worker on purpose, so the queue's lock is let go.
- **Plan (9.89):** a worker stopped inside `Connection._recv_bytes`, past its header, ends without unwinding (`os._exit`; the run is stopping anyway).
- **Revisit when:** 9.89.
- **Noted:** 2026-11-11, the 9.88 W1 review's item (b); the main session's decision, not asked (a deferral). The same in 9.87.

### A library call to `hpcp.hpcp()` on Python 3.6 with an ASCII stdout crashes in the progress bar
- **Kind:** deferred (multiCMD 1.62, then hpcp's next embed).
- **What the operator sees:** `hpcp.hpcp(...)` called from a program on Python 3.6 whose stdout is ASCII (`LC_ALL=C`) copies a small run (30 files) and returns 70 (`General Exception`: `UnicodeEncodeError` in `multiCMD.print_progress_bar`); a larger run aborts at the first bar after the pool starts. The command line is not affected (`main()` escapes what stdout cannot encode). 9.87 does the same.
- **Why:** multiCMD's `print_progress_bar` prints its non-ASCII bar again in its `except Exception` fallback, which fails the same way; hpcp does not reconfigure a library caller's `sys.stdout`.
- **Plan:** multiCMD 1.62 falls back to an ASCII bar (or writes with the stream's encoding and `errors='replace'`); hpcp picks it up through the embed, as do mssh, statblk and statbtrfs.
- **Revisit when:** multiCMD 1.62 ships.
- **Noted:** 2026-11-11, the 9.88 W1 review's item (c); the main session's decision, not asked (a deferral). The same in 9.87.

### Windows: directories on the way to a landing go by path, as in 9.87
- **Kind:** known limitation.
- **What the operator sees:** on Windows the door (9.88's walk that holds each directory on the way to a landing by descriptor) does not exist: hpcp goes by path everywhere, looks at each directory once a process in a run, and checks a destination named with `-d` against its pin once a process, as 9.87 did. So in a run by an administrator into a tree another user can write, a directory on the way swapped for a link or junction mid-run can lead the writes below it elsewhere. The copy itself is whole, and what is written is what 9.87 wrote: not worse than 9.87.
- **Why:** Windows has no descriptor-relative calls (`dir_fd`, `O_PATH`, `/proc/self/fd`) for Python to use, and there is no Windows machine here to build and test another route on.
- **Revisit when:** someone runs hpcp on Windows into trees other users can write.
- **Approved:** 2026-11-11, the user chose "Approve as KL (Recommended)" (asked after the 9.88 fix round added Windows to a deferred entry that had no plan for it).

### macOS: a rerun of a tree of many directories with few files each takes longer than in 9.87
- **Kind:** known limitation.
- **What the operator sees:** on macOS (APFS, Python 3.9.6) a rerun with nothing to copy of a tree of 10,000 directories of one file each takes longer than 9.87's, the versions taking turns. On a loaded Mac (the final review: load 12 to 32 from other work): median 3.36 s against 3.18 s in 11 rounds (+6 %) and 2.83 s against 2.70 s with `-s` in 5 (+5 %). On a quiet one (the fix round: load 2 when it started): 3.26 s against 3.08 s in 11 rounds (+6 %) and 2.82 s against 2.47 s with `-s` in 5 (+14.5 %). Other shapes are close: 50 directories of 1,000 files 5.31 s against 5.28 s (loaded, 5 rounds) and 5.04 s against 4.91 s (quiet, +2.7 %); a first copy of 3,000 directories of one file each 19.28 s against 19.42 s (loaded) and 19.23 s against 18.74 s (quiet; 3 rounds each, one 9.87 round took 17.35 s).
- **Why:** the door opens and closes each directory a process enters (about 20 µs on APFS, against about 0.5 µs on Linux: about 0.2 s for 10,000 directories, so most of the serial run's extra 0.35 s by this estimate); that open is the guard (a directory swapped for a link is never followed), so it cannot go, and the read route macOS needs (no `O_PATH`) holds the directories by read descriptors. The rest is the per-landing Python code, which Linux shows too (the 9.89 deferral "A rerun costs about 6 to 14 µs more CPU a file than in 9.87").
- **Revisit when:** an operator reports it, or the 9.89 trim of the per-landing cost ships and the Mac is measured again.
- **Approved:** 2026-11-11, the user chose "Approve as KL (Recommended)" on the loaded Mac's "about 3 to 8 %", and, asked again with the quiet Mac's numbers (+6 % in the pool, +14.5 % with `-s`), "Keep the approval (Recommended)". The user did not ask for it explicitly; the 9.88 final review found it (N3).

### A rerun costs about 6 to 14 µs more CPU a file than in 9.87
- **Kind:** deferred (9.89).
- **What the operator sees:** a rerun with nothing to copy uses more CPU than 9.87's, in Python, on every platform. Measured on a tmpfs in Docker, Python 3.14, the versions taking turns, host load 8 to 35, so CPU seconds decide, not wall time:
  - serial (`-s`), the final 9.88 build, medians of 11 (the design check's, on c843e7e: +12 % and +11 %): 50 directories of 1,000 files 2.69 s of CPU against 2.40 s (+12 %, +5.8 µs a file); 10,000 directories of one file 1.16 s against 1.02 s (+14 %, +14.5 µs a file);
  - the default pool, the design check on c843e7e: wall time 0.810 s against 0.766 s on the 50-directory tree (9 rounds, +5.8 %; CPU of all processes +11 %), and 1.467 s against 1.462 s on the deep tree (+0.4 %; CPU +5 %); with `-j 2` the 50-directory tree took 1.227 s against 1.120 s (+9.5 %);
  - a first copy is not slower (serial, 10,000 directories of one file, `cp` included: 6.26 s against 6.60 s CPU, 3 rounds, ranges apart).
  On NFS mounted `actimeo=0` a rerun is 2 to 3 times faster in 9.88 in three of the build's four rounds (the README has the numbers). The macOS share of N3 is a known limitation of its own (above).
- **Why:** it is ordinary Python on each landing's way, not the descriptor calls: the door's call is 1.95 µs for a landing in the directory just used and 4.6 µs when the directory changes, against 9.87's 0.4 µs, and the rest is spread over the calls made relative to the held directory (the skip check's open, `_settle_identical`, `_landing_of`, closing the held directories). A variant whose door goes by path cost the same (+5.6 µs a file against the descriptor route's +5.8 µs), so making W1 narrower would not win it back. The plan's bar for W1 was under 2 µs a file; the build's bars measured pool wall time on a tree of 10 files a directory on a 32-core host, where the parent's listing hides the workers' CPU. The build before W1 (c401f23) was level with 9.87 on the wide tree, so the cost is W1's.
- **Plan (9.89):** profile a serial rerun and take the work out of the hot path (the dup and close of the held directory for each landing, a second look, closures made for each file) until its CPU a file is 9.87's within 2 %, measured as above, in CPU seconds serially and at `-j 2`, on Linux and on the Mac.
- **Revisit when:** 9.89.
- **Noted:** 2026-11-11, found by the 9.88 design check (X1); the user chose "Ship, trim in 9.89 (Recommended)".

### A device node copied by a non-root user onto a `-d` link: the message names the failed copy beside, not the refusal
- **Kind:** deferred (9.89).
- **What the operator sees:** `hpcp /dev/null -d out/l` as a non-root user, `out/l` a link to `../other/real`: the link and the file are untouched and the run exits 190, in 9.87 and 9.88 alike, but the message differs. 9.87: `Copy failed: cannot make a device node like /dev/null here, leaving [...] as they are`. 9.88: `Copy failed: cannot copy /dev/null to out/l, leaving out/l as it is: /dev/null is no regular file, which cp makes anew; the new copy beside it failed: cp: cannot create special file '.../other/.hpcp_tmp_...': Operation not permitted`, and two `Task return code error` records.
- **Why (run):** `_can_make_node` asks by doing it (a node of the source's number is made under a name of its own beside a landing that exists, and removed again) and only for a destination it finds there. For a `-d` link the door's directory is the link target's (`other/`) and the lookup is by the link's own name (`l`): it finds nothing, so it does not ask, and `cp` is left to fail on the copy beside.
- **Plan (9.89):** look the landing up by its target's name (`_landing_path`) in `_can_make_node`.
- **Revisit when:** 9.89.
- **Noted:** 2026-11-11, the 9.88 W1 fixer's call 5a, the final review's N4; the main session's decision, not asked (a deferral).

### macOS: a directory's BSD flags are set through its logical path, so a `-d` link gets them
- **Kind:** deferred (9.89).
- **What the operator sees:** on macOS, `-d out/` with `out` a link to a directory: the finish of the destination directory sets the source directory's BSD flags (`os.chflags(dest_path, flags, follow_symlinks=False)`, `_finish_directory`) on the link itself, not on the directory it leads to. Run on macOS 26 (Python 3.9): `src` with the user flag `nodump` copied with `src/ -d out/`, `out` a link to `real`: `out` (the link) gets `nodump` and `real` none, in 9.87 and 9.88 alike.
- **Why:** Python has no `fchflags`, so the flags are set by name, and not following a link is what keeps a link put at the name from taking them.
- **Plan (9.89):** set them through the held descriptor with ctypes `fchflags` (macOS and the BSDs), as the plan for the macOS xattrs does for its calls.
- **Revisit when:** 9.89.
- **Noted:** 2026-11-11, the 9.88 W1 fixer's call 5b, the final review's D1; the main session's decision, not asked (a deferral).

### A move's directory fsyncs go by path
- **Kind:** deferred (9.89).
- **What the operator sees:** after a move, `_sync_moved_directories` fsyncs each directory by its path (`_fsync_file`), as 9.87 did: a directory on the way swapped for a link at the right moment in a root move into a tree another user can write has the fsync land on the directory the link leads to (a flush of a directory elsewhere, so the intended directory's new names are not committed by it; a power cut soon after could then lose them, with the sources gone).
- **Why:** the 9.88 plan kept the by-path fsync; the other directory fsyncs of the move (the finish's, the file's) go through descriptors.
- **Plan (9.89):** fsync each through the door's descriptor for it.
- **Revisit when:** 9.89.
- **Noted:** 2026-11-11, the 9.88 W1 fixer's call 5c, the final review's D1; the main session's decision, not asked (a deferral).

### A pinned destination refused in one process says nothing when another destination takes the files
- **Kind:** deferred (9.89).
- **What the operator sees:** with two destinations (`-d B/ -d A/`) and `B` swapped for a link to another directory mid-run, a process that finds `B` leading elsewhere writes nothing below it (9.88's pin; 9.87 wrote into the victim). It says so only when every destination refuses it, so where the files go to `A` the refusal is mostly silent (root, Python 3.14, 60 files, the victim untouched in each): with forkserver the run exits 213 with 7 clash lines (the directory sync's), and with `-nds` it exits 0 with no line; with fork the workers hold the swapped-out directory and write there (exit 213).
- **Why:** a refusal is reported only when every destination fails, as 9.87 treats every refusal. Better than 9.87 in each case, and the files are whole where they landed.
- **Plan (9.89):** say a pin's refusal once a process even when the file goes elsewhere.
- **Revisit when:** 9.89.
- **Noted:** 2026-11-11, the final review's N7 (second opinion's lead 2); the main session's decision, not asked (a deferral).

### `_fsync_file`'s chmod of a file its owner cannot open follows a link
- **Kind:** deferred (9.89).
- **What the operator sees:** a non-root move of a file of mode 0044 (its owner can neither read nor write it): hpcp gives the owner read for the fsync (`os.chmod(name, mode | 0400, dir_fd=...)`) and restores the mode. A link put at the name between hpcp's `lstat` and that chmod has the chmod land on the link's target if the runner owns it, and the restore then gives that file the copy's mode. Read, not run. The same in 9.87 (by full path).
- **Why:** Linux has no `lchmod`: a chmod by name follows a link at the name.
- **Plan (9.89):** chmod through an `O_PATH` descriptor checked against the `lstat`, as `_carry_over` does.
- **Revisit when:** 9.89.
- **Noted:** 2026-11-11, the final review's N8 (Grok's lead 1); the main session's decision, not asked (a deferral).

### A rewrite of a file whose landing is held under a write lease goes beside it and gets a new inode
- **Kind:** deferred (9.89).
- **What the operator sees:** where a client holds a write lease on a landing (Samba with `kernel oplocks = yes`, an NFS server that gives write delegations, any program with an `fcntl` lease), a rerun of a changed file does not write the old copy in place: its open is non-blocking (`_LANDING_FLAGS`, since 9.86: never waiting on a FIFO or a lease), fails with EAGAIN, and the copy goes beside the old file and is renamed over it (0.3 s in the probe, Python 3.14, an `fcntl` lease): a new inode, and the lease holder keeps the old file. 9.87 and 9.88 alike. The skip check's read of an identical landing waits for the lease to be given up and keeps the inode (9.87; 9.88 again after the final review's N2).
- **Why:** the open must not wait on a FIFO or a lease, so it is non-blocking, and a landing that cannot be opened for writing is rewritten beside. It meets the user's words "I would prefer the inode do not change if possible".
- **Plan (9.89):** decide whether the rewrite's open may wait for a lease's break as the skip check's read does (the same question as N2's flag, with a FIFO swapped in at the landing the other case), then keep the inode.
- **Revisit when:** 9.89.
- **Noted:** 2026-11-11, found by the 9.88 design check; the main session's decision, not asked (a deferral).
