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

### Destination symlink at a directory's path
- **Kind:** deferred.
- **What the operator sees:** with `out/d -> ../elsewhere`, `hpcp src/ -d out/` writes src/d's files into elsewhere and changes elsewhere's mode, and as root its owner, exit 0, as in 9.80. A user who can write the destination can plant `d -> /etc` before root restores into it. `hpcp src/ -d out/ -rme` with `out/d -> ../src/d` removes the link as an extra, and `d` is not mirrored. The same at a file's path, identical or not: with `out/x -> ../t/f`, `t/f` takes the mode, and as root the owner, of `a/x`.
- **Why:** not a regression (a link there that leads onto a source file is refused, exit 213), and it needs its own decision: refuse as `cp -a` does, or write through without touching the target's mode and owner, since someone with a symlinked data directory may rely on writing through.
- **Revisit when:** the next hpcp release.
- **Approved:** 2026-11-05, the user chose "Deferred to 9.82 (Recommended)".

### A stored list used under another spelling can match by chance
- **Kind:** known limitation.
- **What the operator sees:** after `hpcp s -fl S`, `hpcp s/ -fl S -rme` with s/a and s/s/a present shares one name with the list (`s/a` against `a`), so the matches-nothing rule does not fire, and the prompt offers the other files for deletion.
- **Why:** telling a chance match from a real one needs the list to record how it was made, a format change.
- **Revisit when:** someone hits it.
- **Approved:** 2026-11-05, the user chose "Known limitation (Recommended)".

### Positional `-rme` judges the whole destination
- **Kind:** deferred.
- **What the operator sees:** `hpcp b -d out/ -rme` lands in `out/b`, but at the prompt's y it also deletes what lies outside that, such as `out/keep`; `hpcp a/x b -d out/ -rme` does the same. Same as 9.80. A source from a list is already limited to what it covers.
- **Why:** limiting it changes old behaviour and needs its own look; a source ending in a slash mirrors into the destination, where judging all of it is right.
- **Revisit when:** the next hpcp release, limiting it to where the source lands.
- **Approved:** 2026-11-05, the user chose "Deferred to 9.82 (Recommended)".

### A move of a link's directory and a path through the link can end in exit 192
- **Kind:** deferred.
- **What the operator sees:** `printf 'lb/\n' | hpcp -sfl - lb/x -d out/ -rf`, with `lb` a link to `b`: `out/x` and the link `out/lb -> b` land, `lb` goes and `b/x`, `b/y`, `b/s/z` stay, in every run. Some runs then end with "not removing lb: it, or a directory on its path, is no longer what the copy listed" and exit 192, the others exit 0. Nothing is lost: the tree comes out the same either way.
- **Why:** the removal of `lb/x` and of the link `lb` run side by side, and the check before removing sees the link already gone. 7617241 does the same (67 runs in 200, against 66).
- **Revisit when:** the next hpcp release.

### A million-entry list holds its jobs in memory
- **Kind:** deferred.
- **What the operator sees:** the check before the copy keeps every entry's paths at once: a peak of about 741 bytes more per entry than 7617241 with the copy stubbed out (740 MB for a million), and +143 bytes per file measured in a real copy of 300,000. The extra CPU is nearly all garbage collection over the kept jobs: a stubbed million took 19.08, 20.56 and 22.22 s with it on, 19.26, 19.15 and 19.56 s with it off.
- **Why:** it reads the jobs twice, and generators need `_settle_destinations` reshaped; the copy works either way.
- **Revisit when:** the next hpcp release.

### A rerun scans the destination directories one by one
- **Kind:** deferred.
- **What the operator sees:** a copy onto an existing tree reads each directory it lands in, on one worker, before the first file is copied: about 1 us per file on a local disk, longer on network storage.
- **Why:** correct and bounded as it is; a parallel scan is its own change.
- **Revisit when:** the next hpcp release.

### `-rme` can take a source inside the destination for extra
- **Kind:** deferred.
- **What the operator sees:** `hpcp out/sub/ b/ -d out/ -rme`, answered y at a terminal, deletes `out/sub/x`, the only copy when `b/x` replaced it at `out/x`: the file lies in the destination and no source lands on it, so it looks extra. With `-rf` the run then exits 212, and its message says -rme may have removed it. Same in 9.80.
- **Why:** it needs `-rme` to know what the run reads, and that belongs with limiting positional `-rme` to where a source lands.
- **Revisit when:** the next hpcp release, with "Positional `-rme` judges the whole destination".

### A source that opens and then fails on read costs the old copy
- **Kind:** deferred.
- **What the operator sees:** `cp` truncates the destination before its first read, so a source that opens and then fails (`/proc/self/mem` with an old `out/mem`) ends in exit 190 and the old copy is gone. 9.80 left it empty.
  - With several destinations, a part written through a link at the first destination and the file then taken by the second ends exit 0 with the link's target half written (`hpcp a/ -d out/ -d two/` with `out/x -> ../t/f`).
- **Why:** keeping it takes a temporary name and a rename, which is its own design; that covers the link case too. It covers landings in a tree only: a name given by `-d` keeps the write-through, because a rename would replace a FIFO named by `-d` and break a pipe feed that works today (a regular file onto a lone FIFO with a reader).
- **Revisit when:** the next hpcp release.

### A FIFO that blocks the copy
- **Kind:** deferred.
- **What the operator sees:** a regular file onto a lone FIFO at the destination hangs with or without `-co`, because `cp` writes to it and waits for a reader. As on 9.80. A regular file onto a FIFO with other names does not hang: it lands as a regular file at that name, as `-a` always did, and the other names keep the FIFO.
- **Why:** rare, and the hang is old. The temporary name and rename above (tree landings only) keeps the name until the copy is whole, and never opens the FIFO; a FIFO named by `-d` keeps today's behaviour, so a feed that works is not broken.
- **Revisit when:** the next hpcp release.

### A named FIFO or socket is copied and still exits 200
- **Kind:** deferred.
- **What the operator sees:** `hpcp a/p -d out/` makes the node, prints `Source path type error` and ends exit 200. With `-s` the exit is 0 (9.81 too). A file that is refused is not listed, so a clash adds no such line; a named FIFO whose landing is a directory prints it before the clash, and a device source a user may not make before the `Copy failed`. With `-rf` it is 212 and every named source stays, files included; nothing is lost. The same FIFO in a directory source is skipped silently with exit 0. Same on 9.80.
- **Why:** old and safe. A named special should either copy cleanly or be skipped as in a tree, and that is its own decision.
- **Revisit when:** the next hpcp release.

### A regular file writes through a device node at the destination
- **Kind:** deferred.
- **What the operator sees:** a regular source that lands on a device node, in a tree or named by `-d`, is written through it, with or without `-co`: into the device, or the disk behind it, which on a mounted disk writes over live blocks (reasoned, not run). Without `-co` the node also takes the source's mode and times. A link to a device is the device: a regular file named onto `-d lnk` writes the disk behind it. The same on 9.80. A device or FIFO source no longer does this: since 9.82 it is never read, with or without `-co`, and it never replaces a device (or a link to one) that `-d` names.
- **Why:** old, and it needs a node where a file belongs. Leaving a destination that is not a regular file alone belongs with the temporary name and rename above. A device named by `-d` may be what the operator meant to write.
- **Revisit when:** the next hpcp release.

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
- **Approved:** 2026-11-06, the user chose "Known limitation (Recommended)".

### A probe left behind by SIGKILL
- **Kind:** known limitation.
- **What the operator sees:** hpcp killed with SIGKILL in the moment it asks a directory about a name leaves a hidden `.hpcp-case-probe-PID-uuid` directory in the destination, empty or holding a small file for each spelling of the name it was asking about; where the probe asks each made name alone (a spelling the directory will not look up, as on vfat), a second such directory can be left beside it. No data is affected. SIGINT, SIGTERM and SIGHUP wait for the probe to go, and left nothing in 30 kills each.
- **Why:** SIGKILL cannot be handled, and the only way to ask a directory about two names is to make them.
- **Revisit when:** an operator reports finding one.
- **Approved:** 2026-11-06, the user chose "Known limitation (Recommended)".

### A directory that cannot be asked about a new pair
- **Kind:** known limitation.
- **What the operator sees:** two names that fold alike (`X` and `x`), at least one of them new, landing in a directory where nothing can be made, such as an owned mode 555 directory, are refused with 213: `may be one name, and the destination cannot be asked: nothing can be made in DIR to ask it (Permission denied): make it writable and run again` (a read-only mount or a full disk names its own fix). On a case-sensitive filesystem the refusal is false: 9.82 opened an owned directory and landed both. No file is lost. Where nothing can be made, the directory's listing answers: both names in it are two names, so a rerun with both names there passes.
- **Why:** it takes that exact mix, and opening the directory for the probe means another chmod and restore in the path that writes to destinations.
- **Revisit when:** someone hits it.
- **Approved:** 2026-11-06, the user chose "Known limitation (Recommended)".

### An 8.3 alias beside its long name on vfat
- **Kind:** deferred.
- **What the operator sees:** `hpcp a/ b/ -d out/ -rf` onto vfat with `a/longfilename.txt` and `b/LONGFI~1.TXT` exits 0 and one content is gone, in about half of the runs: vfat made `LONGFI~1.TXT` the short name of the first file, so the second source is written onto it. The sources are removed. 9.82 does the same.
- **Why:** a numbered alias such as `LONGFI~1.TXT` is in neither name, so no rule on names catches it, and asking the directory about the pair (`_ask_names`) cannot see that one file already has the other's name: it is the filesystem's invention, not a spelling. (A short name that is a plain fold of the long one, such as `KIZ.TXT` for `kız.txt` with codepage 850 or 857, is caught: dotless `ı` folds with `i`.)
- **Plan:** the same backstop as for legacy names: before a move removes a source, check that its landing holds that source (the size and modification time `cp -a` gave it, and that the landing is not the source itself, asked by doing in the same directory where they share it), and where it does not, keep the source and report a clash.
- **Revisit when:** the next hpcp release.

### A copy into its own inside under another spelling on FUSE exFAT
- **Kind:** deferred.
- **What the operator sees:** on exFAT through FUSE (EL7 and EL8), with `D/f` and `D/sub/f`, `hpcp D/ -d d/sub/` without `-rf` exits 0 and adds `D/sub/sub/f`: `d` is `D`, so the destination is inside the source, and the copy reads the destination's own files and copies them into it, one level deeper each run. The file it adds holds either content, as its copy races with the one that writes `D/f` onto `D/sub/f`. In one spelling (`hpcp D/ -d D/sub/`) the destination is left out of the walk, and `D/sub/f` takes `D/f`'s content as any file of a destination would, so the other spelling adds only the extra copies. The move form, `hpcp D/ -d d/sub/ -rf`, is refused (241), as in one spelling.
- **Why:** the check that a source holds the destination compares real paths, and FUSE exFAT gives each spelling of one directory a number of its own, so `D` and `d` are two real paths there. Kernel filesystems give both one number, and the check sees one directory.
- **Plan:** where a source's real path folds like the start of a destination's, ask the directory whether they are one (`_RealPlaces.same_place`), as the move check does, and leave the destination out of the walk as for one spelling.
- **Revisit when:** the next hpcp release.

### A legacy name read as UTF-8 or ASCII, against its other case
- **Kind:** deferred.
- **What the operator sees:** on a vfat or exFAT mounted with a codepage `iocharset` (`iso8859-1`, `iso8859-9`, `cp1251`, `koi8-r`; not `utf8`), two names that the mount takes for one are not asked about when the net reads one of them as UTF-8 or ASCII, which happens in three ways:
  - one name's bytes happen to be valid UTF-8 and the other's are not: `C3 A9 74` (Latin-1 `Ã©t`, UTF-8 `ét`) against `E3 A9 74`, which a cp1251 mount reads as `Г©t` and `г©t`;
  - both names' bytes are valid UTF-8, but not as each other's case: KOI8-R `бё` and `бЁ` (`C2 A3` and `C2 B3`), CP1251 `EF B8 B8` and `EF A8 B8`; CP1251 alone has 880 such two-byte pairs;
  - an ASCII name against a legacy one: `istanbul` against ISO 8859-9 `İstanbul` (`DD` for `İ`, which that table makes the other case of `i`).
  - two cases of different lengths: EUC-JP `ς` is three bytes (`8F A6 F8`) and `Σ` two (`A6 B2`), and no rule over single bytes can join names of different lengths.
  `hpcp a/ b/ -d out/ -rf` with one in `a/` and the other in `b/` exits 0 with one content gone, and the rename in case (`hpcp out/ONE -d out/OTHER -rf`) exits 0 and the file is gone. Checked on vfat mounted `iocharset=cp1251`, `koi8-r`, `iso8859-1` and `iso8859-9` (`utf8=0`), and kernel exFAT mounted `iocharset=cp1251` (VM 901). Only for such pairs, and only on such a mount: names that are not UTF-8 as a whole fold byte by byte and meet.
- **Why:** a name that is valid UTF-8 is read as UTF-8, which it is on every other mount, and the mount's own table, not UTF-8, says what its other case is. Joining every valid name with every name of the same length would ask about most trees pair by pair again.
- **Plan:** first a backstop that needs no fold rule, the only one that catches names of different lengths: before a move removes a source, check that its landing holds that source (the size and modification time `cp -a` gave it, and that the landing is not the source itself, which is asked by doing, in the same directory, where they share it), and where it does not, keep the source and report a clash. Then catch the pairs before the copy: fold every name byte by byte on a mount with a codepage `iocharset` (read from the mount options), or join byte-level keys across all names.
- **Revisit when:** the next hpcp release.

### Directories that differ only in case on vfat mounted `iocharset=utf8`
- **Kind:** deferred.
- **What the operator sees:** on a vfat mounted with `iocharset=utf8` given explicitly, `hpcp a/ b/ -d out/` with `a/D/f` and `b/d/g` exits 193 (`Directory sync error: .../b/d to .../out/d: [Errno 2] No such file or directory`, then `Copy failed ... No more destination paths to try`), and `g` is not copied; with `-rf` it exits 212 and keeps both sources. Nothing is lost, and 9.82 does the same. Checked on VM 901; vfat with its default charset or `iocharset=iso8859-1`, kernel exFAT, NTFS (`ignore_case`) and ext4 casefold merge the two, exit 0.
- **Why:** vfat mounted that way looks a name up in its own case only (`out/d` is not found) but will not make it either, as `out/D` holds the name; the copy names the directory as the source spells it.
- **Plan:** write a merged directory's files under the spelling the directory holds, which `_RealPlaces` knows from asking it, or report the pair as a clash.
- **Revisit when:** the next hpcp release.

### A landing within about 57 characters of PATH_MAX
- **Kind:** deferred.
- **What the operator sees:** two names that fold alike landing at a path within about 57 characters of the system's path limit (4096 on Linux) are refused with 213, `refuses the name ... (File name too long)`, because the probe directory makes the path longer. Never a loss; 9.83's earlier probe did the same.
- **Why:** the probe makes its names at the probe directory's full path.
- **Plan:** make the probe directory and its names relative to a descriptor of the directory (`dir_fd`), so their length is the name's own.
- **Revisit when:** the next hpcp release.

