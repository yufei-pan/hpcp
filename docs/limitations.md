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

### A move where one source's landing is another source's path
- **Kind:** deferred.
- **What the operator sees:** `hpcp d1 d2/d1/ -d d2/ -rf` (d1/x=A, d2/d1/x=B) is a race in the default copy: in 30 runs on 9.81, A survived 10 times, B 13, and d2/x was 0 bytes 7 times, both contents gone, the sources removed, exit 0 (9.80: 13, 12, 5). With `-s` or `-nb`, A survived all 12 runs on both versions. The list form loses one content too: `cd w/D/c; printf '../../c/r\nr\n' | hpcp -d .. -rf` (w/c/r=OUT, w/D/c/r=IN) leaves only `w/D/r`, holding OUT or IN from run to run, exit 0 (also after the 9.81 landing rule): both entries land in the destination, one at the path of the other, and both sources are removed. `hpcp a out/a/x -d out/ -rf` leaves only out/x. 9.80 loses content here too.
- **Why:** not a regression. It happens in a move where one source's landing is another source's path; the proposed fix (widening the overlap check to source roots and real paths) is untested and adds another rule to this round.
- **Revisit when:** the next hpcp release (9.82).
- **Approved:** 2026-11-05, the user chose "Keep deferred, correct the text (Recommended)" (replacing the text approved as "Deferred to 9.82 (Recommended)").

### Destination symlink at a directory's path
- **Kind:** deferred.
- **What the operator sees:** with `out/d -> ../elsewhere`, `hpcp src/ -d out/` writes src/d's files into elsewhere and changes elsewhere's mode, and as root its owner, exit 0, as in 9.80. A user who can write the destination can plant `d -> /etc` before root restores into it.
- **Why:** not a regression, and it needs its own decision: refuse as `cp -a` does, or write through without touching the target's mode and owner, since someone with a symlinked data directory may rely on writing through.
- **Revisit when:** the next hpcp release (9.82).
- **Approved:** 2026-11-05, the user chose "Deferred to 9.82 (Recommended)".

### Symlink in the destination makes two list entries one file
- **Kind:** known limitation.
- **What the operator sees:** with `out/l -> real`, `printf 'l/x\nreal/x\n' | hpcp -d out -rf` writes both to out/real/x, so one content is lost and both sources are removed, exit 0. 9.80 loses one too, through its flat layout.
- **Why:** catching it means resolving the real parent of every landing on the destination side before copying, and it needs a symlink already in the destination plus a list naming both spellings.
- **Revisit when:** someone hits it, or the 9.82 destination-symlink work makes the check cheap.
- **Approved:** 2026-11-05, the user chose "Known limitation (Recommended)".

### A stored list used under another spelling can match by chance
- **Kind:** known limitation.
- **What the operator sees:** after `hpcp s -fl S`, `hpcp s/ -fl S -rme` with s/a and s/s/a present shares one name with the list (`s/a` against `a`), so the matches-nothing rule does not fire, and the prompt offers the other files for deletion.
- **Why:** telling a chance match from a real one needs the list to record how it was made, a format change.
- **Revisit when:** someone hits it.
- **Approved:** 2026-11-05, the user chose "Known limitation (Recommended)".

### `-nds` with a directory source meeting a file in the destination
- **Kind:** deferred.
- **What the operator sees:** with `-nds`, `hpcp a/x -d out/` where a/x is a directory and out/x a file exits 70 with a `File exists` traceback instead of a clean clash, exit 213. The file is untouched, as in 9.80.
- **Why:** ugly but safe, and old.
- **Revisit when:** the next hpcp release (9.82).
- **Approved:** 2026-11-05, the user chose "Deferred to 9.82 (Recommended)".
