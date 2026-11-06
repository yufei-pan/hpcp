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

### Positional source inside the destination, another source landing on it
- **Kind:** deferred.
- **What the operator sees:** `hpcp d1 d2/d1/ -d d2/ -rf` (d1/x=A, d2/d1/x=B) leaves only d2/x=B and exits 0, so A is gone. `hpcp a out/a/x -d out/ -rf` leaves only out/x. 9.80 loses content here too. A list with no positional source is not affected.
- **Why:** not a regression. The proposed fix (widening the overlap check to source roots and real paths) is untested and adds another rule to this round.
- **Revisit when:** the next hpcp release (9.82).
- **Approved:** 2026-11-05, the user chose "Deferred to 9.82 (Recommended)".

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
