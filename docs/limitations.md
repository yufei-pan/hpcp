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
- **What the operator sees:** with `out/d -> ../elsewhere`, `hpcp src/ -d out/` writes src/d's files into elsewhere and changes elsewhere's mode, and as root its owner, exit 0, as in 9.80. A user who can write the destination can plant `d -> /etc` before root restores into it. `hpcp src/ -d out/ -rme` with `out/d -> ../src/d` removes the link as an extra, and `d` is not mirrored.
- **Why:** not a regression (a link there that leads onto a source file is refused, exit 213), and it needs its own decision: refuse as `cp -a` does, or write through without touching the target's mode and owner, since someone with a symlinked data directory may rely on writing through.
- **Revisit when:** the next hpcp release (9.82).
- **Approved:** 2026-11-05, the user chose "Deferred to 9.82 (Recommended)".

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

### Positional `-rme` judges the whole destination
- **Kind:** deferred.
- **What the operator sees:** `hpcp b -d out/ -rme` lands in `out/b`, but at the prompt's y it also deletes what lies outside that, such as `out/keep`; `hpcp a/x b -d out/ -rme` does the same. Same as 9.80. A source from a list is already limited to what it covers.
- **Why:** limiting it changes old behaviour and needs its own look; a source ending in a slash mirrors into the destination, where judging all of it is right.
- **Revisit when:** the next hpcp release (9.82), limiting it to where the source lands.
- **Approved:** 2026-11-05, the user chose "Deferred to 9.82 (Recommended)".

### A move of a link's directory and a path through the link can end in exit 192
- **Kind:** deferred.
- **What the operator sees:** `printf 'lb/\n' | hpcp -sfl - lb/x -d out/ -rf`, with `lb` a link to `b`: `out/x` and the link `out/lb -> b` land, `lb` goes and `b/x`, `b/y`, `b/s/z` stay, in every run. Some runs then end with "not removing lb: it, or a directory on its path, is no longer what the copy listed" and exit 192, the others exit 0. Nothing is lost: the tree comes out the same either way.
- **Why:** the removal of `lb/x` and of the link `lb` run side by side, and the check before removing sees the link already gone. 7617241 does the same (67 runs in 200, against 66).
- **Revisit when:** the next hpcp release (9.82).

### A million-entry list holds its jobs in memory
- **Kind:** deferred.
- **What the operator sees:** the check before the copy keeps every entry's paths at once: a peak of about 741 bytes more per entry than 7617241 with the copy stubbed out (740 MB for a million), and +143 bytes per file measured in a real copy of 300,000. The extra CPU is nearly all garbage collection over the kept jobs: a stubbed million took 19.08, 20.56 and 22.22 s with it on, 19.26, 19.15 and 19.56 s with it off.
- **Why:** it reads the jobs twice, and generators need `_settle_destinations` reshaped; the copy works either way.
- **Revisit when:** the next hpcp release (9.82).

### A rerun scans the destination directories one by one
- **Kind:** deferred.
- **What the operator sees:** a copy onto an existing tree reads each directory it lands in, on one worker, before the first file is copied: about 1 us per file on a local disk, longer on network storage.
- **Why:** correct and bounded as it is; a parallel scan is its own change.
- **Revisit when:** the next hpcp release (9.82).

### `-ncd` can end in a traceback
- **Kind:** deferred.
- **What the operator sees:** `-sfl` with `-ncd`, when a listed file's directory is missing under the destination: `cp` fails and each retry prints a Python traceback before the run ends with an error code (30 to 56 of 4000 fuzz runs). Nothing is lost.
- **Why:** old, ugly but safe.
- **Revisit when:** the next hpcp release (9.82).

### `-rme` can take a source inside the destination for extra
- **Kind:** deferred.
- **What the operator sees:** `hpcp out/sub/ b/ -d out/ -rme`, answered y at a terminal, deletes `out/sub/x`, the only copy when `b/x` replaced it at `out/x`: the file lies in the destination and no source lands on it, so it looks extra. With `-rf` the run then exits 212, and its message says -rme may have removed it. Same in 9.80.
- **Why:** it needs `-rme` to know what the run reads, and that belongs with limiting positional `-rme` to where a source lands.
- **Revisit when:** the next hpcp release (9.82), with "Positional `-rme` judges the whole destination".

### A landing through a link into an unreachable place blames the wrong directory
- **Kind:** deferred.
- **What the operator sees:** with `out/l -> /root/secret/f`, the 213 says `/root` cannot be searched and offers a `chmod` a non-root user cannot run. The refusal is right; it should name the link.
- **Why:** `_RealPlaces.directory` raises where the walk stopped and no longer knows which landing led there.
- **Revisit when:** the next hpcp release (9.82).

### A named source that matches nothing beside a good one is dropped silently
- **Kind:** deferred.
- **What the operator sees:** `hpcp nosuch ok/g -d out/`, also with `-rf`, copies `ok/g` and exits 0 without a word about `nosuch`, as 9.80 does. Alone it is exit 240, and the list route gives 207.
- **Why:** the glob expansion (near line 756) drops it. It should say so and exit 207 after copying the rest.
- **Revisit when:** the next hpcp release (9.82).

### A source that opens and then fails on read costs the old copy
- **Kind:** deferred.
- **What the operator sees:** `cp` truncates the destination before its first read, so a source that opens and then fails (`/proc/self/mem` with an old `out/mem`) ends in exit 190 and the old copy is gone. 9.80 left it empty.
- **Why:** keeping it takes a temporary name and a rename, which is its own design.
- **Revisit when:** the next hpcp release (9.82).
