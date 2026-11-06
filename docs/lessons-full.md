# Lessons: full entries

Why this project's APIs, formats and agreed behaviours had to change. `lessons.md` has one line per
entry: read that first, and look entries up here by date and name rather than reading this whole. New
entries go last, in the same commit as the change:

```markdown
## YYYY-MM-DD <short name>
- Changed: ...
- Why we had to change: ...
- Original design was wrong because: ...
- Lesson: ...
```

## 2026-11-05 A list of paths names paths, not just files
- Changed: hpcp's -sfl and stdin lists land each path at dest/<path as given>, minus a leading / and any .., the layout `rsync --files-from` and `cp --parents` give (they differ from hpcp in how the directories on the way are treated: hpcp gives a directory it creates its source's mode and owner, and leaves an existing one alone). They used to land by basename.
- Why we had to change: `find src -type f | hpcp -d out` put every file flat, so src/a/x.txt and src/b/x.txt overwrote each other with rc=0, and which one survived varied.
- Original design was wrong because: it fed list entries through the rule for positional arguments, where a basename is what the operator typed. A pipeline's paths carry their directories as information, and the frozenset it read them into lost their order as well.
- Lesson: when a second input route joins an existing one, ask what its input means before reusing the rule. And when two inputs can land on one output, name what the operator meant by each: two named files landing together is a mistake to report, two directories merging is an overlay to honour. Decide the winner once, before any work starts, so no copy mode can answer differently. A rule that compares sources with destinations belongs where destinations are known.

## 2026-11-06 A landing is a path too
- Changed: when a source lies inside the destination and another source would land on it, inside it, or (a directory) on a path that holds it with a file at its place, hpcp names both paths and exits 213 before anything is copied or removed. The first 9.81 build handled only a named file overwritten by a directory's files, and only when two destinations overlapped; a move exited 241 for it.
- Why we had to change: `cd w/D/c; printf '../../c/r\nr\n' | hpcp -d .. -rf` landed one entry on the other entry's source while the second was being copied: 20 runs in 360 left an empty `w/D/r`, with the sources removed and exit 0, and with `-s` or `-nb` one content was always lost. `hpcp d1 d2/d1/ -d d2/ -rf` did the same.
- Original design was wrong because: the settling pass looked at where files land, and ran only when two landings could meet. A landing on a source is not two landings meeting, so the pass never saw it, and a copy that writes a file another copy is reading has no safe order.
- Lesson: for every destination a run writes, ask what else the run reads there. A check on sources against destinations has to cover the destination side of every source, and it runs once before any copy, so every mode refuses the same way.

## 2026-11-06 Judge the thing you protect
- Changed: one rule in `_settle_destinations` replaces `_landing_clash`, `_same_or_inside` and the move-onto-itself loop: a run is refused when a file would land, by real path, on a source file (its own too, in a move), unless, in a copy, that file belongs to a directory that lands on itself. A positional source that holds the destination leaves it out of its copy, as a list's always did, since its copy read files other sources were writing (an empty file, exit 0). `_RealPlaces` gives every check the same real paths. A directory that lands on itself no longer counts as a writer and runs no copy: another source may be writing its files, so even a read for the identity check could fail (exit 190 or 191, 5 runs in 60), while `-rme` still learns what it holds. Forms 7617241 refused now run, forms it let through are refused, and a symlink in the destination that made two list entries one file is a clash (that known limitation is gone).
- Why we had to change: `_landing_clash` compared source roots, by spelling. `hpcp p/q/ -d p/ -rf` lost a file in 120 of 120 runs, because a source was never compared with itself. A link in any path hid the clash, while `_inside_destinations` next to it already compared real paths. And `hpcp */conf -d merged/`, a safe rerun, exited 213, because a directory that copies nothing was still taken for a reader.
- Original design was wrong because: the danger is at file level (a file one copy writes is a file another reads, or a move removes), and a root says nothing about that. Each patch added a case for one shape of root. Its check also looked at what exists (`lexists`), not at what the listing would really copy, so an excluded path could refuse a run.
- Lesson: state the danger in one sentence about the thing at risk, then judge exactly that thing, from the same data the work itself uses. Make it cheap with a gate that proves the expensive question cannot arise, not by asking a cheaper question. A question two checks ask gets one answer, computed once.

## 2026-11-06 A landing never leaves its destination
- Changed: `_source_dest` treats '..' as it treats '.': no name to land under, so the contents land in the destination. `hpcp .. -d out/` used to land in `out/..`.
- Why we had to change: `cd w/sub; hpcp .. -d out/` wrote into `w/sub/` and left `out/` empty, and `cd w/c/s; hpcp ../../x .. -d ../ -rf` swapped two files through it and lost both, in 60 of 60 runs.
- Original design was wrong because: the basename of a source was taken for a name. '.' came out right by accident (joining it changes nothing), and '..' joined as a name climbs out of the destination. `hpcp .. -d out`, without the slash, already gave the contents.
- Lesson: a path part is a name only if it names something. When a layout rule joins user input onto a destination, say what it does with the parts that are not names.
