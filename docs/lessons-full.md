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
