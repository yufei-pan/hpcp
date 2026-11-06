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
