# Lessons

What this project's past changes taught us. Read all of this before changing an API, a format or an
agreed behaviour here. For the full story of a line, find the entry with the same date and name in
`lessons-full.md`.

When one has to change, add the full entry to `lessons-full.md` and one line here, newest last, in the
same commit as the change. Keep this file under 20 KB: at the cap, merge lines before adding.

- **2026-11-05 A list of paths names paths, not just files**: when a second input route joins an existing one, ask what its input means before reusing the rule; when two inputs can land on one output, name what the operator meant by each and decide the winner once, before any work starts.
- **2026-11-06 A landing is a path too**: for every path a run writes, ask what else the run reads there; refuse a source landing on another source's path (or inside it, or on a directory holding it) once, before any copy, in every mode.
- **2026-11-06 Judge the thing you protect**: a safety check on data asks its question of the files, by real path, from what the run will really do; one built on roots and spellings is wrong both ways, and two checks asking one question must share one answer.
- **2026-11-06 A landing never leaves its destination**: a path part with no name of its own ('.', '..') gives nothing to land under; joining it as a name made `hpcp .. -d out/` write beside `out/`.
- **2026-11-06 A write goes to an inode, not a path**: an option that writes in place writes every name of that file; a check by path cannot see a hard link, so the writer has to leave it.
- **2026-11-06 Ask the rule that decided, not the spelling**: 'a source that is the destination' splits on where it lands, so ask the landing rule; `a` lands under itself and can go, `a/` lands on itself and must stay judged, and a copy of the rule on the text gets that backwards.
