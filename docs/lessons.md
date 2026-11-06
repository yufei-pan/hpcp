# Lessons

What this project's past changes taught us. Read all of this before changing an API, a format or an
agreed behaviour here. For the full story of a line, find the entry with the same date and name in
`lessons-full.md`.

When one has to change, add the full entry to `lessons-full.md` and one line here, newest last, in the
same commit as the change. Keep this file under 20 KB: at the cap, merge lines before adding.

- **2026-11-05 A list of paths names paths, not just files**: when a second input route joins an existing one, ask what its input means before reusing the rule; when two inputs can land on one output, name what the operator meant by each and decide the winner once, before any work starts.
- **2026-11-06 A landing is a path too**: for every path a run writes, ask what else the run reads there; refuse a source landing on another source's path (or inside it, or on a directory holding it) once, before any copy, in every mode.
