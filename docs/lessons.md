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
- **2026-11-06 'None' and 'could not look' are two answers**: an empty set from a failed scan reads as 'no links'; a check keeps three answers (no, yes, unknown), asks another way when it can, and refuses, naming the fix, only at a question that needs the answer and nothing can give it.
- **2026-11-06 Clean up what you made**: a cleanup after a failure first proves the failed step made what it removes; a destination that was there before looks just like a part of a copy.
- **2026-11-06 A node is not its contents**: `cp -f` opens what it is given, so a mode that copies contents takes regular files only; and the safe route for a node (`cp -a`) unlinks what is there, so what the operator named as the destination is judged before it, once, by asking `_source_dest` where the source lands.
- **2026-11-06 A refusal is as narrow as its reason**: record a refusal against what the reason is about (one destination, one file), not the job around it; ask what a name leads to (`stat`) and whether the tool runs at all before saying what it would do; a permission that decides data loss is asked by doing it, before the step that takes the old data.
- **2026-11-06 A name that matched nothing is still a name**: a pattern expander that returns only matches loses the names the operator typed; keep the name so the one check that reports a missing source sees it, and decide per route what a missing source means: a name typed now is an error, a list entry may simply be older than the tree.
- **2026-11-06 A name is what the destination reads**: two strings are two names only to a filesystem that says so, and each has its own rules; cast a wide net in memory, then ask the directory about the actual pair by doing it. Asking 'does this destination fold?' once gave a false refusal, missed what lay below the top, and wrote on every run; and a check by name needs a check by identity beside it, or a mirror deletes what the copy just wrote.
- **2026-11-06 Could not compare is not a crash**: a check that may skip work treats a read it was refused as 'not known to be the same' and goes on to the step that reports it; only the refusal, as any other read error reaches a tool that truncates first.
- **2026-11-06 One door for a directory**: the flag that skips a step must not skip what the step refused; the refusal and the making of a directory are one helper, or the second route ends in a traceback.
- **2026-11-06 A refusal offers a fix its reader can run**: name the thing in the reader's own tree (the link), not only where the walk stopped, which can belong to anyone.
- **2026-11-06 The reason travels with the error**: a runner that knows what a tool said puts it in the error it raises, or every caller reports only a return code.
- **2026-11-07 Different numbers prove nothing**: equal st_ino means one file, but a filesystem may number each spelling of one name (exFAT through FUSE, CIFS noserverino); whether two names are one is asked by doing it, and a listing answers only where nothing can be made.
- **2026-11-07 A cheaper filter is no narrower net**: speed up a safety check by making it cheaper, never by a cheaper rule in front of it that is not a superset of it (lowercasing is not folding: it let a landing through a link, and `σ`/`ς`, through).
- **2026-11-07 Ask the whole group at once**: when a wide net groups candidates for a costly question, ask about the whole group in one go and keep the answer for the run, or the cost is the square of a group that safety keeps large; a reason names what failed, the name or the place.

