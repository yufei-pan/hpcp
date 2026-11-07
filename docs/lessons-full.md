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
- Changed: one rule in `_settle_destinations` replaces `_landing_clash`, `_same_or_inside` and the move-onto-itself loop: a run is refused when a file would land, by real path, on a source file (its own too, in a move), unless, in a copy, that file belongs to a directory that lands on itself. A source that holds the destination leaves it out of its copy, an argument now as a list always did, and `-rme` then checks nothing, because what was left out looks extra, since its copy read files other sources were writing (an empty file, exit 0). `_RealPlaces` gives every check the same real paths. A directory that lands on itself no longer counts as a writer and runs no copy: another source may be writing its files, so even a read for the identity check could fail (exit 190 or 191, 5 runs in 60), while `-rme` still learns what it holds. Forms 7617241 refused now run, forms it let through are refused, and a symlink in the destination that made two list entries one file is a clash (that known limitation is gone).
- Why we had to change: `_landing_clash` compared source roots, by spelling. `hpcp p/q/ -d p/ -rf` lost a file in 120 of 120 runs, because a source was never compared with itself. A link in any path hid the clash, while `_inside_destinations` next to it already compared real paths. And `hpcp */conf -d merged/`, a safe rerun, exited 213, because a directory that copies nothing was still taken for a reader.
- Original design was wrong because: the danger is at file level (a file one copy writes is a file another reads, or a move removes), and a root says nothing about that. Each patch added a case for one shape of root. Its check also looked at what exists (`lexists`), not at what the listing would really copy, so an excluded path could refuse a run.
- Lesson: state the danger in one sentence about the thing at risk, then judge exactly that thing, from the same data the work itself uses. Make it cheap with a gate that proves the expensive question cannot arise, not by asking a cheaper question. A question two checks ask gets one answer, computed once.

## 2026-11-06 A landing never leaves its destination
- Changed: `_source_dest` treats '..' as it treats '.': no name to land under, so the contents land in the destination. `hpcp .. -d out/` used to land in `out/..`.
- Why we had to change: `cd w/sub; hpcp .. -d out/` wrote into `w/sub/` and left `out/` empty, and `cd w/c/s; hpcp ../../x .. -d ../ -rf` swapped two files through it and lost both, in 60 of 60 runs.
- Original design was wrong because: the basename of a source was taken for a name. '.' came out right by accident (joining it changes nothing), and '..' joined as a name climbs out of the destination. `hpcp .. -d out`, without the slash, already gave the contents.
- Lesson: a path part is a name only if it names something. When a layout rule joins user input onto a destination, say what it does with the parts that are not names.

## 2026-11-06 A write goes to an inode, not a path
- Changed: `-co` replaces a destination file that has other hard links instead of writing through it; such a file then has the source's mode, not its old one.
- Why we had to change: `hpcp a/ -d out/ -co -rf` with `ln a/y out/x` wrote a/x into a/y's inode: out/y came out holding X, empty or partial in most runs, and the source was then removed. `hpcp a/ b/ -d out/ -co -rf` with `ln b/x out/y` lost B in most runs.
- Original design was wrong because: `-co` swapped `cp -af` for `cp -f` to keep metadata and lost what `-a` also does: `cp -a` unlinks a destination with several names, `cp -f` opens it. The landing rule works by path, and a hard link has no path to compare.
- Lesson: when an option changes how a file is written, list what the old way did besides the thing the option is about. A check by path protects paths; what shares an inode has to be left by the writer.

## 2026-11-06 Ask the rule that decided, not the spelling
- Changed: a directory named as an argument that is a destination and lands below itself is left out of the run, as a list entry is: `hpcp a b -d b/ -rf` was 241 and is now exit 0, `a` moved, `b` left. `hpcp a -d a/ -rf` was 241 and now moves nothing, exit 0. `-rme` then checks nothing.
- Why we had to change: `hpcp * -d backup/` with `backup` in the working directory gave 0, 213, 213: the first run nests `backup/backup`, and the next run's files land on it. 7617241 gave 0, 0, 0, one level deeper each run.
- Original design was wrong because: leaving out a source that holds the destination covers what it holds, and a source that is the destination holds nothing to copy but itself. It stayed a source that read the files others land on. A prototype that tested the trailing slash away made `a/` look named: `hpcp a/ b/ -d a/ -d out/ -rf` then destroyed a/x. `a/` lands on itself, and 241 stays right.
- Lesson: when a case turns on where something lands, ask `_source_dest` and compare its answer with the destination; never re-derive the rule from the spelling. `a`, `a/`, `.` and `..` differ only in that answer.

## 2026-11-06 'None' and 'could not look' are two answers
- Changed: a landing in a directory that can be neither listed nor searched is refused, exit 213, naming it and `chmod u+rx`. `chmod 000 out; hpcp a/ -d out/ -rf` ran in 9.80 and 7617241. A directory that cannot be listed (0300), or has more entries than the run has names to ask about (past max(10000, files)), is asked by name instead; that also stops one file into a 2M-entry directory from reading all of it (1.04 s, back to about 0.1 s). A listing alone answers too, so `chmod 600 out` runs as in 9.80. A first version refused every directory without search permission, in a probe before the listing: a source with `a/locked` at 600 then gave 0, 213, 213, where 9.80 gives 0, 0, 0, because every folder is asked about and the copy made `out/locked` too.
- Why we had to change: the scan answered "no links" when it failed, so `out/y -> ../b/x` in a 0300 `out` was not seen and the run's own directory sync opened the way: b/x lost in 20 of 20 runs.
- Original design was wrong because: one set meant both "I looked and found none" and "I could not look". And `os.path.islink` turns EACCES into False, the same mistake one level down: only a lstat tells them apart. The scan also read a whole directory for one name, since nothing said how much reading a question is worth.
- Lesson: a look that can fail returns three answers. Bound the look by what the answer is worth, and treat "stopped" the same as "failed": ask a narrower way, open the gate, or refuse with the fix. Refuse only at the question that needs the answer: a look that raises by itself refuses for places nothing lands in.

## 2026-11-06 Clean up what you made
- Changed: a failed copy removes the destination file only when the attempt changed it (inode, size or mtime against before), and a source that cannot be opened gets no `cp` at all. Exit codes stay. The state is the path's own (lstat): a link stays through a failed write, so a retry finishes its target.
- Why we had to change: `os.remove(dest_path)` ran after any failure, so an unreadable source deleted the good copy a backup held. And the last retry returned before it removed its part.
- Original design was wrong because: the cleanup assumed the failure was its own doing. It was written for a part of a file, and a destination that was there before looks the same as a part.
- Lesson: before cleaning up after a failure, prove the failed step made what you are about to remove. And `cp -a` acts on the destination before it opens the source, so only not starting it keeps a hard-linked one.

## 2026-11-06 A node is not its contents
- Changed: under `-co` a source that is not a regular file goes through `cp -af`, so a FIFO, socket or device node is made, as in a plain run, with its mode, owner and times, and never read. A source that would take the name of a device node the destination names itself (`hpcp /dev/sda -d /dev/sdb`, `hpcp a/p -d /dev/sdb`, with or without `-co`; a regular file too when the node has other names) is a clash, exit 213, and the node and the source stay. A plain run replaced the node (exit 200); `-co` read the source into it, or hung.
- Why we had to change: `-co` with a device source read it to its end and wrote it through what was at the destination, another disk if the numbers differed, and a FIFO source hung on the first run. Sending those to `cp -a` alone would have unlinked a device named by `-d`, as a plain run already did.
- Original design was wrong because: `-co` was written for files, and "copy the contents" has no meaning for a node: opening one is an act of its own. And nothing separated a node that is a file of the tree, which a source replaces, from the device the operator named as the destination.
- Lesson: when a flag narrows what a copy does, ask what it means for every kind of source, not only the one it was written for. When moving something to a safer route, ask what that route does to the destination: `cp -a` never reads a node, and always unlinks what is there.

## 2026-11-06 A refusal is as narrow as its reason
- Changed: a device node that `-d` names, and that a source would replace, now takes only that destination out of the job (`_copy_job`). With several `-d` (spill-over) the file lands on the others with a note and the run ends as any other run of that source (a move completes); it is a clash, exit 213, and the job copies nothing only when every destination is such a node. A landing on a name any `-d` gives counts as named (`-d out/ -d out/n`). A patch applied to a directory of a mounted filesystem (`-siff`) is flushed with syncfs too, in a `finally`. A link to a disk counts as the disk; a node for the same device (same kind and number) is no clash; a symlink source, which runs no `cp`, is no clash and keeps 9.81's exit 162. The refusal names the fix. A device source that a user without the right to make nodes cannot copy is refused before `cp -a` takes the old file away, asked by making a node of that kind beside the destination.
- Why we had to change: the first version dropped the whole job when one `-d` was such a node, so `hpcp a/n -d out/ -d dev2` copied nothing where 9.81 landed `out/n`. It judged the destination's name (`lstat`), so a link to a disk was replaced by the source's node, and it called a matching pair a clash. A non-root device copy deleted the old file because `cp -a` unlinks before its `mknod`.
- Original design was wrong because: the reason (one destination is a device) was recorded against the job, which has many destinations, and the check asked what the name is, not what it leads to. And a refusal reasoned from `cp`'s documentation said "would replace it" for cases `cp` never runs in.
- Lesson: keep a refusal at the grain of its reason (per destination, per file), and decide what it covers by asking what the thing leads to (`stat`) and whether the tool runs at all. A permission that decides whether data is lost is asked by doing it (make the node, remove it), before the step that takes the old data away.

## 2026-11-06 A name that matched nothing is still a name
- Changed: a source named as an argument that matches nothing is kept by `_named_sources`, so `verify_src_path` reports it: `hpcp nosuch ok/g -d out/` was exit 0 and is 207, with the rest copied. A move counts it as an error (212, everything kept); `-rme` stops for it by arguments and lists alike. A list entry that is gone does not stop a move: 207 and the rest moves, as in 9.82. A removal with no destination goes on, exit 207. A name excluded with `-e` counts for nothing. A name under a directory that cannot be searched says `cannot be reached`. `hpcp nosuch ok/g` was 70 and is 240. A name with glob characters that match nothing as a pattern is that file, where it was dropped. The last word with no `-d` is held back until the destination is known, so a prompt that turns it down reports it.
- Why we had to change: a mistyped source was dropped with exit 0. With `-rme` that is worse: what the mistyped source holds at the destination looks extra.
- Original design was wrong because: `expand_paths` answers "what matches", and `hpcp()` used that as "what was named", so nothing that matched nothing reached the one check that reports it. And the remove gate counted errors from the copy's start. A first version made a gone list entry stop a move too, so both routes agreed; the user chose otherwise, and rightly: a typed name is a mistake to stop for, a list is older than the tree, and a gone entry has nothing to lose. For `-rme` the two agree, as what looks extra may be what the missing source holds.
- Lesson: when a step can drop what the operator typed, keep it until the step that can report it. Two routes share one answer only when the question is the same; ask what the input means on each. A refusal that names a way out (`-e`) has to honour it.

## 2026-11-06 A name is what the destination reads
- Changed: every landing is folded in memory (`_names_may_fold`, a wide net: NFD, casefold, no ignorable code points). Only when two collide does a run ask, in the directory they land in, with those two names (`_RealPlaces.same_place`, `_one_name`). One name is a clash, exit 213, directory sources and one tree too, where the run was exit 0 with one file's content gone. Each destination of several is asked. A directory that cannot be asked gets a 213 that says so. A directory source is now always listed before the copy, which reuses the listing. `_RealPlaces.is_link` asks by name when a link folds like the landing, so `out/X -> ../src/y` with `src/x` is 213. `-rme` compares what it takes for extra with what the copy wrote by identity, on the names the listing does not show.
- Why we had to change: `a/X` and `b/x` onto a casefold directory lost one, and either could win. With `-rme`, `a/x` onto `out/X` deleted the file from both sides, exit 0.
- Original design was wrong because: `_settle_destinations` compared landings as strings, and `-rme` compared listings as strings. The first fix asked each destination once, up front, "do you fold?", and then applied Python's rules. That gave a false 213 (`Straße` and `STRASSE` on vfat), missed a folding directory or mount below a plain top, could not ask an unwritable top, left a file behind in an append-only directory, and wrote into every destination on every run, all to keep a shortcut that skipped a listing the copy makes anyway. It held that the probe had to come before the shortcut; the listing was free, so the shortcut was the thing to give up.
- Lesson: no single call answers what a filesystem does with a name, and no local rule predicts it. Cast the net in memory, wide, since the filesystem judges what it catches, and ask about the actual pair where it lands. Before keeping a shortcut, price what it saves. Anything that compares names at a destination needs a comparison by identity behind it.

## 2026-11-06 Could not compare is not a crash
- Changed: `copy_file` asks `_known_identical`: a `PermissionError` on either side of the identity check means "not known to be the same". An unreadable source then stops at `_can_open`, exit 190 where it was 191, and the old copy stays. An unreadable destination is copied again, exit 0 where it was 191. Any other read error stays a crash.
- Why we had to change: the identical-file check ran before the readability check, so a rerun with a source gone unreadable ended as `Copy process crashed`.
- Original design was wrong because: a check meant to skip work was treated as one that cannot fail. Moving `_can_open` ahead of it would have added an open per skipped file on every rerun, and catching every `OSError` would have sent an I/O error on to `cp`, which truncates the old copy before its first read.
- Lesson: when a shortcut's question cannot be answered, fall through to the step that owns the failure, and only for the failure that step really handles.

## 2026-11-06 One door for a directory
- Changed: `-nds` makes a directory through `_make_dest_directory`, as the sync does: a file in the way is a clash, exit 213, where it was 70 with a traceback; a directory it cannot make is `Directory sync error`, 193, where it was 70.
- Why we had to change: `process_copy` called `_makedirs` itself, outside every handler.
- Original design was wrong because: "skip the sync" was written as a second way of making the directory, without the checks the sync had.
- Lesson: a flag that skips a step skips what the step changes, not the checks around it. Share the part that stays.

## 2026-11-06 A refusal offers a fix its reader can run
- Changed: a 213 for a place that cannot be searched names the link that led there, what it leads to, and `rm` of it, when that place is on the way to the link's target (`_last_link`, on the failure path only). A link that only spells the way (`alias -> real`, `real/out` mode 000) is not blamed. Exit codes stay.
- Why we had to change: `out/l -> /root/secret/f` said `/root cannot be searched` and offered a `chmod` a user cannot run.
- Original design was wrong because: `_RealPlaces` raised from the walk, which no longer knew the path it was asked about. The public calls now add the link on the way out, so the walk stays as it was.
- Lesson: a refusal states what would let it pass, in terms of what its reader controls. Add that where the question was asked, not where the walk stopped.

## 2026-11-06 The reason travels with the error
- Changed: a `Task return code error` carries the last three lines of what the tool printed to stderr, after the return code, so `Copy failed` says `No space left on device`. A tool that says nothing keeps the old sentence. The prefix, and so the exit code, stays.
- Why we had to change: a full disk said only "return code 1".
- Original design was wrong because: the runner had the task, and with it the stderr, and passed on only the number.
- Lesson: the layer that sees a tool's own words puts them in the error. Prefixes decide; the words only inform.

## 2026-11-07 Different numbers prove nothing
- Changed: `_one_name` no longer answers "two names" when both names stat as two files: it asks by making them, every time, and only where nothing can be made does a listing answer (both names listed, two names). `-rme` takes an expected name that the listing lacks but that is there for the file under another spelling, by its folded name, not its number. Directories in another spelling (`-rme`), links in another case (`_link_targets`) and a move onto itself, or into its own inside, in another spelling (241) are asked of the directory too. Names that are not valid UTF-8 meet in the net (`replace`), and the directory decides. A 213 that could not ask says why: nothing could be made there (and what fixes that), or the directory refuses the name.
- Why we had to change: exFAT through FUSE, the only exFAT on EL7 and EL8, gives each spelling of one file its own `st_ino`. On a rerun both names were there with two numbers, so round 2's shortcut said two names: `hpcp a/ b/ -d out/ -rf` was 213, then 0 with one content gone; `hpcp X -d x -rf` lost the file; `-rme` deleted what the copy wrote through. A Latin-1 vfat lost one of `\xc9` and `\xe9`, which the net kept apart.
- Original design was wrong because: it took "these two numbers differ" as "these are two files". That holds on every kernel filesystem and fails on any filesystem that numbers names, not files. And the net, which has to be wide because the filesystem judges what it catches, kept apart names it could not read.
- Lesson: an identity check proves sameness, never difference. To learn whether two names are one, ask the filesystem by doing it; an observation (a number, a listing) is only a fallback where nothing can be done, and only in the direction it can prove.
