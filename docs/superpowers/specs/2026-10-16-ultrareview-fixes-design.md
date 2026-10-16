# Whole-code ultrareview fixes (hpcp 9.73)

## Problem

On 2026-10-16 two `/code-review ultra` runs audited all of `hpcp.py` on main 0be41b5 (hpcp 9.72). Run 1 covered the helpers, file lists, copy/delete/sync engine, remove-extra, orchestration and CLI. Run 2 covered diff images and disk patches. The 18 findings are below, with the fix chosen for each. I checked each one against the code before fixing it. Fixes go in roughly the user's order: first data loss and aborted runs, then wrong behaviour, then nits.

## Decisions

The user asked for this round to run unattended overnight. These decisions are mine and are open to the user's review. The **Questions** section lists the ones worth a second look.

### Group 1: data loss and aborted runs

- **C2 + D6: one failed pool task aborts the whole run (9.64 regression).** Since 9.64 a worker raises its recorded failures (`_call_raising_recorded_failures`), and the five pool loops call `future.result()` unguarded. One unreadable file therefore stops the copy, the dir sync, the parallel listing and `-diff`'s payload copy.
  - Fix: pool tasks go through a new `_run_task`. It returns `(result, failures)` instead of raising, so the parent keeps both the partial result and the errors.
  - A parent helper `_task_result` adds the failures to the parent's `ERRORS` and returns the result. The worker printed them already.
  - An unexpected exception in a task (a crash) is recorded under the site's prefix, and the run goes on.
  - `_replace_files` keeps the raising contract through `_raise_task_failures`. A failed temp copy there must not be renamed over the destination. `_call_with_worker_init` is gone; `_call_raising_recorded_failures` remains for in-process callers.
  - Found while fixing: a pool job crashed by one directory dropped its siblings in the same job. `sync_directory_metadata` now records a failed directory (or destination) as `Directory sync error` (new prefix, rc 193) and goes on. That covers a source directory that vanished after the listing too. As with `cp -a`, a non-root `chown` EPERM is not an error, here and in `copy_file`'s identical-file path (a re-run over unchanged files).
- **New (found while fixing C2): `-rm` and `-rme` run after a failed copy.** With a destination, `-rm` copies and then deletes the sources, a move. In serial mode this already happened when a copy failed, so a file that failed to copy was deleted from the source. Once parallel failures stop aborting, the same would happen in every mode. `-rme` has the same problem: a source subtree that could not be listed makes its destination counterpart look extra.
  - Fix: `hpcp()` takes note of the errors recorded from the start of the copy through symlink creation. If any has a non-zero code, it skips `-rme` and `-rm` and records `Remove skipped: ...`, a new prefix with rc 212.
  - FS-param warnings (rc 0) do not count, and neither do errors recorded before the copy (e.g. 'Permission warning').
- **C3: an unreadable directory aborts listing.** `get_file_list_serial` now catches `OSError` around `os.scandir` and around each entry (the hash read, readlink, `DirEntry` type checks). It records `Listing error: <path>: <error>`, a new prefix with rc 209, and goes on with the other entries. Because of the gate above, a copy with listing errors never goes on to remove anything.
- **C1: `-rme -di` expects files in the wrong place.**
  - In `-di` mode `process_copy` writes each source to `dest + abspath(src)`, for every destination. `_src_to_dest_map` and `remove_extra_dirs` now take `dest_image` and mirror that mapping.
  - A destination directory under `-di` counts as "in the source" when its absolute path (`'/' + relpath(dir, dest)`) exists. This keeps the parent directories `process_copy` creates.
  - Listing scope does not change: `-rme` still mirrors the whole destination (here the image) to the set of sources, as it does without `-di` and as the README warns.
- **C4: a `-d` glob is expanded one character at a time.** `get_dests` calls `expand_paths([dest])`.
- **C10: `remove_extra_dirs` crashes on non-empty directories.**
  - Excludes are matched without the trailing separator, so `*/cache` matches the way the listing matches it.
  - The walk is now top-down and prunes excluded directories. Directories inside an excluded tree are never touched.
  - Symlinks to directories are skipped; `rmdir` would raise ENOTDIR on them.
  - `rmdir` treats ENOTEMPTY/EEXIST as "kept" (a note, no error). Other failures record `Remove directory failed` (rc 211).
  - The source exclusion filter no longer removes items from the caller's list while iterating over it.
- **D3: a non-UTF-8 path printed during an apply aborts it half done.**
  - The four sites print through `_printable_path`. That covers the conflict detail lines (whose states can hold link targets), the restore notes and the forced-removal mount note.
  - As a safety net, `main()` sets `sys.stdout`'s error handler to `backslashreplace` (when the stream supports `reconfigure`), so a CLI print can never crash a run. Library callers of `hpcp()` keep their own streams.

### Group 2: wrong behaviour

- **C5: spawn/forkserver workers run with module defaults.** `_run_task` takes a snapshot of the run settings and applies it in the worker before the task runs. The snapshot covers `CONTENT_ONLY`, `NO_CREATE_DIR`, `RANDOM_DESTINATION_SELECTION`, `HASH_SIZE`, `COMMAND_TIMEOUT`, `BYTES_RATE_LIMIT`, `FILES_RATE_LIMIT`, `MIRROR_FS_PARAMS` and `REMOVE_FILES_WHILE_LISTING`.
  - Every pool site uses it.
  - `_states_chunk` already passes `HASH_SIZE` and is left as is.
- **C6: `-dd` nested `hpcp()` sees the outer `ERRORS`.**
  - The `-dd` branch saves and clears `ERRORS` around each nested partition copy and puts the outer errors back afterwards.
  - The nested rc then reflects only that partition's copy. A failed partition is still reported as `Partition copy error` with the nested rc.
- **C7: the Windows GUI passes the destination as a str.** It passes `[dest]`, or `[]` when the field is empty.
- **C8: symlink link tracking is dead, and links always go to the last destination.** This restores the original (pre-multi-destination) behaviour, carried over to several destinations:
  - Stale links at each destination are removed first.
  - If any destination already holds a real file or directory at the link's path, the link is skipped (as before May 2025). This is recorded as `Failed to create symbolic link` (rc 162), the code the failed `os.symlink` gave in 9.72, so the remove gate stops a move.
  - A relative link goes to the destination where its target already exists.
  - Without `-nlt`, a missing relative target is copied (`copy_files_serial`) and the link placed where the copy landed.
  - Otherwise the first destination is used (a random one with `-rds`).
- **C9: `-s` drops the symlinks of a directory that holds only symlinks.** `copy_files_serial_batch` builds the link map before its early return.
- **D1: the read-write single-image mount has no `nouuid` for xfs.** `_mount_single_fs_image` mounts read-write with `-o nouuid` when `blkid` says xfs, as `_mount_partition` does.
- **D2: `-fl` lists made with `-hs 0` or another hash algorithm.**
  - `_parse_file_list_lines` recognises a file line by any digest hpcp writes: 16 hex (xxh64), 128 hex (blake2b), or empty (`-hs 0`).
  - It keeps the digest as the value only when it is the local algorithm's. Otherwise the value is `None` (unknown), so a list from another host or from `-hs 0` gives no spurious A+D or M rows.

### Group 3: nits

- **D4: a section whose payload directory is a symlink on every volume.** `_preflight_section` refuses a section with rows and no usable payload root (`Incomplete patch`), before anything changes.
  - It refuses even when every row is already done, because a later restore (`_restore_across_sections`) would need the payload too. An automated security review raised this point on the first version of the fix.
- **D5: volume count 0.** `_load_patch` refuses a count below 1 and a volume number outside 1..count (`Patch format error`).
- **D7: unregistered prefix.** `Patch error` is registered with rc 252.
- **D8: dead code.** `_resolve_conflicts` is removed.

## Post-round code review

A local `/code-review high` of the whole branch found 10 items. I checked each against the code and fixed the valid ones, each with a test that failed first:

1. **Valid, data loss.** `-rm` with `-d` destinations that all fall away (a glob that matches nothing) took the remove-only branch and deleted the sources without copying them. It also showed an older bug: `-rm -di NEW.img` deleted the sources and never created the image. Now a new `-di` image is created first, and requested destinations that give none raise `No valid destination` (202).
2. **Valid.** Tracking a link into an excluded directory copied the excluded target. The target is now checked together with its parent directories (`_excluded_or_below`).
3. **Valid.** `_task_result` let a `BrokenProcessPool` (a dead worker) escape. It is now recorded under the site's prefix.
4. **Valid.** A name ending in `:` made an unhashed list "hashed". An empty digest now counts only when every file and link line has a `:`.
5. **Valid (fragile, not live).** The remove gate could miss the errors of a listing cached before the copy. `hpcp()` now drops the listing cache when the copy starts.
6. **Valid.** A source directory gone after its listing reached `copy_file` ("Copy process crashed"); the first test had patched `isdir`. It is now a `Directory sync error`.
7. **Valid.** `remove_extra_dirs` walked into mount points below the destination. They are now pruned.
8. **Valid.** Ignoring a non-root `chown` EPERM dropped the group. `_chown_like_cp` retries with the group only, as `cp -a` does.
9. **Partly valid.** The `-dd` destination is not affected: its xfs UUID is set after the copy, through `delayed_commands`. The `-dd` source mount and the duplicated logic were real, so all read-write mounts now use `_rw_mount_command`.
10. **Cleanup, accepted:** `_is_failure` and `_link_dests`. I kept both C4 fixes on purpose: `get_dests` passes `[dest]`, and `expand_paths` takes a str for any other caller.

## Scoped re-review

A second `/code-review high` checked the README claims of the release commit against the code. Its findings, all fixed with tests (#4 was already fixed by the time of this review):

1. `-rme` deleted the destination copy of a source path excluded by an **absolute** `-e`: patterns name source paths, but they were matched against destination paths. That also made the 'Remove skipped' advice ("exclude those paths with -e") delete the backup copy of the very subtree the gate protected. Each destination path is now also mapped back to its source path (`_excluded_at_source`).
2. Link targets that tracking placed in the destination looked extra to `-rme`. `create_sym_links` now returns them and `remove_extra_files` keeps them.
3. `-rme -di` kept another source's directories (they exist on the host) while removing its files. Directories now follow this run's sources, and the README says that `-rme` mirrors the whole image.
4. Already fixed (b539e91): a non-root copy keeps the group like `cp -a`. The README says so now.
5. A non-root skip re-applied setuid/setgid bits for an owner or group it could not keep. They are now dropped, as `cp -a` does.
6. and 8. The dangling-link note, the intermediate symlink retry lines, the dir-sync 'maybe a mounted dir' note and 'Destination path excluded' were recorded as rc-128 errors whenever a path held a ':'. With the gate, that refused a clean move. They are plain stderr notes now.
7. A tracked tree was listed by its un-normalized path ('src/../other'), so absolute excludes inside it never matched. It is normalized now.

## Third review (of the re-review wave)

Fixed, with tests:
- **The -rme keep list protected every relative link's target.** An in-tree link (`X11 -> .`, `current -> releases/v3`) shielded its subtree, and a dangling link kept a stale file. Now only targets that exist at the source are tracked, and only those outside the copied trees are kept (`_kept_paths`, checked through a set by walking up a path's parents).
- **`_excluded_at_source` tested the source root itself,** which the copy listing never does for a directory (it lists `path/`). `-e cache` with `/home/u/cache/` as the source shielded the whole destination.
- **`remove_extra_dirs` keeps tracked target directories too.**
- **Lexical '..' resolution.** A tracked target is used by its normalized spelling, so excludes match, but only when the kernel resolves the link to the same file. A `..` after a symlinked directory is not tracked (a note).
- **Setid bits.** `_chown_like_cp` reads back the owner and group the destination really has. Directory sync drops the set-user-ID / set-group-ID bits it could not keep the owner / group for.
- **Performance.** `remove_extra_files` builds the source mapping once, and the directory walk checks only a directory's own source path.

Declined: when several sources map to one destination path and one of them excludes it, `-rme` keeps it. Keeping is the safe answer when the sources disagree.

## Fourth review

Fixed, with tests:
- An exclude that drops a whole source root as the listing names it (`-e 'cache/'` matches `…/cache/`) left nothing copied, and `-rme` then emptied the destination. This was also true on 9.72. The root is now tested in that `path/` form.
- A tracked target that holds a copied tree (`up -> ..`) kept everything. Such targets are left out of the keep list.
- Tracked targets are (destination, source) pairs. One inside another source's copied tree is kept; one that is just its own tree's copy is judged like any other path.
- The destination copy of an excluded link target is kept.
- The link notes print only when a copy would have been tried, and `realpath` runs only for a target with a `..`.
- One `_path_and_parents` walk replaces three loops. `_chown_like_cp` has a non-root unit test.

Declined: a target tracked through a symlinked directory (`sub/../t`) that an earlier run copied by its lexical path. 9.72 never tracked links (the code was dead), and the lexical path is not the link's real target.

Most findings after the first review came from `-rme` meeting the revived link tracking. That bears on Question 1.

## Fifth review, and why the review loop stops here

Fixed, with tests:
- A tracked directory that holds another source's copy lost its other contents. Only a target that holds its own link is dropped now, and the nearest container decides.
- Tracking checked a target's parents above the source tree. Now only below the common ancestor of the link and its target.
- A symlink source root is judged without a trailing `/`.
- Excluded roots are computed once per pass.
- The symlinked-directory note always prints.

Declined:
- Several sources merged into one destination with one of them excluded whole keeps every path there, since any of them may be its copy.
- An excluded link target's destination copy is kept whether or not the source still has it.

Each review after the first found fewer and narrower issues, almost all where `-rme` meets excludes, merged sources and the revived link tracking. More rounds would mostly refine behaviour that depends on Question 1. If tracking goes opt-in, the keep list (`create_sym_links`' triples, `_kept_paths`, `_is_kept`) only matters with the option and can be reviewed then.

## Security review and performance of the move removal

- The automated security review flagged path-based deletion in the new move removal: a directory swapped for a symlink during the copy (a tree other users can write; hpcp often runs as root) led the unlink outside the source (reproduced). Each listed entry is now removed relative to a descriptor of its directory, reached from the source root with `O_NOFOLLOW` (`_descend`), and each source root is pinned: its `(st_dev, st_ino)` is taken before the copy, and it is reopened from its parent with `O_NOFOLLOW` and checked. Entries are tied to the outermost source root containing them.
- The user asked that hpcp not get slower. The first descriptor version reopened the parent chain for every file (about depth + 1 opens per file). `_remove_listed_files` now keeps the current directory's descriptors open over the sorted entries and does not stat before unlinking; the empty-directory pass is one descriptor walk. Removal phase, 9.72 against now: about 55k against 29k metadata syscalls for 20k files (23.6k against 905 stats); FUSE 0.87-0.91 s against 0.72-0.76 s single-threaded and 0.93 s against 0.25 s with 32 workers; tmpfs 0.34 s against 0.26 s and 0.35 s against 0.12 s for 100k files. A test pins the open count.
- Left as before (path-based, pre-existing): the plain `-rm` listing that removes while it lists, and `-rme`'s removals in the destination.

## Local reviews after the ultrareview

- `/code-review high eaee294` reviewed that one commit (link tracking opt-in): an excluded link target's copy was removed by `-rme` without `-lt` (reproduced; fixed: the excluded-target keep no longer depends on tracking), `-lt` with `-dd` was ignored silently (now said), `target_is_directory` was resolved against the CWD (fixed), tests now run with and without tracking, README contradictions fixed. Declined: one argument for `link_tracking` / `no_link_tracking` (both are library API).
- `/code-review high eaee294..HEAD` (the fix wave): a move onto itself (`hpcp -rf /data/w/x -d /data/w/`) deleted the source (reproduced, also in 9.72; now refused, 241); root matching was quadratic with many roots (`-sfl`); a nested root was walked twice; `ln/` pinned the target's identity; `-diff`'s base side refused for errors inside excluded subtrees; a `-hs 0` list of links only read as unhashed (the live tree now decides); a plain `-rm -rwloff` exited 209 for paths rmtree removes anyway; `-e` without `-lt` cost two realpath calls per `..` link. All fixed with tests. Declined: merging the empty-directory walk with `_remove_tree_around_mounts` (that one removes files and skips mounts).

## Not changed (noted for the user)

- `-rme` in `-di` mode still mirrors the whole image to the sources, as `-rme` does for a plain destination.
- `remove_extra_files` (unlike `remove_extra_dirs` now) still lists into filesystems mounted below the destination, so `-rme` can delete extra files there. This predates 9.73 and was left for the user.

## Questions for the user

1. C8 restores default link tracking. Without `-nlt`, a relative symlink whose target is missing at the destination gets the target copied next to it, even when that is outside the destination tree (e.g. `../shared/lib`). A link to a large external tree (`data -> ../../mnt/big`) copies that tree as well. This was the original design and is what `-nlt`'s help text describes, but it has been dead since the multi-destination change of 2025-05-08 (499ee42). Keep the restored default, or make tracking opt-in?
2. The new remove gate skips `-rm` and `-rme` after any copy-phase error, including `Listing error` on an unreadable directory the user does not care about. The fix is `-e`. Is that the right default, or should listing errors outside the copied paths be ignored?
3. Should cas-toolbox pick up 9.73 in Monday's roll-up, or as a mid-week hotfix? The `-rm` after a failed copy, `-rme -di` and `-d` glob bugs exist in released 9.72.

## Decisions (user check-in, 2026-10-16)

1. **Link tracking is opt-in.** New `-lt --link_tracking`; `-nlt` is kept and overrides `-lt` (so an alias with `-lt` can still turn it off). `hpcp()` takes `link_tracking=False` next to `no_link_tracking`; `create_sym_links` takes `link_tracking=False`. The default copies symlinks as they are, as `cp -a` does and as hpcp has behaved in effect since May 2025. The C8 fixes stay (links go where their target is; tracking works with `-lt`). Without `-lt` the keep list is empty, so `-rme` removes a target copy an earlier `-lt` run placed, like any other extra. The tracking tests now pass `link_tracking=True`.
2. **The remove gate stays strict**: any copy-phase failure skips `-rm` and `-rme` (`Remove skipped`, 212), as rsync's `--delete` skips deletion after I/O errors. `-e` is the way around an unreadable path.
3. **cas-toolbox: mid-week hotfix 2026.42.2** pinning `hpcp==9.73` (2026.42.1, published, ships 9.72 and its source-deleting `-rm` bugs).
4. **Release order:** the user runs `/code-review ultra` on this branch first; its findings are fixed; then main is fast-forwarded and pushed, hpcp 9.73 published, cas-toolbox 2026.42.2 published, and the `ultrareview-*` branches and the worktree deleted.

5. **After the ultrareview:** finding 7's mid-move case is fixed in 9.73, not deferred (list-based move removal), and the commits after eaee294 get a scoped local review before the release-gate approval is recorded.

## Ultrareview of the fix branch (eaee294 against main)

The user ran `/code-review ultra main` on the branch after the decisions above. Seven findings, all handled:

1. **Normal, a regression of C3.** The listing goes on past a directory it cannot read, so `-diff` (and `-cfl`/`-dfl`) built a map without that directory's entries, and every base file under it became a delete row: applying the patch would delete them. `build_entry_map` now raises `Listing error` when its listing recorded a failure, before any patch exists (9.72 ended there too, with a crash).
2. **Normal, a regression of D2.** An unhashed list whose every name has a ':' (a source root named `snap:…`) was read as a `-hs 0` list, its lines as `path:target` links. A list now counts as `-hs 0` only when a line is a bare `path:` (what `-hs 0` writes for a file) and every line has a ':'. A `-hs 0` list of links only reads as unhashed.
3. **Nit, a regression of D1.** The `-dd` source was mounted read-write with `nouuid`, so a duplicate-UUID xfs clone (refused, and the run stopped, in 9.72) was mounted and written. A source partition that is not mounted is now mounted read-only (`ro`, plus `nouuid` for xfs). Recovery is kept (not `norecovery`/`noload`, as the reviewer proposed), so a dirty source is copied in its recovered state rather than inconsistent. The first version of this fix mounted every source read-only, which broke the documented live-disk clone: the kernel refuses a read-only mount of a partition mounted read-write ("would change RO state"), checked on this kernel for ext4 and xfs. A partition that is mounted is now mounted the way it is live (plainly when read-write, as in 9.72). A real test clones a loop disk whose partition is mounted read-write.
4. **Nit.** `_link_dests` now builds the link destinations at all four copy paths.
5. **Pre-existing.** `-rme` removed an empty `lost+found` at a filesystem root (a `-di` image, an ext4 destination), and would delete the orphans fsck puts there. A filesystem's own `lost+found`, directly at a mount point, and everything in it are left alone.
6. **Pre-existing, left by D1.** An existing `-di` image was mounted without `nouuid`; a cloned xfs image then went to the repair prompt, whose default runs `xfs_repair -L`. Both mounts in `get_dest_from_image` use `_rw_mount_command`.
7. **Pre-existing.** `-rm` removed whole source trees, so what the copy skipped without an error was deleted with no copy: FIFOs, sockets and device nodes, paths over 4096 characters, files written to a source after the copy listed it, and (found while fixing this) every file of a `-do` (directories only) copy. The user chose to fix it in 9.73. A move (`-rm` with a destination) now removes only what the copy listed: `remove_moved_sources` unlinks the listed files and links (`_remove_listed_files` never removes a directory, so a listed path that became one keeps its contents), then `rmdir`s each source directory that is empty, deepest first. What stays, excluded paths aside, is named in a `Move left entries` note (rc 0). A special file is noted when listed (`Special file skipped`, rc 0) and simply stays; an overlong path is a `Listing error` and blocks the move like any copy error. A file changed after it was copied is still removed, as with `mv`. Without a destination, `-rm` keeps the bulk delete (remove while listing, then `rmtree`), which is what makes it fast on Lustre.

Found in the self-review of this wave: without d_type (xfs `ftype=0`, some NFS) `is_file()` stats an entry, and one deleted meanwhile is neither file nor directory nor link; the special-file check would have turned it into a `Listing error`. A deleted entry is skipped, as in 9.72.

## Testing

- Every fix starts with a test that fails on 9.72.
- C5 uses a spawn-context pool.
- D1 uses real xfs images: root and loop devices are available here.
- C6 reuses the `test_dd_errors.py` harness.
- C7 drives `hpcp_gui` with a fake `tkinter`.
- The full suite passes before the version bump and again after it.
