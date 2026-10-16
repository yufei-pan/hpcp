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

## Not changed (noted for the user)

- `-rme` in `-di` mode still mirrors the whole image to the sources, as `-rme` does for a plain destination.

## Questions for the user

1. C8 restores default link tracking. Without `-nlt`, a relative symlink whose target is missing at the destination gets the target copied next to it, even when that is outside the destination tree (e.g. `../shared/lib`). A link to a large external tree (`data -> ../../mnt/big`) copies that tree as well. This was the original design and is what `-nlt`'s help text describes, but it has been dead since the multi-destination change of 2025-05-08 (499ee42). Keep the restored default, or make tracking opt-in?
2. The new remove gate skips `-rm` and `-rme` after any copy-phase error, including `Listing error` on an unreadable directory the user does not care about. The fix is `-e`. Is that the right default, or should listing errors outside the copied paths be ignored?
3. Should cas-toolbox pick up 9.73 in Monday's roll-up, or as a mid-week hotfix? The `-rm` after a failed copy, `-rme -di` and `-d` glob bugs exist in released 9.72.

## Testing

- Every fix starts with a test that fails on 9.72.
- C5 uses a spawn-context pool.
- D1 uses real xfs images: root and loop devices are available here.
- C6 reuses the `test_dd_errors.py` harness.
- C7 drives `hpcp_gui` with a fake `tkinter`.
- The full suite passes before the version bump and again after it.
