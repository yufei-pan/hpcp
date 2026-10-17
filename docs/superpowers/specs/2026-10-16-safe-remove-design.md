# hpcp 9.74: plain -rm and -rme remove by directory descriptors

## Goal

9.73 made a move (`-rm` with a destination) remove by directory descriptors without following symlinks. The user asked for the same in the other two removal paths: plain `-rm` (no destination) and `-rme`. hpcp must not get slower.

## The problem

All of these removed by full path:

- **Plain `-rm`, serial and `-pfl`:** the listing removed files as it went, with `os.remove(entry.path)`, then `shutil.rmtree` ran on the root. With `-e`, a recursion by path ran instead.
- **Plain `-rm -rwloff`:** the listing came first, then pool jobs ran `os.remove` or `shutil.rmtree` on each listed path.
- **`-rme` files:** after the y/n prompt (a window of any length), `os.remove` or `shutil.rmtree` per extra path, serial or in pool jobs.
- **`-rme` directories:** `os.walk`, then `os.rmdir` per extra directory.

The attack: hpcp often runs as root on trees other users can write to. Another user swaps a directory in the tree for a symlink, say to `/etc`, between the listing and the removal. The removal then follows the symlink and deletes files outside the tree. Reproduced for `-rme` files, `-rme` directories and `-rwloff`. The serial listing's window was narrow (between `scandir` and `os.remove`) but real.

A root swapped for a symlink was not followed: the listing removed the symlink and said nothing.

## Design

The 9.73 move helpers carry over:
- `_path_identity` / `_source_identities` pin each root;
- `_open_source_root` reopens a root from its parent with `O_NOFOLLOW` and checks it against the pin;
- `_descend` / `_parent_below` walk one component at a time with `O_NOFOLLOW`;
- `_remove_listed_files` is the descriptor cursor over sorted entries.

**Plain `-rm` (`process_remove`).** Each source is pinned before the confirmation prompt.
- **Serial (the default, as in 9.73 without `-pfl`):** `_remove_tree_fd(remove_files=True)` makes one descriptor walk per source, deepest first, as `shutil.rmtree` does.
  - Each directory is opened relative to its parent with `O_NOFOLLOW`.
  - Entries are read with `d_type` (`scandir` on a descriptor), so nothing is `stat`ed per entry.
  - Files are unlinked, and each directory is removed once its entries are gone.
  - Excluded paths are not entered or removed.
- **`-pfl`:** pool jobs each empty one directory (`_remove_directory_entries`), and the serial walk then removes the directories.
  - A job opens its directory by path with `O_NOFOLLOW` and checks it against the identity its parent's listing saw (`fstatat`).
  - So a directory above it that was swapped for a symlink gives another identity, and the job refuses.
  - A directory that was moved elsewhere keeps its identity. Removing in it then is still removing the right files.
- **`-rwloff`:** the trees are listed first (read-only), then the listed files go through the cursor in pool jobs, and the walk does the rest.
- A file or link source is removed with `_remove_one_listed`, which checks its identity.
- A replaced root removes nothing and is reported (`Remove process exception`).

**`-rme`.**
- `remove_extra_files` pins each destination before listing it and records, for each extra, the destination it was found in. After the prompt, the extras go through the cursor.
- `remove_extra_dirs` does the same for directories, through the cursor's new `rmdir` mode, deepest first.

**The listing only reads now.** Removal while listing, `_remove_file_or_link` and `delete_files_parallel` are gone. The listing functions return 4-tuples. `-rwloff` keeps its flag and now only picks plain `-rm`'s mode.

**`hpcp -rm /` is refused,** as `rm --preserve-root` does: the root is compared by device and inode, so a bind mount of `/` is refused too. 9.73 hung in a loop on `/`, and the rewrite would otherwise have removed it. The test stubs every removal function to fail first.

## Behaviour changes

- The plain `-rm` summary prints `Removed N files and links ... in T seconds`. Sizes are gone; they needed a `stat` per file.
- An `-rme` extra that turned into a directory before removal is left. 9.73 ran `rmtree` on it.
- A source or destination root replaced during the run is reported; 9.73 removed a symlink left at the root without a word.
- `get_file_list*` lose `remove_files_while_listing` and `return_removed_count`. Nothing outside hpcp in this repo uses them.

## Performance

Removal phase, 20,000 files in 900 directories.

**Syscalls**, single process:

| | 9.73 | 9.74 |
|---|---|---|
| plain `-rm` | 52.6k, of which 23.6k stats | 24.5k, of which 6 stats |
| `-rme` | 44.5k | 25.4k |

**FUSE (mergerfs)**, median of 5 to 7 alternating runs at normal load, 9.73 against 9.74:

| Mode | 9.73 | 9.74 |
|---|---|---|
| plain `-rm`, serial | 0.914 s | 0.822 s |
| plain `-rm -pfl`, 32 workers | 0.291 s | 0.263 s |
| plain `-rm -rwloff`, 32 workers | 0.549 s | 0.538 s |
| `-rme`, serial | 0.621 s | 0.623 s |
| `-rme`, 32 workers | 0.143 s | 0.142 s |

`-rme` with 32 workers was 3% slower until `_items_below` stopped re-resolving the listed root for every entry.

**tmpfs**, 50,000 files: plain `-rm` serial 0.16 s against 0.10 s.

A test pins one open per directory and no `stat` per file.

## Review of main..HEAD (local /code-review)

Fixed, with tests:
- **`-rme` into a symlinked destination (`link/`) removed nothing:** it was pinned with `lstat` and opened with `O_NOFOLLOW`. A destination as given is trusted, so its real path is pinned now (`_destination_roots`).
  - The security review that followed found a gap: an extra equal to the destination itself (a destination replaced by a symlink to `/etc/shadow`) then resolved to the target. The destination itself is removed as the name given, pinned by `lstat`.
- **Legacy `-fl -rme` with a relative destination dropped its extras silently.** `_items_below` now takes paths from the CWD, treats the root as `'.'`, and reports a path that is not below its root.
- **Plain `-rm` ignored `-F`.** The walk now paces each unlink, and `-pfl` queues a job only while under the limit, as 9.73's paced listing did.
- **`-pfl` recorded a false error for a directory too deep to open by path.** Such a directory is left to the descriptor walk.
- **Type changes mid-walk:**
  - an entry that became a directory is removed as one;
  - a failing type check no longer skips its directory;
  - without descriptor support, a symlink to a directory is unlinked.
- **Without descriptor support, the root itself is no longer `root/.`,** which `rmdir` refuses with EINVAL. This also affected the move's fallback in 9.73.
- **The listing cache is cleared after plain `-rm`, a move and `-rme`.**
- **Shared helpers `_items_below` and `_remove_items` replace four copies.** `batch` is documented as implied.

Declined:
- **A single parallel pass for `-pfl`:** 9.73 also made two passes (a parallel listing, then `rmtree`), and 9.74 measured faster than it.

## Re-review of 418ca7e..HEAD (user: "re-review then release")

Fixed, with tests:
- **Destinations are pinned before the copy, as sources are.** A destination swapped for a symlink between the copy and `-rme` used to be followed; one that now resolves elsewhere is not cleaned (`Remove skipped`, `_checked_destination_roots`).
- **Legacy `-fl -rme` with a bare relative destination** (`data`, `.`) built its extras under `/`. 9.73 removed `/data/extra`; the previous fix refused everything. The fallback is now the CWD.
- **`-F` is honoured everywhere:**
  - `-pfl` jobs pace each unlink from one shared counter, so a wide directory is paced too (checked under spawn, forkserver and fork);
  - the serial cursor (moves, `-rme`), file sources, and the walk after `-rwloff` (paced from its own start) are paced;
  - under `-F`, pool jobs keep `-fpj`'s size.
- **An entry whose type check fails is tried as a directory first.**
- **Cleanup:** the move builds its items with `_items_below`; roots map values are always 3-tuples; paths sort natively; a single macOS EPERM check; root messages print the root, not `root/.`.

## Testing

`tests/test_safe_remove.py`:
- a root replaced while the prompt waits, in each mode;
- a subdirectory swapped after its parent was read (serial);
- a `-pfl` job refusing a path that now leads elsewhere;
- swaps after the listing (`-rwloff`), during the `-rme` prompt, and after `remove_extra_dirs`' walk;
- open and `stat` counts;
- removal counts, with no failure recorded, so a crashed pool job can't hide;
- a symlink source;
- a warm listing cache;
- `/` refused, and a link to `/` removed as a link.

The existing `test_remove_preserves_excluded_entries` covers excludes in every mode combination.
