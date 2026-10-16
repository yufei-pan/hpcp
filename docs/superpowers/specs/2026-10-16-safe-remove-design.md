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

Measured on the removal phase.

**Syscalls**, 20,000 files in 900 directories, single process:

| | 9.73 | 9.74 |
|---|---|---|
| plain `-rm` | 52.6k, of which 23.6k stats | 24.5k, of which 6 stats |
| `-rme` | 44.5k | 25.4k |

**FUSE (mergerfs)**, two runs, 9.73 against 9.74:
- plain `-rm`, serial: 0.86–0.89 s against 0.81–0.82 s;
- `-pfl`, 32 workers: 0.28–0.29 s against 0.25–0.26 s;
- `-rwloff`, 32 workers: 0.54 s against 0.53–0.55 s;
- `-rme`, serial: 0.63 s against 0.61–0.62 s;
- `-rme`, 32 workers: 0.14 s in both.

**tmpfs**, 50,000 files: plain `-rm` serial 0.16 s against 0.10 s; the other modes are equal or faster.

A test pins one open per directory and no `stat` per file.

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
