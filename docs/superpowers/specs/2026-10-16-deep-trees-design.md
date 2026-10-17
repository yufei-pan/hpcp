# hpcp 9.76: trees deeper than Python's recursion limit

## Goal

This was a follow-up noted in 9.75: `get_file_list_serial`, the listing behind the copy, `-rme`, `-diff` and list comparisons, called itself once per directory level. A tree about 1,000 levels deep raised `RecursionError`. Such a tree can be valid: 1,500 levels of a one-letter name stay under the 4,096-character path limit. The user asked for it to be fixed.

## What recursed

- **`get_file_list_serial`:** once per level. It now uses an explicit stack. Each directory below the root gets exactly the checks the recursive call made for it: the length limit, exclude, link, file, directory, and a "Source path type error" for one that vanished. Results go into one set each. The recursive version copied each subtree's sets into its parent's, once per level above it, so the change is also a little faster: listing 40,000 files on tmpfs takes 0.061 s against 0.067 s (median of 8 alternating runs), and FUSE runs are level.
- **Standard-library calls that recurse on the Pythons hpcp supports (3.6+):**
  - `os.makedirs` recurses once per missing level. It is replaced by `_makedirs`, with the same contract: the mode applies to the last directory only, as since 3.7, and an existing path is fine only with `exist_ok` when it is a directory. All 11 calls in hpcp's own code were replaced; the embedded multiCMD blob is not touched.
  - `os.walk` recursed before 3.12. It is replaced by `_walk`: top-down, symlinks to directories listed but not entered, unreadable directories skipped, and in-place pruning of `dirnames` honoured.
  - `shutil.rmtree` recursed before 3.13. It is replaced by `_rmtree`, which goes through the existing descriptor walk `_remove_tree_fd` and raises its first error. `_remove_entry`, used in patch building, and `delete_file_bulk` use it. `delete_file_bulk`'s exclude recursion goes through the same walk.
- **Not changed:** `create_sym_links` recurses once per chain of nested tracked links, not per tree level; `_stable_fs_params` and the `-dd` nested `hpcp()` call are unrelated.

## Review of the first version (local /code-review)

Fixed, with tests:
- **The descriptor walks held one directory descriptor per level** (`_remove_tree_fd`, and the removal cursor for moves and `-rme`). A tree deeper than the open-files limit (commonly 1024, and 256 on macOS) failed with EMFILE (reproduced). Both now hold at most 32 (`_WALK_FD_BUDGET`):
  - the walk lets the shallowest go, keeping each one's identity, and reopens it through its child's `..`, checked against that identity; a directory moved meanwhile stops the walk with an error;
  - the cursor reopens a released descriptor by name with `O_NOFOLLOW`, from the deepest one still held.

  Ordinary trees never reach the budget, so they pay nothing extra.
- **`delete_file_bulk`'s excludes were matched against absolute paths**, so relative patterns stopped protecting anything (reproduced). They are matched against the paths as the caller spelled them again. Every failure is recorded, and the function shares `_rmtree`.
- **`_rmtree` keeps `shutil.rmtree`'s contract:** a missing path raises; a tree that is not gone raises with the full path. Parents are opened with `O_PATH`, so search permission is enough, as with `rmtree`; a mode-0333 drop box works as `nobody`.
- **`_makedirs` follows `os.makedirs` step for step,** including `x/y/.`, the empty path, and a dangling symlink on the way.
- **The listing gives each subdirectory one `lstat`** instead of `islink`, `isfile` and `isdir`. The root keeps its checks.

Measured on FUSE, median of 5 alternating runs, 9.75 against 9.76:
- listing: 0.299 s against 0.291 s;
- plain `-rm`, serial: 0.810 s against 0.811 s;
- whole `-rme`, serial: 0.875 s against 0.882 s;
- whole `-rme`, 32 workers: 0.292 s against 0.290 s.

## Re-review of c4d89da..HEAD (user: "re-review then release")

Fixed, with tests:
- **The cursor reopened every released directory from the root by name** (reproduced): 540,257 opens for a 1,500-level `-rme` directory pass, and 150,213 for an 800-level move chain, against about one per level in 9.75. Nor did it check the device or the identity of a reopened directory. The walk and the cursor now share one helper, `_HeldDirs`:
  - it holds at most 32 descriptors below the root, as a suffix of the path, so each step is O(1);
  - a released directory is reopened through its child's `..`, one open per level, which stays in the child's filesystem; if the child is gone, it is reopened by name from the deepest held directory;
  - every reopen is checked against the identity kept when the directory was released (ESTALE when it was moved or mounted on).
- **`..` of a directory removed meanwhile is ENOENT.** The walk now reopens the parent by name and goes on, where it used to stop and leave every sibling; a parent that is gone too is skipped. `fstat` failures are recorded and no descriptor leaks.
- **`_rmtree` takes the path as the kernel resolves it.** `''` used to remove the current directory; `link/..` was collapsed lexically.
- **The fallback without descriptors spells paths as given,** so relative excludes match there too.
- **`_makedirs` returns early for a `.` tail only after making a missing head,** as `os.makedirs` does.
- **`-pfl` folders listed by their parent get one `lstat` (`known_dir`).**

## Testing

`tests/test_deep_trees.py` uses a 1,500-level tree. It covers:
- listing, serial and parallel;
- copying, serial, parallel and with `-nds` (`_makedirs`);
- `-rme`;
- plain `-rm` and a move, serial and with `-pfl`;
- `build_entry_map` for `-diff`;
- `_makedirs`' contract and its match with `os.makedirs`;
- removal under an open-files limit of 256;
- `delete_file_bulk` with relative excludes, and reporting every failure;
- `_rmtree`'s contract and the drop box, as `nobody`;
- one `lstat` per subdirectory.

An autouse fixture removes the deep trees with `_rmtree` before pytest's own `shutil.rmtree` cleanup would recurse on them under Python older than 3.13. The file passes on Python 3.10, 3.11, 3.12, 3.13 and 3.14.
