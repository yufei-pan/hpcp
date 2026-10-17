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

## Testing

`tests/test_deep_trees.py` uses a 1,500-level tree. It covers:
- listing, serial and parallel;
- copying, serial, parallel and with `-nds` (`_makedirs`);
- `-rme`;
- plain `-rm` and a move, serial and with `-pfl`;
- `build_entry_map` for `-diff`;
- `_makedirs`' contract.

An autouse fixture removes the deep trees with `_rmtree` before pytest's own `shutil.rmtree` cleanup would recurse on them under Python older than 3.13. The file passes on Python 3.10, 3.11, 3.12, 3.13 and 3.14.
