# Bugs found while building the hpcp test suite

These bugs were recorded during test-suite development and have now been fixed in a subsequent product change. Their regression tests run normally rather than being marked as expected failures.

## BUG-1: `-rf` does not imply `--remove`

- **Symbol:** `get_args` / `hpcp()` / `main`
- **Repro:** `hpcp -rf /path` (no `-rm`)
- **Observed:** argparse stores `remove_force=True` and `remove=False`. `hpcp()` only calls `process_remove` when `remove=True`.
- **Expected (per README):** "`-rf --remove_force` implies `--remove`."
- **Resolution:** `hpcp()` makes `remove_force` imply removal before resolving destinations. `tests/test_remove.py::test_remove_force_implies_remove` and `test_remove_force_alone_removes_without_copying` cover the behavior.

## BUG-2: `--tar_diff_file_list` writes dest extras, not src updates

- **Symbol:** `compare_file_list`
- **Repro:** compare a src list that has `new.txt` with a dest list that has `extra.txt`; pass `tar_diff_file_list=True`.
- **Observed:** the diff file lists dest-only paths (`extra.txt`) and omits src-only paths (`new.txt`).
- **Expected (per `-h`):** "Generate a tar compatible diff file list. ( update / new files only )" — files in src not in dest.
- **Resolution:** Tar diffs use source-minus-destination entries. `tests/test_file_lists.py::test_tar_diff_file_list_is_new_or_updated_only` checks new and updated files while excluding unchanged and destination-only files.

## BUG-3: missing `--src_image` is not skipped

- **Symbol:** `mount_src_image`
- **Repro:** `mount_src_image(['/no/such.img'], src_paths, mount_points, loop_devices)`
- **Observed:** after `eprint` + `src_images.remove(src)`, the loop continues and runs `losetup` on the missing file (rc 1).
- **Expected:** skip the missing image and continue.
- **Resolution:** Missing images are logged and skipped without modifying the input list. `tests/test_imaging.py::test_mount_src_image_skips_missing_file` and `test_mount_src_image_keeps_valid_image_after_missing_entries` cover skipping and continuation.
