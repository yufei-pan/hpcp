# Bugs found while building the hpcp test suite

Do **not** fix these in `hpcp.py` as part of the test-suite work. Report here; product fixes are a separate change.

## BUG-1: `-rf` does not imply `--remove`

- **Symbol:** `get_args` / `hpcp()` / `main`
- **Repro:** `hpcp -rf /path` (no `-rm`)
- **Observed:** argparse stores `remove_force=True` and `remove=False`. `hpcp()` only calls `process_remove` when `remove=True`.
- **Expected (per README):** "`-rf --remove_force` implies `--remove`."
- **Suite handling:** `tests/test_cli_args.py::test_remove_force_flag` documents argparse as-is. `tests/test_remove.py::test_remove_force_implies_remove` is `xfail` (strict=False).

## BUG-2: `--tar_diff_file_list` writes dest extras, not src updates

- **Symbol:** `compare_file_list`
- **Repro:** compare a src list that has `new.txt` with a dest list that has `extra.txt`; pass `tar_diff_file_list=True`.
- **Observed:** the diff file lists dest-only paths (`extra.txt`) and omits src-only paths (`new.txt`).
- **Expected (per `-h`):** "Generate a tar compatible diff file list. ( update / new files only )" — files in src not in dest.
- **Suite handling:** `tests/test_file_lists.py::test_tar_diff_file_list_is_new_or_updated_only` calls `pytest.xfail` when current behavior is seen.

## BUG-3: missing `--src_image` is not skipped

- **Symbol:** `mount_src_image`
- **Repro:** `mount_src_image(['/no/such.img'], src_paths, mount_points, loop_devices)`
- **Observed:** after `eprint` + `src_images.remove(src)`, the loop continues and runs `losetup` on the missing file (rc 1).
- **Expected:** skip the missing image and continue.
- **Suite handling:** `tests/test_imaging.py::test_mount_src_image_skips_missing_file` xfails on the `losetup` error.
