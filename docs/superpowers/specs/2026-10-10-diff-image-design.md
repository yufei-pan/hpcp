# Differential patch images (`-diff` / `-siff`)

Date: 2026-10-10  
Status: Approved design (pending implementation plan)

## Problem

`hpcp` declares two flags that do nothing:

- `-siff` / `--load_diff_image` — "Not implemented. Load diff images and apply the changes to the destination."
- `-diff` / `--get_diff_image` — "Not implemented. Compare the source and destination file list, create a diff image …"

Both are parsed in `get_args()` and never passed to `hpcp()`, so `hpcp -diff src -d dst` silently performs a
normal copy instead.

The intended use is differential updates of a whole filesystem image: ship one self-contained patch that turns
an old tree (a directory or a single-filesystem image) into a new one, instead of re-shipping the full image.

The existing building blocks are almost there but do not connect:

- `-dfl` / `--diff_file_list` already set-diffs two file lists, but its output cannot drive a patch:
  - The non-tar output writes **dest-only** paths plainly and **src-only** paths with a `-\0` prefix — the
    opposite direction from "what turns dest into src".
  - It recovers each path by splitting `path:hash` at the last `:` (`file.rpartition(':')[0]`). Directory
    entries carry no hash, so `a:b/` becomes `a`; symlink entries are `path:target`, so a target containing
    `:` is cut; with `-nhfl` every name containing `:` is cut.
  - A modified file appears once in each set (different hash), so it is reported as both added and deleted.
- Change detection is too weak for a patch. File-list entries are `path:<hash of last HASH_SIZE bytes>`, without
  size, mtime, mode, owner, or xattrs, so a same-size edit outside the hashed tail, a `chmod`, or an ACL change
  is invisible.
- `-di` creates and fills filesystem images (`create_image`, `-dis` splitting), and `-si` mounts them read-only
  (`mount_src_image`), but nothing writes or reads a deletion list.

## Goal

1. `hpcp -diff [PATCH.img] <new> <base>` writes a patch image that contains every added and modified entry as a
   whole file, plus a change list that records deletions and the expected before/after state of every touched
   path.
2. `hpcp -siff PATCH.img <target>` checks the target against the patch, asks before overriding conflicts
   (default: abort), applies deletions then the payload, and verifies the result. `-fa` forces the apply for
   unattended scripts while still logging every conflict as a warning.
3. `-dfl` writes the same change-list format the patch image carries, and stops truncating names that contain `:`.
4. `is_file_identical()` can report metadata-only differences (mode, owner, xattrs including ACLs); copy closes
   its one metadata gap (stale xattrs) without changing its in-place sync strategy.

Success criterion: for any `new` and `base` trees, applying `-diff new base` to a copy of `base` yields a tree in
which every path's state equals its state in `new` (within the compared subtree, in `store` layout).

## Non-goals

- **dd-mode / whole-disk patches.** Phase 2, separate spec. This design reserves room for it (manifest
  `section` lines) but implements a single filesystem per side.
- **Block- or delta-level diffs.** A modified file is carried whole, even for a one-byte change. Accepted: hpcp's
  target trees are dominated by small files, and a plain-file payload stays readable without hpcp.
- **Move/rename detection.** A move is a `D` plus an `A`. Accepted for simplicity.
- **Hardlink preservation.** Each hardlinked path is carried as its own file; applying breaks the link.
- **Changing the `-fl` stored file-list format.** `-fl` keeps writing `path:hash` lines.
- **Windows.** Patch generation and application need loop devices and mounts, which hpcp already refuses on
  Windows.

## Decisions

| Topic | Decision |
|---|---|
| Patch container | A normal hpcp image (partitionless xfs/ext4 from `create_image`) with the payload at its real paths and a reserved `.hpcp_diff/` directory at the image root. Alternatives rejected: overlayfs whiteouts (root-only to create, copy engine must special-case device nodes, deletions invisible to `ls`); a sidecar deletion file (two artifacts that must travel together). |
| Scope per side | One filesystem: a directory, or a partitionless / single-partition image. Multi-partition images use their largest partition, with a warning pointing at `-dd`. |
| Path layout | Follows `-di`'s image-store layout: a directory new side against a `-di` base is keyed by its absolute path and compared only with that subtree of the store. All other combinations compare root to root. See "Path layout". |
| Change granularity | Any difference in type, size, mtime, mode, owner, content hash, or xattrs (ACLs included) makes the entry "modified", carried whole. |
| Change-list format | One TSV format shared by `-dfl` and `.hpcp_diff/changes.tsv`. |
| Apply safety | Check only the paths the patch touches. Conflicts prompt `[y/N]` (default abort); non-tty aborts; `-fa` forces and prints every conflict as a warning. |
| Re-apply | Paths already in their `new` state are skipped, so an interrupted apply can be re-run. |
| Copy metadata | Copy keeps `compare_metadata=False` and its in-place metadata sync; it additionally removes destination-only xattrs. |
| `-dis` splitting | Supported. Every volume carries the same `.hpcp_diff/` with a shared `patch_id` and `volume k/n`. |

## CLI

```
hpcp -diff [PATCH.img]  <new: src_path | -si IMG>  <base: -d DIR | -di IMG>  [-fh] [-hs N] [-dis SIZE] [-e PAT] [-x FILE] [-j N]
hpcp -siff PATCH.img [-siff PATCH_1.img ...]  <target: -d DIR | -di IMG | single positional path>  [-fa] [-j N]
```

Argument changes in `get_args()`:

- `-diff` / `--get_diff_image`: `store_true` → `type=str, nargs='?', const='auto', default=None` (same shape as
  `-dfl`). With no value the output is `DIFF_{src_str}_TO_{base_str}_{int(time.time())}.img`, reusing the
  `src_str` / `dest_folder_name` naming already used for `-dfl`.
- `-siff` / `--load_diff_image`: unchanged shape (`action='append'`); values go through `expand_paths()`.
- New `-fa` / `--force_apply` (`store_true`). `-f` alone is not used, to avoid confusion with `-F`.
- `main()` and `hpcp()` gain `get_diff_image=None`, `load_diff_image=None`, `force_apply=False`.

Help text (replacing "Not implemented"):

- `-diff`: "Generate a patch image that updates the base (-d / -di) to the source (src_path / -si): added and
  modified files in full plus a change list under .hpcp_diff/. Optionally name the output; default
  DIFF_<src>_TO_<base>_<time>.img. Honors -dis, -fh, -hs and -e."
- `-siff`: "Apply a patch image made by -diff to the target (-d / -di or a single path). Repeat for each -dis
  volume. Every path the patch touches is checked first; conflicts prompt before being overridden."
- `-fa`: "With -siff, apply even when the target differs from the state the patch expects. Conflicts are still
  printed as warnings."

In `-diff` mode, `-di` names the **base** image, never the output.

Incompatible combinations fail before any mount with a `"Invalid arguments: …"` error:

- `-diff` with `-siff`.
- `-diff` or `-siff` with `-dd`, `-rm` / `-rf`, `-rme`, `-fl`, `-cfl`, `-dfl`, `-tdfl`, or `-co`.
  (`-siff` with `-dd` says dd support is planned.)
- `-diff` with more than one new tree (several positional paths, several `-si`, or a mix) or more than one base.
- `-fa` without `-siff`.

## Path layout

`-di` writes files into an image at their **absolute source path**: `hpcp /data/proj -di store.img` stores
`/data/proj/a.txt` as `store.img:/data/proj/a.txt` and syncs the metadata of the parent directories (`/data`,
`/data/proj`) into the image (`process_copy()`, `if dest_image:` branch). One store can hold several trees.
`-diff` follows that layout so a patch made against a store lands where the store keeps the files.

Each side of a diff is an **entry map** `{key: absolute_path}` built from a root directory and a key prefix:

| New side | Base side | New map | Base map | Patch layout |
|---|---|---|---|---|
| directory `P` | `-di` image store mounted at `M` | `P`'s tree keyed `rel(P,'/')/…` (e.g. `data/proj/a.txt`), plus each parent directory of `P` except `/` (`data/`) | the same keys resolved under `M` (`M/data/`, `M/data/proj/…`) — only that subtree of the store is listed | `store` |
| directory `P` | directory `B` | `P`'s contents keyed relative to `P` | `B`'s contents keyed relative to `B` | `root` |
| `-si` image mounted at `S` | `-di` image mounted at `M` | `S`'s contents keyed relative to `S` | `M`'s contents keyed relative to `M` | `root` |
| `-si` image mounted at `S` | directory `B` | `S`'s contents keyed relative to `S` | `B`'s contents keyed relative to `B` | `root` |

Rules:

- The new side must be a directory or an image; a single file is rejected.
- In `store` layout, entries elsewhere in the store (other trees, other parents) are never listed, so they are
  never reported as deleted. If `P`'s subtree does not exist in the store yet, every entry is `A`.
- In `root` layout the patch root `./` is the root directory of each side. Copy's rule of appending the source
  folder's name to a `-d` destination does not apply to `-diff`: the two roots are compared as given.
- The patch payload uses the same keys, so it mirrors the base's layout. The manifest records the layout
  (`layout	store|root`).
- Apply always treats the target (`-d DIR`, `-di IMG` root, or positional path) as the patch root. A `store`
  patch applies to the store image with `-siff patch.img -di store.img`, or to a live system with
  `-siff patch.img -d /`. When a `store` patch is applied to a `-d` directory other than `/`, apply prints a
  note that keys such as `data/proj/…` are resolved under that directory.

## Formats

All `.hpcp_diff/` files and `-dfl` output are UTF-8 text written and read with `errors='surrogateescape'`, so
arbitrary byte names round-trip.

### Field escaping

Applied to every path and symlink target:

| Character | Written as |
|---|---|
| `\` | `\\` |
| tab | `\t` |
| newline | `\n` |
| carriage return | `\r` |

All other bytes pass through unchanged. Unescaping is the exact inverse; an unknown escape is a format error.

### Paths

- Relative to the patch root (see "Path layout"), `/`-separated.
- Directories end with `/`; the patch root itself is `./`.
- A path that changes type appears twice: `foo` (`D`, file) and `foo/` (`A`, dir).

### Entry states

| State | Meaning |
|---|---|
| `-` | absent |
| `?` | unknown (only produced by `-dfl` against a stored `-fl` list) |
| `f:<size>:<mtime_ns>:<mode>:<uid>:<gid>:<hash>:<xattr>` | regular file |
| `d:<mode>:<uid>:<gid>:<xattr>` | directory (no mtime: it changes whenever the contents do) |
| `l:<target>` | symlink; `<target>` is escaped and is everything after the first `:` |

- `<mode>` is `st_mode` in octal (`100644`, `40755`).
- `<hash>` is `hash_file()` output under the recorded hash settings, or `-` when hashing is disabled
  (`HASH_SIZE == 0`).
- `<xattr>` is the hex digest (same algorithm as `<hash>`) of the canonical serialization of all readable xattrs
  read with `follow_symlinks=False`: for each name in sorted byte order, `name + b'\0' + len(value) as 8-byte
  big-endian + value`. A file with no xattrs gets the digest of empty input. `-` means xattrs were not read
  (platform without `os.listxattr`, or the filesystem returned `ENOTSUP`).

State equality (used for detection, apply checks, and the post-apply verify):

- Different type letter → different.
- Files: `size`, `mode`, `uid`, `gid`, `hash` equal; `mtime_ns` within the `is_file_identical()` tolerance
  (`2 × max(get_timestamp_precision(a), get_timestamp_precision(b))`); `xattr` equal unless either side is `-`.
- Directories: `mode`, `uid`, `gid` equal; `xattr` as for files.
- Symlinks: target equal.

### `changes.tsv`

```
#hpcp-changes	1
#op	path	old	new
D	foo	f:10:1717000000123456789:100644:0:0:9f2c…:-	-
D	usr/lib/libfoo.so	l:libfoo.so.1	-
D	var/cache/old/	d:40755:0:0:-	-
A	foo/	-	d:40755:0:0:-
A	foo/bar	-	f:34:1717000500000000000:100644:0:0:77ab…:-
M	usr/bin/tool	f:2000:…:100755:0:0:aa01…:-	f:2100:…:100755:0:0:bb02…:3c9e…
```

- Lines starting with `#` are headers/comments. The first line must be `#hpcp-changes<TAB><version>`; readers
  reject a version greater than 1.
- Rows: `op<TAB>path<TAB>old<TAB>new`. `op` is `A` (old `-`), `M` (old and new present, same type), or `D`
  (new `-`).
- Every entry is listed individually, including each file under an added or deleted directory.
- Rows are ordered `D`, then `A`, then `M`; within each op, `natural_sort()` by path. Output is deterministic.

### `manifest`

Key/value TSV, one pair per line; `section` is the only key that repeats and the only one with more than one
value:

```
format	hpcp-diff
format_version	1
patch_id	3f6c0d2e9b8a4c1e8f0a7b6c5d4e3f21
volume	1/2
created	2026-10-10T12:34:56Z
hpcp_version	9.65
new	<src_str>
base	<base_str>
hash_algo	xxh64
hash_size	65536
full_hash	0
xattrs	1
layout	store
section	root	.	changes.tsv
added	12
modified	3
deleted	5
payload_bytes	123456
```

- `patch_id` is `uuid.uuid4().hex`, identical on every volume.
- `volume` is `k/n`; volumes are numbered from 1 in the order of `_split_dest_image_name()` indices.
- `hash_algo` is `xxh64` or `blake2b`, matching `hash_file()`.
- `xattrs` is `1` if xattrs were read on both trees, `0` otherwise.
- `layout` is `store` or `root` (see "Path layout").
- `section<TAB>name<TAB>payload_dir<TAB>changes_file`: payload root relative to the volume root, and the change
  list relative to `.hpcp_diff/`. Phase 1 writes exactly `section	root	.	changes.tsv`. Phase 2 (dd) will add
  one section per partition.
- Readers ignore unknown keys and reject `format` ≠ `hpcp-diff` or `format_version` > 1.

## Change detection

### `is_file_identical()`

New keyword `compare_metadata=False`. When `True`, after the existing size / mtime / hash checks it also
requires equal `st_mode`, `st_uid`, `st_gid`, and xattr digest (skipped when either side cannot report xattrs).
Existing callers and behavior are unchanged with the default.

### Copy: stale xattr removal

`copy_file()`'s identical-file branch and `sync_directory_metadata()` already call `shutil.copystat()`, which on
Linux copies mode, times, and all source xattrs onto the destination — but never removes an xattr (or ACL) that
exists only on the destination. A new helper `_remove_stale_xattrs(src, dest)` removes destination xattrs whose
names the source does not have. It runs after `copystat()` in both places, is skipped under `CONTENT_ONLY`, and
silently ignores `ENOTSUP`, `EPERM`, `ENODATA`, and `EACCES`, matching `shutil._copyxattr()`.

### `get_entry_state(path, full_hash=False, read_xattrs=True)`

Returns the state string for one path using a single `os.lstat()`, `hash_file()` for regular files, and the
xattr digest. `ENOENT` and `ENOTDIR` (a parent component is not a directory) mean absent and return `-` without
an error. Any other read failure returns `-` and records an error through `eprint`.

Entries that are not regular files, directories, or symlinks (FIFOs, sockets, device nodes) are already skipped
by `get_file_list_serial()` and therefore never reach `get_entry_state()`; patches ignore them, as copy does.

### `build_entry_map(root, prefix='', parents=(), exclude=None, max_workers=…, parallel_file_listing=False)`

Lists `root` with `get_file_list(append_hash=False)` (honoring `exclude`) and returns `{key: absolute_path}`:
files, links, and folders keyed `prefix + relpath(entry, root)`, directories with a trailing `/`, and `root`
itself keyed `prefix` (or `./` when `prefix` is empty). `parents` adds the `store`-layout parent directories
(`{'data/': '/data'}` on the new side, `{'data/': M + '/data'}` on the base side). A `root` that does not exist
yields only the `parents` that exist — this is the "subtree not in the store yet" case.

### `compute_changes(new_map, base_map, full_hash=False, max_workers=…)`

1. Take the two entry maps built according to "Path layout".
2. Compute every state with `get_entry_state()` in chunks on a `ProcessPoolExecutor`, the pattern the copy
   engine uses. Every file on both sides is hashed: that is what catches a same-size, same-mtime edit, and the
   states are recorded for apply.
3. Pair by key:
   - only in new → `A`
   - only in base → `D` (together with the previous rule this covers file ↔ directory changes, because
     directory paths end in `/`: `foo` is deleted and `foo/` is added)
   - in both with a different type letter (file ↔ symlink) → a `D` row and an `A` row for the same path
   - in both, same type, unequal states → `M`
4. Return rows in the order defined above.

`compute_changes()` never lists a directory itself; the layout decision lives entirely in how the two maps are
built.

## Generating a patch (`process_get_diff_image`)

1. **Resolve inputs.**
   - New: a single `src_path` directory, or a single `-si` image.
   - Base: a single `-d` directory, or a single `-di` image.
   - Images are mounted **read-only** through a new helper
     `_mount_single_fs_image(image, read_only, mount_points, loop_devices)`: attach a loop device, pick
     `get_largest_partition()`, warn if the image has more than one partition, mount, verify with
     `os.path.ismount()`. A missing image is an error; nothing is ever created on the base side.
   - Build the two entry maps according to "Path layout".
   - Fail before any work if the new map contains the key `.hpcp_diff/` (or anything under it), or the output
     path already exists.
   - Warn (as other image paths do) when not running as root.
2. **Compute changes** with `compute_changes(new_map, base_map, …)`. Print `A` / `M` / `D` counts and the
   payload size (sum of `A` and `M` regular-file sizes). If there are no rows, print "No differences" and exit 0 without creating an image.
3. **Create volumes.** `create_image()` gains an `init_size=None` override; when given, it skips listing
   `src_paths` and sizes from `init_size` (payload bytes plus the change-list size). `_plan_dest_images()`,
   `-dis` splitting, and the xfs-above-300-MiB / ext4-below choice are unchanged. A target mount point is
   created with `tempfile.mkdtemp()` and registered in `mount_points`.
4. **Copy the payload** with `build_patch_tree(rows, new_map, volume_roots, meta)`:
   - `A` / `M` regular files via `copy_file_list_parallel()`, each copied from `new_map[key]` to `key` under a
     volume root; the copy engine already places each file on a volume with free space.
   - `A` / `M` symlinks via `create_sym_links()`.
   - Then every `A` / `M` directory and every parent directory of a payload entry, on every volume, via
     `sync_directory_metadata()` from the matching `new_map` entry (or the live parent directory for keys above
     the map, e.g. `data/` in `store` layout), deepest first, so directory mtimes are not disturbed by later
     writes.
   - Rate limits, `-j`, and `ERRORS` collection behave as in a normal copy.
5. **Write `.hpcp_diff/`** on every volume: `changes.tsv` first, then `manifest`. The manifest is written last
   so a patch interrupted at any earlier point has no manifest and `-siff` rejects it.
6. **Clean up** through the existing `clean_up()`; return `get_rc_from_error()`.

## Applying a patch (`process_load_diff_image`)

1. **Load.** Mount every `-siff` volume read-only with `_mount_single_fs_image()`. Read each
   `.hpcp_diff/manifest`. Fail if:
   - a volume has no manifest ("Incomplete patch: …"),
   - `patch_id`s differ,
   - a volume number is missing (the message names the expected file using `_split_dest_image_name()` on
     volume 1's path),
   - `format` / `format_version` is unsupported.

   Read `changes.tsv` from volume 1. If `hash_algo` is unavailable locally, warn and compare states without
   `<hash>`. Set `HASH_SIZE` from `hash_size` and use `full_hash` from the manifest for every state computed
   during this apply; restore the previous `HASH_SIZE` afterwards.
2. **Resolve the target.** `-d DIR`, `-di IMG` (mounted read-write with `_mount_single_fs_image()`), or a single
   positional path. It must already exist; apply never creates a target. The target is the patch root; for a
   `store` patch applied to a `-d` directory other than `/`, print the note described in "Path layout".
3. **Check** with `check_patch_target(rows, target_root, meta)`, computing target states in parallel:

   | Row | target == `old` | target == `new` | otherwise |
   |---|---|---|---|
   | `D` | apply (delete) | target absent → skip | conflict |
   | `A` | target absent → apply (add) | skip | conflict: exists |
   | `M` | apply (replace) | skip | conflict: changed locally or missing |

   A `D` directory that still contains entries not listed as `D` is also a conflict.

   `A` and `M` rows are checked against the target **as it will be after the applicable `D` rows**: a path
   removed by a `D` row (same path, or the same name with/without the trailing `/`), or lying under a path
   removed by a `D` row, counts as absent. This is what lets a file → directory or file → symlink change pass
   the check without a conflict.
4. **Conflicts.** Print every conflict (op, path, expected, found).
   - tty, no `-fa`: prompt `Apply anyway? [y/N]` with `multiCMD.input_with_timeout_and_countdown(60)`; anything
     other than `y`/`yes` aborts with a non-zero exit.
   - no tty, no `-fa`: abort with a non-zero exit (same rule as `-rm`).
   - `-fa`: print each conflict as `Warning: forcing …`, then continue. Forced semantics: `D` deletes anyway
     (`shutil.rmtree` for directories with unlisted entries); `A` / `M` remove a target entry of the wrong type
     first and then overwrite; missing `M` targets are created.
5. **Apply.**
   1. Delete applicable `D` rows: files and symlinks first, then directories deepest first.
   2. Copy applicable `A` / `M` regular files: locate each path in the volume that holds it and copy with
      `copy_file_list_parallel()`. Because only listed paths are copied, `.hpcp_diff/` is never touched.
   3. Recreate applicable `A` / `M` symlinks with `create_sym_links()`.
   4. Sync metadata of applicable `A` / `M` directories (and the root `./` if listed) from the payload, deepest
      first.
6. **Verify.** Recompute the state of every applied path and compare with `new`; each mismatch is recorded with
   `eprint` (non-zero exit). Print a summary: deleted, added, modified, skipped, forced, failed.
7. **Clean up** through `clean_up()`.

An apply interrupted mid-way can be re-run: completed paths match `new` and are skipped. A file cut off during
`cp` matches neither state and surfaces as a conflict that needs `-fa`.

## Effects on existing list options

- `-dfl` / `--diff_file_list` writes `changes.tsv` format via `write_changes()`, computed with
  `compute_changes()` when the base is a live tree (`-d` / `-di`). Entry maps follow "Path layout", extended to
  the several `src_path`s `-dfl` accepts: with a `-di` base each source contributes its own `store`-layout
  subtree (so other trees in the store are no longer reported as dest-only); with a `-d` base both sides are
  keyed exactly as `process_compare_file_list()` keys them today (`trim_paths()` against each root, sources
  merged into one map).
- `-dfl` against a stored `-fl` list: the new side is live, the base side is the list. The base is parsed with
  fixed rules: a line ending in `/` is a directory; otherwise the text after the last `:` is treated as a hash
  only if it is a valid hex digest for the current `hash_algo`, else the whole line is the path. Base states
  are `?`. The new side is keyed as `store_file_list()` keys it today. Rows: path only in new → `A`; only in
  list → `D`; in both and the list carries a hash that differs → `M` (old `?`); in both without a hash in the
  list → no row.
  Symlink lines (`path:target`) whose target contains `:` remain ambiguous — a limitation of the `-fl` format,
  which this design does not change.
- `-cfl` / `--compare_file_list` prints the `A`, `M`, and `D` lists and counts instead of "in src / in dest".
- `-tdfl` / `--tar_diff_file_list` keeps its role and format: plain newline-separated paths for `tar -T`,
  now taken from the `A` and `M` rows.
- The public name `compare_file_list` (alias of `_compare_file_lists`) is kept and routed through the new
  helpers.

## Code units (all in `hpcp.py`)

| Unit | Responsibility |
|---|---|
| `_escape_field` / `_unescape_field` | field escaping |
| `encode_state` / `decode_state`, `states_equal` | state strings and equality rules |
| `_xattr_digest(path)` | canonical xattr digest or `None` when unsupported |
| `_remove_stale_xattrs(src, dest)` | copy's stale-xattr fix |
| `get_entry_state(path, full_hash, read_xattrs)` | one state |
| `build_entry_map(root, prefix, parents, …)` | one side's `{key: absolute_path}` per "Path layout" |
| `compute_changes(new_map, base_map, …)` | sorted rows |
| `write_changes` / `read_changes` | `changes.tsv` and `-dfl` output |
| `write_manifest` / `read_manifest` | `manifest` |
| `build_patch_tree(rows, new_map, volume_roots, meta)` | payload copy + `.hpcp_diff/` (manifest last) |
| `check_patch_target(rows, target_root, meta)` | per-row apply / skip / conflict |
| `apply_patch_tree(volume_roots, target_root, force)` | load, check, prompt/force, delete, copy, verify |
| `_mount_single_fs_image(image, read_only, mount_points, loop_devices)` | largest-partition mount with warning |
| `process_get_diff_image` / `process_load_diff_image` | image wrappers called from `hpcp()` |

The tree-level units (`build_entry_map`, `compute_changes`, `build_patch_tree`, `check_patch_target`,
`apply_patch_tree`) take plain directories and maps, so they are testable without root or loop devices. Image handling lives only in the two
`process_*` wrappers.

Rows are passed as a small `namedtuple('ChangeRow', 'op path old new')`.

## Error handling

- All failures raise or report through the existing `"Category: detail"` `RuntimeError` / `eprint` style; the
  exit code comes from `get_rc_from_error()`.
- Mounts and loop devices are always released by `clean_up()` in `hpcp()`'s `finally`.
- Per-entry read failures during `compute_changes()` are recorded with `eprint` and the run continues; the
  resulting patch is still written but the run exits non-zero.
- Generation never modifies the new or base tree (both mounted read-only when they are images).

## Testing

New `tests/test_diff_image.py`, plus additions to existing test files. Tabs for indentation; plain `test_*`
functions; existing fixtures (`hpcp_mod`, `tmp_tree`, `reset_hpcp_globals`).

No root required:

- Escaping round-trips: tab, newline, CR, backslash, non-UTF-8 bytes, names containing `:`.
- `encode_state` / `decode_state` round-trips; `states_equal` tolerance and xattr `-` rules.
- `compute_changes` cases: add, delete, modify, file → directory, directory → file, mode-only, mtime-only,
  xattr-only (skipped when the filesystem lacks `user.*` xattrs), symlink retarget, directory chmod, same-size
  edit inside the hash window, excluded paths ignored on both sides.
- `build_entry_map`: `root` keys, `store` keys with parents, missing subtree yields only parents, `exclude`.
- `store` layout on directories (a directory standing in for a mounted store): a store holding two trees
  (`data/a/`, `data/b/`) diffed against a new `data/a`; only `data/a/…` rows appear, and applying leaves
  `data/b/` untouched.
- `build_patch_tree` → `apply_patch_tree` on directories: after applying to a copy of base, every path's state
  equals new (the success criterion), including the type-change cases.
- Conflicts: tty prompt aborts by default (simulated input), non-tty aborts, `force=True` applies and prints
  `Warning: forcing` lines.
- Re-applying a patch is a no-op (everything skipped, exit 0).
- A patch tree without `manifest` is rejected; mismatched `patch_id` and missing volume are rejected.
- `.hpcp_diff` at the new tree's root is rejected.
- `is_file_identical(compare_metadata=True)` detects mode, owner (when root), and xattr differences; the default
  ignores them.
- Copy removes destination-only xattrs on identical files and directories; not under `-co`.
- `-dfl` writes the new format; `-dfl` against a stored `-fl` list with `:` in directory and file names
  (regression test for the truncation bug); `-tdfl` still lists only `A` / `M` paths; `-cfl` prints `A` / `M` /
  `D`.
- CLI: `-diff` with and without a value, incompatible combinations rejected, `-fa` without `-siff` rejected.

Root + loop (marked `root` and `loop`, skipped like `tests/test_imaging.py` when unavailable):

- Directory new → directory base → patch image → apply to a directory copy of base.
- Directory new → `-di` store image (made with `hpcp DIR -di store.img`, then the directory changed) → patch
  image → `-siff` onto the store image; the store's other tree is unchanged.
- Image new → image base → patch image → apply to a `-di` image target.
- `-dis` two-volume patch: applies with both volumes; fails with a named missing volume when one is omitted.
- Base image is left unmodified after generation (checksum before/after).

## Release and docs

- Replace the "Not implemented" help text for `-diff` / `-siff`; add `-fa`.
- Regenerate the README help block (`COLUMNS=80`) and add a short "Patch images" section describing the
  generate → apply workflow, the `.hpcp_diff/` layout, and the conflict / `-fa` behavior.
- Bump `version` to `9.65` and sync `COMMIT_DATE`. The pending rate-limit help-text fix ships in the same
  release.
- The `cas-toolbox` pin rides the next Monday roll-up.

## Phase 2 notes (dd mode, not in this design)

- One section per partition (`part1/`, `part2/`, …) in the same patch image, each with its own change list.
- Partition pairing by number; refuse when the layout differs (partition count, fs type).
- Partitions hpcp cannot mount (BIOS boot, swap, unknown fs) would need a raw image payload when they differ.
- Filesystem identity changes (UUID, label, mkfs parameters) are not file changes and must be recorded or
  warned about.

## Known limitations

- Hardlinks are not preserved through a patch.
- FIFOs, sockets, and device nodes are ignored (hpcp's listing skips them today).
- An apply interrupted while a file is being written leaves that file in neither state; re-running needs `-fa`
  for it.
- Small patches occupy at least ~55 MiB because of `create_image()`'s 50 MiB filesystem slag.
- `-dfl` against a stored `-fl` list cannot disambiguate symlink lines whose target contains `:`.
- Image generation and application require root (loop devices, mounts) and are unavailable on Windows.
