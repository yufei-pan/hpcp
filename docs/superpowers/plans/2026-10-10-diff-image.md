# Differential Patch Images Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `hpcp -diff` write a self-contained patch image (whole added/modified files plus a change list) that turns a base tree into a new tree, make `hpcp -siff` check and apply it, and move `-dfl` onto the same change-list format.

**Architecture:** A new "Diff Image" section in `hpcp.py` holds tree-level units that work on plain directories: format primitives (escaping, entry states), `changes.tsv` / `manifest` I/O, entry maps, `compute_changes`, `build_patch_tree`, `check_patch_target`, and `apply_patch_tree`. Thin wrappers (`process_get_diff_image`, `process_load_diff_image`) add loop-device mounting and `create_image`, and `hpcp()` dispatches to them. `-dfl` / `-cfl` / `-tdfl` reuse the same rows and writer.

**Tech Stack:** Python 3 stdlib only (`os` xattr calls, `uuid`, `struct`, `namedtuple`), the existing hpcp copy engine (`copy_file_list_parallel`, `sync_directory_metadata`, `create_image`), pytest.

**Spec:** `docs/superpowers/specs/2026-10-10-diff-image-design.md`

## Global Constraints

- **Single-file module.** All production code goes in `hpcp.py`. No new modules or packages; the tool must stay copy-transportable.
- **Tabs for indentation** in `hpcp.py` and in `tests/`, matching the existing files.
- **Python 3.6 syntax floor** (`requires-python = ">=3.6"` in the script header): no walrus operator, no `str.removeprefix`, no `dict | dict`, no `f"{x=}"`.
- **No new runtime dependencies.** Only stdlib additions (`uuid`, `collections.namedtuple`).
- **Errors vs. warnings.** `eprint()` records any message containing `:` in `ERRORS`, and the exit code is looked up from the text before the first `:` in `ERROR_TO_RETURNCODE_TABLE` (unknown → 128). Use `eprint("Category: detail")` only for real failures. Print warnings and notes with `print(..., file=sys.stderr)` or `print(...)` so they don't change the exit code.
- **New error categories** (Task 1): `Invalid arguments` 241, `Diff image error` 242, `Patch format error` 243, `Incomplete patch` 244, `Patch conflict` 245, `Patch verify error` 246, `Entry state error` 247.
- **Reserved name:** `.hpcp_diff` at the patch root. Change-list header `#hpcp-changes<TAB>1`; manifest `format hpcp-diff`, `format_version 1`; phase 1 writes exactly `section	root	.	changes.tsv`.
- **Hashing:** use `hash_file.__wrapped__(path, size, full_hash)` for entry states. `hash_file` is `lru_cache`d and returns stale hashes for a file rewritten with the same size.
- **Version target:** `9.65`. Bump only in the final task.
- Run tests from the `hpcp/` directory with `python -m pytest tests/ -q`. Root/loop tests are marked `root` and `loop` and skip when not root or tools are missing; this machine runs as root, so they execute.
- Every commit message ends with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  ```

## Review Focus

1. **`-diff` written before the source path** (`hpcp -diff /new -d /base`): argparse makes `/new` the patch name. Expected: a `-diff` value that is an existing directory is used as the source, and the patch name is generated. Test: Task 11 `test_diff_value_that_is_a_directory_becomes_the_source`.
2. **Absolute `-e` patterns written for the new tree** (`-e /data/proj/cache`): the base side lives under a different path (a store mount or another directory), so a physical match misses it, the excluded content shows up as `D`, and apply deletes it. Expected: the base side is filtered by the equivalent new-tree path. Test: Task 5 `test_compute_changes_honours_exclude_on_both_sides`.
3. **Filenames with spaces, tabs, newlines, `:` or invalid UTF-8**: expected to survive compute → patch → apply unchanged. Tests: Task 9 `test_apply_patch_tree_round_trip` (via `_build_scenario`) and Task 10 `test_dir_patch_image_round_trip`.
4. **A file rewritten in place with the same size**: a cached hash would make detection miss it or make the post-apply verify fail. Tests: Task 2 `test_get_entry_state_sees_same_size_rewrite`; Task 9 round trip (`same_size.bin`).
5. **A directory replaced by a symlink or a file, and the reverse**: the delete and add on one name must pass the pre-apply check, apply in order, verify, and re-apply as a no-op. Tests: Task 5 `test_compute_changes_type_changes`; Task 9 `test_apply_patch_tree_round_trip` and `test_apply_patch_tree_is_idempotent`.

## Deviations from the spec (decided while planning)

- `build_patch_tree(rows, new_base, volume_roots, meta, …)` takes `new_base` (the root that keys are relative to: `/` in `store` layout, the new root in `root` layout) instead of `new_map`, because `copy_file_list_parallel` needs a source root. Every entry map satisfies `key == relpath(path, new_base)`.
- Patch symlinks are created with `os.symlink` (`_place_symlink`), not `create_sym_links`. The latter follows relative link targets and copies the files they point to.
- `-fl` parsing for `-dfl` is refined. In a hashed `-fl` list every non-directory line without a hex suffix is a symlink (`path:target`). Its path is matched against the live keys, or cut at the first `:`. The spec's rule alone would misparse every symlink line.
- Live `-dfl` / `-cfl` comparisons always use full entry states, so `-nhfl` no longer turns off hashing there. `-nhfl` still controls what `-fl` writes.
- Two safety checks the spec didn't list. `read_changes` rejects unsafe keys (`..`, absolute, empty components). Apply refuses to write or delete through a parent directory that resolves outside the target.

---

## File Structure

| File | Responsibility |
|---|---|
| `hpcp.py` (modify) | New "Diff Image" section (all new units), error codes, `is_file_identical(compare_metadata=)`, stale-xattr removal in `copy_file` / `sync_directory_metadata`, `_natural_key`, `-dfl`/`-cfl`/`-tdfl` rewrite, `create_image(init_size=)`, CLI flags, `hpcp()` / `main()` wiring |
| `tests/test_diff_image.py` (create) | Format, states, entry maps, compute, build, check, apply, wrappers, validation, root/loop end-to-end |
| `tests/test_identity_hash.py` (modify) | `is_file_identical(compare_metadata=True)` |
| `tests/test_copy.py` (modify) | Stale xattr removal |
| `tests/test_file_lists.py` (modify) | `-dfl` / `-tdfl` / `-cfl` on the new format, `:` regression |
| `tests/test_cli_args.py` (modify) | `-diff` / `-siff` / `-fa` parsing; `main()` passes them |
| `README.md` (modify) | Patch-image section, development status, regenerated help block |

**Where new code goes:** the new section starts with the comment `#%% ---- Diff Image ----` and sits **immediately before the line `#%% ---- Remove Extra ----`** (currently right after `compare_file_list = _compare_file_lists`). Unless a step says otherwise, each task appends its new functions to the end of this section, just above `#%% ---- Remove Extra ----`.

---

### Task 0: Commit the pending help-text fix

The working tree already holds a verified fix to the `-L` / `-F` help text in `hpcp.py` and `README.md`. Commit it on its own so later commits stay focused.

**Files:**
- Modify (already modified): `hpcp.py` (the `-L` / `-F` `add_argument` lines), `README.md` (the matching help block lines)

- [ ] **Step 1: Confirm the diff is only the help text**

Run: `git diff --stat && git diff hpcp.py README.md | grep '^[-+]' | grep -v '^[-+][-+]'`
Expected: only the `-L` / `-F` help strings change. The "do not work in single thread / serial mode" notes are gone, and the examples read `10K for 10000 files/s, 1Ki for 1024 files/s`.

- [ ] **Step 2: Run the CLI and limit tests**

Run: `python -m pytest tests/test_cli_args.py tests/test_limits.py -q`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add hpcp.py README.md
git commit -m "fix(cli): correct rate-limit help text

Serial mode is rate limited for both bytes and files, and 10K / 1Mi parse
as 10000 / 1048576.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 1: Format primitives and error codes

**Files:**
- Modify: `hpcp.py` (import line `from collections import deque`; `ERROR_TO_RETURNCODE_TABLE`; new Diff Image section)
- Create: `tests/test_diff_image.py`

**Interfaces:**
- Consumes: `get_timestamp_precision(num)` (existing).
- Produces:
  - `DIFF_DIR_NAME = '.hpcp_diff'`, `DIFF_FORMAT_VERSION = 1`, `CHANGES_HEADER = '#hpcp-changes'`, `CHANGES_VERSION = 1`
  - `ChangeRow = namedtuple('ChangeRow', 'op path old new')` (all four fields are `str`)
  - `_escape_field(text: str) -> str`, `_unescape_field(text: str) -> str` (raises `ValueError`)
  - `decode_state(state: str) -> dict | None` (raises `ValueError`); dict keys: files `type,size,mtime_ns,mode,uid,gid,hash,xattr`, dirs `type,mode,uid,gid,xattr`, links `type,target`, unknown `{'type': '?'}`
  - `encode_state(info: dict | None) -> str`
  - `states_equal(a: str, b: str, compare_hash: bool = True) -> bool`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_diff_image.py`:

```python
import contextlib
import hashlib
import os
import shutil
import stat
import sys
import pytest


def test_escape_field_round_trips_special_characters(hpcp_mod):
	samples = ['plain.txt', 'tab\tname', 'new\nline', 'cr\rname', 'back\\slash', 'colon:name',
		os.fsdecode(b'bad\xffbyte'), '\\t literal']
	for text in samples:
		escaped = hpcp_mod._escape_field(text)
		assert '\t' not in escaped and '\n' not in escaped and '\r' not in escaped
		assert hpcp_mod._unescape_field(escaped) == text


def test_escape_field_exact_output(hpcp_mod):
	assert hpcp_mod._escape_field('a\tb\nc\\d\re') == 'a\\tb\\nc\\\\d\\re'


def test_unescape_field_rejects_unknown_escape(hpcp_mod):
	with pytest.raises(ValueError):
		hpcp_mod._unescape_field('bad\\q')
	with pytest.raises(ValueError):
		hpcp_mod._unescape_field('trailing\\')


@pytest.mark.parametrize('state', [
	'-', '?',
	'f:10:1717000000123456789:100644:0:0:9f2c00aa11bb22cc:-',
	'f:0:5:100600:1000:1000:-:abcdef0123456789',
	'd:40755:0:0:-',
	'l:target',
	'l:../with\\ttab:and:colons',
])
def test_state_round_trip(hpcp_mod, state):
	assert hpcp_mod.encode_state(hpcp_mod.decode_state(state)) == state


def test_decode_state_fields(hpcp_mod):
	info = hpcp_mod.decode_state('f:10:1717000000123456789:100644:0:0:9f2c:-')
	assert info == {'type': 'f', 'size': 10, 'mtime_ns': 1717000000123456789, 'mode': 0o100644,
		'uid': 0, 'gid': 0, 'hash': '9f2c', 'xattr': '-'}
	assert hpcp_mod.decode_state('-') is None
	assert hpcp_mod.decode_state('l:a\\tb') == {'type': 'l', 'target': 'a\tb'}


def test_decode_state_rejects_garbage(hpcp_mod):
	for bad in ['x:1', 'f:1:2', 'd:abc:0:0:-', '']:
		with pytest.raises(ValueError):
			hpcp_mod.decode_state(bad)


def test_states_equal_rules(hpcp_mod):
	eq = hpcp_mod.states_equal
	base = 'f:10:1717000000000000000:100644:0:0:aaaa:-'
	assert eq(base, base)
	# whole-second timestamps get a 2 x 2 s tolerance, like is_file_identical()
	assert eq(base, 'f:10:1717000001000000000:100644:0:0:aaaa:-')
	assert not eq(base, 'f:10:1717000010000000000:100644:0:0:aaaa:-')
	assert not eq(base, 'f:11:1717000000000000000:100644:0:0:aaaa:-')
	assert not eq(base, 'f:10:1717000000000000000:100755:0:0:aaaa:-')
	assert not eq(base, 'f:10:1717000000000000000:100644:1:0:aaaa:-')
	assert not eq(base, 'f:10:1717000000000000000:100644:0:0:bbbb:-')
	assert eq(base, 'f:10:1717000000000000000:100644:0:0:bbbb:-', compare_hash=False)
	# xattr '-' on either side means "not compared"
	assert eq(base, 'f:10:1717000000000000000:100644:0:0:aaaa:1234')
	assert not eq('f:10:1717000000000000000:100644:0:0:aaaa:1234', 'f:10:1717000000000000000:100644:0:0:aaaa:5678')
	# directories ignore mtime and compare mode / owner / xattr
	assert eq('d:40755:0:0:-', 'd:40755:0:0:-')
	assert not eq('d:40755:0:0:-', 'd:40700:0:0:-')
	assert eq('l:a', 'l:a') and not eq('l:a', 'l:b')
	assert eq('-', '-') and not eq('-', base)
	assert not eq('?', '?')
	assert not eq('d:40755:0:0:-', 'l:a')
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_diff_image.py -q`
Expected: FAIL with `AttributeError: module 'hpcp' has no attribute '_escape_field'` (and similar).

- [ ] **Step 3: Implement**

In `hpcp.py`, change the import line `from collections import deque` to:

```python
from collections import deque, namedtuple
```

In `ERROR_TO_RETURNCODE_TABLE`, directly after the line `'No source paths': 240,` add:

```python
	'Invalid arguments': 241,
	'Diff image error': 242,
	'Patch format error': 243,
	'Incomplete patch': 244,
	'Patch conflict': 245,
	'Patch verify error': 246,
	'Entry state error': 247,
```

Immediately before the line `#%% ---- Remove Extra ----`, insert the new section:

```python
#%% ---- Diff Image ----
DIFF_DIR_NAME = '.hpcp_diff'
DIFF_FORMAT_VERSION = 1
CHANGES_HEADER = '#hpcp-changes'
CHANGES_VERSION = 1
ChangeRow = namedtuple('ChangeRow', 'op path old new')
_FIELD_UNESCAPES = {'\\': '\\', 't': '\t', 'n': '\n', 'r': '\r'}

def _escape_field(text):
	"""Escape backslash, tab, newline and carriage return so text fits in one TSV field."""
	return text.replace('\\', '\\\\').replace('\t', '\\t').replace('\n', '\\n').replace('\r', '\\r')

def _unescape_field(text):
	"""Exact inverse of _escape_field. Raises ValueError on an unknown or dangling escape."""
	if '\\' not in text:
		return text
	out = []
	index = 0
	while index < len(text):
		char = text[index]
		if char != '\\':
			out.append(char)
			index += 1
			continue
		if index + 1 >= len(text) or text[index + 1] not in _FIELD_UNESCAPES:
			raise ValueError(f'bad escape in {text!r}')
		out.append(_FIELD_UNESCAPES[text[index + 1]])
		index += 2
	return ''.join(out)

def decode_state(state):
	"""Parse an entry state string.

	Returns None for '-' (absent), {'type': '?'} for '?' (unknown), otherwise a dict:
	files f:size:mtime_ns:mode:uid:gid:hash:xattr, directories d:mode:uid:gid:xattr,
	symlinks l:target (target escaped). Raises ValueError on anything else.
	"""
	if state == '-':
		return None
	if state == '?':
		return {'type': '?'}
	kind, sep, rest = state.partition(':')
	if not sep:
		raise ValueError(f'bad entry state {state!r}')
	if kind == 'l':
		return {'type': 'l', 'target': _unescape_field(rest)}
	fields = rest.split(':')
	if kind == 'f' and len(fields) == 7:
		size, mtime_ns, mode, uid, gid, file_hash, xattr = fields
		return {'type': 'f', 'size': int(size), 'mtime_ns': int(mtime_ns), 'mode': int(mode, 8),
			'uid': int(uid), 'gid': int(gid), 'hash': file_hash, 'xattr': xattr}
	if kind == 'd' and len(fields) == 4:
		mode, uid, gid, xattr = fields
		return {'type': 'd', 'mode': int(mode, 8), 'uid': int(uid), 'gid': int(gid), 'xattr': xattr}
	raise ValueError(f'bad entry state {state!r}')

def encode_state(info):
	"""Inverse of decode_state()."""
	if info is None:
		return '-'
	kind = info['type']
	if kind == '?':
		return '?'
	if kind == 'l':
		return 'l:' + _escape_field(info['target'])
	if kind == 'd':
		return f"d:{info['mode']:o}:{info['uid']}:{info['gid']}:{info['xattr']}"
	if kind == 'f':
		return (f"f:{info['size']}:{info['mtime_ns']}:{info['mode']:o}:{info['uid']}:{info['gid']}:"
			f"{info['hash']}:{info['xattr']}")
	raise ValueError(f'bad entry state type {kind!r}')

def _mtime_ns_close(a_ns, b_ns):
	"""Same tolerance as is_file_identical(): 2 x the coarser timestamp precision."""
	precision = max(get_timestamp_precision(a_ns / 1e9), get_timestamp_precision(b_ns / 1e9))
	return abs(a_ns - b_ns) <= precision * 2 * 1e9

def states_equal(a, b, compare_hash=True):
	"""True when two entry states describe the same entry.

	Directories ignore mtime; files allow the timestamp-precision tolerance; an xattr
	digest of '-' on either side is not compared; '?' never equals anything.
	"""
	if a == '?' or b == '?':
		return False
	if a == b:
		return True
	info_a, info_b = decode_state(a), decode_state(b)
	if info_a is None or info_b is None or info_a['type'] != info_b['type']:
		return False
	if info_a['type'] == 'l':
		return info_a['target'] == info_b['target']
	if (info_a['mode'], info_a['uid'], info_a['gid']) != (info_b['mode'], info_b['uid'], info_b['gid']):
		return False
	if info_a['xattr'] != '-' and info_b['xattr'] != '-' and info_a['xattr'] != info_b['xattr']:
		return False
	if info_a['type'] == 'd':
		return True
	if info_a['size'] != info_b['size']:
		return False
	if compare_hash and info_a['hash'] != info_b['hash']:
		return False
	return _mtime_ns_close(info_a['mtime_ns'], info_b['mtime_ns'])

```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_diff_image.py tests/test_run_hpcp_doctests.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_diff_image.py
git commit -m "feat(diff): add change-list field escaping and entry states

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Entry states and metadata-aware `is_file_identical`

**Files:**
- Modify: `hpcp.py` (`is_file_identical`, currently near line 2276; Diff Image section)
- Test: `tests/test_diff_image.py`, `tests/test_identity_hash.py`

**Interfaces:**
- Consumes: `encode_state` (Task 1), `hash_file` (existing, use `.__wrapped__`), `xxhash_available` (existing).
- Produces:
  - `_hash_algo_name() -> str` (`'xxh64'` or `'blake2b'`)
  - `_new_hasher()` (hashlib-style object with `update` / `hexdigest`)
  - `_xattr_digest(path: str) -> str | None` (None when the platform or filesystem doesn't support xattrs)
  - `get_entry_state(path: str, full_hash: bool = False, read_xattrs: bool = True) -> str`
  - `is_file_identical(src_path, dest_path, src_size, full_hash=False, compare_metadata=False) -> bool`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_diff_image.py`:

```python
def _set_xattr_or_skip(path, name='user.hpcp_test', value=b'1'):
	try:
		os.setxattr(path, name, value, follow_symlinks=False)
	except (OSError, AttributeError) as e:
		pytest.skip(f'user xattrs unsupported here: {e}')


def test_get_entry_state_file_dir_link_absent(tmp_path, hpcp_mod, reset_hpcp_globals):
	hpcp_mod.HASH_SIZE = 65536
	f = tmp_path / 'f.txt'
	f.write_bytes(b'hello')
	os.chmod(f, 0o640)
	d = tmp_path / 'd'
	d.mkdir()
	os.chmod(d, 0o750)
	os.symlink('f.txt', tmp_path / 'l')
	fs = hpcp_mod.decode_state(hpcp_mod.get_entry_state(str(f)))
	st = os.lstat(f)
	assert fs['type'] == 'f' and fs['size'] == 5 and fs['mtime_ns'] == st.st_mtime_ns
	assert (fs['mode'], fs['uid'], fs['gid']) == (st.st_mode, st.st_uid, st.st_gid)
	assert fs['hash'] == hpcp_mod.hash_file.__wrapped__(str(f), 5, False)
	ds = hpcp_mod.decode_state(hpcp_mod.get_entry_state(str(d) + os.sep))
	assert ds['type'] == 'd' and ds['mode'] == os.lstat(d).st_mode
	assert hpcp_mod.get_entry_state(str(tmp_path / 'l')) == 'l:f.txt'
	assert hpcp_mod.get_entry_state(str(tmp_path / 'missing')) == '-'
	# a path below a regular file is absent, not an error
	assert hpcp_mod.get_entry_state(str(f / 'child')) == '-'
	assert hpcp_mod.ERRORS == []


def test_get_entry_state_sees_same_size_rewrite(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Review Focus 4: hash_file is lru_cached; entry states must not reuse a stale hash."""
	hpcp_mod.HASH_SIZE = 65536
	f = tmp_path / 'f.bin'
	f.write_bytes(b'AAAA')
	first = hpcp_mod.get_entry_state(str(f))
	hpcp_mod.hash_file(str(f), 4, False)
	f.write_bytes(b'BBBB')
	second = hpcp_mod.get_entry_state(str(f))
	assert hpcp_mod.decode_state(first)['hash'] != hpcp_mod.decode_state(second)['hash']


def test_get_entry_state_hash_disabled(tmp_path, hpcp_mod, reset_hpcp_globals):
	hpcp_mod.HASH_SIZE = 0
	f = tmp_path / 'f.bin'
	f.write_bytes(b'AAAA')
	assert hpcp_mod.decode_state(hpcp_mod.get_entry_state(str(f)))['hash'] == '-'


def test_xattr_digest_reflects_xattrs(tmp_path, hpcp_mod, reset_hpcp_globals):
	plain = tmp_path / 'plain.txt'
	tagged = tmp_path / 'tagged.txt'
	plain.write_text('x')
	tagged.write_text('x')
	_set_xattr_or_skip(str(tagged))
	assert hpcp_mod._xattr_digest(str(plain)) != hpcp_mod._xattr_digest(str(tagged))
	state = hpcp_mod.decode_state(hpcp_mod.get_entry_state(str(tagged)))
	assert state['xattr'] == hpcp_mod._xattr_digest(str(tagged))
	assert hpcp_mod.decode_state(hpcp_mod.get_entry_state(str(tagged), read_xattrs=False))['xattr'] == '-'
```

Append to `tests/test_identity_hash.py`:

```python
def _identical_pair(tmp_tree):
	src = tmp_tree.add_file('m.txt', 'same', under='src')
	dst = tmp_tree.add_file('m.txt', 'same', under='dst')
	t = time.time() - 1000
	os.utime(src, (t, t))
	os.utime(dst, (t, t))
	return src, dst


def test_is_file_identical_compare_metadata_sees_mode(tmp_tree, hpcp_mod, reset_hpcp_globals):
	src, dst = _identical_pair(tmp_tree)
	os.chmod(src, 0o644)
	os.chmod(dst, 0o600)
	hpcp_mod.HASH_SIZE = 65536
	assert hpcp_mod.is_file_identical(src, dst, os.path.getsize(src)) is True
	assert hpcp_mod.is_file_identical(src, dst, os.path.getsize(src), compare_metadata=True) is False


@pytest.mark.skipif(os.name == 'nt' or os.geteuid() != 0, reason='chown needs root')
def test_is_file_identical_compare_metadata_sees_owner(tmp_tree, hpcp_mod, reset_hpcp_globals):
	src, dst = _identical_pair(tmp_tree)
	os.chown(dst, 4321, 4321)
	hpcp_mod.HASH_SIZE = 65536
	assert hpcp_mod.is_file_identical(src, dst, os.path.getsize(src)) is True
	assert hpcp_mod.is_file_identical(src, dst, os.path.getsize(src), compare_metadata=True) is False


def test_is_file_identical_compare_metadata_sees_xattr(tmp_tree, hpcp_mod, reset_hpcp_globals):
	src, dst = _identical_pair(tmp_tree)
	try:
		os.setxattr(dst, 'user.hpcp_test', b'1')
	except (OSError, AttributeError) as e:
		pytest.skip(f'user xattrs unsupported here: {e}')
	hpcp_mod.HASH_SIZE = 65536
	assert hpcp_mod.is_file_identical(src, dst, os.path.getsize(src)) is True
	assert hpcp_mod.is_file_identical(src, dst, os.path.getsize(src), compare_metadata=True) is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_diff_image.py tests/test_identity_hash.py -q`
Expected: FAIL. The new `test_diff_image.py` tests fail with `AttributeError` (`get_entry_state` / `_xattr_digest` missing), and the identity tests fail with `TypeError: is_file_identical() got an unexpected keyword argument 'compare_metadata'`.

- [ ] **Step 3: Implement**

Replace the whole `is_file_identical` function with:

```python
def is_file_identical(src_path, dest_path,src_size,full_hash=False,compare_metadata=False):
	dst_size = os.path.getsize(dest_path)
	# try to find the mtime are different
	if src_size != dst_size:
		return False
	try:
		src_mtime = os.path.getmtime(src_path)
		dst_mtime = os.path.getmtime(dest_path)
		precision = max(get_timestamp_precision(src_mtime), get_timestamp_precision(dst_mtime))
		if abs(src_mtime - dst_mtime) > precision * 2:
			return False
	except Exception:
		pass
	if hash_file(src_path,src_size,full_hash) != hash_file(dest_path,dst_size,full_hash):
		return False
	if compare_metadata:
		src_st = os.lstat(src_path)
		dst_st = os.lstat(dest_path)
		if (src_st.st_mode, src_st.st_uid, src_st.st_gid) != (dst_st.st_mode, dst_st.st_uid, dst_st.st_gid):
			return False
		src_xattr = _xattr_digest(src_path)
		dst_xattr = _xattr_digest(dest_path)
		if src_xattr is not None and dst_xattr is not None and src_xattr != dst_xattr:
			return False
	return True
```

Append to the Diff Image section:

```python
def _hash_algo_name():
	return 'xxh64' if xxhash_available else 'blake2b'

def _new_hasher():
	return xxhash.xxh64() if xxhash_available else hashlib.blake2b()

def _xattr_digest(path):
	"""Hex digest of every readable xattr of path (symlinks not followed); ACLs are xattrs too.

	Returns None when the platform or the filesystem does not support xattrs.
	"""
	if not hasattr(os, 'listxattr'):
		return None
	try:
		names = os.listxattr(path, follow_symlinks=False)
	except OSError as e:
		if e.errno in (errno.ENOTSUP, errno.EOPNOTSUPP):
			return None
		raise
	hasher = _new_hasher()
	for name in sorted(names, key=os.fsencode):
		try:
			value = os.getxattr(path, name, follow_symlinks=False)
		except OSError as e:
			if e.errno in (errno.ENODATA, errno.EPERM, errno.EACCES, errno.ENOTSUP):
				continue
			raise
		hasher.update(os.fsencode(name) + b'\0' + struct.pack('>Q', len(value)) + value)
	return hasher.hexdigest()

def get_entry_state(path, full_hash=False, read_xattrs=True):
	"""State string of one path (see decode_state). Missing paths, including ones below a
	regular file, are '-'; any other read failure is recorded with eprint and returns '-'."""
	path = path.rstrip(os.sep) or os.sep
	try:
		st = os.lstat(path)
		if stat.S_ISLNK(st.st_mode):
			return encode_state({'type': 'l', 'target': os.readlink(path)})
		xattr = (_xattr_digest(path) if read_xattrs else None) or '-'
		if stat.S_ISDIR(st.st_mode):
			return encode_state({'type': 'd', 'mode': st.st_mode, 'uid': st.st_uid, 'gid': st.st_gid, 'xattr': xattr})
		if stat.S_ISREG(st.st_mode):
			# hash_file is lru_cached and would return a stale hash for a same-size rewrite.
			file_hash = hash_file.__wrapped__(path, st.st_size, full_hash) or '-'
			return encode_state({'type': 'f', 'size': st.st_size, 'mtime_ns': st.st_mtime_ns, 'mode': st.st_mode,
				'uid': st.st_uid, 'gid': st.st_gid, 'hash': file_hash, 'xattr': xattr})
	except (FileNotFoundError, NotADirectoryError):
		return '-'
	except OSError as e:
		eprint(f'Entry state error: cannot read {path}: {e}')
		return '-'
	return '-'

```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_diff_image.py tests/test_identity_hash.py tests/test_copy.py -q`
Expected: all pass (xattr tests may report SKIPPED on filesystems without `user.*` xattrs).

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_diff_image.py tests/test_identity_hash.py
git commit -m "feat(diff): add entry states and metadata-aware is_file_identical

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Copy removes stale xattrs

**Files:**
- Modify: `hpcp.py` (`copy_file` identical-file branch, currently near line 2732; `sync_directory_metadata`, currently near line 3359; Diff Image section)
- Test: `tests/test_copy.py`

**Interfaces:**
- Consumes: `CONTENT_ONLY` global.
- Produces: `_remove_stale_xattrs(src: str, dest: str) -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_copy.py`:

```python
def _set_xattr_or_skip(path, name, value=b'1'):
	try:
		os.setxattr(path, name, value, follow_symlinks=False)
	except (OSError, AttributeError) as e:
		pytest.skip(f'user xattrs unsupported here: {e}')


def _xattr_names(path):
	return set(os.listxattr(path, follow_symlinks=False))


def _identical_files(tmp_tree):
	src = tmp_tree.add_file('x.txt', 'same', under='src')
	dst = tmp_tree.add_file('x.txt', 'same', under='dst')
	t = time.time() - 1000
	os.utime(src, (t, t))
	os.utime(dst, (t, t))
	return src, dst


def test_copy_file_removes_stale_xattr_on_identical_file(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	src, dst = _identical_files(tmp_tree)
	_set_xattr_or_skip(dst, 'user.stale')
	_set_xattr_or_skip(src, 'user.keep')
	hpcp_mod.CONTENT_ONLY = False
	hpcp_mod.HASH_SIZE = 65536
	hpcp_mod.copy_file(src, [dst])
	names = _xattr_names(dst)
	assert 'user.stale' not in names
	assert 'user.keep' in names


def test_copy_file_keeps_stale_xattr_under_content_only(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	src, dst = _identical_files(tmp_tree)
	_set_xattr_or_skip(dst, 'user.stale')
	hpcp_mod.CONTENT_ONLY = True
	hpcp_mod.HASH_SIZE = 65536
	hpcp_mod.copy_file(src, [dst])
	assert 'user.stale' in _xattr_names(dst)


def test_sync_directory_metadata_removes_stale_xattr(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	src = tmp_tree.add_dir('d', under='src')
	dst = tmp_tree.add_dir('d', under='dst')
	_set_xattr_or_skip(dst, 'user.stale')
	hpcp_mod.CONTENT_ONLY = False
	hpcp_mod.sync_directory_metadata(src, [dst])
	assert 'user.stale' not in _xattr_names(dst)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_copy.py -q -k stale`
Expected: `test_copy_file_removes_stale_xattr_on_identical_file` and `test_sync_directory_metadata_removes_stale_xattr` FAIL (`'user.stale'` still present); the content-only test passes.

- [ ] **Step 3: Implement**

Append to the Diff Image section:

```python
def _remove_stale_xattrs(src, dest):
	"""Remove xattrs (ACLs included) that dest has and src does not.

	shutil.copystat() copies and overwrites xattrs but never removes one.
	Skipped under CONTENT_ONLY; unsupported or forbidden removals are ignored.
	"""
	if CONTENT_ONLY or not hasattr(os, 'listxattr'):
		return
	try:
		src_names = set(os.listxattr(src, follow_symlinks=False))
		dest_names = os.listxattr(dest, follow_symlinks=False)
	except OSError:
		return
	for name in dest_names:
		if name in src_names:
			continue
		try:
			os.removexattr(dest, name, follow_symlinks=False)
		except OSError as e:
			if e.errno not in (errno.ENOTSUP, errno.EPERM, errno.ENODATA, errno.EACCES):
				raise

```

In `copy_file`, in the identical-file branch, change:

```python
				if not CONTENT_ONLY:
					st = os.stat(src_path,follow_symlinks=False)
					shutil.copystat(src_path, dest,follow_symlinks=False)
```

to:

```python
				if not CONTENT_ONLY:
					st = os.stat(src_path,follow_symlinks=False)
					_remove_stale_xattrs(src_path, dest)
					shutil.copystat(src_path, dest,follow_symlinks=False)
```

In `sync_directory_metadata`, change:

```python
		if not CONTENT_ONLY:
			shutil.copystat(src_path, dest_path)
```

to:

```python
		if not CONTENT_ONLY:
			_remove_stale_xattrs(src_path, dest_path)
			shutil.copystat(src_path, dest_path)
```

(Removing first lets `copystat()` re-apply the source's own xattrs and then `chmod`, so an ACL removal can't leave stale group bits behind.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_copy.py tests/test_content_only.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_copy.py
git commit -m "fix(copy): remove destination-only xattrs when syncing metadata

copystat() adds and overwrites xattrs but never removes one, so an ACL or
xattr deleted on the source survived on identical files and directories.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `changes.tsv` and `manifest` I/O

**Files:**
- Modify: `hpcp.py` (`natural_sort`, currently near line 2164; Diff Image section)
- Test: `tests/test_diff_image.py`

**Interfaces:**
- Consumes: `ChangeRow`, `_escape_field`, `_unescape_field`, `decode_state` (Task 1).
- Produces:
  - `_natural_key(text: str) -> list` (shared sort key; `natural_sort` uses it)
  - `sort_change_rows(rows) -> list[ChangeRow]` (D, then A, then M; natural path order within each)
  - `_check_key(key: str) -> None` (raises `ValueError` for unsafe keys)
  - `write_changes(target: str, rows) -> None`, `read_changes(source: str) -> list[ChangeRow]` (raises `RuntimeError('Patch format error: …')`)
  - `write_manifest(target: str, meta: dict) -> None`, `read_manifest(source: str) -> dict` (values are `str`; `meta['sections']` is a list of `(name, payload_dir, changes_file)` tuples)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_diff_image.py`:

```python
def test_changes_round_trip_and_order(tmp_path, hpcp_mod):
	R = hpcp_mod.ChangeRow
	rows = [
		R('M', 'b10.txt', 'f:1:1:100644:0:0:aa:-', 'f:2:1:100644:0:0:bb:-'),
		R('A', 'dir/', '-', 'd:40755:0:0:-'),
		R('D', 'odd\tname\nhere', 'f:1:1:100644:0:0:aa:-', '-'),
		R('M', 'b9.txt', 'f:1:1:100644:0:0:aa:-', 'f:2:1:100644:0:0:bb:-'),
		R('A', 'link', '-', 'l:tar\\tget'),
		R('A', os.fsdecode(b'bad\xffbyte'), '-', 'f:1:1:100644:0:0:aa:-'),
		R('M', './', 'd:40755:0:0:-', 'd:40700:0:0:-'),
	]
	out = tmp_path / 'changes.tsv'
	hpcp_mod.write_changes(str(out), rows)
	lines = out.read_bytes().split(b'\n')
	assert lines[0] == b'#hpcp-changes\t1'
	assert lines[1] == b'#op\tpath\told\tnew'
	got = hpcp_mod.read_changes(str(out))
	assert [r.op for r in got] == ['D', 'A', 'A', 'A', 'M', 'M', 'M']
	assert [r.path for r in got if r.op == 'M'][-2:] == ['b9.txt', 'b10.txt']
	assert set(got) == set(rows)


def test_read_changes_rejects_bad_input(tmp_path, hpcp_mod):
	path = tmp_path / 'c.tsv'

	def read(text):
		path.write_text(text)
		return hpcp_mod.read_changes(str(path))

	with pytest.raises(RuntimeError, match='Patch format error'):
		read('not a change list\n')
	with pytest.raises(RuntimeError, match='newer'):
		read('#hpcp-changes\t2\n')
	with pytest.raises(RuntimeError, match='Patch format error'):
		read('#hpcp-changes\t1\nX\ta\t-\t-\n')
	with pytest.raises(RuntimeError, match='Patch format error'):
		read('#hpcp-changes\t1\nA\ta\t-\tz:1\n')
	for unsafe in ['../x', '/etc/passwd', 'a//b', 'a/./b', 'a/../b', '']:
		with pytest.raises(RuntimeError, match='Patch format error'):
			read(f'#hpcp-changes\t1\nA\t{unsafe}\t-\tl:x\n')


def test_manifest_round_trip(tmp_path, hpcp_mod):
	meta = {'format': 'hpcp-diff', 'format_version': 1, 'patch_id': 'abc', 'volume': '1/2',
		'new': 'name\twith tab', 'sections': [('root', '.', 'changes.tsv')]}
	path = tmp_path / 'manifest'
	hpcp_mod.write_manifest(str(path), meta)
	got = hpcp_mod.read_manifest(str(path))
	assert got['format'] == 'hpcp-diff' and got['format_version'] == '1'
	assert got['new'] == 'name\twith tab' and got['volume'] == '1/2'
	assert got['sections'] == [('root', '.', 'changes.tsv')]


def test_read_manifest_rejects_foreign_or_newer(tmp_path, hpcp_mod):
	path = tmp_path / 'manifest'
	path.write_text('format\tsomething-else\nformat_version\t1\n')
	with pytest.raises(RuntimeError, match='Patch format error'):
		hpcp_mod.read_manifest(str(path))
	path.write_text('format\thpcp-diff\nformat_version\t2\n')
	with pytest.raises(RuntimeError, match='newer'):
		hpcp_mod.read_manifest(str(path))
	path.write_text('format\thpcp-diff\nformat_version\t1\nunknown_key\tfine\n')
	assert hpcp_mod.read_manifest(str(path))['unknown_key'] == 'fine'
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_diff_image.py -q -k "changes or manifest"`
Expected: FAIL with `AttributeError: module 'hpcp' has no attribute 'write_changes'`.

- [ ] **Step 3: Implement**

Replace `natural_sort` (keep its docstring and doctest) with:

```python
def _natural_key(text):
	"""Sort key splitting digit runs into ints. Odd positions of re.split are always ASCII digit runs."""
	return [int(part) if index % 2 else part.lower() for index, part in enumerate(re.split('([0-9]+)', text))]

def natural_sort(L): 
	"""
	Sorts a list of strings naturally, considering both numeric and alphabetic characters.

	Args:
		L (list): The list of strings to be sorted.

	Returns:
		list: The sorted list of strings.

	Examples:
		>>> natural_sort(['file10.txt', 'file2.txt', 'file1.txt'])
		['file1.txt', 'file2.txt', 'file10.txt']
	"""
	return sorted(L, key=_natural_key)
```

Append to the Diff Image section:

```python
_OP_ORDER = {'D': 0, 'A': 1, 'M': 2}

def sort_change_rows(rows):
	"""Deterministic order: D, then A, then M; natural path order within each op."""
	return sorted(rows, key=lambda row: (_OP_ORDER[row.op], _natural_key(row.path)))

def _check_key(key):
	"""Reject keys that could escape the patch root."""
	if key == './':
		return
	if not key or key.startswith('/') or '\0' in key:
		raise ValueError(f'unsafe path {key!r}')
	if any(part in ('', '.', '..') for part in key.rstrip('/').split('/')):
		raise ValueError(f'unsafe path {key!r}')

def write_changes(target, rows):
	"""Write rows to target in changes.tsv format (also the -dfl output format)."""
	with open(target, 'w', encoding='utf-8', errors='surrogateescape', newline='\n') as f:
		f.write(f'{CHANGES_HEADER}\t{CHANGES_VERSION}\n')
		f.write('#op\tpath\told\tnew\n')
		for row in sort_change_rows(rows):
			f.write(f'{row.op}\t{_escape_field(row.path)}\t{row.old}\t{row.new}\n')

def read_changes(source):
	"""Read a changes.tsv file into ChangeRows. Raises RuntimeError('Patch format error: ...')."""
	rows = []
	with open(source, 'r', encoding='utf-8', errors='surrogateescape', newline='\n') as f:
		header = f.readline().rstrip('\n').split('\t')
		if len(header) != 2 or header[0] != CHANGES_HEADER:
			raise RuntimeError(f'Patch format error: {source} is not an hpcp change list')
		try:
			change_version = int(header[1])
		except ValueError:
			raise RuntimeError(f'Patch format error: {source} has a bad change list version {header[1]!r}')
		if change_version > CHANGES_VERSION:
			raise RuntimeError(f'Patch format error: {source} has change list version {change_version}, newer than supported {CHANGES_VERSION}')
		for line_number, line in enumerate(f, start=2):
			line = line.rstrip('\n')
			if not line or line.startswith('#'):
				continue
			fields = line.split('\t')
			if len(fields) != 4 or fields[0] not in _OP_ORDER:
				raise RuntimeError(f'Patch format error: {source} line {line_number} is malformed')
			try:
				path = _unescape_field(fields[1])
				_check_key(path)
				decode_state(fields[2])
				decode_state(fields[3])
			except ValueError as e:
				raise RuntimeError(f'Patch format error: {source} line {line_number}: {e}')
			rows.append(ChangeRow(fields[0], path, fields[2], fields[3]))
	return rows

def write_manifest(target, meta):
	"""Write meta (key -> value, plus 'sections': [(name, payload_dir, changes_file)]) as key/value TSV."""
	with open(target, 'w', encoding='utf-8', errors='surrogateescape', newline='\n') as f:
		for key, value in meta.items():
			if key != 'sections':
				f.write(f'{key}\t{_escape_field(str(value))}\n')
		for name, payload_dir, changes_file in meta.get('sections', []):
			f.write(f'section\t{_escape_field(name)}\t{_escape_field(payload_dir)}\t{_escape_field(changes_file)}\n')

def read_manifest(source):
	"""Read a manifest. Values are strings; unknown keys are kept and ignored by callers."""
	meta = {'sections': []}
	with open(source, 'r', encoding='utf-8', errors='surrogateescape', newline='\n') as f:
		for line in f:
			line = line.rstrip('\n')
			if not line or line.startswith('#'):
				continue
			try:
				fields = [_unescape_field(field) for field in line.split('\t')]
			except ValueError as e:
				raise RuntimeError(f'Patch format error: {source}: {e}')
			if fields[0] == 'section' and len(fields) == 4:
				meta['sections'].append(tuple(fields[1:]))
			elif fields[0] != 'section' and len(fields) == 2:
				meta[fields[0]] = fields[1]
			else:
				raise RuntimeError(f'Patch format error: {source} has a malformed line {line!r}')
	if meta.get('format') != 'hpcp-diff':
		raise RuntimeError(f'Patch format error: {source} is not an hpcp-diff manifest')
	try:
		format_version = int(meta.get('format_version', ''))
	except ValueError:
		raise RuntimeError(f'Patch format error: {source} has a bad format_version')
	if format_version > DIFF_FORMAT_VERSION:
		raise RuntimeError(f'Patch format error: {source} uses format version {format_version}, newer than supported {DIFF_FORMAT_VERSION}')
	return meta

```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_diff_image.py tests/test_run_hpcp_doctests.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_diff_image.py
git commit -m "feat(diff): read and write changes.tsv and manifest

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Entry maps and `compute_changes`

**Files:**
- Modify: `hpcp.py` (Diff Image section)
- Test: `tests/test_diff_image.py`

**Interfaces:**
- Consumes: `get_file_list` (existing; pass `drop_cache=True`), `is_excluded` (existing), `_call_with_worker_init` is **not** used here, `get_entry_state`, `states_equal`, `sort_change_rows`, `ChangeRow`.
- Produces:
  - `build_entry_map(root, prefix='', parents=None, exclude=None, logical_root=None, max_workers=…, parallel_file_listing=False) -> dict[str, str]`
  - `build_store_maps(new_dir, store_root, exclude=None, max_workers=…, parallel_file_listing=False) -> (new_map, base_map)` (`store` layout, keys relative to `/`)
  - `build_root_maps(new_root, base_root, exclude=None, max_workers=…, parallel_file_listing=False) -> (new_map, base_map)` (`root` layout)
  - `_states_for_paths(paths, full_hash=False, read_xattrs=True, max_workers=1) -> dict[path, state]`
  - `compute_changes(new_map, base_map, full_hash=False, max_workers=…, read_xattrs=True) -> list[ChangeRow]` (sorted)
  - Invariant used by later tasks: every map key equals `relpath(path, new_base)` with `/` separators (plus a trailing `/` for directories), where `new_base` is `/` for store maps and the new root for root maps.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_diff_image.py`:

```python
T0 = 1700000000  # whole even second: predictable 4 s mtime tolerance


def _write(path, data=b'x', mtime=T0):
	path = str(path)
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'wb') as f:
		f.write(data if isinstance(data, bytes) else data.encode('utf-8', 'surrogateescape'))
	os.utime(path, (mtime, mtime))
	return path


@pytest.fixture
def trees(tmp_path, hpcp_mod, reset_hpcp_globals):
	hpcp_mod.HASH_SIZE = 65536
	new = tmp_path / 'new'
	base = tmp_path / 'base'
	new.mkdir()
	base.mkdir()
	return new, base


def _changes(hpcp_mod, new, base, **kwargs):
	new_map, base_map = hpcp_mod.build_root_maps(str(new), str(base), max_workers=1, **kwargs)
	return hpcp_mod.compute_changes(new_map, base_map, max_workers=1)


def _ops(rows):
	return {(row.op, row.path) for row in rows}


def test_build_entry_map_keys(tmp_path, hpcp_mod, reset_hpcp_globals):
	root = tmp_path / 'r'
	_write(root / 'a.txt', 'a')
	_write(root / 'sub' / 'b.txt', 'b')
	os.symlink('a.txt', root / 'ln')
	assert set(hpcp_mod.build_entry_map(str(root), max_workers=1)) == {'./', 'a.txt', 'sub/', 'sub/b.txt', 'ln'}
	prefixed = hpcp_mod.build_entry_map(str(root), prefix='data/r/', max_workers=1)
	assert set(prefixed) == {'data/r/', 'data/r/a.txt', 'data/r/sub/', 'data/r/sub/b.txt', 'data/r/ln'}
	assert prefixed['data/r/sub/b.txt'] == str(root / 'sub' / 'b.txt')


def test_build_store_maps_parents_and_missing_subtree(tmp_path, hpcp_mod, reset_hpcp_globals):
	new = tmp_path / 'live' / 'proj'
	_write(new / 'a.txt', 'a')
	store = tmp_path / 'store'
	store.mkdir()
	new_map, base_map = hpcp_mod.build_store_maps(str(new), str(store), max_workers=1)
	prefix = os.path.relpath(str(new), '/').replace(os.sep, '/') + '/'
	assert new_map[prefix] == str(new)
	assert new_map[prefix + 'a.txt'] == str(new / 'a.txt')
	expected_parents = set()
	parent = os.path.dirname(str(new))
	while parent != '/':
		expected_parents.add(os.path.relpath(parent, '/') + '/')
		parent = os.path.dirname(parent)
	assert {key for key in new_map if not key.startswith(prefix)} == expected_parents
	assert base_map == {}  # the empty store has neither the subtree nor its parents


def test_compute_changes_add_modify_delete(trees, hpcp_mod):
	new, base = trees
	_write(new / 'same.txt', 'same')
	_write(base / 'same.txt', 'same')
	_write(new / 'mod.txt', 'new content')
	_write(base / 'mod.txt', 'old')
	_write(new / 'add.txt', 'added')
	_write(base / 'gone.txt', 'bye')
	rows = _changes(hpcp_mod, new, base)
	assert _ops(rows) == {('A', 'add.txt'), ('M', 'mod.txt'), ('D', 'gone.txt')}
	modified = [row for row in rows if row.op == 'M'][0]
	assert hpcp_mod.decode_state(modified.old)['size'] == 3
	assert hpcp_mod.decode_state(modified.new)['size'] == 11
	assert [row.op for row in rows] == ['D', 'A', 'M']


def test_compute_changes_metadata_only(trees, hpcp_mod):
	new, base = trees
	_write(new / 'mtime.txt', 'x', mtime=T0 + 100)
	_write(base / 'mtime.txt', 'x')
	_write(new / 'mode.sh', 'x')
	_write(base / 'mode.sh', 'x')
	os.chmod(new / 'mode.sh', 0o755)
	os.chmod(base / 'mode.sh', 0o644)
	(new / 'sub').mkdir()
	(base / 'sub').mkdir()
	os.chmod(new / 'sub', 0o700)
	os.chmod(base / 'sub', 0o755)
	# same size, same mtime, different bytes inside the hash window
	_write(new / 'same_size.bin', b'AAAA')
	_write(base / 'same_size.bin', b'BBBB')
	rows = _changes(hpcp_mod, new, base)
	assert _ops(rows) == {('M', 'mtime.txt'), ('M', 'mode.sh'), ('M', 'sub/'), ('M', 'same_size.bin')}


def test_compute_changes_xattr_only(trees, hpcp_mod):
	new, base = trees
	_write(new / 'x.txt', 'x')
	_write(base / 'x.txt', 'x')
	_set_xattr_or_skip(str(new / 'x.txt'))
	assert _ops(_changes(hpcp_mod, new, base)) == {('M', 'x.txt')}


def test_compute_changes_type_changes(trees, hpcp_mod):
	"""Review Focus 5: file <-> dir <-> symlink on one name."""
	new, base = trees
	_write(base / 'f2d', 'file')
	_write(new / 'f2d' / 'child.txt', 'c')
	_write(base / 'f2l', 'file')
	os.symlink('target', new / 'f2l')
	_write(base / 'd2l' / 'inner.txt', 'i')
	os.symlink('elsewhere', new / 'd2l')
	os.symlink('old', base / 'relink')
	os.symlink('new', new / 'relink')
	rows = _changes(hpcp_mod, new, base)
	assert _ops(rows) == {
		('D', 'f2d'), ('A', 'f2d/'), ('A', 'f2d/child.txt'),
		('D', 'f2l'), ('A', 'f2l'),
		('D', 'd2l/'), ('D', 'd2l/inner.txt'), ('A', 'd2l'),
		('M', 'relink'),
	}
	relink = [row for row in rows if row.path == 'relink'][0]
	assert (relink.old, relink.new) == ('l:old', 'l:new')


def test_compute_changes_honours_exclude_on_both_sides(trees, hpcp_mod):
	"""Review Focus 2: an absolute pattern written for the new tree also hides the base subtree."""
	new, base = trees
	_write(new / 'keep.txt', 'k')
	_write(base / 'keep.txt', 'k')
	_write(new / 'a.log', 'new log')
	_write(base / 'b.log', 'old log')
	_write(base / 'cache' / 'blob', 'cached')
	exclude = hpcp_mod.format_exclude(['*.log', str(new / 'cache')])
	assert _changes(hpcp_mod, new, base, exclude=exclude) == []


def test_store_layout_ignores_other_trees(tmp_path, hpcp_mod, reset_hpcp_globals):
	hpcp_mod.HASH_SIZE = 65536
	new = tmp_path / 'live' / 'a'
	other = tmp_path / 'live' / 'b'
	_write(new / 'f.txt', 'new')
	store = tmp_path / 'store'
	rel_new = os.path.relpath(str(new), '/')
	rel_other = os.path.relpath(str(other), '/')
	_write(store / rel_new / 'f.txt', 'old')
	_write(store / rel_other / 'g.txt', 'other tree')
	new_map, base_map = hpcp_mod.build_store_maps(str(new), str(store), max_workers=1)
	rows = hpcp_mod.compute_changes(new_map, base_map, max_workers=1)
	assert ('M', rel_new.replace(os.sep, '/') + '/f.txt') in _ops(rows)
	assert not any(row.path.startswith(rel_other.replace(os.sep, '/')) for row in rows)


def test_compute_changes_parallel_matches_serial(trees, hpcp_mod):
	new, base = trees
	for i in range(300):
		_write(new / f'f{i}.txt', f'new {i}')
		_write(base / f'f{i}.txt', f'old {i}' if i % 3 == 0 else f'new {i}')
	new_map, base_map = hpcp_mod.build_root_maps(str(new), str(base), max_workers=1)
	serial = hpcp_mod.compute_changes(new_map, base_map, max_workers=1)
	parallel = hpcp_mod.compute_changes(new_map, base_map, max_workers=4)
	assert serial == parallel and len(serial) == 100
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_diff_image.py -q -k "entry_map or store or compute_changes"`
Expected: FAIL with `AttributeError: module 'hpcp' has no attribute 'build_root_maps'` (and similar).

- [ ] **Step 3: Implement**

Append to the Diff Image section:

```python
def _logically_excluded(rel, logical_root, exclude):
	"""True when rel (relative to a listed root) or one of its ancestors matches exclude
	once placed under logical_root."""
	if rel in ('', '.'):
		return False
	parts = rel.split(os.sep)
	for depth in range(1, len(parts) + 1):
		if is_excluded(os.path.join(logical_root, *parts[:depth]), exclude):
			return True
	return False

def build_entry_map(root, prefix='', parents=None, exclude=None, logical_root=None,
		max_workers=multiprocessing.cpu_count(), parallel_file_listing=False):
	"""Map patch keys to absolute paths for one side of a diff.

	Keys are prefix + the path relative to root ('/'-separated, directories end with '/');
	root itself is keyed prefix, or './' when prefix is empty. parents adds extra
	{key: directory} entries (store-layout ancestors) when they exist. With logical_root,
	exclude is matched against logical_root + relative path instead of the physical path,
	so a base tree honours patterns written for the new tree. A missing root yields only
	the existing parents.
	"""
	entries = {}
	for key, path in (parents or {}).items():
		if os.path.isdir(path) and not os.path.islink(path):
			entries[key] = path
	root = os.path.abspath(root)
	if not os.path.isdir(root) or os.path.islink(root):
		return entries
	post_filter = bool(exclude) and logical_root is not None
	files, links, _, folders = get_file_list(root, exclude=None if post_filter else exclude, max_workers=max_workers,
		drop_cache=True, parallel_file_listing=parallel_file_listing)
	entries[prefix or './'] = root
	for paths, is_dir in ((files, False), (links, False), (folders, True)):
		for path in paths:
			rel = os.path.relpath(path, root)
			if rel == '.':
				continue
			if post_filter and _logically_excluded(rel, logical_root, exclude):
				continue
			entries[prefix + rel.replace(os.sep, '/') + ('/' if is_dir else '')] = path
	return entries

def _store_layout_parents(new_dir, store_root):
	"""Ancestors of new_dir except '/', as ({key: live directory}, {key: directory inside the store})."""
	new_parents, base_parents = {}, {}
	parent = os.path.dirname(os.path.abspath(new_dir))
	while parent != os.path.dirname(parent):
		rel = os.path.relpath(parent, os.sep)
		key = rel.replace(os.sep, '/') + '/'
		new_parents[key] = parent
		base_parents[key] = os.path.join(store_root, rel)
		parent = os.path.dirname(parent)
	return new_parents, base_parents

def build_store_maps(new_dir, store_root, exclude=None, max_workers=multiprocessing.cpu_count(), parallel_file_listing=False):
	"""Entry maps for a directory against a -di image store mounted at store_root ('store' layout).

	-di keeps files at their absolute source path, so keys are relative to '/' and the base
	side is limited to new_dir's own subtree (plus its parent directories) inside the store.
	"""
	new_dir = os.path.abspath(new_dir)
	rel = os.path.relpath(new_dir, os.sep)
	prefix = '' if rel == '.' else rel.replace(os.sep, '/') + '/'
	new_parents, base_parents = _store_layout_parents(new_dir, store_root)
	new_map = build_entry_map(new_dir, prefix, new_parents, exclude, None, max_workers, parallel_file_listing)
	base_root = store_root if rel == '.' else os.path.join(store_root, rel)
	base_map = build_entry_map(base_root, prefix, base_parents, exclude, new_dir, max_workers, parallel_file_listing)
	return new_map, base_map

def build_root_maps(new_root, base_root, exclude=None, max_workers=multiprocessing.cpu_count(), parallel_file_listing=False):
	"""Entry maps for two trees compared root to root ('root' layout)."""
	new_map = build_entry_map(new_root, '', None, exclude, None, max_workers, parallel_file_listing)
	base_map = build_entry_map(base_root, '', None, exclude, os.path.abspath(new_root), max_workers, parallel_file_listing)
	return new_map, base_map

def _states_chunk(paths, full_hash, read_xattrs):
	"""Worker body for _states_for_paths: states plus the errors recorded while computing them."""
	error_start = len(ERRORS)
	states = {path: get_entry_state(path, full_hash, read_xattrs) for path in paths}
	errors = ERRORS[error_start:]
	del ERRORS[error_start:]
	return states, errors

def _states_for_paths(paths, full_hash=False, read_xattrs=True, max_workers=1):
	"""{path: state} for every path; parallel chunks on a ProcessPoolExecutor for large sets."""
	paths = list(paths)
	if max_workers <= 1 or len(paths) < 256:
		return {path: get_entry_state(path, full_hash, read_xattrs) for path in paths}
	chunk_size = max(64, len(paths) // (max_workers * 4) + 1)
	states = {}
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		futures = [executor.submit(_states_chunk, paths[index:index + chunk_size], full_hash, read_xattrs)
			for index in range(0, len(paths), chunk_size)]
		for future in concurrent.futures.as_completed(futures):
			chunk_states, chunk_errors = future.result()
			states.update(chunk_states)
			ERRORS.extend(chunk_errors)
	return states

def compute_changes(new_map, base_map, full_hash=False, max_workers=multiprocessing.cpu_count(), read_xattrs=True):
	"""Rows that turn base_map's tree into new_map's tree, sorted D / A / M."""
	states = _states_for_paths(set(new_map.values()) | set(base_map.values()), full_hash=full_hash,
		read_xattrs=read_xattrs, max_workers=max_workers)
	rows = []
	for key in set(new_map) | set(base_map):
		new_state = states[new_map[key]] if key in new_map else '-'
		old_state = states[base_map[key]] if key in base_map else '-'
		if new_state == '-' and old_state == '-':
			continue
		if old_state == '-':
			rows.append(ChangeRow('A', key, '-', new_state))
		elif new_state == '-':
			rows.append(ChangeRow('D', key, old_state, '-'))
		elif old_state[0] != new_state[0]:
			rows.append(ChangeRow('D', key, old_state, '-'))
			rows.append(ChangeRow('A', key, '-', new_state))
		elif not states_equal(old_state, new_state):
			rows.append(ChangeRow('M', key, old_state, new_state))
	return sort_change_rows(rows)

```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_diff_image.py -q`
Expected: all pass (the xattr test may be SKIPPED).

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_diff_image.py
git commit -m "feat(diff): build entry maps and compute change rows

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: `-dfl` / `-cfl` / `-tdfl` on the shared format

**Files:**
- Modify: `hpcp.py` (`_compare_file_lists`, currently near line 3674; `store_file_list`, currently near line 3944; `process_compare_file_list`, currently near line 4187; `hpcp()` around the `mount_src_image` and `process_compare_file_list` calls; Diff Image section)
- Test: `tests/test_file_lists.py`

**Interfaces:**
- Consumes: `build_entry_map`, `build_store_maps`, `compute_changes`, `_states_for_paths`, `write_changes`, `sort_change_rows`, `decode_state`, `ChangeRow`, `load_file_list` (existing).
- Produces:
  - `report_changes(rows, diff_file_list=None, tar_diff_file_list=False) -> None`
  - `_parse_file_list_lines(lines, live_keys=()) -> dict[key, value|None]`
  - `_rows_from_list_entries(new_entries, base_entries, new_states=None) -> list[ChangeRow]`
  - `_value_from_state(state) -> str | None`
  - `_trim_prefix(path) -> str`
  - `_compare_file_lists(file_list, file_list2, diff_file_list=None, tar_diff_file_list=False) -> list[ChangeRow]` (public alias `compare_file_list` unchanged)
  - `process_compare_file_list(..., src_image_roots=())` (new keyword)

- [ ] **Step 1: Update and add the tests**

In `tests/test_file_lists.py`, add at the top (after the imports):

```python
import time


def _h(hpcp_mod, char):
	"""A digest-shaped string for the current hash algorithm."""
	return char * (16 if hpcp_mod.xxhash_available else 128)
```

Replace `test_compare_file_list_writes_diff` and `test_tar_diff_file_list_is_new_or_updated_only` with:

```python
def test_compare_file_list_writes_diff(tmp_path, hpcp_mod, reset_hpcp_globals):
	diff = tmp_path / 'diff.txt'
	a = _h(hpcp_mod, 'a')
	src = {f'a.txt:{a}', f'only_src.txt:{a}'}
	dst = {f'a.txt:{a}', f'only_dst.txt:{a}'}
	hpcp_mod.compare_file_list(src, dst, diff_file_list=str(diff), tar_diff_file_list=False)
	rows = hpcp_mod.read_changes(str(diff))
	assert {(r.op, r.path) for r in rows} == {('A', 'only_src.txt'), ('D', 'only_dst.txt')}


def test_tar_diff_file_list_is_new_or_updated_only(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Help text: tar-compatible diff lists update/new files only (in src not dest)."""
	diff = tmp_path / 'tar.txt'
	a, b, c = _h(hpcp_mod, 'a'), _h(hpcp_mod, 'b'), _h(hpcp_mod, 'c')
	src = {f'new.txt:{a}', f'updated.txt:{b}', f'same.txt:{a}'}
	dst = {f'same.txt:{a}', f'updated.txt:{c}', f'extra.txt:{a}'}
	hpcp_mod.compare_file_list(src, dst, diff_file_list=str(diff), tar_diff_file_list=True)
	assert set(diff.read_text().splitlines()) == {'new.txt', 'updated.txt'}
```

In `test_compare_stored_list_preserves_baseline_and_writes_diff`, replace the last line:

```python
	assert diff.read_bytes() == b'-\x00new.txt\n'
```

with:

```python
	rows = hpcp_mod.read_changes(str(diff))
	assert [(r.op, r.path) for r in rows] == [('A', 'new.txt')]
	assert rows[0].new.startswith('f:')
```

Append:

```python
def test_diff_file_list_keeps_names_with_colons(tmp_tree, hpcp_mod, reset_hpcp_globals):
	"""Regression: -dfl cut every entry at its last ':' (directory 'a:b/' became 'a')."""
	tmp_tree.add_file('a:b/file:1.txt', 'x')
	tmp_tree.add_file('plain.txt', 'p')
	baseline = tmp_tree.root / 'baseline.txt'
	baseline.write_text('./\nplain.txt\nold:dir/\n')
	diff = tmp_tree.root / 'diff.tsv'
	rc = hpcp_mod.hpcp(
		[str(tmp_tree.src) + os.sep], target_file_list=str(baseline), diff_file_list=str(diff),
		append_hash_to_file_list=False, single_thread=True,
	)
	assert rc == 0
	ops = {(r.op, r.path) for r in hpcp_mod.read_changes(str(diff))}
	assert ops == {('A', 'a:b/'), ('A', 'a:b/file:1.txt'), ('D', 'old:dir/')}


def test_hashed_list_symlink_lines_parse_to_their_path(hpcp_mod, reset_hpcp_globals):
	a = _h(hpcp_mod, 'a')
	lines = [f'f.txt:{a}', 'ln:target:with:colons', 'dir/']
	entries = hpcp_mod._parse_file_list_lines(lines)
	assert entries == {'f.txt': a, 'ln': 'target:with:colons', 'dir/': None}
	# a live key that itself contains ':' wins over the first-colon split
	entries = hpcp_mod._parse_file_list_lines([f'f.txt:{a}', 'my:link:dest'], live_keys={'my:link'})
	assert entries['my:link'] == 'dest'


def test_hpcp_diff_file_list_against_dest_reports_changes(tmp_tree, hpcp_mod, reset_hpcp_globals, copy_args, require_linux, capsys):
	t = time.time() - 1000
	for path in (tmp_tree.add_file('keep.txt', 'k', under='src'), tmp_tree.add_file('keep.txt', 'k', under='dst')):
		os.utime(path, (t, t))
	tmp_tree.add_file('changed.txt', 'new!', under='src')
	tmp_tree.add_file('changed.txt', 'old', under='dst')
	tmp_tree.add_file('only_src.txt', 's', under='src')
	tmp_tree.add_file('only_dst.txt', 'd', under='dst')
	diff = str(tmp_tree.root / 'cmp.tsv')
	srcs, opts = copy_args(diff_file_list=diff)
	assert hpcp_mod.hpcp(srcs, **opts) == 0
	ops = {(r.op, r.path) for r in hpcp_mod.read_changes(diff)}
	assert ops == {('A', 'only_src.txt'), ('M', 'changed.txt'), ('D', 'only_dst.txt')}
	assert 'Modified:' in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_file_lists.py -q`
Expected: FAIL. The rewritten and new tests fail: `read_changes` rejects the old `-\0` output as `Patch format error`, and `_parse_file_list_lines` is missing.

- [ ] **Step 3: Implement**

Replace the whole `_compare_file_lists` function (keep the alias line `compare_file_list = _compare_file_lists` below it) with:

```python
def _compare_file_lists(file_list, file_list2, diff_file_list=None, tar_diff_file_list = False):
	"""Compare two -fl style entry sets (src, dest) and report the rows that turn dest into src."""
	new_entries = _parse_file_list_lines(file_list)
	base_entries = _parse_file_list_lines(file_list2, live_keys=new_entries)
	print('-'*80)
	print(f"Number of files in src: {len(new_entries)}")
	print(f"Number of files in dest: {len(base_entries)}")
	rows = _rows_from_list_entries(new_entries, base_entries)
	report_changes(rows, diff_file_list, tar_diff_file_list)
	return rows
```

Append to the Diff Image section:

```python
def report_changes(rows, diff_file_list=None, tar_diff_file_list=False):
	"""Print the A / M / D lists and counts; optionally write diff_file_list.

	The file is changes.tsv format, or with tar_diff_file_list a plain newline list of
	added / modified paths for `tar -T` (the root './' is left out).
	"""
	labels = (('A', 'Added (in src, not in dest)'), ('M', 'Modified'), ('D', 'Deleted (in dest, not in src)'))
	print('-'*80)
	for op, label in labels:
		print(f"{label}:")
		for row in rows:
			if row.op == op:
				print(row.path)
		print('-'*80)
	for op, label in labels:
		print(f"{label} count: {sum(1 for row in rows if row.op == op)}")
	if diff_file_list:
		if tar_diff_file_list:
			with open(diff_file_list, 'w', encoding='utf-8', errors='surrogateescape', newline='\n') as f:
				for row in rows:
					if row.op in ('A', 'M') and row.path != './':
						f.write(row.path + '\n')
		else:
			write_changes(diff_file_list, rows)
		print(f"Diff file list written to {diff_file_list}")

def _hash_hex_re():
	return re.compile(r'[0-9a-f]{16}' if xxhash_available else r'[0-9a-f]{128}')

def _parse_file_list_lines(lines, live_keys=()):
	"""Parse -fl style lines into {key: value}.

	value is the hash for files, the link target for symlinks, None when the list holds
	neither. A line ending in '/' is a directory. A line ending in ':<digest>' is a file.
	In a hashed list (any line has a digest) the remaining lines are symlinks 'path:target':
	the longest live key followed by ':' is the path, else the text before the first ':'.
	"""
	hex_re = _hash_hex_re()
	lines = [line for line in lines if line]
	hashed = any(':' in line and not line.endswith('/') and hex_re.fullmatch(line.rpartition(':')[2]) for line in lines)
	live_keys = set(live_keys)
	entries = {}
	for line in lines:
		if line.endswith('/'):
			entries[line] = None
			continue
		path, sep, tail = line.rpartition(':')
		if sep and hex_re.fullmatch(tail):
			entries[path] = tail
			continue
		if hashed and ':' in line:
			key = None
			for index, char in enumerate(line):
				if char == ':' and line[:index] in live_keys:
					key = line[:index]
			if key is None:
				key = line.partition(':')[0]
			entries[key] = line[len(key) + 1:]
			continue
		entries[line] = None
	return entries

def _value_from_state(state):
	"""What a -fl list records for an entry: file hash, link target, or None."""
	info = decode_state(state)
	if info is None or info['type'] in ('d', '?'):
		return None
	if info['type'] == 'l':
		return info['target']
	return None if info['hash'] == '-' else info['hash']

def _rows_from_list_entries(new_entries, base_entries, new_states=None):
	"""Rows from -fl style entries; base states are unknown ('?')."""
	new_states = new_states or {}
	rows = []
	for key in set(new_entries) | set(base_entries):
		if key not in base_entries:
			rows.append(ChangeRow('A', key, '-', new_states.get(key, '?')))
		elif key not in new_entries:
			rows.append(ChangeRow('D', key, '?', '-'))
		else:
			new_value, base_value = new_entries[key], base_entries[key]
			if new_value is not None and base_value is not None and new_value != base_value:
				rows.append(ChangeRow('M', key, '?', new_states.get(key, '?')))
	return sort_change_rows(rows)

def _trim_prefix(path):
	"""Key prefix matching trim_paths(): a path without a trailing separator contributes its folder name."""
	if path.endswith(os.sep):
		return ''
	name = os.path.basename(os.path.normpath(path))
	return '' if name in ('', '.') else name + '/'

```

In `store_file_list`, replace everything from the line `	fileList = set()` to the end of the function with:

```python
	if compare_file_list or diff_file_list:
		# This means we have a file_list and a src_path so we compare them
		print(f"Comparing file list from {src_paths} with {file_list}")
		if diff_file_list == 'auto':
			if not src_str:
				src_str = '-'.join([os.path.basename(os.path.realpath(src)) for src in src_paths])
			diff_file_list = f'DIFF_{src_str}_TO_{os.path.basename(os.path.realpath(file_list))}_{int(time.time())}_{"tar_" if tar_diff_file_list else ""}file_list.txt'
		workers = 1 if single_thread else max_workers
		new_map = {}
		for src in src_paths:
			new_map.update(build_entry_map(src, _trim_prefix(src), exclude=exclude, max_workers=workers, parallel_file_listing=parallel_file_listing))
		states = _states_for_paths(set(new_map.values()), full_hash=full_hash, max_workers=workers)
		new_states = {key: states[path] for key, path in new_map.items()}
		new_entries = {key: _value_from_state(state) for key, state in new_states.items()}
		base_entries = _parse_file_list_lines(load_file_list(file_list), live_keys=new_entries)
		print(f"Number of entries in src: {len(new_entries)}")
		print(f"Number of entries in {file_list}: {len(base_entries)}")
		report_changes(_rows_from_list_entries(new_entries, base_entries, new_states), diff_file_list, tar_diff_file_list)
	else:
		fileList = set()
		for src in src_paths:
			files, links, _,folders  = get_file_list(src, max_workers=max_workers, exclude=exclude,append_hash=append_hash,full_hash=full_hash,parallel_file_listing=parallel_file_listing)
			fileList.update(trim_paths(files, src))
			fileList.update(trim_paths(links, src))
			fileList.update([folder_path + os.path.sep for folder_path in trim_paths(folders, src)])
		print(f"Number of files: {len(fileList)}")
		print(f"Writing file list to {fileList}")
		# sort the file list
		fileList = natural_sort(fileList)
		with open(file_list, 'w') as f:
			for file in fileList:
				f.write(file + '\n')
```

Replace the whole `process_compare_file_list` function with:

```python
def process_compare_file_list(src_paths: list, dests, max_workers = multiprocessing.cpu_count(),
							  parallel_file_listing = False,exclude=None,dest_image = None,diff_file_list = None,tar_diff_file_list = False,
							  append_hash = True,full_hash = False,src_image_roots = ()):
	# append_hash is kept for signature compatibility: live comparisons always use full entry states.
	dests = [dest for dest in dests if dest and os.path.exists(dest) and os.path.isdir(dest)]
	if not dests:
		raise RuntimeError(f"No destination paths in cpl: Destination image {dest_image} does not exist or dests {dests} is empty when comparing file lists, exiting")
	print(f"Comparing file list from {src_paths} with {dests}")
	new_map, base_map = {}, {}
	if dest_image:
		for src in src_paths:
			if src in src_image_roots:
				new_map.update(build_entry_map(src, '', exclude=exclude, max_workers=max_workers, parallel_file_listing=parallel_file_listing))
				base_map.update(build_entry_map(dests[0], '', exclude=exclude, logical_root=os.path.abspath(src), max_workers=max_workers, parallel_file_listing=parallel_file_listing))
			else:
				src_new, src_base = build_store_maps(src, dests[0], exclude=exclude, max_workers=max_workers, parallel_file_listing=parallel_file_listing)
				new_map.update(src_new)
				base_map.update(src_base)
	else:
		for src in src_paths:
			new_map.update(build_entry_map(src, _trim_prefix(src), exclude=exclude, max_workers=max_workers, parallel_file_listing=parallel_file_listing))
		for dest in dests:
			base_map.update(build_entry_map(dest, _trim_prefix(dest), exclude=exclude, max_workers=max_workers, parallel_file_listing=parallel_file_listing))
	print('-'*80)
	rows = compute_changes(new_map, base_map, full_hash=full_hash, max_workers=max_workers)
	report_changes(rows, diff_file_list, tar_diff_file_list)
```

In `hpcp()`:
- after `src_paths = []` add `src_image_roots = []`;
- after the line `src_str = mount_src_image(src_images,src_paths,mount_points,loop_devices)` add `src_image_roots = list(src_paths)` (same indentation);
- in the `process_compare_file_list(...)` call, add `src_image_roots=src_image_roots` after `full_hash=full_hash`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_file_lists.py tests/test_orchestrator_smoke.py tests/test_diff_image.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_file_lists.py
git commit -m "feat(dfl): write -dfl in the changes.tsv format and stop truncating ':' names

-dfl, -cfl and -tdfl now share the patch-image change rows. Entries are keyed
by path instead of re-splitting 'path:hash' at the last ':', which cut
directory names and symlink targets containing ':'.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Build a patch tree

**Files:**
- Modify: `hpcp.py` (add `import uuid` after `import time`; Diff Image section)
- Test: `tests/test_diff_image.py`

**Interfaces:**
- Consumes: `copy_file_list_parallel(file_list, links, src_path, dest_paths, max_workers, full_hash=False, verbose=False, files_per_job=1, estimated_size=0)` (existing), `sync_directory_metadata(src_path, dest_paths)` (existing), `write_changes`, `write_manifest`, `decode_state`, `HASH_SIZE`, `version`.
- Produces:
  - `_is_reserved_key(key) -> bool`
  - `_key_path(root, key) -> str` (`'./'` → `normpath(root)`)
  - `_key_depth(key) -> int`, `_parent_keys(key) -> list[str]` (shallowest first)
  - `_remove_entry(path) -> None`, `_place_symlink(target, dest) -> None`
  - `make_patch_meta(rows, new_name, base_name, layout, full_hash=False) -> dict` (`patch_id`, counts as `int`, `payload_bytes`, `sections`)
  - `write_patch_metadata(volume_roots, rows, meta) -> None` (per volume: `changes.tsv`, then `manifest` with `volume k/n`)
  - `build_patch_tree(rows, new_base, volume_roots, meta, max_workers=1, verbose=False) -> None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_diff_image.py`:

```python
def _patch_from(hpcp_mod, new, base, volumes, layout='root'):
	new_map, base_map = hpcp_mod.build_root_maps(str(new), str(base), max_workers=1)
	rows = hpcp_mod.compute_changes(new_map, base_map, max_workers=1)
	meta = hpcp_mod.make_patch_meta(rows, 'new', 'base', layout)
	for volume in volumes:
		os.makedirs(str(volume), exist_ok=True)
	hpcp_mod.build_patch_tree(rows, str(new), [str(v) for v in volumes], meta, max_workers=1)
	return rows, meta


def test_make_patch_meta_counts(hpcp_mod):
	R = hpcp_mod.ChangeRow
	rows = [
		R('A', 'a', '-', 'f:5:1:100644:0:0:aa:-'),
		R('M', 'b', 'f:1:1:100644:0:0:aa:-', 'f:7:1:100644:0:0:bb:-'),
		R('D', 'c', 'f:1:1:100644:0:0:aa:-', '-'),
		R('A', 'd/', '-', 'd:40755:0:0:-'),
	]
	meta = hpcp_mod.make_patch_meta(rows, 'new', 'base', 'root')
	assert (meta['added'], meta['modified'], meta['deleted'], meta['payload_bytes']) == (2, 1, 1, 12)
	assert len(meta['patch_id']) == 32 and meta['hash_algo'] in ('xxh64', 'blake2b')
	assert meta['sections'] == [('root', '.', 'changes.tsv')] and meta['layout'] == 'root'


def test_build_patch_tree_writes_payload_and_metadata(trees, hpcp_mod, tmp_path):
	new, base = trees
	_write(new / 'add.txt', 'added')
	_write(new / 'mod.txt', 'new content')
	_write(base / 'mod.txt', 'old')
	_write(base / 'gone.txt', 'bye')
	_write(new / 'same.txt', 's')
	_write(base / 'same.txt', 's')
	_write(new / 'deep' / 'er' / 'f.txt', 'deep')
	os.chmod(new / 'deep', 0o700)
	os.symlink('add.txt', new / 'ln')
	vol = tmp_path / 'vol'
	rows, meta = _patch_from(hpcp_mod, new, base, [vol])
	assert (vol / 'add.txt').read_text() == 'added'
	assert (vol / 'mod.txt').read_text() == 'new content'
	assert os.stat(vol / 'mod.txt').st_mtime_ns == os.stat(new / 'mod.txt').st_mtime_ns
	assert (vol / 'deep' / 'er' / 'f.txt').read_text() == 'deep'
	assert stat.S_IMODE(os.stat(vol / 'deep').st_mode) == 0o700
	assert os.readlink(vol / 'ln') == 'add.txt'
	assert not (vol / 'gone.txt').exists() and not (vol / 'same.txt').exists()
	assert hpcp_mod.read_changes(str(vol / '.hpcp_diff' / 'changes.tsv')) == rows
	got = hpcp_mod.read_manifest(str(vol / '.hpcp_diff' / 'manifest'))
	assert got['volume'] == '1/1' and got['patch_id'] == meta['patch_id']
	assert (got['added'], got['modified'], got['deleted']) == (str(meta['added']), str(meta['modified']), str(meta['deleted']))
	assert got['layout'] == 'root' and got['sections'] == [('root', '.', 'changes.tsv')]


def test_build_patch_tree_two_volumes_share_patch_id(trees, hpcp_mod, tmp_path):
	new, base = trees
	for i in range(6):
		_write(new / f'f{i}.txt', f'file {i}')
	volumes = [tmp_path / 'v1', tmp_path / 'v2']
	_patch_from(hpcp_mod, new, base, volumes)
	metas = [hpcp_mod.read_manifest(str(v / '.hpcp_diff' / 'manifest')) for v in volumes]
	assert [m['volume'] for m in metas] == ['1/2', '2/2']
	assert metas[0]['patch_id'] == metas[1]['patch_id']
	for i in range(6):
		assert sum((v / f'f{i}.txt').exists() for v in volumes) == 1


def test_build_patch_tree_rejects_reserved_name(trees, hpcp_mod, tmp_path):
	new, base = trees
	_write(new / '.hpcp_diff' / 'x', 'nope')
	with pytest.raises(RuntimeError, match='reserved'):
		_patch_from(hpcp_mod, new, base, [tmp_path / 'vol'])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_diff_image.py -q -k "patch_meta or build_patch_tree"`
Expected: FAIL with `AttributeError: module 'hpcp' has no attribute 'make_patch_meta'`.

- [ ] **Step 3: Implement**

Add `import uuid` after `import time` in the import block.

Append to the Diff Image section:

```python
def _is_reserved_key(key):
	return key in (DIFF_DIR_NAME, DIFF_DIR_NAME + '/') or key.startswith(DIFF_DIR_NAME + '/')

def _key_path(root, key):
	"""Absolute path of key under root ('./' is root itself)."""
	parts = [part for part in key.rstrip('/').split('/') if part not in ('', '.')]
	return os.path.normpath(os.path.join(root, *parts)) if parts else os.path.normpath(root)

def _key_depth(key):
	return 0 if key == './' else key.rstrip('/').count('/') + 1

def _parent_keys(key):
	"""Directory keys above key, shallowest first: 'a/b/c' -> ['a/', 'a/b/']."""
	parts = key.rstrip('/').split('/')
	return ['/'.join(parts[:index]) + '/' for index in range(1, len(parts))]

def _remove_entry(path):
	"""Remove a file, symlink or whole directory tree; a missing path is fine."""
	try:
		if os.path.isdir(path) and not os.path.islink(path):
			shutil.rmtree(path)
		else:
			os.unlink(path)
	except FileNotFoundError:
		pass

def _place_symlink(target, dest):
	if os.path.lexists(dest):
		_remove_entry(dest)
	os.makedirs(os.path.dirname(dest), exist_ok=True)
	os.symlink(target, dest)

def _payload_bytes(rows):
	return sum(decode_state(row.new)['size'] for row in rows if row.op in ('A', 'M') and row.new.startswith('f:'))

def make_patch_meta(rows, new_name, base_name, layout, full_hash=False):
	"""Manifest fields for a new patch; 'volume' is filled in per volume by write_patch_metadata()."""
	return {
		'format': 'hpcp-diff',
		'format_version': DIFF_FORMAT_VERSION,
		'patch_id': uuid.uuid4().hex,
		'volume': '1/1',
		'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
		'hpcp_version': version,
		'new': new_name,
		'base': base_name,
		'hash_algo': _hash_algo_name(),
		'hash_size': HASH_SIZE,
		'full_hash': int(bool(full_hash)),
		'xattrs': int(hasattr(os, 'listxattr')),
		'layout': layout,
		'added': sum(1 for row in rows if row.op == 'A'),
		'modified': sum(1 for row in rows if row.op == 'M'),
		'deleted': sum(1 for row in rows if row.op == 'D'),
		'payload_bytes': _payload_bytes(rows),
		'sections': [('root', '.', 'changes.tsv')],
	}

def write_patch_metadata(volume_roots, rows, meta):
	"""Write .hpcp_diff/changes.tsv then .hpcp_diff/manifest on every volume (manifest last)."""
	total = len(volume_roots)
	for index, volume_root in enumerate(volume_roots, start=1):
		diff_dir = os.path.join(volume_root, DIFF_DIR_NAME)
		os.makedirs(diff_dir, exist_ok=True)
		write_changes(os.path.join(diff_dir, 'changes.tsv'), rows)
		volume_meta = dict(meta)
		volume_meta['volume'] = f'{index}/{total}'
		write_manifest(os.path.join(diff_dir, 'manifest'), volume_meta)

def build_patch_tree(rows, new_base, volume_roots, meta, max_workers=1, verbose=False):
	"""Copy the A / M payload of rows from new_base onto the volume roots, then write .hpcp_diff/.

	Every map key is relpath(path, new_base), so new_base is '/' for store layout and the
	new tree's root for root layout.
	"""
	for row in rows:
		if _is_reserved_key(row.path):
			raise RuntimeError(f'Diff image error: {row.path} uses the reserved name {DIFF_DIR_NAME}')
	payload = [row for row in rows if row.op in ('A', 'M')]
	dir_keys = set()
	for row in payload:
		dir_keys.update(_parent_keys(row.path))
		if row.new.startswith('d:'):
			dir_keys.add(row.path)
	for key in sorted(dir_keys, key=_key_depth):
		for volume_root in volume_roots:
			os.makedirs(_key_path(volume_root, key), exist_ok=True)
	files = [_key_path(new_base, row.path) for row in payload if row.new.startswith('f:')]
	if files:
		copy_file_list_parallel(files, [], new_base, list(volume_roots), max(1, max_workers), verbose=verbose,
			estimated_size=meta.get('payload_bytes', 0))
	for row in payload:
		if row.new.startswith('l:'):
			_place_symlink(decode_state(row.new)['target'], _key_path(volume_roots[0], row.path))
	# Directories last and deepest first, so later writes cannot disturb their mtimes.
	for key in sorted(dir_keys, key=_key_depth, reverse=True):
		sync_directory_metadata(_key_path(new_base, key), [_key_path(volume_root, key) for volume_root in volume_roots])
	write_patch_metadata(volume_roots, rows, meta)

```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_diff_image.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_diff_image.py
git commit -m "feat(diff): write patch payload and .hpcp_diff metadata to volumes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Check a target against a patch

**Files:**
- Modify: `hpcp.py` (Diff Image section)
- Test: `tests/test_diff_image.py`

**Interfaces:**
- Consumes: `_states_for_paths`, `states_equal`, `_key_path`, `ChangeRow`.
- Produces:
  - `PatchCheck = namedtuple('PatchCheck', 'row action found reason')` (`action` ∈ `'apply' | 'skip' | 'conflict'`)
  - `check_patch_target(rows, target_root, compare_hash=True, full_hash=False, max_workers=1) -> list[PatchCheck]` (same order as `rows`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_diff_image.py`:

```python
def _clone(source, target):
	shutil.copytree(str(source), str(target), symlinks=True)


def _actions(checks):
	return {(check.row.op, check.row.path): check.action for check in checks}


def test_check_patch_target_classifies_rows(trees, hpcp_mod, tmp_path):
	new, base = trees
	_write(new / 'mod.txt', 'new content')
	_write(base / 'mod.txt', 'old')
	_write(base / 'gone.txt', 'bye')
	_write(new / 'add.txt', 'added')
	_write(base / 'f2d', 'file')
	_write(new / 'f2d' / 'child.txt', 'c')
	rows = _changes(hpcp_mod, new, base)
	target = tmp_path / 'target'
	_clone(base, target)
	assert set(_actions(hpcp_mod.check_patch_target(rows, str(target))).values()) == {'apply'}
	_write(target / 'mod.txt', 'local edit')
	_write(target / 'add.txt', 'someone else')
	os.unlink(target / 'gone.txt')
	actions = _actions(hpcp_mod.check_patch_target(rows, str(target)))
	assert actions[('M', 'mod.txt')] == 'conflict'
	assert actions[('A', 'add.txt')] == 'conflict'
	assert actions[('D', 'gone.txt')] == 'skip'
	assert actions[('D', 'f2d')] == 'apply'
	assert actions[('A', 'f2d/')] == 'apply' and actions[('A', 'f2d/child.txt')] == 'apply'


def test_check_patch_target_flags_unlisted_children(trees, hpcp_mod, tmp_path):
	new, base = trees
	_write(base / 'olddir' / 'a.txt', 'a')
	rows = _changes(hpcp_mod, new, base)
	target = tmp_path / 'target'
	_clone(base, target)
	_write(target / 'olddir' / 'local.txt', 'mine')
	actions = _actions(hpcp_mod.check_patch_target(rows, str(target)))
	assert actions[('D', 'olddir/')] == 'conflict'
	assert actions[('D', 'olddir/a.txt')] == 'apply'


def test_check_patch_target_skips_already_applied(trees, hpcp_mod, tmp_path):
	new, base = trees
	_write(base / 'f2d', 'file')
	_write(new / 'f2d' / 'child.txt', 'c')
	_write(new / 'mod.txt', 'new content')
	_write(base / 'mod.txt', 'old')
	rows = _changes(hpcp_mod, new, base)
	target = tmp_path / 'target'
	_clone(new, target)
	assert set(_actions(hpcp_mod.check_patch_target(rows, str(target))).values()) == {'skip'}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_diff_image.py -q -k check_patch_target`
Expected: FAIL with `AttributeError: module 'hpcp' has no attribute 'check_patch_target'`.

- [ ] **Step 3: Implement**

Append to the Diff Image section:

```python
PatchCheck = namedtuple('PatchCheck', 'row action found reason')

def _under_removed(key, removed_names):
	"""True when key or one of its ancestors is a name some D row will remove."""
	parts = key.rstrip('/').split('/')
	return any('/'.join(parts[:index]) in removed_names for index in range(1, len(parts) + 1))

def _unlisted_children(key, path, deleted_keys):
	"""Children of directory path whose keys are not in deleted_keys."""
	prefix = '' if key == './' else key
	try:
		names = os.listdir(path)
	except OSError:
		return []
	unlisted = []
	for name in names:
		child = os.path.join(path, name)
		child_key = prefix + name + ('/' if os.path.isdir(child) and not os.path.islink(child) else '')
		if child_key not in deleted_keys:
			unlisted.append(child_key)
	return unlisted

def check_patch_target(rows, target_root, compare_hash=True, full_hash=False, max_workers=1):
	"""Classify every row against target_root as 'apply', 'skip' or 'conflict'.

	D: target == old -> apply; absent, or already the replacing A row's state -> skip.
	A: absent -> apply; == new -> skip. M: == old -> apply; == new -> skip.
	Anything else is a conflict, as is a D directory holding entries the patch does not delete.
	A and M rows are judged as the target will be after the applicable D rows: a name a D
	row removes (with or without the trailing '/'), and everything below it, counts as absent.
	"""
	paths = {row.path: _key_path(target_root, row.path) for row in rows}
	states = _states_for_paths(set(paths.values()), full_hash=full_hash, max_workers=max_workers)
	added_by_name = {row.path.rstrip('/'): row for row in rows if row.op == 'A'}
	deleted_keys = {row.path for row in rows if row.op == 'D'}
	checks = {}
	removed_names = set()
	for row in rows:
		if row.op != 'D':
			continue
		found = states[paths[row.path]]
		replacement = added_by_name.get(row.path.rstrip('/'))
		if found == '-':
			checks[row] = PatchCheck(row, 'skip', found, 'already absent')
			continue
		if replacement is not None and states_equal(found, replacement.new, compare_hash):
			checks[row] = PatchCheck(row, 'skip', found, 'already replaced')
			continue
		removed_names.add(row.path.rstrip('/'))
		if not states_equal(found, row.old, compare_hash):
			checks[row] = PatchCheck(row, 'conflict', found, 'changed on target')
		elif found.startswith('d:') and _unlisted_children(row.path, paths[row.path], deleted_keys):
			checks[row] = PatchCheck(row, 'conflict', found, 'directory holds entries the patch does not delete')
		else:
			checks[row] = PatchCheck(row, 'apply', found, '')
	for row in rows:
		if row.op == 'D':
			continue
		found = '-' if _under_removed(row.path, removed_names) else states[paths[row.path]]
		if row.op == 'A':
			if found == '-':
				checks[row] = PatchCheck(row, 'apply', found, '')
			elif states_equal(found, row.new, compare_hash):
				checks[row] = PatchCheck(row, 'skip', found, 'already in new state')
			else:
				checks[row] = PatchCheck(row, 'conflict', found, 'already exists on target')
		elif states_equal(found, row.old, compare_hash):
			checks[row] = PatchCheck(row, 'apply', found, '')
		elif states_equal(found, row.new, compare_hash):
			checks[row] = PatchCheck(row, 'skip', found, 'already in new state')
		else:
			checks[row] = PatchCheck(row, 'conflict', found, 'missing on target' if found == '-' else 'changed on target')
	return [checks[row] for row in rows]

```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_diff_image.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_diff_image.py
git commit -m "feat(diff): check a target against a patch before applying

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Apply a patch tree

**Files:**
- Modify: `hpcp.py` (Diff Image section)
- Test: `tests/test_diff_image.py`

**Interfaces:**
- Consumes: `read_manifest`, `read_changes`, `check_patch_target`, `PatchCheck`, `_key_path`, `_key_depth`, `_remove_entry`, `_place_symlink`, `_states_for_paths`, `states_equal`, `sort_change_rows`, `decode_state`, `_split_dest_image_name(dest_image, index)` (existing), `copy_file_list_parallel`, `sync_directory_metadata`, `multiCMD.input_with_timeout_and_countdown(timeout, prompt)` (returns the stripped input, or `None` on timeout).
- Produces:
  - `load_patch_volumes(volume_roots, volume_names=None) -> (meta: dict, rows: list[ChangeRow], ordered_roots: list[str])`
  - `apply_patch_tree(volume_roots, target_root, force=False, max_workers=1, verbose=False, volume_names=None, store_note=False) -> dict` with keys `deleted, added, modified, skipped, forced, failed`
  - helpers `_resolve_conflicts`, `_inside_root`, `_apply_rows`, `_verify_rows`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_diff_image.py`:

```python
ODD_NAMES = ('with space.txt', 'tab\tname', 'new\nline', os.fsdecode(b'bad\xffbyte'), 'colon:name')


def _build_scenario(new, base):
	_write(new / 'same.txt', 'same')
	_write(base / 'same.txt', 'same')
	_write(new / 'mod.txt', 'new content')
	_write(base / 'mod.txt', 'old')
	_write(new / 'same_size.bin', b'NEW!')
	_write(base / 'same_size.bin', b'OLD!')
	_write(base / 'gone.txt', 'bye')
	_write(base / 'olddir' / 'a.txt', 'a')
	_write(base / 'olddir' / 'sub' / 'b.txt', 'b')
	_write(new / 'newdir' / 'sub' / 'f.txt', 'fresh')
	_write(new / 'run.sh', '#!/bin/sh')
	_write(base / 'run.sh', '#!/bin/sh')
	os.chmod(new / 'run.sh', 0o755)
	os.chmod(base / 'run.sh', 0o644)
	os.symlink('same.txt', new / 'ln')
	os.symlink('mod.txt', base / 'ln')
	_write(base / 'f2d', 'file')
	_write(new / 'f2d' / 'child.txt', 'c')
	_write(base / 'f2l', 'file')
	os.symlink('same.txt', new / 'f2l')
	_write(base / 'd2l' / 'inner.txt', 'i')
	os.symlink('same.txt', new / 'd2l')
	for name in ODD_NAMES:
		_write(new / name, name)


def test_apply_patch_tree_round_trip(trees, hpcp_mod, tmp_path):
	"""Success criterion + Review Focus 3, 4, 5."""
	new, base = trees
	_build_scenario(new, base)
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert summary['failed'] == 0 and summary['forced'] == 0
	assert _changes(hpcp_mod, new, target) == []
	assert hpcp_mod.ERRORS == []


def test_apply_patch_tree_is_idempotent(trees, hpcp_mod, tmp_path):
	new, base = trees
	_build_scenario(new, base)
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	second = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert (second['deleted'], second['added'], second['modified'], second['failed']) == (0, 0, 0, 0)
	assert second['skipped'] == len(rows)


class _FakeStdin:
	def __init__(self, tty):
		self._tty = tty

	def isatty(self):
		return self._tty


def _conflicted(hpcp_mod, new, base, tmp_path):
	_write(new / 'mod.txt', 'new content')
	_write(base / 'mod.txt', 'old')
	_write(base / 'gone.txt', 'bye')
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	_write(target / 'mod.txt', 'local edit')
	return vol, target


def test_apply_aborts_on_conflict_without_tty(trees, hpcp_mod, tmp_path, monkeypatch):
	new, base = trees
	vol, target = _conflicted(hpcp_mod, new, base, tmp_path)
	monkeypatch.setattr(sys, 'stdin', _FakeStdin(False))
	with pytest.raises(RuntimeError, match='Patch conflict'):
		hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert (target / 'gone.txt').exists()
	assert (target / 'mod.txt').read_text() == 'local edit'


def test_apply_prompt_defaults_to_abort(trees, hpcp_mod, tmp_path, monkeypatch):
	new, base = trees
	vol, target = _conflicted(hpcp_mod, new, base, tmp_path)
	monkeypatch.setattr(sys, 'stdin', _FakeStdin(True))
	monkeypatch.setattr(hpcp_mod.multiCMD, 'input_with_timeout_and_countdown', lambda timeout, prompt='': None)
	with pytest.raises(RuntimeError, match='Patch conflict'):
		hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert (target / 'gone.txt').exists()


def test_apply_prompt_yes_applies(trees, hpcp_mod, tmp_path, monkeypatch, capsys):
	new, base = trees
	vol, target = _conflicted(hpcp_mod, new, base, tmp_path)
	monkeypatch.setattr(sys, 'stdin', _FakeStdin(True))
	monkeypatch.setattr(hpcp_mod.multiCMD, 'input_with_timeout_and_countdown', lambda timeout, prompt='': 'y')
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert summary['forced'] == 1
	assert (target / 'mod.txt').read_text() == 'new content'
	assert not (target / 'gone.txt').exists()
	assert 'Warning: forcing M mod.txt' in capsys.readouterr().err


def test_apply_force_applies_and_warns(trees, hpcp_mod, tmp_path, monkeypatch, capsys):
	new, base = trees
	vol, target = _conflicted(hpcp_mod, new, base, tmp_path)
	monkeypatch.setattr(sys, 'stdin', _FakeStdin(False))
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), force=True, max_workers=1)
	assert summary['forced'] == 1 and summary['failed'] == 0
	assert 'Warning: forcing M mod.txt' in capsys.readouterr().err
	assert _changes(hpcp_mod, new, target) == []


def test_apply_rejects_incomplete_or_mixed_volumes(trees, hpcp_mod, tmp_path):
	new, base = trees
	for i in range(3):
		_write(new / f'f{i}.txt', 'x')
	v1, v2 = tmp_path / 'v1', tmp_path / 'v2'
	_patch_from(hpcp_mod, new, base, [v1, v2])
	target = tmp_path / 'target'
	_clone(base, target)
	with pytest.raises(RuntimeError, match=r'Incomplete patch: missing volume.*patch_1\.img'):
		hpcp_mod.apply_patch_tree([str(v1)], str(target), volume_names=['/x/patch.img'])
	other = tmp_path / 'other'
	_patch_from(hpcp_mod, new, base, [other])
	with pytest.raises(RuntimeError, match='different patches'):
		hpcp_mod.apply_patch_tree([str(v1), str(other)], str(target))
	os.unlink(v2 / '.hpcp_diff' / 'manifest')
	with pytest.raises(RuntimeError, match='Incomplete patch'):
		hpcp_mod.apply_patch_tree([str(v1), str(v2)], str(target))


def test_apply_refuses_to_write_through_symlinked_parent(tmp_path, hpcp_mod, reset_hpcp_globals):
	outside = tmp_path / 'outside'
	outside.mkdir()
	target = tmp_path / 'target'
	target.mkdir()
	os.symlink(str(outside), target / 'esc')
	vol = tmp_path / 'vol'
	_write(vol / 'esc' / 'x.txt', 'payload')
	rows = [hpcp_mod.ChangeRow('A', 'esc/x.txt', '-', hpcp_mod.get_entry_state(str(vol / 'esc' / 'x.txt')))]
	hpcp_mod.write_patch_metadata([str(vol)], rows, hpcp_mod.make_patch_meta(rows, 'n', 'b', 'root'))
	hpcp_mod.apply_patch_tree([str(vol)], str(target), force=True, max_workers=1)
	assert not (outside / 'x.txt').exists()
	assert any(error.startswith('Patch format error') for error in hpcp_mod.ERRORS)


def test_apply_store_layout_leaves_other_trees(tmp_path, hpcp_mod, reset_hpcp_globals):
	hpcp_mod.HASH_SIZE = 65536
	new = tmp_path / 'live' / 'a'
	other = tmp_path / 'live' / 'b'
	_write(new / 'f.txt', 'new')
	_write(new / 'added.txt', 'added')
	store = tmp_path / 'store'
	rel_new = os.path.relpath(str(new), '/')
	rel_other = os.path.relpath(str(other), '/')
	_write(store / rel_new / 'f.txt', 'old')
	_write(store / rel_new / 'gone.txt', 'bye')
	_write(store / rel_other / 'g.txt', 'other tree')
	new_map, base_map = hpcp_mod.build_store_maps(str(new), str(store), max_workers=1)
	rows = hpcp_mod.compute_changes(new_map, base_map, max_workers=1)
	vol = tmp_path / 'vol'
	vol.mkdir()
	hpcp_mod.build_patch_tree(rows, '/', [str(vol)], hpcp_mod.make_patch_meta(rows, 'a', 'store', 'store'), max_workers=1)
	hpcp_mod.apply_patch_tree([str(vol)], str(store), max_workers=1)
	assert (store / rel_other / 'g.txt').read_text() == 'other tree'
	assert hpcp_mod.compute_changes(*hpcp_mod.build_store_maps(str(new), str(store), max_workers=1), max_workers=1) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_diff_image.py -q -k apply`
Expected: FAIL with `AttributeError: module 'hpcp' has no attribute 'apply_patch_tree'`.

- [ ] **Step 3: Implement**

Append to the Diff Image section:

```python
def load_patch_volumes(volume_roots, volume_names=None):
	"""Read and cross-check every volume's manifest.

	Returns (meta of volume 1, rows of volume 1, volume roots ordered by volume number).
	volume_names (image paths, parallel to volume_roots) only improve error messages.
	"""
	names = list(volume_names) if volume_names else [str(root) for root in volume_roots]
	metas = []
	for volume_root, name in zip(volume_roots, names):
		manifest = os.path.join(volume_root, DIFF_DIR_NAME, 'manifest')
		if not os.path.isfile(manifest):
			raise RuntimeError(f'Incomplete patch: {name} has no {DIFF_DIR_NAME}/manifest')
		metas.append(read_manifest(manifest))
	patch_ids = sorted({str(meta.get('patch_id')) for meta in metas})
	if len(patch_ids) != 1:
		raise RuntimeError(f'Incomplete patch: volumes belong to different patches ({", ".join(patch_ids)})')
	by_number = {}
	total = None
	for volume_root, name, meta in zip(volume_roots, names, metas):
		number, _, count = str(meta.get('volume', '')).partition('/')
		try:
			number, count = int(number), int(count)
		except ValueError:
			raise RuntimeError(f'Patch format error: {name} has a bad volume field {meta.get("volume")!r}')
		if total is None:
			total = count
		elif count != total:
			raise RuntimeError(f'Incomplete patch: volumes disagree on the volume count ({total} vs {count})')
		by_number[number] = (volume_root, name, meta)
	missing = [number for number in range(1, total + 1) if number not in by_number]
	if missing:
		if 1 in by_number:
			expected = ', '.join(_split_dest_image_name(by_number[1][1], number - 1) for number in missing)
		else:
			expected = ', '.join(f'volume {number}' for number in missing)
		raise RuntimeError(f'Incomplete patch: missing volume(s) {expected} of {total}')
	ordered = [by_number[number] for number in range(1, total + 1)]
	meta = ordered[0][2]
	if meta['sections'] != [('root', '.', 'changes.tsv')]:
		raise RuntimeError(f'Patch format error: unsupported sections {meta["sections"]}; this hpcp applies single-filesystem patches only')
	rows = read_changes(os.path.join(ordered[0][0], DIFF_DIR_NAME, 'changes.tsv'))
	return meta, rows, [entry[0] for entry in ordered]

def _resolve_conflicts(conflicts, force):
	"""Print conflicts, then prompt (default abort), abort without a tty, or force with warnings."""
	print(f"{len(conflicts)} path(s) on the target differ from the state this patch expects:")
	for check in conflicts:
		expected = '-' if check.row.op == 'A' else check.row.old
		print(f"  {check.row.op} {check.row.path!r}: {check.reason} (expected {expected}, found {check.found})")
	if not force:
		if not sys.stdin.isatty():
			raise RuntimeError(f'Patch conflict: {len(conflicts)} conflicting path(s) and stdin is not a tty; nothing applied (use -fa to force)')
		answer = multiCMD.input_with_timeout_and_countdown(60, 'Apply anyway? [y/N]')
		if (answer or '').strip().lower() not in ('y', 'yes'):
			raise RuntimeError(f'Patch conflict: {len(conflicts)} conflicting path(s); nothing applied')
	for check in conflicts:
		print(f"Warning: forcing {check.row.op} {check.row.path}: {check.reason}", file=sys.stderr)

def _inside_root(root, path):
	"""True when path's parent directory resolves inside root (no symlinked escape)."""
	root_real = os.path.realpath(root)
	parent_real = os.path.realpath(os.path.dirname(path))
	return parent_real == root_real or parent_real.startswith(root_real.rstrip(os.sep) + os.sep)

def _apply_rows(rows, volume_roots, target_root, max_workers=1, verbose=False):
	"""Carry out checked rows: deletions, then directories, files, symlinks, directory metadata."""
	def target_of(row):
		path = _key_path(target_root, row.path)
		if row.path != './' and not _inside_root(target_root, path):
			eprint(f'Patch format error: {row.path} resolves outside {target_root}; skipped')
			return None
		return path
	deletes = [row for row in rows if row.op == 'D']
	payload = [row for row in rows if row.op in ('A', 'M')]
	# Non-directories first, then directories deepest first.
	for row in sorted(deletes, key=lambda r: (r.path.endswith('/'), -_key_depth(r.path))):
		path = target_of(row)
		if path is None:
			continue
		try:
			_remove_entry(path)
		except OSError as e:
			eprint(f'Remove file failed: {row.path}: {e}')
	targets = {}
	for row in payload:
		path = target_of(row)
		if path is None:
			continue
		targets[row.path] = path
		if not os.path.lexists(path):
			continue
		if row.new.startswith('d:') and os.path.isdir(path) and not os.path.islink(path):
			continue
		# Files are re-copied fresh (drops stale xattrs); wrong-type entries make way.
		try:
			_remove_entry(path)
		except OSError as e:
			eprint(f'Remove file failed: {row.path}: {e}')
	dir_rows = [row for row in payload if row.new.startswith('d:') and row.path in targets]
	for row in sorted(dir_rows, key=lambda r: _key_depth(r.path)):
		os.makedirs(targets[row.path], exist_ok=True)
	by_volume = {}
	for row in payload:
		if not row.new.startswith('f:') or row.path not in targets:
			continue
		for volume_root in volume_roots:
			source = _key_path(volume_root, row.path)
			if os.path.lexists(source):
				by_volume.setdefault(volume_root, []).append(source)
				break
		else:
			eprint(f'Incomplete patch: payload for {row.path} is missing from every volume')
	for volume_root, files in by_volume.items():
		copy_file_list_parallel(files, [], volume_root, [target_root], max(1, max_workers), verbose=verbose)
	for row in payload:
		if row.new.startswith('l:') and row.path in targets:
			try:
				_place_symlink(decode_state(row.new)['target'], targets[row.path])
			except OSError as e:
				eprint(f'Failed to create symbolic link: {row.path}: {e}')
	for row in sorted(dir_rows, key=lambda r: _key_depth(r.path), reverse=True):
		sync_directory_metadata(_key_path(volume_roots[0], row.path), [targets[row.path]])

def _verify_rows(rows, target_root, compare_hash=True, full_hash=False, max_workers=1):
	"""Re-read every applied path and compare with its final expected state; returns the failure count."""
	final = {}
	for row in sort_change_rows(rows):
		# D sorts before A, so for a D + A pair on one name the A row decides the final state.
		final[row.path.rstrip('/')] = row
	paths = {name: _key_path(target_root, row.path) for name, row in final.items()}
	states = _states_for_paths(set(paths.values()), full_hash=full_hash, max_workers=max_workers)
	failed = 0
	for name, row in final.items():
		expected = '-' if row.op == 'D' else row.new
		found = states[paths[name]]
		ok = found == '-' if expected == '-' else states_equal(found, expected, compare_hash)
		if not ok:
			failed += 1
			eprint(f'Patch verify error: {row.op} {row.path}: expected {expected}, found {found}')
	return failed

def apply_patch_tree(volume_roots, target_root, force=False, max_workers=1, verbose=False, volume_names=None, store_note=False):
	"""Apply a patch whose volumes are mounted (or unpacked) at volume_roots onto directory target_root.

	Returns {'deleted', 'added', 'modified', 'skipped', 'forced', 'failed'}. Raises RuntimeError
	for a malformed or incomplete patch, or when conflicts are not overridden.
	"""
	global HASH_SIZE
	meta, rows, volume_roots = load_patch_volumes(volume_roots, volume_names)
	if store_note and meta.get('layout') == 'store':
		example = rows[0].path if rows else ''
		print(f"Note: this patch uses image-store layout; paths such as {example!r} are resolved under {target_root}")
	compare_hash = meta.get('hash_algo') == _hash_algo_name()
	if not compare_hash:
		print(f"Warning: patch was hashed with {meta.get('hash_algo')}, which is not available here; file hashes are not compared", file=sys.stderr)
	full_hash = meta.get('full_hash') == '1'
	saved_hash_size = HASH_SIZE
	try:
		HASH_SIZE = int(meta.get('hash_size', HASH_SIZE))
		checks = check_patch_target(rows, target_root, compare_hash, full_hash, max_workers)
		conflicts = [check for check in checks if check.action == 'conflict']
		if conflicts:
			_resolve_conflicts(conflicts, force)
		todo = [check.row for check in checks if check.action != 'skip']
		_apply_rows(todo, volume_roots, target_root, max_workers, verbose)
		failed = _verify_rows(todo, target_root, compare_hash, full_hash, max_workers)
	finally:
		HASH_SIZE = saved_hash_size
	summary = {
		'deleted': sum(1 for row in todo if row.op == 'D'),
		'added': sum(1 for row in todo if row.op == 'A'),
		'modified': sum(1 for row in todo if row.op == 'M'),
		'skipped': len(checks) - len(todo),
		'forced': len(conflicts),
		'failed': failed,
	}
	print('Patch summary: ' + ', '.join(f'{key} {value}' for key, value in summary.items()))
	return summary

```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_diff_image.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_diff_image.py
git commit -m "feat(diff): apply a patch tree with conflict prompt, force and verify

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Image wrappers

**Files:**
- Modify: `hpcp.py` (`create_image` signature and size block, currently near line 4222; Diff Image section)
- Test: `tests/test_diff_image.py`

**Interfaces:**
- Consumes: `create_loop_device(image_path, read_only=False)`, `get_partitions(disk)`, `get_largest_partition(disk)`, `run_command_in_multicmd_with_path_check(command, strict=False)`, `expand_paths(paths)`, `create_image(...)`, `clean_up(mount_points, loop_devices)` (all existing), plus everything from Tasks 5–9.
- Produces:
  - `create_image(..., dest_image_size=0, init_size=None)`: when `init_size` is not `None`, skips listing `src_paths`
  - `_mount_single_fs_image(image, read_only, mount_points, loop_devices) -> str` (mount point with a trailing separator)
  - `process_get_diff_image(src_paths, src_image, dest_paths, dest_image, output, mount_points, loop_devices, exclude=None, full_hash=False, dest_image_size=0, max_workers=…, parallel_file_listing=False, verbose=False) -> str | None` (patch path, or `None` when there are no differences)
  - `process_load_diff_image(load_diff_image, src_paths, dest_paths, dest_image, mount_points, loop_devices, force=False, max_workers=…, verbose=False) -> dict` (the `apply_patch_tree` summary)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_diff_image.py`:

```python
_IMAGE_TOOLS = ('losetup', 'mkfs.ext4', 'mount', 'umount', 'lsblk')
requires_root_and_image_tools = pytest.mark.skipif(
	os.name == 'nt' or os.geteuid() != 0 or any(shutil.which(tool) is None for tool in _IMAGE_TOOLS),
	reason='patch image tests need root and losetup/mkfs.ext4/mount/umount/lsblk')


@contextlib.contextmanager
def _mount_session(hpcp_mod):
	mount_points, loop_devices = [], []
	try:
		yield mount_points, loop_devices
	finally:
		hpcp_mod.clean_up(mount_points, loop_devices)


def _sha256(path):
	digest = hashlib.sha256()
	with open(path, 'rb') as f:
		for chunk in iter(lambda: f.read(1 << 20), b''):
			digest.update(chunk)
	return digest.hexdigest()


def test_process_get_diff_image_no_differences_creates_nothing(trees, hpcp_mod, tmp_path):
	new, base = trees
	_write(new / 'same.txt', 's')
	_write(base / 'same.txt', 's')
	out = tmp_path / 'patch.img'
	assert hpcp_mod.process_get_diff_image([str(new)], None, [str(base)], None, str(out), [], [], max_workers=1) is None
	assert not out.exists()


def test_process_get_diff_image_refuses_existing_output(trees, hpcp_mod, tmp_path):
	new, base = trees
	_write(new / 'a.txt', 'a')
	out = tmp_path / 'patch.img'
	out.write_text('keep me')
	with pytest.raises(RuntimeError, match='already exists'):
		hpcp_mod.process_get_diff_image([str(new)], None, [str(base)], None, str(out), [], [], max_workers=1)
	assert out.read_text() == 'keep me'


def test_process_get_diff_image_refuses_reserved_name(trees, hpcp_mod, tmp_path):
	new, base = trees
	_write(new / '.hpcp_diff' / 'x', 'x')
	with pytest.raises(RuntimeError, match='reserves'):
		hpcp_mod.process_get_diff_image([str(new)], None, [str(base)], None, str(tmp_path / 'p.img'), [], [], max_workers=1)


@requires_root_and_image_tools
@pytest.mark.root
@pytest.mark.loop
def test_dir_patch_image_round_trip(trees, hpcp_mod, tmp_path):
	new, base = trees
	_build_scenario(new, base)
	out = str(tmp_path / 'patch.img')
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		assert hpcp_mod.process_get_diff_image([str(new)], None, [str(base)], None, out, mount_points, loop_devices, max_workers=2) == out
	target = tmp_path / 'target'
	_clone(base, target)
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		summary = hpcp_mod.process_load_diff_image([out], [], [str(target)], None, mount_points, loop_devices, max_workers=2)
	assert summary['failed'] == 0
	assert _changes(hpcp_mod, new, target) == []


@requires_root_and_image_tools
@pytest.mark.root
@pytest.mark.loop
def test_store_image_patch_applies_in_place(tmp_path, hpcp_mod, reset_hpcp_globals):
	hpcp_mod.HASH_SIZE = 65536
	tree_a = tmp_path / 'live' / 'a'
	tree_b = tmp_path / 'live' / 'b'
	_write(tree_a / 'f.txt', 'v1')
	_write(tree_a / 'gone.txt', 'bye')
	_write(tree_b / 'g.txt', 'other')
	store = str(tmp_path / 'store.img')
	rc = hpcp_mod.hpcp([str(tree_a), str(tree_b)], dest_image=store, single_thread=True, max_workers=1,
		batch=True, do_not_remove_files_while_listing=True)
	assert rc in (None, 0)
	_write(tree_a / 'f.txt', 'v2 changed')
	os.unlink(tree_a / 'gone.txt')
	_write(tree_a / 'new.txt', 'n')
	before = _sha256(store)
	out = str(tmp_path / 'patch.img')
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		hpcp_mod.process_get_diff_image([str(tree_a)], None, [], store, out, mount_points, loop_devices, max_workers=2)
	assert _sha256(store) == before
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		summary = hpcp_mod.process_load_diff_image([out], [], [], store, mount_points, loop_devices, max_workers=2)
	assert summary['failed'] == 0
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		root = hpcp_mod._mount_single_fs_image(store, True, mount_points, loop_devices)
		assert hpcp_mod.compute_changes(*hpcp_mod.build_store_maps(str(tree_a), root, max_workers=1), max_workers=1) == []
		with open(os.path.join(root, os.path.relpath(str(tree_b), '/'), 'g.txt')) as f:
			assert f.read() == 'other'


@requires_root_and_image_tools
@pytest.mark.root
@pytest.mark.loop
def test_split_patch_needs_every_volume(trees, hpcp_mod, tmp_path):
	new, base = trees
	for i in range(40):
		_write(new / f'f{i:02d}.bin', os.urandom(2 * 1024 * 1024))
	out = str(tmp_path / 'patch.img')
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		hpcp_mod.process_get_diff_image([str(new)], None, [str(base)], None, out, mount_points, loop_devices,
			dest_image_size=80 * 1024 * 1024, max_workers=2)
	second = str(tmp_path / 'patch_1.img')
	assert os.path.exists(second)
	target = tmp_path / 'target'
	_clone(base, target)
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		with pytest.raises(RuntimeError, match=r'Incomplete patch: missing volume.*patch_1\.img'):
			hpcp_mod.process_load_diff_image([out], [], [str(target)], None, mount_points, loop_devices, max_workers=2)
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		summary = hpcp_mod.process_load_diff_image([out, second], [], [str(target)], None, mount_points, loop_devices, max_workers=2)
	assert summary['failed'] == 0
	assert _changes(hpcp_mod, new, target) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_diff_image.py -q -k "process_get_diff_image or patch_image or store_image or split_patch"`
Expected: FAIL with `AttributeError: module 'hpcp' has no attribute 'process_get_diff_image'`.

- [ ] **Step 3: Implement**

In `create_image`, change the signature line:

```python
def create_image(dest_image,target_mount_point,loop_devices: list,src_paths: list,mount_points, max_workers = multiprocessing.cpu_count(),
				parallel_file_listing = False,exclude=None,dest_image_size=0):
```

to:

```python
def create_image(dest_image,target_mount_point,loop_devices: list,src_paths: list,mount_points, max_workers = multiprocessing.cpu_count(),
				parallel_file_listing = False,exclude=None,dest_image_size=0,init_size=None):
```

and replace its size block:

```python
		init_size = 0
		for src in src_paths:
			src = os.path.abspath(src + os.path.sep)
			_,_,size,_ = get_file_list(src, max_workers=max_workers,exclude=exclude,parallel_file_listing=parallel_file_listing)
			init_size += size
```

with:

```python
		if init_size is None:
			init_size = 0
			for src in src_paths:
				src = os.path.abspath(src + os.path.sep)
				_,_,size,_ = get_file_list(src, max_workers=max_workers,exclude=exclude,parallel_file_listing=parallel_file_listing)
				init_size += size
```

Append to the Diff Image section:

```python
def _mount_single_fs_image(image, read_only, mount_points, loop_devices):
	"""Attach image, mount its largest partition (or the whole partitionless image), return the mount point + os.sep."""
	image = os.path.abspath(os.path.expanduser(image))
	if not os.path.isfile(image):
		raise RuntimeError(f'Diff image error: image {image} does not exist')
	if os.geteuid() != 0:
		print(f"Warning: mounting {image} usually needs root privileges", file=sys.stderr)
	loop_device = create_loop_device(image, read_only=read_only)
	loop_devices.append(loop_device)
	partitions = get_partitions(loop_device)
	if len(partitions) > 1:
		print(f"Warning: {image} has {len(partitions)} partitions; using the largest one. Use -dd for whole-disk images.", file=sys.stderr)
	target = get_largest_partition(loop_device) if partitions else loop_device
	mount_point = tempfile.mkdtemp()
	mount_points.append(mount_point)
	command = ['mount', '-o', 'ro', target, mount_point] if read_only else ['mount', target, mount_point]
	run_command_in_multicmd_with_path_check(command, strict=False)
	if not os.path.ismount(mount_point):
		raise RuntimeError(f'Diff image error: cannot mount {target} from {image}')
	return mount_point + os.sep

def process_get_diff_image(src_paths, src_image, dest_paths, dest_image, output, mount_points, loop_devices,
		exclude=None, full_hash=False, dest_image_size=0, max_workers=multiprocessing.cpu_count(),
		parallel_file_listing=False, verbose=False):
	"""Write a patch image that turns the base (-d DIR / -di IMG) into the new tree (src_path DIR / -si IMG).

	Returns the patch path, or None when the trees do not differ (no image is created).
	"""
	if src_image:
		images = expand_paths(src_image)
		if len(images) != 1:
			raise RuntimeError(f'Invalid arguments: -diff needs exactly one -si image, got {images}')
		new_root = _mount_single_fs_image(images[0], True, mount_points, loop_devices)
		new_name, new_is_dir = os.path.basename(images[0]), False
	else:
		paths = expand_paths(src_paths)
		if len(paths) != 1 or not os.path.isdir(paths[0]):
			raise RuntimeError(f'Invalid arguments: -diff needs exactly one existing source directory, got {src_paths}')
		new_root = os.path.abspath(paths[0])
		new_name, new_is_dir = os.path.basename(os.path.normpath(new_root)) or 'root', True
	if dest_image:
		base_root = _mount_single_fs_image(dest_image, True, mount_points, loop_devices)
		base_name, base_is_image = os.path.basename(dest_image), True
	else:
		bases = expand_paths(dest_paths or [])
		if len(bases) != 1 or not os.path.isdir(bases[0]):
			raise RuntimeError(f'Invalid arguments: -diff needs exactly one existing base directory (-d), got {dest_paths}')
		base_root = os.path.abspath(bases[0])
		base_name, base_is_image = os.path.basename(os.path.normpath(base_root)) or 'root', False
	if new_is_dir and base_is_image:
		new_map, base_map = build_store_maps(new_root, base_root, exclude, max_workers, parallel_file_listing)
		new_base, layout = os.sep, 'store'
	else:
		new_map, base_map = build_root_maps(new_root, base_root, exclude, max_workers, parallel_file_listing)
		new_base, layout = new_root, 'root'
	if any(_is_reserved_key(key) for key in new_map):
		raise RuntimeError(f'Diff image error: {new_root} contains {DIFF_DIR_NAME}, which hpcp reserves for patch metadata')
	if not output or output == 'auto':
		output = f'DIFF_{new_name}_TO_{base_name}_{int(time.time())}.img'
	output = os.path.abspath(os.path.expanduser(output))
	if os.path.lexists(output):
		raise RuntimeError(f'Diff image error: output {output} already exists')
	rows = compute_changes(new_map, base_map, full_hash=full_hash, max_workers=max_workers)
	meta = make_patch_meta(rows, new_name, base_name, layout, full_hash)
	print(f"Changes: {meta['added']} added, {meta['modified']} modified, {meta['deleted']} deleted; payload {format_bytes(meta['payload_bytes'])}B")
	if not rows:
		print(f"No differences between {new_root} and {base_root}; no patch image created.")
		return None
	change_list_bytes = sum(len(row.path) + len(row.old) + len(row.new) + 8 for row in rows)
	if os.geteuid() != 0:
		print("Warning: creating a patch image usually needs root privileges (loop devices, mkfs, mount)", file=sys.stderr)
	target_mount_point = tempfile.mkdtemp()
	mount_points.append(target_mount_point)
	volume_roots = create_image(output, target_mount_point, loop_devices, [], mount_points, max_workers=max_workers,
		dest_image_size=dest_image_size, init_size=meta['payload_bytes'] + 2 * change_list_bytes)
	build_patch_tree(rows, new_base, volume_roots, meta, max_workers=max_workers, verbose=verbose)
	extra = f" (+{len(volume_roots) - 1} more volume(s))" if len(volume_roots) > 1 else ''
	print(f"Patch image written to {output}{extra}")
	return output

def process_load_diff_image(load_diff_image, src_paths, dest_paths, dest_image, mount_points, loop_devices,
		force=False, max_workers=multiprocessing.cpu_count(), verbose=False):
	"""Mount every -siff volume and apply the patch to -di IMG, -d DIR, or a single positional path."""
	images = expand_paths(load_diff_image)
	if not images:
		raise RuntimeError(f'Incomplete patch: no patch image found for {load_diff_image}')
	volume_roots = [_mount_single_fs_image(image, True, mount_points, loop_devices) for image in images]
	if dest_image:
		target_root = _mount_single_fs_image(dest_image, False, mount_points, loop_devices)
		store_note = False
	else:
		targets = expand_paths(dest_paths) if dest_paths else expand_paths(src_paths or [])
		if len(targets) != 1 or not os.path.isdir(targets[0]):
			raise RuntimeError(f'Invalid arguments: -siff needs exactly one existing target directory, got {dest_paths or src_paths}')
		target_root = os.path.abspath(targets[0])
		store_note = target_root != os.path.abspath(os.sep)
	return apply_patch_tree(volume_roots, target_root, force=force, max_workers=max_workers, verbose=verbose,
		volume_names=[os.path.abspath(image) for image in images], store_note=store_note)

```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_diff_image.py tests/test_imaging.py -q`
Expected: all pass (as root; the root/loop tests report SKIPPED elsewhere).

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_diff_image.py
git commit -m "feat(diff): generate and apply patch images on loop devices

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: CLI flags, validation, and `hpcp()` wiring

**Files:**
- Modify: `hpcp.py` (`get_args` lines for `-siff` / `-diff`, currently near lines 4577 and 4582; `hpcp()` signature and the start of its `try:` block; `main()`; Diff Image section)
- Test: `tests/test_cli_args.py`, `tests/test_diff_image.py`

**Interfaces:**
- Consumes: `process_get_diff_image`, `process_load_diff_image` (Task 10).
- Produces:
  - CLI: `-diff/--get_diff_image [PATCH]` (`nargs='?'`, `const='auto'`, `default=None`), `-siff/--load_diff_image PATCH` (append), `-fa/--force_apply`
  - `hpcp(..., get_diff_image=None, load_diff_image=None, force_apply=False)`
  - `_validate_diff_args(get_diff_image, load_diff_image, force_apply, src_path, src_image, dest_paths, dest_image, dd=False, remove=False, remove_extra=False, target_file_list=None, compare_file_list=False, diff_file_list=None, tar_diff_file_list=False, content_only=False) -> (get_diff_image, src_path)`

- [ ] **Step 1: Update and add the tests**

In `tests/test_cli_args.py`, replace `test_image_and_unimplemented_flags_parse` and `test_main_does_not_pass_unimplemented_image_flags` with:

```python
def test_image_and_diff_flags_parse(hpcp_mod, restore_argv, reset_hpcp_globals):
	args = _parse(hpcp_mod, [
		'hpcp', '-si', '/tmp/src.img', '-siff', 'diff.img', '-diff',
		'-dd', '-ddr', '10G', '-d', '/tmp/out.img', '/tmp/src.img',
	])
	assert args.src_image == ['/tmp/src.img']
	assert args.load_diff_image == ['diff.img']
	assert args.get_diff_image == 'auto'
	assert args.disk_dump is True
	assert args.dd_resize == ['10G']


def test_diff_flag_takes_optional_patch_name(hpcp_mod, restore_argv, reset_hpcp_globals):
	args = _parse(hpcp_mod, ['hpcp', '-diff', 'out.img', '/tmp/new', '-d', '/tmp/base'])
	assert args.get_diff_image == 'out.img'
	assert args.src_path == ['/tmp/new']
	assert _parse(hpcp_mod, ['hpcp', '/tmp/new', '-d', '/tmp/base', '-diff']).get_diff_image == 'auto'
	assert _parse(hpcp_mod, ['hpcp', '/tmp/new']).get_diff_image is None


def test_force_apply_flag(hpcp_mod, restore_argv, reset_hpcp_globals):
	args = _parse(hpcp_mod, ['hpcp', '-siff', 'p.img', '-fa', '-d', '/tmp/t'])
	assert args.force_apply is True
	assert _parse(hpcp_mod, ['hpcp', '/tmp/src']).force_apply is False


def test_main_passes_diff_image_flags(hpcp_mod):
	import inspect
	src = inspect.getsource(hpcp_mod.main)
	for text in ('get_diff_image = args.get_diff_image', 'load_diff_image = args.load_diff_image', 'force_apply = args.force_apply'):
		assert text in src
```

Append to `tests/test_diff_image.py`:

```python
def test_hpcp_rejects_bad_patch_arguments(tmp_path, hpcp_mod, reset_hpcp_globals, capsys):
	new, base = str(tmp_path / 'new'), str(tmp_path / 'base')
	os.makedirs(new)
	os.makedirs(base)
	cases = [
		(dict(src_path=[new], dest_paths=[base], force_apply=True), '-fa'),
		(dict(src_path=[new], dest_paths=[base], get_diff_image='auto', load_diff_image=['p.img']), 'cannot be used together'),
		(dict(src_path=[new], dest_paths=[base], get_diff_image='auto', dd=True), '-dd'),
		(dict(src_path=[], dest_paths=[base], load_diff_image=['p.img'], remove=True), '-rm'),
		(dict(src_path=[new, base], dest_paths=[base], get_diff_image='auto'), 'exactly one source'),
		(dict(src_path=[new], get_diff_image='auto'), 'exactly one base'),
		(dict(src_path=[], load_diff_image=['p.img']), 'exactly one target'),
		(dict(src_path=[new], dest_paths=[base], get_diff_image='auto', content_only=True), '-co'),
	]
	for kwargs, message in cases:
		src_path = kwargs.pop('src_path')
		rc = hpcp_mod.hpcp(src_path, single_thread=True, **kwargs)
		err = capsys.readouterr().err
		assert rc == 241, (kwargs, err)
		assert 'Invalid arguments' in err and message in err, (kwargs, err)


def test_diff_value_that_is_a_directory_becomes_the_source(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Review Focus 1: `hpcp -diff /new -d /base` must not treat /new as the patch name."""
	new = tmp_path / 'new'
	new.mkdir()
	got = hpcp_mod._validate_diff_args(str(new), None, False, [], None, [str(tmp_path)], None)
	assert got == ('auto', [str(new)])


@requires_root_and_image_tools
@pytest.mark.root
@pytest.mark.loop
def test_hpcp_generates_and_applies_patch(trees, hpcp_mod, tmp_path):
	new, base = trees
	_build_scenario(new, base)
	out = str(tmp_path / 'p.img')
	assert hpcp_mod.hpcp([str(new)], dest_paths=[str(base)], get_diff_image=out, max_workers=2) == 0
	target = tmp_path / 'target'
	_clone(base, target)
	assert hpcp_mod.hpcp([], dest_paths=[str(target)], load_diff_image=[out], max_workers=2) == 0
	assert _changes(hpcp_mod, new, target) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_cli_args.py tests/test_diff_image.py -q -k "diff or force or patch_arguments"`
Expected: FAIL. `args.get_diff_image` is `True`, not `'auto'`; `force_apply` is missing; `hpcp()` rejects the `get_diff_image` keyword.

- [ ] **Step 3: Implement**

In `get_args`, replace the two "Not implemented" lines:

```python
	parser.add_argument('-siff','--load_diff_image',action='append', type=str, help='Not implemented. Load diff images and apply the changes to the destination.')
```

```python
	parser.add_argument('-diff', '--get_diff_image', action='store_true', help='Not implemented. Compare the source and destination file list, create a diff image of that will update the destination to source.')
```

with (keep each at its current position):

```python
	parser.add_argument('-siff','--load_diff_image',action='append', type=str, help='Apply a patch image made by -diff to the target (-d / -di or a single path). Repeat for each -dis volume. Every path the patch touches is checked first; conflicts prompt before being overridden.')
```

```python
	parser.add_argument('-diff', '--get_diff_image', type=str, nargs='?', const='auto', default=None, help='Generate a patch image that updates the base (-d / -di) to the source (src_path / -si): added and modified files in full plus a change list under .hpcp_diff/. Optionally name the output; default DIFF_<src>_TO_<base>_<time>.img. Honors -dis, -fh, -hs and -e.')
	parser.add_argument('-fa', '--force_apply', action='store_true', help='With -siff, apply even when the target differs from the state the patch expects. Conflicts are still printed as warnings.')
```

Append to the Diff Image section:

```python
def _validate_diff_args(get_diff_image, load_diff_image, force_apply, src_path, src_image, dest_paths, dest_image,
		dd=False, remove=False, remove_extra=False, target_file_list=None, compare_file_list=False,
		diff_file_list=None, tar_diff_file_list=False, content_only=False):
	"""Check -diff / -siff / -fa combinations before anything is mounted.

	Returns (get_diff_image, src_path). A -diff value that is an existing directory is the
	source path that argparse swallowed (`hpcp -diff /new -d /base`); it is moved back.
	"""
	src_path = list(src_path or [])
	dest_paths = list(dest_paths or [])
	if force_apply and not load_diff_image:
		raise RuntimeError('Invalid arguments: -fa / --force_apply only applies together with -siff')
	if get_diff_image and load_diff_image:
		raise RuntimeError('Invalid arguments: -diff and -siff cannot be used together')
	mode = '-siff' if load_diff_image else '-diff'
	if dd:
		raise RuntimeError(f'Invalid arguments: {mode} with -dd is not supported yet (dd-mode patches are planned)')
	clashes = [flag for flag, enabled in (('-rm/-rf', remove), ('-rme', remove_extra), ('-fl', target_file_list),
		('-cfl', compare_file_list), ('-dfl', diff_file_list), ('-tdfl', tar_diff_file_list), ('-co', content_only)) if enabled]
	if clashes:
		raise RuntimeError(f'Invalid arguments: {mode} cannot be combined with {", ".join(clashes)}')
	if load_diff_image:
		if src_image:
			raise RuntimeError('Invalid arguments: -siff takes no -si image; give the patch volumes with -siff')
		targets = (1 if dest_image else 0) + len(dest_paths)
		if targets and src_path:
			raise RuntimeError(f'Invalid arguments: -siff takes no source paths when -d / -di is given, got {src_path}')
		if not targets:
			targets = len(src_path)
		if targets != 1:
			raise RuntimeError(f'Invalid arguments: -siff needs exactly one target (-d DIR, -di IMG or one path), got {targets}')
		return get_diff_image, src_path
	if get_diff_image not in (None, 'auto') and os.path.isdir(os.path.expanduser(get_diff_image)):
		print(f"Note: -diff value {get_diff_image} is an existing directory; using it as the source and generating the patch name")
		src_path = [get_diff_image] + src_path
		get_diff_image = 'auto'
	sources = len(src_path) + len(src_image or [])
	if sources != 1:
		raise RuntimeError(f'Invalid arguments: -diff needs exactly one source (a directory or one -si image), got {sources}')
	bases = (1 if dest_image else 0) + len(dest_paths)
	if bases != 1:
		raise RuntimeError(f'Invalid arguments: -diff needs exactly one base (-d DIR or -di IMG), got {bases}')
	return get_diff_image, src_path

```

In `hpcp()`, extend the signature's last line:

```python
			exit_not_enough_space = False,do_not_remove_files_while_listing = False,no_fs_param_mirror = False):
```

to:

```python
			exit_not_enough_space = False,do_not_remove_files_while_listing = False,no_fs_param_mirror = False,
			get_diff_image = None, load_diff_image = None, force_apply = False):
```

In `hpcp()`, insert as the **first statements inside `try:`** (before `if dd:`):

```python
		if get_diff_image or load_diff_image or force_apply:
			if os.name == 'nt':
				raise RuntimeError("Platform error: patch images are not supported on Windows, exiting")
			get_diff_image, src_path = _validate_diff_args(get_diff_image, load_diff_image, force_apply, src_path, src_image,
				dest_paths, dest_image, dd=dd, remove=remove, remove_extra=remove_extra, target_file_list=target_file_list,
				compare_file_list=compare_file_list, diff_file_list=diff_file_list, tar_diff_file_list=tar_diff_file_list,
				content_only=content_only)
			workers = 1 if single_thread else max_workers
			if load_diff_image:
				process_load_diff_image(load_diff_image, src_path, dest_paths, dest_image, mount_points, loop_devices,
					force=force_apply, max_workers=workers, verbose=verbose)
				raise RuntimeError("Exiting after applying patch image.")
			process_get_diff_image(src_path, src_image, dest_paths, dest_image, get_diff_image, mount_points, loop_devices,
				exclude=exclude, full_hash=full_hash, dest_image_size=dest_image_size, max_workers=workers,
				parallel_file_listing=parallel_file_listing, verbose=verbose)
			raise RuntimeError("Exiting after generating patch image.")
```

(The `raise RuntimeError("Exiting after …")` pattern matches the existing list and compare modes. The message has no `:`, so it prints without being recorded, and `finally` still runs `clean_up`.)

In `main()`, change the last argument line of the `hpcp(...)` call:

```python
			 no_fs_param_mirror = args.no_fs_param_mirror)
```

to:

```python
			 no_fs_param_mirror = args.no_fs_param_mirror, get_diff_image = args.get_diff_image,
			 load_diff_image = args.load_diff_image, force_apply = args.force_apply)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_cli_args.py tests/test_diff_image.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_cli_args.py tests/test_diff_image.py
git commit -m "feat(cli): wire -diff, -siff and -fa into hpcp

-diff now takes an optional patch name, -siff applies patch volumes, and -fa
forces an apply past conflicts. Bad combinations fail with Invalid arguments
before anything is mounted.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Docs, version, and full verification

**Files:**
- Modify: `hpcp.py` (`version`, `COMMIT_DATE` near line 128)
- Modify: `README.md` (Development status, new section, help block)

- [ ] **Step 1: Bump the version and date**

```bash
sed -i "s/^version = '9.64'$/version = '9.65'/" hpcp.py
sed -i "s/^COMMIT_DATE = '.*'$/COMMIT_DATE = '$(date +%F)'/" hpcp.py
grep -nE "^version|^COMMIT_DATE" hpcp.py
```

Expected: `version = '9.65'` and today's date.

- [ ] **Step 2: Update the README prose**

In `README.md`, under `## Development status`, replace:

```markdown
Imaging functionality (source/destination as `.img` files) will be extended with differential image support (differential backup). Imaging is only available on Linux—similar to `tar`, but uses disk images.
```

with:

```markdown
Imaging functionality (source/destination as `.img` files) includes differential patch images (`-diff` / `-siff`, see below). Imaging is only available on Linux—similar to `tar`, but uses disk images.
```

Insert this section immediately before the line `## Remove Extra Feature Note`:

````markdown
## Patch Images Feature Note

Only available on Linux (loop devices and mounts, so usually root).

`-diff [PATCH.img]` compares a new tree (a directory, or a single-filesystem image with `-si`) against a base (`-d DIR` or `-di IMG`) and writes a patch image. The patch is an ordinary xfs/ext4 image, split with `-dis` like `-di` output. It holds every added or modified file **whole** at its real path, plus `.hpcp_diff/manifest` and `.hpcp_diff/changes.tsv`, which list every added, modified and deleted path with its before and after state. A moved file appears as one delete plus one add. Any metadata change (mode, owner, mtime, ACL or xattr) counts as a modification. Without a value the patch is named `DIFF_<src>_TO_<base>_<time>.img`.

A `-di` image store keeps files at their absolute source path. When the base is such a store, a directory is compared only with its own subtree inside the store, and the patch uses the same layout. So `hpcp -siff patch.img -di store.img` updates the store in place, and `hpcp -siff patch.img -d /` updates the live system.

`-siff PATCH.img` applies a patch to `-d DIR`, `-di IMG` or a single path. Repeat it for every `-dis` volume. It first checks every path the patch touches. If the target differs from what the patch expects, it lists the conflicts and asks `Apply anyway? [y/N]`; the default, and any non-interactive run, aborts. `-fa --force_apply` applies anyway and prints each conflict as a warning. Re-running a patch is safe because already-updated paths are skipped, and every touched path is verified afterwards.

`changes.tsv` is also what `-dfl` writes:

```
#hpcp-changes	1
#op	path	old	new
D	gone.txt	f:<size>:<mtime_ns>:<mode>:<uid>:<gid>:<hash>:<xattr>	-
A	newdir/	-	d:<mode>:<uid>:<gid>:<xattr>
M	link	l:old-target	l:new-target
```

Paths escape `\`, tab, newline and carriage return as `\\`, `\t`, `\n`, `\r`.

````

- [ ] **Step 3: Regenerate the README help block**

```bash
python3 - <<'EOF'
import os, subprocess
env = dict(os.environ, COLUMNS='80')
help_text = subprocess.run(['python3', 'hpcp.py', '-h'], capture_output=True, text=True, env=env, check=True).stdout
readme = open('README.md').read()
marker = '```bash\n$ hpcp -h\n'
start = readme.index(marker) + len(marker)
end = readme.index('```', start)
open('README.md', 'w').write(readme[:start] + help_text + readme[end:])
EOF
grep -n "force_apply\|load_diff_image\|get_diff_image" README.md
```

Expected: the help block shows `-diff`, `-siff` and `-fa` with the new help text and no "Not implemented".

- [ ] **Step 4: Run the full test suite**

Run: `python -m pytest tests/ -q`
Expected: all pass (as root, the root/loop tests execute; elsewhere they report SKIPPED).

Run: `python3 hpcp.py -V`
Expected: `hpcp.py 9.65 @ <today> …`

- [ ] **Step 5: Commit**

```bash
git add hpcp.py README.md
git commit -m "docs: document patch images and release 9.65

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Releasing (uvpack, cas-toolbox pin) is out of scope for this plan. The pin rides the next Monday roll-up.
