import contextlib
import errno
import hashlib
import os
import shutil
import stat
import struct
import subprocess
import sys
import time
import pytest
import re


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


def test_xattr_digest_and_stale_removal_ignore_non_portable_names(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""F5: trusted.* (like NFSv4 / Lustre internals) is neither hashed nor removed; user.* still is."""
	plain = tmp_path / 'plain.txt'
	tagged = tmp_path / 'tagged.txt'
	plain.write_text('x')
	tagged.write_text('x')
	try:
		os.setxattr(str(tagged), 'trusted.hpcp_test', b'1', follow_symlinks=False)
	except (OSError, AttributeError) as e:
		pytest.skip(f'trusted xattrs unsupported here (needs root): {e}')
	assert hpcp_mod._xattr_digest(str(tagged)) == hpcp_mod._xattr_digest(str(plain))
	_set_xattr_or_skip(str(tagged), 'user.stale')
	hpcp_mod._remove_stale_xattrs(str(plain), str(tagged))
	names = os.listxattr(str(tagged), follow_symlinks=False)
	assert 'trusted.hpcp_test' in names and 'user.stale' not in names


def test_special_file_at_added_path_is_a_conflict(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""F6: a FIFO where the patch adds a file is not 'absent'."""
	if not hasattr(os, 'mkfifo'):
		pytest.skip('no FIFOs on this platform')
	hpcp_mod.HASH_SIZE = 65536
	rows = [hpcp_mod.ChangeRow('A', 'pipe', '-', hpcp_mod.get_entry_state(_write(tmp_path / 'new' / 'pipe', 'regular')))]
	target = tmp_path / 'target'
	target.mkdir()
	os.mkfifo(str(target / 'pipe'))
	assert hpcp_mod.get_entry_state(str(target / 'pipe')) == '?'
	checks = hpcp_mod.check_patch_target(rows, str(target))
	assert {(check.row.op, check.row.path): check.action for check in checks} == {('A', 'pipe'): 'conflict'}
	assert hpcp_mod.ERRORS == []


@pytest.mark.skipif(sys.version_info < (3, 7), reason='Python 3.6 pools only fork: no mp_context to spawn with')
def test_states_for_paths_workers_use_callers_hash_size(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""F15: workers that do not inherit module globals (spawn) still hash with the caller's HASH_SIZE."""
	import functools
	import multiprocessing
	hpcp_mod.HASH_SIZE = 4
	paths = [str(tmp_path / f'f{i:03d}') for i in range(300)]
	for i, path in enumerate(paths):
		with open(path, 'w') as f:
			f.write(f'{i:03d}-differing-head-same-tail')
	serial = hpcp_mod._states_for_paths(paths, max_workers=1)
	spawn_pool = functools.partial(hpcp_mod.ProcessPoolExecutor, mp_context=multiprocessing.get_context('spawn'))
	monkeypatch.setattr(hpcp_mod, 'ProcessPoolExecutor', spawn_pool)
	assert hpcp_mod._states_for_paths(paths, max_workers=2) == serial


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
	path.write_text('format\thpcp-diff\nformat_version\t4\n')
	with pytest.raises(RuntimeError, match='newer'):
		hpcp_mod.read_manifest(str(path))
	path.write_text('format\thpcp-diff\nformat_version\t1\nunknown_key\tfine\n')
	assert hpcp_mod.read_manifest(str(path))['unknown_key'] == 'fine'


def test_read_manifest_rejects_unsafe_sections_and_duplicate_keys(tmp_path, hpcp_mod):
	path = tmp_path / 'manifest'
	head = 'format\thpcp-diff\nformat_version\t1\n'
	bad_bodies = [
		'section\troot\t/etc\tchanges.tsv\n',
		'section\troot\t.\t../../foo\n',
		'section\troot\t.\tsub/changes.tsv\n',
		'section\troot\t.\t..\n',
		'sections\tfoo\n',
		'patch_id\tabc\npatch_id\tdef\n',
	]
	for body in bad_bodies:
		path.write_text(head + body)
		with pytest.raises(RuntimeError, match='Patch format error'):
			hpcp_mod.read_manifest(str(path))
	path.write_text(head + 'section\troot\t.\tchanges.tsv\n')
	assert hpcp_mod.read_manifest(str(path))['sections'] == [('root', '.', 'changes.tsv')]


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


def test_build_entry_map_lists_through_a_symlinked_root(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""F2: a root that is a symlink to a directory is listed from its target, keyed from the given path."""
	real = tmp_path / 'real'
	_write(real / 'a.txt', 'a')
	_write(real / 'sub' / 'b.txt', 'b')
	os.symlink('real', tmp_path / 'link')
	got = hpcp_mod.build_entry_map(str(tmp_path / 'link'), prefix='data/link/', max_workers=1)
	assert set(got) == {'data/link/', 'data/link/a.txt', 'data/link/sub/', 'data/link/sub/b.txt'}
	assert hpcp_mod.get_entry_state(got['data/link/']).startswith('d:')
	os.symlink('real/a.txt', tmp_path / 'filelink')
	assert hpcp_mod.build_entry_map(str(tmp_path / 'filelink'), prefix='x/', max_workers=1) == {}


def test_store_maps_for_a_symlinked_source_keep_the_stored_subtree(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""F2: `-diff /data/link -di store.img` compares contents instead of deleting what -di stored."""
	hpcp_mod.HASH_SIZE = 65536
	real = tmp_path / 'real'
	_write(real / 'a.txt', 'a')
	_write(real / 'sub' / 'b.txt', 'b')
	link = tmp_path / 'link'
	os.symlink('real', link)
	store = tmp_path / 'store'
	shutil.copytree(str(real), str(store / os.path.relpath(str(link), '/')))
	rows = hpcp_mod.compute_changes(*hpcp_mod.build_store_maps(str(link), str(store), max_workers=1), max_workers=1)
	assert [row for row in rows if row.op != 'M'] == []


def test_patch_from_a_symlinked_new_root_round_trips(trees, hpcp_mod, tmp_path):
	"""F2: the root directory's metadata comes from the symlink's target, not the symlink."""
	_, base = trees
	real = tmp_path / 'real'
	_write(real / 'a.txt', 'a')
	os.chmod(str(real), 0o700)
	link = tmp_path / 'link'
	os.symlink('real', link)
	rows = hpcp_mod.compute_changes(*hpcp_mod.build_root_maps(str(link), str(base), max_workers=1), max_workers=1)
	assert ('M', './') in _ops(rows)
	vol = tmp_path / 'vol'
	vol.mkdir()
	hpcp_mod.build_patch_tree(rows, str(link), [str(vol)], hpcp_mod.make_patch_meta(rows, 'n', 'b', 'root'), max_workers=1)
	target = tmp_path / 'target'
	_clone(base, target)
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert summary['failed'] == 0
	assert stat.S_IMODE(os.stat(str(target)).st_mode) == 0o700


def test_process_get_diff_image_validates_rows_before_creating_images(trees, hpcp_mod, tmp_path, monkeypatch):
	"""F2: rows that any apply would reject fail before an output name is claimed or mkfs runs."""
	new, base = trees
	_write(new / 'a.txt', 'a')
	_, dstate, _ = _states(hpcp_mod, tmp_path)
	monkeypatch.setattr(hpcp_mod, 'compute_changes', lambda *a, **k: [hpcp_mod.ChangeRow('D', './', dstate, '-')])

	def fail(*args, **kwargs):
		raise AssertionError('create_image must not be reached')

	monkeypatch.setattr(hpcp_mod, 'create_image', fail)
	out = tmp_path / 'p.img'
	with pytest.raises(RuntimeError, match='Patch format error'):
		hpcp_mod.process_get_diff_image([str(new)], None, [str(base)], None, str(out), [], [], max_workers=1)
	assert not out.exists()


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


def test_get_entry_state_read_failure_is_unknown(tmp_path, hpcp_mod, reset_hpcp_globals):
	path = os.path.join(str(tmp_path), 'x' * 300)  # ENAMETOOLONG is an unexpected read failure
	assert hpcp_mod.get_entry_state(path) == '?'
	assert any(e.startswith('Entry state error') for e in hpcp_mod.ERRORS)


def test_compute_changes_skips_unreadable_entries(trees, hpcp_mod):
	new, base = trees
	_write(new / 'added.txt', 'a')
	new_map, base_map = hpcp_mod.build_root_maps(str(new), str(base), max_workers=1)
	new_map['bad.txt'] = os.path.join(str(new), 'y' * 300)
	base_map['bad.txt'] = os.path.join(str(base), 'y' * 300)
	base_map['base_only_bad.txt'] = os.path.join(str(base), 'z' * 300)
	# readable on one side, unreadable on the other: neither may become a D or A row
	_write(base / 'bad_new.txt', 'real base file')
	new_map['bad_new.txt'] = os.path.join(str(new), 'n' * 300)
	base_map['bad_new.txt'] = str(base / 'bad_new.txt')
	_write(new / 'bad_base.txt', 'real new file')
	new_map['bad_base.txt'] = str(new / 'bad_base.txt')
	base_map['bad_base.txt'] = os.path.join(str(base), 'b' * 300)
	before = len(hpcp_mod.ERRORS)
	rows = hpcp_mod.compute_changes(new_map, base_map, max_workers=1)
	assert _ops(rows) == {('A', 'added.txt')}
	assert not any(row.path in ('bad_new.txt', 'bad_base.txt') for row in rows)
	assert any(e.startswith('Entry state error') for e in hpcp_mod.ERRORS[before:])


def test_states_for_paths_parallel_reports_worker_errors(tmp_path, hpcp_mod, reset_hpcp_globals):
	hpcp_mod.HASH_SIZE = 65536
	paths = [_write(tmp_path / f'f{i}.txt', f'{i}') for i in range(300)]
	bad = os.path.join(str(tmp_path), 'x' * 300)
	before = len(hpcp_mod.ERRORS)
	states = hpcp_mod._states_for_paths(paths + [bad], max_workers=2)
	assert states[bad] == '?'
	assert all(states[path].startswith('f:') for path in paths)
	assert any(e.startswith('Entry state error') for e in hpcp_mod.ERRORS[before:])


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


def test_build_patch_tree_refuses_incomplete_payload(trees, hpcp_mod, tmp_path, monkeypatch):
	new, base = trees
	_write(new / 'add.txt', 'added')
	monkeypatch.setattr(hpcp_mod, 'copy_file_list_parallel', lambda *a, **k: (0, 0, {}, frozenset()))
	vol = tmp_path / 'vol'
	with pytest.raises(RuntimeError, match='payload incomplete'):
		_patch_from(hpcp_mod, new, base, [vol])
	assert not (vol / '.hpcp_diff' / 'manifest').exists()


_CHANGED_WHILE_BUILT = (r'^Diff image error: 1 file\(s\) changed while the patch was built \(e\.g\. live\.log\); exclude them \(-e\) '
	r'or generate from an offline copy; patch metadata not written$')


@pytest.mark.parametrize('change', ['same-size rewrite', 'touch', 'size'])
def test_build_patch_tree_refuses_a_source_changed_after_the_scan(trees, hpcp_mod, tmp_path, change):
	"""A live file rewritten between the change scan and the payload copy would make a patch every node fails to verify.

	A copy of another size whose source no longer has the recorded size (a growing log) is a change, not an incomplete payload."""
	new, base = trees
	_write(new / 'live.log', 'version 1')
	_write(new / 'calm.txt', 'calm')
	new_map, base_map = hpcp_mod.build_root_maps(str(new), str(base), max_workers=1)
	rows = hpcp_mod.compute_changes(new_map, base_map, max_workers=1)
	assert _ops(rows) == {('A', 'live.log'), ('A', 'calm.txt')}
	if change == 'same-size rewrite':
		_write(new / 'live.log', 'version 2')
	elif change == 'touch':
		os.utime(str(new / 'live.log'), (T0 + 60, T0 + 60))
	else:
		_write(new / 'live.log', 'version 10')
	vol = tmp_path / 'vol'
	vol.mkdir()
	with pytest.raises(RuntimeError, match=_CHANGED_WHILE_BUILT):
		hpcp_mod.build_patch_tree(rows, str(new), [str(vol)], hpcp_mod.make_patch_meta(rows, 'n', 'b', 'root'), max_workers=1)
	assert not (vol / '.hpcp_diff').exists()


def _scan_then(hpcp_mod, new, base, tmp_path, change, monkeypatch=None, after_copy=None):
	"""Rows of new vs base, then change() the sources and build the patch (after_copy() runs after the payload copy)."""
	new_map, base_map = hpcp_mod.build_root_maps(str(new), str(base), max_workers=1)
	rows = hpcp_mod.compute_changes(new_map, base_map, max_workers=1)
	change()
	if after_copy:
		real_copy = hpcp_mod.copy_file_list_parallel
		def copy(*args, **kwargs):
			result = real_copy(*args, **kwargs)
			after_copy()
			return result
		monkeypatch.setattr(hpcp_mod, 'copy_file_list_parallel', copy)
	vol = tmp_path / 'vol'
	vol.mkdir()
	hpcp_mod.build_patch_tree(rows, str(new), [str(vol)], hpcp_mod.make_patch_meta(rows, 'n', 'b', 'root'), max_workers=1)


def test_payload_check_names_every_changed_file_at_once(trees, hpcp_mod, tmp_path):
	"""One rebuild per changed file would mean a full copy for each; all of them are named in one error."""
	new, base = trees
	for name in ('a.log', 'b.log', 'calm.txt'):
		_write(new / name, 'version 1')
	def change():
		_write(new / 'b.log', 'version 2')
		os.utime(str(new / 'a.log'), (T0 + 60, T0 + 60))
	with pytest.raises(RuntimeError, match=r'^Diff image error: 2 file\(s\) changed while the patch was built \(e\.g\. a\.log, b\.log\); '
			r'exclude them \(-e\) or generate from an offline copy; patch metadata not written$'):
		_scan_then(hpcp_mod, new, base, tmp_path, change)
	assert not (tmp_path / 'vol' / '.hpcp_diff').exists()


def test_payload_check_lists_ten_changed_files_and_counts_the_rest(trees, hpcp_mod, tmp_path):
	new, base = trees
	for index in range(13):
		_write(new / f'f{index:02}.log', 'version 1')
	def change():
		for index in range(13):
			_write(new / f'f{index:02}.log', 'version 2')
	shown = ', '.join(f'f{index:02}\\.log' for index in range(10))
	with pytest.raises(RuntimeError, match=rf'^Diff image error: 13 file\(s\) changed while the patch was built \(e\.g\. {shown} and 3 more\); '):
		_scan_then(hpcp_mod, new, base, tmp_path, change)


def test_payload_check_reports_a_truncated_copy_as_incomplete(trees, hpcp_mod, tmp_path, monkeypatch):
	"""A copy cut short (a full patch volume) whose source still matches its row is an incomplete payload,
	not a file that changed while the patch was built (which would point at -e)."""
	new, base = trees
	_write(new / 'big.bin', b'B' * 4096)
	_write(new / 'calm.txt', 'calm')
	with pytest.raises(RuntimeError, match=r'^Diff image error: payload incomplete for 1 file\(s\), e\.g\. big\.bin; patch metadata not written$'):
		_scan_then(hpcp_mod, new, base, tmp_path, lambda: None, monkeypatch, lambda: os.truncate(str(tmp_path / 'vol' / 'big.bin'), 1000))
	assert not (tmp_path / 'vol' / '.hpcp_diff').exists()


def test_payload_check_reports_incomplete_and_changed_files_together(trees, hpcp_mod, tmp_path, monkeypatch):
	new, base = trees
	_write(new / 'big.bin', b'B' * 4096)
	_write(new / 'live.log', 'version 1')
	_write(new / 'more.log', 'version 1')
	def change():
		_write(new / 'live.log', 'version 2')
		_write(new / 'more.log', 'version 10')
	with pytest.raises(RuntimeError, match=r'^Diff image error: payload incomplete for 1 file\(s\), e\.g\. big\.bin; '
			r'2 file\(s\) changed while the patch was built \(e\.g\. live\.log, more\.log\); '
			r'exclude them \(-e\) or generate from an offline copy; patch metadata not written$'):
		_scan_then(hpcp_mod, new, base, tmp_path, change, monkeypatch, lambda: os.truncate(str(tmp_path / 'vol' / 'big.bin'), 1000))
	assert not (tmp_path / 'vol' / '.hpcp_diff').exists()


@pytest.mark.parametrize('full_hash', [False, True])
def test_payload_check_hashes_like_the_rows(trees, hpcp_mod, tmp_path, monkeypatch, full_hash):
	"""The payload is hashed with the rows' settings: a change outside the hashed tail is caught only with a full hash."""
	new, base = trees
	monkeypatch.setattr(hpcp_mod, 'HASH_SIZE', 16)
	_write(new / 'live.log', b'H' * 64)
	new_map, base_map = hpcp_mod.build_root_maps(str(new), str(base), max_workers=1)
	rows = hpcp_mod.compute_changes(new_map, base_map, full_hash=full_hash, max_workers=1)
	_write(new / 'live.log', b'h' + b'H' * 63)
	vol = tmp_path / 'vol'
	vol.mkdir()
	meta = hpcp_mod.make_patch_meta(rows, 'n', 'b', 'root', full_hash=full_hash)
	if full_hash:
		with pytest.raises(RuntimeError, match=_CHANGED_WHILE_BUILT):
			hpcp_mod.build_patch_tree(rows, str(new), [str(vol)], meta, max_workers=1)
		assert not (vol / '.hpcp_diff').exists()
	else:
		hpcp_mod.build_patch_tree(rows, str(new), [str(vol)], meta, max_workers=1)
		assert (vol / '.hpcp_diff' / 'manifest').exists()


def test_payload_check_runs_in_parallel_for_many_files(trees, hpcp_mod, tmp_path):
	new, base = trees
	for index in range(300):
		_write(new / 'many' / f'f{index:03}.txt', f'file {index:03}')
	_write(new / 'live.log', 'version 1')
	new_map, base_map = hpcp_mod.build_root_maps(str(new), str(base), max_workers=1)
	rows = hpcp_mod.compute_changes(new_map, base_map, max_workers=2)
	_write(new / 'live.log', 'version 2')
	vol = tmp_path / 'vol'
	vol.mkdir()
	with pytest.raises(RuntimeError, match=_CHANGED_WHILE_BUILT):
		hpcp_mod.build_patch_tree(rows, str(new), [str(vol)], hpcp_mod.make_patch_meta(rows, 'n', 'b', 'root'), max_workers=2)
	assert not (vol / '.hpcp_diff').exists()


def test_build_patch_tree_root_row_and_nested_symlink(trees, hpcp_mod, tmp_path):
	new, base = trees
	os.chmod(new, 0o700)
	_write(new / 'x.txt', 'x')
	_write(new / 'sub' / 'keep.txt', 'k')
	os.symlink('../x.txt', new / 'sub' / 'ln')
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	assert any(row.path == './' and row.op == 'M' for row in rows)
	assert stat.S_IMODE(os.stat(vol).st_mode) == 0o700
	assert os.readlink(vol / 'sub' / 'ln') == '../x.txt'


def test_build_patch_tree_syncs_directory_mtimes(trees, hpcp_mod, tmp_path):
	new, base = trees
	_write(new / 'deep' / 'er' / 'f.txt', 'deep')
	os.utime(new / 'deep' / 'er', ns=(1717000000000000000, 1717000000000000000))
	os.utime(new / 'deep', ns=(1717000001000000000, 1717000001000000000))
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	assert os.stat(vol / 'deep' / 'er').st_mtime_ns == os.stat(new / 'deep' / 'er').st_mtime_ns
	assert os.stat(vol / 'deep').st_mtime_ns == os.stat(new / 'deep').st_mtime_ns


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


def _temp_leftovers(root):
	return [os.path.join(d, n) for d, dirs, files in os.walk(str(root)) for n in dirs + files if n.startswith('.hpcp_tmp_')]


@pytest.mark.parametrize('workers', [1, 2])
def test_apply_patch_tree_round_trip(trees, hpcp_mod, tmp_path, workers):
	"""Success criterion + Review Focus 3, 4, 5; workers=2 runs the copies on a process pool."""
	new, base = trees
	_build_scenario(new, base)
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=workers)
	assert summary['failed'] == 0 and summary['forced'] == 0
	assert _changes(hpcp_mod, new, target) == []
	assert hpcp_mod.ERRORS == []
	assert _temp_leftovers(target) == []


def test_apply_replaces_files_atomically(trees, hpcp_mod, tmp_path):
	"""F1: a process holding the old file keeps the old inode and content; no temp file is left."""
	new, base = trees
	_write(new / 'lib.so', 'new library')
	_write(base / 'lib.so', 'old library')
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	with open(str(target / 'lib.so'), 'rb') as held:
		old_inode = os.fstat(held.fileno()).st_ino
		summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
		held.seek(0)
		assert held.read() == b'old library'
	assert summary['failed'] == 0
	assert os.stat(str(target / 'lib.so')).st_ino != old_inode
	assert (target / 'lib.so').read_text() == 'new library'
	assert _temp_leftovers(target) == []


@pytest.mark.parametrize('failure', ['raises', 'short'])
def test_failed_copy_keeps_destination_and_leaves_no_temp(trees, hpcp_mod, tmp_path, monkeypatch, failure):
	"""F1: a cp that dies mid-write (or writes the wrong size) never touches the live file."""
	new, base = trees
	_write(new / 'mod.txt', 'new content')
	_write(base / 'mod.txt', 'old')
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	real_run = hpcp_mod.run_command_in_multicmd_with_path_check

	def broken_cp(command, *args, **kwargs):
		if command and os.path.basename(str(command[0])) == 'cp':
			with open(command[-1], 'wb') as f:
				f.write(b'partial')
			if failure == 'raises':
				raise RuntimeError('simulated cp failure')
			return []
		return real_run(command, *args, **kwargs)

	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', broken_cp)
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert summary['failed'] == 1
	assert (target / 'mod.txt').read_text() == 'old'
	assert _temp_leftovers(target) == []
	assert any(error.startswith('Copy failed: mod.txt') for error in hpcp_mod.ERRORS)


def test_apply_places_symlinks_through_a_temp_name(trees, hpcp_mod, tmp_path, monkeypatch):
	"""F1: a symlink is created beside its target and renamed over it, never removed first."""
	new, base = trees
	os.symlink('new-target', new / 'ln')
	os.symlink('old-target', base / 'ln')
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	real_symlink = os.symlink
	created = []

	def recording_symlink(src, dst, *args, **kwargs):
		created.append(os.path.basename(str(dst)))
		return real_symlink(src, dst, *args, **kwargs)

	monkeypatch.setattr(os, 'symlink', recording_symlink)
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	monkeypatch.setattr(os, 'symlink', real_symlink)
	assert summary['failed'] == 0
	assert os.readlink(str(target / 'ln')) == 'new-target'
	assert created and all(name.startswith('.hpcp_tmp_') for name in created)
	assert _temp_leftovers(target) == []


def test_unforced_directory_delete_keeps_late_entries(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""F3: an entry that appears after the check survives a non-forced D of its directory."""
	_, dstate, _ = _states(hpcp_mod, tmp_path)
	target = tmp_path / 'target'
	_write(target / 'olddir' / 'late.txt', 'created after the check')
	row = hpcp_mod.ChangeRow('D', 'olddir/', dstate, '-')
	hpcp_mod._apply_rows([row], [str(tmp_path / 'vol')], str(target))
	assert (target / 'olddir' / 'late.txt').read_text() == 'created after the check'
	assert any(error.startswith('Remove directory failed') for error in hpcp_mod.ERRORS)
	hpcp_mod.ERRORS.clear()
	hpcp_mod._apply_rows([row], [str(tmp_path / 'vol')], str(target), forced_rows={row})
	assert not (target / 'olddir').exists()
	assert hpcp_mod.ERRORS == []


def test_unlistable_directory_delete_is_a_conflict(trees, hpcp_mod, tmp_path, monkeypatch):
	"""F3: a listdir failure is not 'no children'."""
	new, base = trees
	_write(base / 'olddir' / 'a.txt', 'a')
	rows = _changes(hpcp_mod, new, base)
	target = tmp_path / 'target'
	_clone(base, target)
	real_listdir = os.listdir
	blocked = str(target / 'olddir')

	def failing_listdir(path='.'):
		if os.path.normpath(str(path)) == blocked:
			raise PermissionError(13, 'Permission denied', blocked)
		return real_listdir(path)

	monkeypatch.setattr(os, 'listdir', failing_listdir)
	checks = {(check.row.op, check.row.path): check for check in hpcp_mod.check_patch_target(rows, str(target))}
	monkeypatch.setattr(os, 'listdir', real_listdir)
	assert checks[('D', 'olddir/')].action == 'conflict'
	assert checks[('D', 'olddir/')].reason == 'cannot list directory'


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

	def close(self):
		# multiprocessing workers call sys.stdin.close() on startup.
		pass


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
	with pytest.raises(RuntimeError, match='Patch conflict: 1 path\\(s\\) resolve outside the target'):
		hpcp_mod.apply_patch_tree([str(vol)], str(target), force=True, max_workers=1)
	assert not (outside / 'x.txt').exists()


def test_apply_through_symlinked_parent_inside_target(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""G2: a symlinked directory that resolves inside the target is written through, as it is deleted through."""
	target = tmp_path / 'target'
	_write(target / 'real' / 'mod.txt', 'old')
	_write(target / 'real' / 'gone.txt', 'bye')
	os.symlink('real', target / 'link')
	vol = tmp_path / 'vol'
	_write(vol / 'link' / 'mod.txt', 'new content')
	_write(vol / 'link' / 'add.txt', 'added')
	R = hpcp_mod.ChangeRow
	state = hpcp_mod.get_entry_state
	rows = [
		R('D', 'link/gone.txt', state(str(target / 'real' / 'gone.txt')), '-'),
		R('A', 'link/add.txt', '-', state(str(vol / 'link' / 'add.txt'))),
		R('M', 'link/mod.txt', state(str(target / 'real' / 'mod.txt')), state(str(vol / 'link' / 'mod.txt'))),
	]
	hpcp_mod.write_patch_metadata([str(vol)], rows, hpcp_mod.make_patch_meta(rows, 'n', 'b', 'root'))
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert summary['failed'] == 0 and summary['forced'] == 0
	assert (target / 'real' / 'mod.txt').read_text() == 'new content'
	assert (target / 'real' / 'add.txt').read_text() == 'added'
	assert not (target / 'real' / 'gone.txt').exists()
	assert os.readlink(str(target / 'link')) == 'real'
	assert hpcp_mod.ERRORS == [] and _temp_leftovers(target) == []


@pytest.mark.parametrize('force', [False, True])
def test_apply_refuses_symlinked_parent_outside_target_up_front(tmp_path, hpcp_mod, reset_hpcp_globals, force):
	"""G2: a delete or write through a symlink leaving the target refuses the whole patch; -fa cannot force it."""
	outside = tmp_path / 'outside'
	_write(outside / 'old.txt', 'secret')
	target = tmp_path / 'target'
	_write(target / 'keep.txt', 'keep')
	os.symlink(str(outside), target / 'esc')
	vol = tmp_path / 'vol'
	_write(vol / 'esc' / 'new.txt', 'payload')
	R = hpcp_mod.ChangeRow
	rows = [
		R('D', 'esc/old.txt', hpcp_mod.get_entry_state(str(outside / 'old.txt')), '-'),
		R('A', 'esc/new.txt', '-', hpcp_mod.get_entry_state(str(vol / 'esc' / 'new.txt'))),
	]
	hpcp_mod.write_patch_metadata([str(vol)], rows, hpcp_mod.make_patch_meta(rows, 'n', 'b', 'root'))
	if force:
		# Make the D row a conflict, so -fa really forces something before the refusal.
		_write(outside / 'old.txt', 'changed since the patch was made')
	before = _snapshot(target)
	with pytest.raises(RuntimeError, match=r'Patch conflict: 2 path\(s\) resolve outside the target through symlinks'):
		hpcp_mod.apply_patch_tree([str(vol)], str(target), force=force, max_workers=1)
	assert (outside / 'old.txt').exists() and not (outside / 'new.txt').exists()
	assert _snapshot(target) == before


@pytest.mark.parametrize('force', [False, True])
def test_apply_refuses_dangling_symlink_parent_outside_up_front(tmp_path, hpcp_mod, reset_hpcp_globals, force):
	"""G2 follow-up: a parent that is a dangling symlink leading outside the target is refused before any delete."""
	target = tmp_path / 'target'
	_write(target / 'keep' / 'x.txt', 'still here')
	os.symlink(str(tmp_path / 'missing_outside'), target / 'esc')
	vol = tmp_path / 'vol'
	_write(vol / 'esc' / 'new.txt', 'payload')
	R = hpcp_mod.ChangeRow
	rows = [
		R('D', 'keep/x.txt', hpcp_mod.get_entry_state(str(target / 'keep' / 'x.txt')), '-'),
		R('A', 'esc/new.txt', '-', hpcp_mod.get_entry_state(str(vol / 'esc' / 'new.txt'))),
	]
	hpcp_mod.write_patch_metadata([str(vol)], rows, hpcp_mod.make_patch_meta(rows, 'n', 'b', 'root'))
	with pytest.raises(RuntimeError, match='resolve outside the target'):
		hpcp_mod.apply_patch_tree([str(vol)], str(target), force=force, max_workers=1)
	assert (target / 'keep' / 'x.txt').read_text() == 'still here'
	assert not os.path.lexists(str(tmp_path / 'missing_outside'))


@pytest.mark.parametrize('case', ['patch_deletes_link', 'forced_over_link'])
def test_apply_replaces_an_outside_symlink_with_a_directory(trees, hpcp_mod, tmp_path, case):
	"""G2 guard: rows below a symlink the patch removes first are judged after that removal, not through the old link.

	patch_deletes_link: the base had the symlink (D data + A data/). forced_over_link: the
	target has a symlink where the patch adds a directory, and the conflict is forced.
	"""
	new, base = trees
	outside = tmp_path / 'outside'
	outside.mkdir()
	if case == 'patch_deletes_link':
		os.symlink(str(outside), base / 'data')
	_write(new / 'data' / 'f.txt', 'now a real directory')
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	expected = {('A', 'data/'), ('A', 'data/f.txt')}
	assert _ops(rows) == (expected | {('D', 'data')} if case == 'patch_deletes_link' else expected)
	target = tmp_path / 'target'
	_clone(base, target)
	if case == 'forced_over_link':
		os.symlink(str(outside), target / 'data')
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), force=case == 'forced_over_link', max_workers=1)
	assert summary['failed'] == 0
	assert summary['forced'] == (1 if case == 'forced_over_link' else 0)
	assert (target / 'data').is_dir() and not (target / 'data').is_symlink()
	assert os.listdir(str(outside)) == []
	assert _changes(hpcp_mod, new, target) == []


def test_unsafe_parent_allows_only_directories_inside_the_target(tmp_path, hpcp_mod):
	"""G2: the write-time check follows a symlinked parent that stays inside the target, and nothing else."""
	target = tmp_path / 'target'
	(target / 'real').mkdir(parents=True)
	(tmp_path / 'outside').mkdir()
	os.symlink('real', target / 'link')
	os.symlink(str(tmp_path / 'outside'), target / 'esc')
	_write(target / 'file', 'f')
	assert hpcp_mod._unsafe_parent(str(target), str(target / 'real' / 'x')) is None
	assert hpcp_mod._unsafe_parent(str(target), str(target / 'link' / 'x')) is None
	assert 'resolves outside' in hpcp_mod._unsafe_parent(str(target), str(target / 'esc' / 'x'))
	assert 'not a directory' in hpcp_mod._unsafe_parent(str(target), str(target / 'file' / 'x'))
	assert 'not a directory' in hpcp_mod._unsafe_parent(str(target), str(target / 'missing' / 'x'))


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
	left = hpcp_mod.compute_changes(*hpcp_mod.build_store_maps(str(new), str(store), max_workers=1), max_workers=1)
	if hasattr(os, 'geteuid') and os.geteuid() != 0:
		# without root, apply cannot give the store's copy of a directory above the tree (/tmp) its owner,
		# nor a group the user is not in: its row is let through when the owner is all that differs (a lost
		# sticky bit still fails)
		groups = set(os.getgroups()) | {os.getegid()}
		foreign = {os.path.relpath(str(p), '/') + '/' for p in new.parents
			if str(p) != '/' and (os.stat(str(p)).st_uid != os.geteuid() or os.stat(str(p)).st_gid not in groups)}

		def unowned(state):
			info = hpcp_mod.decode_state(state)
			return {k: v for k, v in info.items() if k not in ('uid', 'gid')} if info else info
		left = [row for row in left
			if not (row.op == 'M' and row.path in foreign and unowned(row.old) == unowned(row.new))]
	assert left == []


def _states(hpcp_mod, tmp_path):
	"""A (file, dir, symlink) state triple taken from real entries."""
	scratch = tmp_path / 'scratch'
	_write(scratch / 'f.txt', 'payload')
	(scratch / 'd').mkdir()
	os.symlink('/nonexistent', scratch / 'l')
	return tuple(hpcp_mod.get_entry_state(str(scratch / name)) for name in ('f.txt', 'd', 'l'))


def _snapshot(root):
	return sorted((os.path.relpath(os.path.join(d, n), str(root)), os.path.islink(os.path.join(d, n)))
		for d, dirs, files in os.walk(str(root)) for n in dirs + files)


@pytest.mark.parametrize('case', ['root_symlink', 'slash_with_link', 'nonslash_with_dir', 'add_with_old',
	'modify_changes_type', 'nested_under_symlink', 'metadata_escape'])
def test_apply_rejects_malformed_rows(case, tmp_path, hpcp_mod, reset_hpcp_globals):
	fstate, dstate, lstate = _states(hpcp_mod, tmp_path)
	R = hpcp_mod.ChangeRow
	rows = {
		'root_symlink': [R('M', './', dstate, lstate)],
		'slash_with_link': [R('A', 'x/', '-', lstate)],
		'nonslash_with_dir': [R('A', 'x', '-', dstate)],
		'add_with_old': [R('A', 'x', fstate, fstate)],
		'modify_changes_type': [R('M', 'x', fstate, lstate)],
		'nested_under_symlink': [R('A', 'a', '-', lstate), R('A', 'a/f.txt', '-', fstate)],
		'metadata_escape': [R('A', 'a', '-', lstate), R('A', 'a/sub/', '-', dstate)],
	}[case]
	vol = tmp_path / 'vol'
	vol.mkdir()
	hpcp_mod.write_patch_metadata([str(vol)], rows, hpcp_mod.make_patch_meta(rows, 'n', 'b', 'root'))
	target = tmp_path / 'target'
	_write(target / 'keep.txt', 'keep')
	before = _snapshot(target)
	with pytest.raises(RuntimeError, match='Patch format error'):
		hpcp_mod.apply_patch_tree([str(vol)], str(target), force=True, max_workers=1)
	assert _snapshot(target) == before and (target / 'keep.txt').read_text() == 'keep'


def _fake_mounts(hpcp_mod, monkeypatch, *mounts):
	"""Make mountinfo list each (directory, fstype) of mounts as a mount point, and nothing else."""
	entries = [{'devnum': (0, 100 + index), 'root': '/', 'mount_point': os.path.realpath(str(path)), 'options': ['rw'],
		'fstype': fs_type, 'source': 'none'} for index, (path, fs_type) in enumerate(mounts)]
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda *args, **kwargs: [dict(entry) for entry in entries])


def _case_only_rename(new, base, prefix=''):
	"""new renames base's EFI/BOOT (under prefix) to EFI/Boot, a change only in case."""
	_write(new / prefix / 'EFI' / 'Boot' / 'BOOTX64.EFI', 'loader')
	_write(base / prefix / 'EFI' / 'BOOT' / 'BOOTX64.EFI', 'loader')
	_write(new / prefix / 'EFI' / 'grub.cfg', 'new cfg')
	_write(base / prefix / 'EFI' / 'grub.cfg', 'old cfg')


def _moved_file(new, base):
	"""new moves base's x/f to y/f (both directories exist on both sides): a D x/f + A y/f pair."""
	(new / 'x').mkdir()
	_write(new / 'y' / 'f', 'moved')
	_write(base / 'x' / 'f', 'moved')
	(base / 'y').mkdir()


def test_apply_deletes_one_hard_link_and_writes_another(trees, hpcp_mod, tmp_path):
	"""Hard links are separate entries of one inode: deleting one name and writing another (git's linked commands
	when one is dropped) is not one entry reached twice, and applies."""
	new, base = trees
	_write(new / 'bin' / 'git', 'v2')
	_write(base / 'bin' / 'git', 'v1')
	os.link(str(base / 'bin' / 'git'), str(base / 'bin' / 'git-old'))
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	assert _ops(rows) == {('D', 'bin/git-old'), ('M', 'bin/git')}
	target = tmp_path / 'target'
	_clone(base, target)
	os.unlink(str(target / 'bin' / 'git-old'))
	os.link(str(target / 'bin' / 'git'), str(target / 'bin' / 'git-old'))
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert (summary['deleted'], summary['modified'], summary['failed']) == (1, 1, 0)
	assert _changes(hpcp_mod, new, target) == []


def test_apply_replaces_a_symlink_into_a_deleted_directory_with_a_directory(trees, hpcp_mod, tmp_path):
	"""base has y -> x and x/f; new has a real directory y with y/f and no x. y/f reaches x/f through the symlink
	today, but the patch removes y first and writes y/f into a new directory: applies, not refused."""
	new, base = trees
	_write(new / 'y' / 'f', 'moved')
	_write(base / 'x' / 'f', 'moved')
	os.symlink('x', str(base / 'y'))
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	assert _ops(rows) == {('D', 'x/'), ('D', 'x/f'), ('D', 'y'), ('A', 'y/'), ('A', 'y/f')}
	target = tmp_path / 'target'
	_clone(base, target)
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert summary['failed'] == 0
	assert (target / 'y').is_dir() and not (target / 'y').is_symlink() and not os.path.lexists(str(target / 'x'))
	assert _changes(hpcp_mod, new, target) == []


def test_apply_refuses_symlinked_parent_inside_patch_volume(tmp_path, hpcp_mod, reset_hpcp_globals):
	outside = tmp_path / 'outside'
	_write(outside / 'x.txt', 'secret')
	state = hpcp_mod.get_entry_state(str(outside / 'x.txt'))
	vol = tmp_path / 'vol'
	vol.mkdir()
	os.symlink(str(outside), vol / 'etc')
	rows = [hpcp_mod.ChangeRow('A', 'etc/x.txt', '-', state)]
	hpcp_mod.write_patch_metadata([str(vol)], rows, hpcp_mod.make_patch_meta(rows, 'n', 'b', 'root'))
	target = tmp_path / 'target'
	target.mkdir()
	with pytest.raises(RuntimeError, match='Incomplete patch'):
		hpcp_mod.apply_patch_tree([str(vol)], str(target), force=True, max_workers=1)
	assert os.listdir(str(target)) == []


def test_apply_refuses_payload_missing_from_every_volume(trees, hpcp_mod, tmp_path):
	new, base = trees
	vol, target = _conflicted(hpcp_mod, new, base, tmp_path)
	_write(target / 'mod.txt', 'old')
	os.unlink(vol / 'mod.txt')
	with pytest.raises(RuntimeError, match='Incomplete patch'):
		hpcp_mod.apply_patch_tree([str(vol)], str(target), force=True, max_workers=1)
	assert (target / 'mod.txt').read_text() == 'old' and (target / 'gone.txt').exists()


def test_apply_multi_volume_out_of_order(trees, hpcp_mod, tmp_path):
	new, base = trees
	_build_scenario(new, base)
	for i in range(6):
		_write(new / f'extra{i}.txt', f'extra {i}')
	v1, v2 = tmp_path / 'v1', tmp_path / 'v2'
	_patch_from(hpcp_mod, new, base, [v1, v2])
	target = tmp_path / 'target'
	_clone(base, target)
	summary = hpcp_mod.apply_patch_tree([str(v2), str(v1)], str(target), max_workers=1)
	assert summary['failed'] == 0
	assert _changes(hpcp_mod, new, target) == []


def test_apply_hash_algo_mismatch_still_applies(trees, hpcp_mod, tmp_path, capsys):
	new, base = trees
	_build_scenario(new, base)
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	other = 'blake2b' if hpcp_mod._hash_algo_name() == 'xxh64' else 'xxh64'
	manifest = str(vol / '.hpcp_diff' / 'manifest')
	meta = hpcp_mod.read_manifest(manifest)
	meta['hash_algo'] = other
	hpcp_mod.write_manifest(manifest, meta)
	target = tmp_path / 'target'
	_clone(base, target)
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert summary['failed'] == 0 and summary['forced'] == 0
	assert 'xattr digests are not compared' in capsys.readouterr().err


def test_apply_restores_hash_size_after_exception(trees, hpcp_mod, tmp_path, monkeypatch):
	new, base = trees
	monkeypatch.setattr(hpcp_mod, 'HASH_SIZE', 65536)
	vol, target = _conflicted(hpcp_mod, new, base, tmp_path)
	monkeypatch.setattr(hpcp_mod, 'HASH_SIZE', 1234)
	monkeypatch.setattr(sys, 'stdin', _FakeStdin(False))
	with pytest.raises(RuntimeError, match='Patch conflict'):
		hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert hpcp_mod.HASH_SIZE == 1234


def test_apply_rerun_after_interrupted_copy(trees, hpcp_mod, tmp_path, monkeypatch):
	new, base = trees
	_build_scenario(new, base)
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	restore = _break_copies(hpcp_mod, monkeypatch)
	monkeypatch.setattr(sys, 'stdin', _FakeStdin(False))
	first = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert first['failed'] > 0
	assert (target / 'mod.txt').read_text() == 'old'
	assert _temp_leftovers(target) == []
	restore()
	second = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert second['failed'] == 0 and second['forced'] == 0
	assert _changes(hpcp_mod, new, target) == []


def test_apply_forced_delete_of_directory_with_unlisted_entries(trees, hpcp_mod, tmp_path):
	new, base = trees
	_write(base / 'olddir' / 'a.txt', 'a')
	_write(new / 'other.txt', 'o')
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	_write(target / 'olddir' / 'unlisted.txt', 'extra')
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), force=True, max_workers=1)
	assert summary['failed'] == 0
	assert not (target / 'olddir').exists()


def test_apply_rows_never_removes_target_root(tmp_path, hpcp_mod, reset_hpcp_globals):
	fstate, dstate, _ = _states(hpcp_mod, tmp_path)
	target = tmp_path / 'target'
	_write(target / 'keep.txt', 'keep')
	rows = [hpcp_mod.ChangeRow('D', './', dstate, '-')]
	hpcp_mod._apply_rows(rows, [str(tmp_path / 'vol')], str(target))
	assert (target / 'keep.txt').exists()
	assert any(error.startswith('Patch format error') for error in hpcp_mod.ERRORS)


def test_states_equal_compare_hash_false_ignores_xattr_digest(hpcp_mod):
	eq = hpcp_mod.states_equal
	a = 'f:10:1717000000000000000:100644:0:0:aaaa:1234'
	b = 'f:10:1717000000000000000:100644:0:0:bbbb:5678'
	assert eq(a, b, compare_hash=False) and not eq(a, b, compare_hash=True)
	assert eq('d:40755:0:0:1234', 'd:40755:0:0:5678', compare_hash=False)
	assert not eq('d:40755:0:0:1234', 'd:40755:0:0:5678', compare_hash=True)


def test_apply_multi_volume_spread_payload_reversed(trees, hpcp_mod, tmp_path, reset_hpcp_globals):
	new, base = trees
	hpcp_mod.RANDOM_DESTINATION_SELECTION = True
	_build_scenario(new, base)
	for i in range(12):
		_write(new / f'extra{i}.txt', f'extra {i}')
	v1, v2 = tmp_path / 'v1', tmp_path / 'v2'
	_patch_from(hpcp_mod, new, base, [v1, v2])
	for volume in (v1, v2):
		assert any(p.is_file() and '.hpcp_diff' not in p.parts for p in volume.rglob('*'))
	target = tmp_path / 'target'
	_clone(base, target)
	summary = hpcp_mod.apply_patch_tree([str(v2), str(v1)], str(target), max_workers=1)
	assert summary['failed'] == 0
	assert _changes(hpcp_mod, new, target) == []


def test_apply_rejects_missing_patch_id_and_bad_hash_size(trees, hpcp_mod, tmp_path):
	new, base = trees
	vol, target = _conflicted(hpcp_mod, new, base, tmp_path)
	manifest = str(vol / '.hpcp_diff' / 'manifest')
	good = hpcp_mod.read_manifest(manifest)
	before = _snapshot(target)
	meta = dict(good)
	del meta['patch_id']
	hpcp_mod.write_manifest(manifest, meta)
	with pytest.raises(RuntimeError, match='Incomplete patch'):
		hpcp_mod.apply_patch_tree([str(vol)], str(target), force=True, max_workers=1)
	meta = dict(good)
	meta['hash_size'] = 'abc'
	hpcp_mod.write_manifest(manifest, meta)
	size_before = hpcp_mod.HASH_SIZE
	with pytest.raises(RuntimeError, match='Patch format error'):
		hpcp_mod.apply_patch_tree([str(vol)], str(target), force=True, max_workers=1)
	assert hpcp_mod.HASH_SIZE == size_before
	assert _snapshot(target) == before and (target / 'mod.txt').read_text() == 'local edit'


def test_apply_full_hash_same_size_same_mtime_change(trees, hpcp_mod, tmp_path, monkeypatch):
	new, base = trees
	monkeypatch.setattr(hpcp_mod, 'HASH_SIZE', 4)
	_write(new / 'big.bin', b'N' + b'x' * 200000)
	_write(base / 'big.bin', b'O' + b'x' * 200000)
	for root in (new, base):
		os.utime(root / 'big.bin', ns=(T0 * 10**9, T0 * 10**9))
	new_map, base_map = hpcp_mod.build_root_maps(str(new), str(base), max_workers=1)
	rows = hpcp_mod.compute_changes(new_map, base_map, full_hash=True, max_workers=1)
	assert ('M', 'big.bin') in _ops(rows)
	vol = tmp_path / 'vol'
	vol.mkdir()
	hpcp_mod.build_patch_tree(rows, str(new), [str(vol)], hpcp_mod.make_patch_meta(rows, 'n', 'b', 'root', full_hash=True), max_workers=1)
	target = tmp_path / 'target'
	_clone(base, target)
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert summary['failed'] == 0
	assert (target / 'big.bin').read_bytes() == b'N' + b'x' * 200000


def test_apply_breaks_hard_links_instead_of_writing_through(trees, hpcp_mod, tmp_path):
	new, base = trees
	_write(new / 'a', 'new content')
	_write(base / 'a', 'old')
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	ext = tmp_path / 'ext'
	os.link(target / 'a', ext)
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert summary['failed'] == 0
	assert (target / 'a').read_text() == 'new content'
	assert ext.read_text() == 'old'


# A default ACL: user::rwx, user:1234:r--, group::r-x, mask::r-x, other::r-x.
_DEFAULT_ACL = struct.pack('<I', 2) + b''.join(struct.pack('<HHI', tag, perm, ident) for tag, perm, ident in [
	(0x01, 7, 0xFFFFFFFF), (0x02, 4, 1234), (0x04, 5, 0xFFFFFFFF), (0x10, 5, 0xFFFFFFFF), (0x20, 5, 0xFFFFFFFF)])


@pytest.mark.parametrize('workers', [1, 2])
def test_apply_strips_inherited_default_acl(trees, hpcp_mod, tmp_path, workers):
	"""G1: a temporary file created under a default ACL inherits an access ACL the payload does not have."""
	new, base = trees
	_write(new / 'shared' / 'mod.txt', 'new content')
	_write(base / 'shared' / 'mod.txt', 'old')
	_write(new / 'shared' / 'add.txt', 'added')
	os.chmod(str(new / 'shared' / 'add.txt'), 0o640)
	try:
		# On both sides, so shared/ itself is unchanged; its files were created before the ACL and carry none.
		for directory in (new / 'shared', base / 'shared'):
			os.setxattr(str(directory), 'system.posix_acl_default', _DEFAULT_ACL)
	except OSError as e:
		pytest.skip(f'cannot set a default ACL here: {e}')
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	assert _ops(rows) == {('M', 'shared/mod.txt'), ('A', 'shared/add.txt')}
	target = tmp_path / 'target'
	_clone(base, target)
	os.setxattr(str(target / 'shared'), 'system.posix_acl_default', _DEFAULT_ACL)
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=workers)
	assert summary['failed'] == 0
	for name in ('mod.txt', 'add.txt'):
		assert 'system.posix_acl_access' not in os.listxattr(str(target / 'shared' / name))
	assert _changes(hpcp_mod, new, target) == []
	assert _temp_leftovers(target) == []


def _break_copies(hpcp_mod, monkeypatch):
	"""Make every apply-side copy_file call raise (max_workers=1 copies in-process); returns restore()."""
	real = hpcp_mod.copy_file

	def broken(*args, **kwargs):
		raise RuntimeError('simulated copy failure')

	monkeypatch.setattr(hpcp_mod, 'copy_file', broken)
	return lambda: monkeypatch.setattr(hpcp_mod, 'copy_file', real)


def test_interrupted_copy_keeps_old_xattrs(trees, hpcp_mod, tmp_path, monkeypatch):
	new, base = trees
	_write(new / 'm.txt', 'new content')
	_write(base / 'm.txt', 'old')
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	_set_xattr_or_skip(str(target / 'm.txt'), 'user.tag', b'1')
	# the base side carries the xattr too, so the patch is a plain content change
	restore = _break_copies(hpcp_mod, monkeypatch)
	first = hpcp_mod.apply_patch_tree([str(vol)], str(target), force=True, max_workers=1)
	assert first['failed'] > 0
	assert 'user.tag' in os.listxattr(str(target / 'm.txt'))
	restore()
	second = hpcp_mod.apply_patch_tree([str(vol)], str(target), force=True, max_workers=1)
	assert second['failed'] == 0
	assert (target / 'm.txt').read_text() == 'new content'


def test_directory_metadata_applied_after_files(trees, hpcp_mod, tmp_path, monkeypatch):
	new, base = trees
	_write(new / 'secret' / 'f.txt', 'inside')
	os.chmod(new / 'secret', 0o755)
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	real = hpcp_mod.copy_file
	modes = []

	def recording(*args, **kwargs):
		modes.append(stat.S_IMODE(os.stat(target / 'secret').st_mode))
		return real(*args, **kwargs)

	monkeypatch.setattr(hpcp_mod, 'copy_file', recording)
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert summary['failed'] == 0
	assert modes and all(mode == 0o700 for mode in modes)
	assert stat.S_IMODE(os.stat(target / 'secret').st_mode) == 0o755


def test_apply_rejects_reserved_key(tmp_path, hpcp_mod, reset_hpcp_globals):
	fstate, _, _ = _states(hpcp_mod, tmp_path)
	vol = tmp_path / 'vol'
	vol.mkdir()
	rows = [hpcp_mod.ChangeRow('A', '.hpcp_diff/x', '-', fstate)]
	hpcp_mod.write_patch_metadata([str(vol)], rows, hpcp_mod.make_patch_meta(rows, 'n', 'b', 'root'))
	target = tmp_path / 'target'
	target.mkdir()
	with pytest.raises(RuntimeError, match='Patch format error'):
		hpcp_mod.apply_patch_tree([str(vol)], str(target), force=True, max_workers=1)
	assert os.listdir(str(target)) == []


def test_apply_rows_symlink_pass_never_replaces_root(tmp_path, hpcp_mod, reset_hpcp_globals):
	_, dstate, _ = _states(hpcp_mod, tmp_path)
	target = tmp_path / 'target'
	_write(target / 'keep.txt', 'keep')
	rows = [hpcp_mod.ChangeRow('M', './', dstate, 'l:elsewhere')]
	hpcp_mod._apply_rows(rows, [str(tmp_path / 'vol')], str(target))
	assert target.is_dir() and not target.is_symlink() and (target / 'keep.txt').exists()


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


def test_process_get_diff_image_refuses_unchanged_reserved_name(trees, hpcp_mod, tmp_path):
	new, base = trees
	_write(new / '.hpcp_diff' / 'x', 'x')
	_write(base / '.hpcp_diff' / 'x', 'x')
	_write(new / 'a.txt', 'a')
	out = tmp_path / 'p.img'
	with pytest.raises(RuntimeError, match='reserves'):
		hpcp_mod.process_get_diff_image([str(new)], None, [str(base)], None, str(out), [], [], max_workers=1)
	assert not out.exists()


def test_process_get_diff_image_claims_every_volume_name(trees, hpcp_mod, tmp_path, monkeypatch):
	new, base = trees
	_write(new / 'a.txt', 'a')
	out = tmp_path / 'p.img'
	second = tmp_path / 'p_1.img'
	second.write_text('keep me')
	monkeypatch.setattr(hpcp_mod, '_plan_dest_images', lambda init_size, dest_image_size=0: (2, 1 << 20))

	def fail(*args, **kwargs):
		raise AssertionError('create_image must not be reached')

	monkeypatch.setattr(hpcp_mod, 'create_image', fail)
	with pytest.raises(RuntimeError, match=r'p_1\.img already exists'):
		hpcp_mod.process_get_diff_image([str(new)], None, [str(base)], None, str(out), [], [], max_workers=1)
	assert second.read_text() == 'keep me'
	assert not out.exists()


def test_process_get_diff_image_sizes_volumes_by_allocation(trees, hpcp_mod, tmp_path, monkeypatch):
	new, base = trees
	for i in range(200):
		_write(new / f'f{i:03d}', 'x')
	captured = {}

	class Sentinel(Exception):
		pass

	def capture(*args, **kwargs):
		captured['init_size'] = kwargs['init_size']
		raise Sentinel()

	monkeypatch.setattr(hpcp_mod, 'create_image', capture)
	with pytest.raises(Sentinel):
		hpcp_mod.process_get_diff_image([str(new)], None, [str(base)], None, str(tmp_path / 'p.img'), [], [], max_workers=1)
	assert captured['init_size'] >= 200 * 4096
	# F9: every A/M row also reserves 512 bytes of inode overhead (xfs).
	rows = _changes(hpcp_mod, new, base)
	payload = [row for row in rows if row.op in ('A', 'M')]
	allocation = sum(hpcp_mod._align_up(max(hpcp_mod.decode_state(row.new)['size'], 1)) if row.new.startswith('f:')
		else hpcp_mod._IMAGE_BLOCK for row in payload)
	listing = 2 * sum(len(row.path) + len(row.old) + len(row.new) + 8 for row in rows)
	assert captured['init_size'] == allocation + listing + 512 * len(payload)


def _fake_image_mounts(hpcp_mod, monkeypatch, roots):
	"""Make _mount_single_fs_image 'mount' image i at the prepared directory roots[image path]."""
	def fake_mount(image, read_only, mount_points, loop_devices):
		root = str(roots[os.path.abspath(image)])
		mount_points.append(root)
		return root + os.sep
	monkeypatch.setattr(hpcp_mod, '_mount_single_fs_image', fake_mount)


def _rows_before_mkfs(hpcp_mod, monkeypatch, *args, **kwargs):
	"""The rows process_get_diff_image would write, captured before anything is created."""
	captured = {}

	class Sentinel(Exception):
		pass

	def capture(rows, *rest, **options):
		captured['rows'] = rows
		raise Sentinel()

	monkeypatch.setattr(hpcp_mod, 'make_patch_meta', capture)
	with pytest.raises(Sentinel):
		hpcp_mod.process_get_diff_image(*args, **kwargs)
	return captured['rows']


def test_image_roots_ignore_lost_and_found(trees, hpcp_mod, tmp_path, monkeypatch):
	"""F10: an ext4 image's lost+found/ is not part of the tree it holds."""
	new, base = trees
	image_root = tmp_path / 'image_root'
	_write(image_root / 'a.txt', 'a')
	_write(image_root / 'lost+found' / '#1234', 'orphan')
	_write(base / 'a.txt', 'a')
	image = tmp_path / 'new.img'
	image.write_bytes(b'')
	_fake_image_mounts(hpcp_mod, monkeypatch, {str(image): image_root})
	rows = _rows_before_mkfs(hpcp_mod, monkeypatch, [], [str(image)], [str(base)], None, str(tmp_path / 'p.img'), [], [], max_workers=1)
	assert not [row for row in rows if row.path.startswith('lost+found')]
	assert hpcp_mod._drop_lost_found({'./': '/m', 'lost+found/': '/m/lost+found', 'lost+found/x': '/m/lost+found/x',
		'lost+foundling': '/m/lost+foundling'}) == {'./': '/m', 'lost+foundling': '/m/lost+foundling'}


def test_image_side_drops_lost_and_found_on_both_sides(trees, hpcp_mod, tmp_path, monkeypatch):
	"""G3: `-si golden.img -d /` must not delete (or touch) the live root's own lost+found/."""
	new, base = trees
	image_root = tmp_path / 'image_root'
	_write(image_root / 'a.txt', 'a')
	_write(image_root / 'added.txt', 'new file')
	_write(image_root / 'lost+found' / 'x', 'image orphan')
	_write(base / 'a.txt', 'a')
	_write(base / 'lost+found' / 'y', 'live orphan')
	image = tmp_path / 'golden.img'
	image.write_bytes(b'')
	_fake_image_mounts(hpcp_mod, monkeypatch, {str(image): image_root})
	rows = _rows_before_mkfs(hpcp_mod, monkeypatch, [], [str(image)], [str(base)], None, str(tmp_path / 'p.img'), [], [], max_workers=1)
	assert not [row for row in rows if 'lost+found' in row.path]
	assert ('A', 'added.txt') in _ops(rows)


def test_diff_ignores_its_own_image_mounted_inside_the_base(trees, hpcp_mod, tmp_path, monkeypatch):
	"""F14: `-si new.img -d /` must not list new.img's own mount point as part of the base."""
	new, base = trees
	_write(base / 'keep.txt', 'k')
	mounted = base / 'tmpmnt'
	_write(mounted / 'keep.txt', 'k')
	_write(mounted / 'inside.txt', 'image content')
	os.utime(str(mounted / 'keep.txt'), ns=(os.stat(str(base / 'keep.txt')).st_atime_ns, os.stat(str(base / 'keep.txt')).st_mtime_ns))
	image = tmp_path / 'new.img'
	image.write_bytes(b'')
	_fake_image_mounts(hpcp_mod, monkeypatch, {str(image): mounted})
	rows = _rows_before_mkfs(hpcp_mod, monkeypatch, [], [str(image)], [str(base)], None, str(tmp_path / 'p.img'), [], [], max_workers=1)
	assert not [row for row in rows if row.path.startswith('tmpmnt')]
	assert ('A', 'inside.txt') in _ops(rows)


def test_diff_ignores_the_store_mounted_inside_the_source(trees, hpcp_mod, tmp_path, monkeypatch):
	"""F14: `-diff / -di store.img` must not list the store's own mount point as part of the source."""
	new, base = trees
	_write(new / 'a.txt', 'a')
	store_root = new / 'storemnt'
	_write(store_root / os.path.relpath(str(new), '/') / 'a.txt', 'a')
	image = tmp_path / 'store.img'
	image.write_bytes(b'')
	_fake_image_mounts(hpcp_mod, monkeypatch, {str(image): store_root})
	rows = _rows_before_mkfs(hpcp_mod, monkeypatch, [str(new)], None, [], str(image), str(tmp_path / 'p.img'), [], [], max_workers=1)
	prefix = os.path.relpath(str(new), '/').replace(os.sep, '/') + '/'
	assert not [row for row in rows if row.path.startswith(prefix + 'storemnt')]


def test_mount_single_fs_image_probes_type_with_blkid(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""F7: the read-only options come from blkid -p; a failed probe falls back to plain ro and records nothing."""
	image = tmp_path / 'fs.img'
	image.write_bytes(b'')
	monkeypatch.setattr(hpcp_mod, 'create_loop_device', lambda path, read_only=False: '/dev/loop-fake')
	monkeypatch.setattr(hpcp_mod, 'get_partitions', lambda device: [])
	monkeypatch.setattr(os.path, 'ismount', lambda path: True)
	commands = []
	blkid = {}

	def fake_run(command, *args, **kwargs):
		commands.append(list(command))
		if command[0] == 'blkid':
			if blkid.get('fail'):
				hpcp_mod.ERRORS.append(f"Task return code error: Command '{command}' failed with return code 2.")
				raise RuntimeError('blkid failed')
			return ['xfs']
		return ['']

	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', fake_run)
	mount_points = []
	try:
		hpcp_mod._mount_single_fs_image(str(image), True, mount_points, [])
		assert ['blkid', '-p', '-s', 'TYPE', '-o', 'value', '/dev/loop-fake'] in commands
		assert commands[-1][:3] == ['mount', '-o', 'ro,nouuid,norecovery']
		blkid['fail'] = True
		hpcp_mod._mount_single_fs_image(str(image), True, mount_points, [])
		assert commands[-1][:3] == ['mount', '-o', 'ro']
		assert hpcp_mod.ERRORS == []
	finally:
		for mount_point in mount_points:
			os.rmdir(mount_point)


@requires_root_and_image_tools
@pytest.mark.root
@pytest.mark.loop
def test_cloned_image_mounts_for_diff(trees, hpcp_mod, tmp_path, reset_hpcp_globals):
	hpcp_mod.HASH_SIZE = 65536
	tree = tmp_path / 'live' / 'a'
	_write(tree / 'f.txt', 'v1')
	base_img = str(tmp_path / 'base.img')
	rc = hpcp_mod.hpcp([str(tree)], dest_image=base_img, single_thread=True, max_workers=1,
		batch=True, do_not_remove_files_while_listing=True)
	assert rc in (None, 0)
	clone = str(tmp_path / 'clone.img')
	shutil.copyfile(base_img, clone)
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		assert hpcp_mod.process_get_diff_image([], [clone], [], base_img, str(tmp_path / 'p.img'), mount_points, loop_devices, max_workers=1) is None


def _make_xfs_image(hpcp_mod, path, populate):
	"""A sparse 320 MiB xfs image (hpcp formats images above 300 MiB as xfs) whose root holds populate(root)."""
	with open(path, 'wb') as f:
		f.truncate(320 * 1024 * 1024)
	subprocess.run(['mkfs.xfs', '-f', '-q', path], check=True)
	_change_image(hpcp_mod, path, populate)


def _change_image(hpcp_mod, path, change):
	"""Mount path read-write, run change(root), unmount."""
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		change(hpcp_mod._mount_single_fs_image(path, False, mount_points, loop_devices))


@requires_root_and_image_tools
@pytest.mark.skipif(shutil.which('mkfs.xfs') is None, reason='needs mkfs.xfs')
@pytest.mark.root
@pytest.mark.loop
def test_xfs_image_to_image_patch_applies_to_the_base_image(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""F4: -si new.img -di base.img on byte-identical xfs clones (same UUID), then -siff onto base.img."""
	hpcp_mod.HASH_SIZE = 65536
	base_img, new_img, out = str(tmp_path / 'base.img'), str(tmp_path / 'new.img'), str(tmp_path / 'patch.img')

	def populate(root):
		_write(os.path.join(root, 'keep.txt'), 'keep')
		_write(os.path.join(root, 'mod.txt'), 'old')
		_write(os.path.join(root, 'gone.txt'), 'bye')
		_write(os.path.join(root, 'olddir', 'a.txt'), 'a')
		_write(os.path.join(root, 'sub', 'x.txt'), 'x')

	def change(root):
		_write(os.path.join(root, 'mod.txt'), 'new content', mtime=T0 + 100)
		_write(os.path.join(root, 'add.txt'), 'added')
		os.unlink(os.path.join(root, 'gone.txt'))
		shutil.rmtree(os.path.join(root, 'olddir'))
		_write(os.path.join(root, 'newdir', 'f.txt'), 'fresh')
		os.chmod(os.path.join(root, 'sub'), 0o700)

	_make_xfs_image(hpcp_mod, base_img, populate)
	subprocess.run(['cp', '--sparse=always', base_img, new_img], check=True)
	_change_image(hpcp_mod, new_img, change)
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		assert hpcp_mod.process_get_diff_image([], [new_img], [], base_img, out, mount_points, loop_devices, max_workers=2) == out
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		summary = hpcp_mod.process_load_diff_image([out], [], [], base_img, mount_points, loop_devices, max_workers=2)
	assert summary['failed'] == 0
	assert (summary['added'], summary['deleted'], summary['modified']) == (3, 3, 2)
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		new_root = hpcp_mod._mount_single_fs_image(new_img, True, mount_points, loop_devices)
		base_root = hpcp_mod._mount_single_fs_image(base_img, True, mount_points, loop_devices)
		assert hpcp_mod.compute_changes(*hpcp_mod.build_root_maps(new_root, base_root, max_workers=1), max_workers=1) == []


def test_hpcp_rejects_bad_patch_arguments(tmp_path, hpcp_mod, reset_hpcp_globals, capsys):
	new, base = str(tmp_path / 'new'), str(tmp_path / 'base')
	os.makedirs(new)
	os.makedirs(base)
	cases = [
		(dict(src_path=[new], dest_paths=[base], force_apply=True), '-fa'),
		(dict(src_path=[new], dest_paths=[base], get_diff_image='auto', load_diff_image=['p.img']), 'cannot be used together'),
		(dict(src_path=[new], dest_paths=[base], get_diff_image='auto', dd=True, src_image=['x.img']), '-dd'),
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


def _restore_scenario(hpcp_mod, tmp_path):
	"""The target spells EFI/BOOT/x.efi, and EFI/Boot is a symlink to BOOT: a lookup of EFI/Boot/x.efi finds the
	same file, as on a case-insensitive filesystem. The patch deletes EFI/BOOT/x.efi and adds EFI/Boot/x.efi with
	the same content (a case-only rename)."""
	target, vol = tmp_path / 'target', tmp_path / 'vol'
	_write(target / 'EFI' / 'BOOT' / 'x.efi', 'loader')
	os.symlink('BOOT', str(target / 'EFI' / 'Boot'))
	_write(vol / 'EFI' / 'Boot' / 'x.efi', 'loader')
	state = hpcp_mod.get_entry_state
	R = hpcp_mod.ChangeRow
	rows = [R('D', 'EFI/BOOT/x.efi', state(str(target / 'EFI' / 'BOOT' / 'x.efi')), '-'),
		R('A', 'EFI/Boot/x.efi', '-', state(str(vol / 'EFI' / 'Boot' / 'x.efi')))]
	return target, vol, rows


def test_apply_restores_a_skipped_entry_a_delete_took_away(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""The add is skipped as already done (the file is there under the other spelling), then the delete removes that
	very file. The repair step writes the add again; verify accepts the deleted name because it is the same entry as
	the written one, differing only by case."""
	hpcp_mod.HASH_SIZE = 65536
	target, vol, rows = _restore_scenario(hpcp_mod, tmp_path)
	plan = hpcp_mod._plan_section('root', rows, [str(vol)], str(target), True, False, 1)
	assert {check.row.path: check.action for check in plan.checks} == {'EFI/BOOT/x.efi': 'apply', 'EFI/Boot/x.efi': 'skip'}
	summary = hpcp_mod._apply_section(plan, 1, False, False, True)
	assert summary == {'deleted': 1, 'added': 0, 'modified': 0, 'skipped': 0, 'restored': 1, 'forced': 0, 'failed': 0}
	assert hpcp_mod.ERRORS == []
	assert (target / 'EFI' / 'Boot' / 'x.efi').read_text() == 'loader'


def test_contradictory_patch_on_an_aliased_target_fails_at_verify(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""D x/f + A y/f where y is a symlink to x: names that differ by more than case. The write is restored (the content
	is kept) and the delete's verify fails: loud, not silent."""
	hpcp_mod.HASH_SIZE = 65536
	target, vol = tmp_path / 'target', tmp_path / 'vol'
	_write(target / 'x' / 'f', 'moved')
	os.symlink('x', str(target / 'y'))
	_write(vol / 'y' / 'f', 'moved')
	state = hpcp_mod.get_entry_state
	R = hpcp_mod.ChangeRow
	rows = [R('D', 'x/f', state(str(target / 'x' / 'f')), '-'), R('A', 'y/f', '-', state(str(vol / 'y' / 'f')))]
	plan = hpcp_mod._plan_section('root', rows, [str(vol)], str(target), True, False, 1)
	summary = hpcp_mod._apply_section(plan, 1, False, False, True)
	assert summary['restored'] == 1 and summary['failed'] == 1
	assert (target / 'x' / 'f').read_text() == 'moved'
	assert len(hpcp_mod.ERRORS) == 1 and hpcp_mod.ERRORS[0].startswith('Patch verify error: D x/f: expected -')


def test_entry_lost_to_a_forced_type_change_is_reported(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""-fa replacing directory x with a file removes x's tree; y/g, reached through the symlink y -> x and skipped as
	already done, goes with it and cannot be restored (y now points at a file): a failure, not a silent loss."""
	hpcp_mod.HASH_SIZE = 65536
	target, vol = tmp_path / 'target', tmp_path / 'vol'
	_write(target / 'x' / 'g', 'g')
	os.symlink('x', str(target / 'y'))
	_write(vol / 'x', 'now a file')
	_write(vol / 'y' / 'g', 'g')
	state = hpcp_mod.get_entry_state
	R = hpcp_mod.ChangeRow
	rows = [R('A', 'x', '-', state(str(vol / 'x'))), R('A', 'y/g', '-', state(str(vol / 'y' / 'g')))]
	plan = hpcp_mod._plan_section('root', rows, [str(vol)], str(target), True, False, 1)
	assert {check.row.path: check.action for check in plan.checks} == {'x': 'conflict', 'y/g': 'skip'}
	summary = hpcp_mod._apply_section(plan, 1, False, False, True)
	assert summary['forced'] == 1 and summary['restored'] == 1 and summary['failed'] >= 1
	assert (target / 'x').read_text() == 'now a file'
	assert any('y/g' in error for error in hpcp_mod.ERRORS)


def test_restore_without_a_payload_is_a_failure(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Review Focus 2: a skipped row's payload is not pre-checked (nothing is written for it normally); when it must be
	restored and the volume lacks it, that is an error and a verify failure."""
	hpcp_mod.HASH_SIZE = 65536
	target, vol, rows = _restore_scenario(hpcp_mod, tmp_path)
	os.unlink(str(vol / 'EFI' / 'Boot' / 'x.efi'))
	plan = hpcp_mod._plan_section('root', rows, [str(vol)], str(target), True, False, 1)
	summary = hpcp_mod._apply_section(plan, 1, False, False, True)
	assert summary['restored'] == 1 and summary['failed'] == 1
	assert any(error.startswith('Incomplete patch: payload for EFI/Boot/x.efi') for error in hpcp_mod.ERRORS)


def test_restore_across_sections_does_not_retry_a_failed_sections_own_restore(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""The section's own repair step already tried (and counted) the restore; the pass across sections, which runs
	because the section removed something, leaves it alone."""
	hpcp_mod.HASH_SIZE = 65536
	target, vol, rows = _restore_scenario(hpcp_mod, tmp_path)
	os.unlink(str(vol / 'EFI' / 'Boot' / 'x.efi'))
	plan = hpcp_mod._plan_section('root', rows, [str(vol)], str(target), True, False, 1)
	outcome = {}
	summary = hpcp_mod._apply_section(plan, 1, False, False, True, outcome)
	assert (outcome['removed'], outcome['restored']) == (True, [rows[1]])
	hpcp_mod._restore_across_sections([(plan, summary, outcome)], 1, False, False, True)
	assert (summary['restored'], summary['failed']) == (1, 1)
	assert len([error for error in hpcp_mod.ERRORS if error.startswith('Incomplete patch')]) == 1


def test_rerun_after_a_completed_apply_restores_nothing(trees, hpcp_mod, tmp_path):
	new, base = trees
	_write(base / 'gone.txt', 'bye')
	_write(new / 'kept.txt', 'kept')
	_write(base / 'kept.txt', 'kept')
	_write(new / 'add.txt', 'add')
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	first = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert (first['deleted'], first['added'], first['restored'], first['failed']) == (1, 1, 0, 0)
	again = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert (again['deleted'], again['added'], again['restored'], again['failed']) == (0, 0, 0, 0)


def test_fold_name_keeps_names_fat_keeps_apart(hpcp_mod):
	fold = hpcp_mod._fold_name
	assert fold('EFI/Boot/bootx64.efi') == fold('efi/BOOT/BOOTX64.EFI') == 'EFI/BOOT/BOOTX64.EFI'
	# 'ß'.upper() is 'SS' (two characters): kept, so 'Straße' and 'STRASSE' stay two names, as FAT keeps them.
	assert fold('Straße') == 'STRAßE' != fold('STRASSE')


def test_repair_leaves_surviving_skipped_rows_alone(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""A delete really removes something, but the skipped row's file survives: it is not restored (not rewritten)."""
	hpcp_mod.HASH_SIZE = 65536
	target, vol = tmp_path / 'target', tmp_path / 'vol'
	_write(target / 'gone.txt', 'bye')
	_write(target / 'keep.txt', 'keep')
	_write(vol / 'keep.txt', 'keep')
	state = hpcp_mod.get_entry_state
	R = hpcp_mod.ChangeRow
	rows = [R('D', 'gone.txt', state(str(target / 'gone.txt')), '-'), R('A', 'keep.txt', '-', state(str(vol / 'keep.txt')))]
	before = os.stat(str(target / 'keep.txt'))
	plan = hpcp_mod._plan_section('root', rows, [str(vol)], str(target), True, False, 1)
	assert {check.row.path: check.action for check in plan.checks} == {'gone.txt': 'apply', 'keep.txt': 'skip'}
	summary = hpcp_mod._apply_section(plan, 1, False, False, True)
	assert summary == {'deleted': 1, 'added': 0, 'modified': 0, 'skipped': 1, 'restored': 0, 'forced': 0, 'failed': 0}
	after = os.stat(str(target / 'keep.txt'))
	assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
	assert not (target / 'gone.txt').exists()


def test_failed_forced_rmtree_still_runs_the_repair_step(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	hpcp_mod.HASH_SIZE = 65536
	target, vol = tmp_path / 'target', tmp_path / 'vol'
	_write(target / 'd' / 'a', 'a')
	_write(target / 'keep.txt', 'k')
	_write(vol / 'keep.txt', 'k')
	state = hpcp_mod.get_entry_state
	R = hpcp_mod.ChangeRow
	row = R('D', 'd/', state(str(target / 'd')), '-')
	skipped = R('A', 'keep.txt', '-', state(str(vol / 'keep.txt')))
	def partial_walk(path, *args, **kwargs):
		os.unlink(str(target / 'keep.txt'))
		return [], ['a: boom']
	monkeypatch.setattr(hpcp_mod, '_remove_tree_around_mounts', partial_walk)
	restored, _ = hpcp_mod._apply_rows([row], [str(vol)], str(target), forced_rows=[row], skipped=[skipped])
	assert restored == [skipped] and (target / 'keep.txt').read_text() == 'k'


@pytest.mark.parametrize('fs_type', ['vfat', 'exfat', 'hfsplus', 'fuseblk', 'cifs', '', 'ext4', 'nfs4'])
def test_single_fs_case_only_rename_applies_whatever_the_reported_type(trees, hpcp_mod, tmp_path, reset_hpcp_globals,
		monkeypatch, fs_type):
	"""A case-only rename is a delete plus an add; it applies on any filesystem (no refusal, -fa or not)."""
	new, base = trees
	_case_only_rename(new, base)
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	assert _ops(rows) == {('D', 'EFI/BOOT/'), ('D', 'EFI/BOOT/BOOTX64.EFI'), ('A', 'EFI/Boot/'), ('A', 'EFI/Boot/BOOTX64.EFI'),
		('M', 'EFI/grub.cfg')}
	target = tmp_path / 'target'
	_clone(base, target)
	_fake_mounts(hpcp_mod, monkeypatch, (target, fs_type))
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert (summary['deleted'], summary['added'], summary['modified'], summary['failed']) == (2, 2, 1, 0)
	assert _changes(hpcp_mod, new, target) == []


def _mount_entry(point, fs_type, source='none', ids=None):
	entry = {'devnum': (0, 1), 'root': '/', 'mount_point': os.path.realpath(str(point)), 'options': ['rw'],
		'fstype': fs_type, 'source': source}
	if ids:
		entry['mount_id'], entry['parent_id'] = ids
	return entry


def test_mount_scopes_follow_mountinfo_order(tmp_path, hpcp_mod):
	"""The deepest visible mount holding the real path is the target's. A mount is hidden when path lookup does not
	reach it: a mount made over its place (its mount point or a parent path) that it does not sit on in the parent
	chain. Mounts below the target come after, deepest last. Entries without ids are never hidden."""
	scopes = hpcp_mod._mount_scopes
	esp = tmp_path / 'mnt' / 'esp'
	esp.mkdir(parents=True)
	link = tmp_path / 'link'
	os.symlink(str(esp), str(link))
	# Without ids: nothing is hidden; of equal depth the later one wins; a link is looked up by its real path.
	assert scopes(str(link), [_mount_entry('/', 'ext4'), _mount_entry(tmp_path / 'mnt', 'xfs'), _mount_entry(esp, 'vfat')])[0][1] == 'vfat'
	assert scopes(str(esp), [_mount_entry('/', 'ext4'), _mount_entry(esp, 'ext4'), _mount_entry(esp, 'vfat')])[0][1] == 'vfat'
	assert scopes(str(esp), [_mount_entry('/', 'ext4'), _mount_entry(esp, 'vfat'), _mount_entry(tmp_path / 'mnt', 'xfs')])[0][1] == 'vfat'
	# (b) A true overmount: the xfs on mnt (parent: the root) is made over the vfat on mnt/esp, whose parent is the root too.
	entries = [_mount_entry('/', 'ext4', ids=(1, 0)), _mount_entry(esp, 'vfat', ids=(2, 1)), _mount_entry(tmp_path / 'mnt', 'xfs', ids=(3, 1))]
	assert scopes(str(esp), entries)[0][1] == 'xfs'
	# ... but a vfat made on the xfs (its parent) is visible.
	entries = [_mount_entry('/', 'ext4', ids=(1, 0)), _mount_entry(tmp_path / 'mnt', 'xfs', ids=(3, 1)), _mount_entry(esp, 'vfat', ids=(2, 3))]
	assert scopes(str(esp), entries)[0][1] == 'vfat'
	# (c) Two mounts on one mount point, the second made on the first: the second is on top.
	entries = [_mount_entry('/', 'ext4', ids=(1, 0)), _mount_entry(esp, 'ext4', ids=(2, 1)), _mount_entry(esp, 'vfat', ids=(3, 2))]
	assert scopes(str(esp), entries)[0][1] == 'vfat'
	# (a) initramfs style: mounts made before the root switch are listed before the root they sit on.
	sys_like = tmp_path / 'sys-like'
	sys_like.mkdir()
	entries = [_mount_entry(sys_like, 'sysfs', ids=(20, 35)), _mount_entry(tmp_path / 'run', 'tmpfs', ids=(21, 35)),
		_mount_entry('/', 'ext4', ids=(35, 1))]
	assert scopes(str(sys_like), entries)[0][1] == 'sysfs'
	assert scopes('/', entries)[0][1] == 'ext4'
	assert [scope[1] for scope in scopes(str(tmp_path), [_mount_entry(tmp_path, 'ext4', ids=(35, 1))] + entries[:1])] == ['ext4', 'sysfs']
	entries = [_mount_entry(tmp_path, 'ext4', '/dev/sda2'), _mount_entry(tmp_path / 'boot' / 'efi', 'vfat', '/dev/sda1'),
		_mount_entry(tmp_path / 'boot' / 'efi' / 'deeper', 'exfat'), _mount_entry(tmp_path / 'a' / 'b', 'vfat'),
		_mount_entry(tmp_path / 'a', 'tmpfs')]
	assert scopes(str(tmp_path), entries) == [('', 'ext4', '/dev/sda2'), ('a', 'tmpfs', 'none'), ('a/b', 'vfat', 'none'),
		('boot/efi', 'vfat', '/dev/sda1'), ('boot/efi/deeper', 'exfat', 'none')]
	assert scopes(str(tmp_path), []) == [('', '', '')]


def test_mountinfo_entries_parse_the_mount_and_parent_ids(tmp_path, hpcp_mod):
	path = tmp_path / 'mountinfo'
	path.write_text('35 1 8:2 / / rw,relatime shared:1 - ext4 /dev/sda2 rw\n'
		'20 35 0:5 / /sys rw,nosuid shared:7 - sysfs sysfs rw\n'
		'garbage\n')
	entries = hpcp_mod._mountinfo_entries(str(path))
	assert [(e['mount_id'], e['parent_id'], e['mount_point'], e['fstype'], e['source'], e['devnum']) for e in entries] == [
		(35, 1, '/', 'ext4', '/dev/sda2', (8, 2)), (20, 35, '/sys', 'sysfs', 'sysfs', (0, 5))]


def test_mount_scopes_follow_what_path_lookup_reaches(tmp_path, hpcp_mod):
	"""A buried mount stays hidden and does not hide what is mounted after the overmount."""
	scopes = hpcp_mod._mount_scopes
	mnt = tmp_path / 'mnt'
	(mnt / 'boot').mkdir(parents=True)
	root = _mount_entry(tmp_path, 'ext4', ids=(1897, 1))
	# ESP at mnt/boot, root-ish xfs mounted over mnt, the ESP mounted at mnt/boot again (on the xfs).
	entries = [root, _mount_entry(mnt / 'boot', 'hsrc', ids=(1925, 1897)), _mount_entry(mnt, 'xsrc', ids=(1926, 1897)),
		_mount_entry(mnt / 'boot', 'msrc', ids=(1927, 1926))]
	assert scopes(str(mnt / 'boot'), entries)[0][1] == 'msrc'
	assert scopes(str(mnt), entries) == [('', 'xsrc', 'none'), ('boot', 'msrc', 'none')]
	# Deeper: mnt2/data buried by a mount over mnt2, then a new mount at mnt2/data/sub.
	mnt2 = tmp_path / 'mnt2'
	(mnt2 / 'data' / 'sub').mkdir(parents=True)
	entries = [root, _mount_entry(mnt2 / 'data', 'old', ids=(10, 1897)), _mount_entry(mnt2, 'over', ids=(11, 1897)),
		_mount_entry(mnt2 / 'data' / 'sub', 'new', ids=(12, 11))]
	assert scopes(str(mnt2 / 'data' / 'sub'), entries)[0][1] == 'new'
	assert scopes(str(mnt2 / 'data'), entries)[0][1] == 'over'
	# A cycle of parent ids ends: both count as reached, but each is stacked on the other, so neither is visible.
	assert scopes(str(mnt2), [_mount_entry(mnt2, 'a', ids=(1, 2)), _mount_entry(mnt2, 'b', ids=(2, 1))])[0][1] == ''


def test_every_member_of_a_parent_cycle_counts_as_reached(tmp_path, hpcp_mod):
	"""A parent-id cycle (impossible on a real system) counts as reached for every member, in whatever order the
	entries come: B at x and A at x/a are each other's parent, and C, another child of B on A's mount point, would
	hide A were A reached only through B."""
	import itertools
	x = tmp_path / 'x'
	cycle = [_mount_entry(x, 'b', ids=(2, 1)), _mount_entry(x / 'a', 'a', ids=(1, 2))]
	sibling = _mount_entry(x / 'a', 'c', ids=(3, 2))
	for entries in itertools.permutations(cycle + [sibling]):
		hidden = hpcp_mod._hidden_mounts(list(entries))
		assert [entry['fstype'] for index, entry in enumerate(entries) if index in hidden] == ['c']


def test_self_parented_root_mount_hides_nothing(tmp_path, hpcp_mod):
	"""The kernel lists a namespace's root mount with its own id as parent."""
	esp = tmp_path / 'boot' / 'efi'
	esp.mkdir(parents=True)
	entries = [_mount_entry('/', 'ext4', ids=(35, 35)), _mount_entry(tmp_path / 'boot', 'xfs', ids=(36, 35)),
		_mount_entry(esp, 'vfat', ids=(37, 36))]
	assert hpcp_mod._mount_scopes(str(esp), entries)[0][1] == 'vfat'
	assert hpcp_mod._mount_scopes(str(tmp_path / 'boot'), entries)[0][1] == 'xfs'
	assert hpcp_mod._mount_scopes('/', entries)[0][1] == 'ext4'
	assert hpcp_mod._compare_metadata((str(esp),), entries) is False


def test_mount_visibility_scales_with_deep_chains_and_many_siblings(tmp_path, hpcp_mod):
	import time
	# A 5000-deep chain listed children-first: every mount on the same point of its parent, only the last is visible.
	point = tmp_path / 'deep'
	entries = [_mount_entry(point, f'fs{i}', ids=(i + 100, i + 99)) for i in range(5000, 0, -1)]
	hidden = hpcp_mod._hidden_mounts(entries)
	assert [entries[i]['fstype'] for i in range(len(entries)) if i not in hidden] == ['fs5000']
	assert hpcp_mod._mount_scopes(str(point), entries)[0][1] == 'fs5000'
	# 20000 siblings under one parent resolve quickly.
	root = _mount_entry('/', 'ext4', ids=(1, 0))
	siblings = [_mount_entry(tmp_path / 'sib' / f'm{i}', 'tmpfs', ids=(i + 10, 1)) for i in range(20000)]
	start = time.time()
	assert hpcp_mod._hidden_mounts([root] + siblings) == set()
	assert time.time() - start < 2


def test_fuseblk_scopes_are_probed_only_where_used(tmp_path, hpcp_mod, monkeypatch):
	probed = []
	def probe(device):
		probed.append(device)
		return 'ntfs'
	monkeypatch.setattr(hpcp_mod, '_probe_fs_type', probe)
	new, base = tmp_path / 'new', tmp_path / 'base'
	new.mkdir()
	base.mkdir()
	entries = [_mount_entry('/', 'ext4'), _mount_entry(new, 'fuseblk', '/dev/fakeX'), _mount_entry(base / 'below', 'fuseblk', '/dev/fakeY')]
	assert hpcp_mod._mount_scopes(str(new), entries)[0][1] == 'fuseblk' and probed == []
	assert hpcp_mod._scope_fs_type('fuseblk.exfat', '/dev/fakeX') == 'ntfs'
	assert hpcp_mod._compare_metadata((str(new), str(base)), entries) is False
	assert probed == ['/dev/fakeX', '/dev/fakeX']
	probed.clear()
	assert hpcp_mod._scope_fs_type('fuseblk', 'none') == 'fuseblk' and probed == []


def test_apply_probes_only_the_fuseblk_scopes_it_plans(trees, hpcp_mod, tmp_path, reset_hpcp_globals, monkeypatch):
	new, base = trees
	_write(new / 'a' / 'f', 'new')
	_write(base / 'a' / 'f', 'old')
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	(target / 'unused').mkdir()
	(target / 'ntfs').mkdir()
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda *args, **kwargs: [
		_mount_entry(target, 'ext4'), _mount_entry(target / 'unused', 'fuseblk', '/dev/unused'),
		_mount_entry(target / 'a', 'fuseblk', '/dev/used')])
	probed = []
	monkeypatch.setattr(hpcp_mod, '_probe_fs_type', lambda device: probed.append(device) or 'ntfs')
	hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert probed == ['/dev/used']


def test_apply_stops_after_a_scope_that_failed(trees, hpcp_mod, tmp_path, reset_hpcp_globals, monkeypatch):
	new, base = trees
	_write(new / 'etc' / 'conf', 'new conf')
	_write(base / 'etc' / 'conf', 'old conf')
	_write(new / 'boot' / 'efi' / 'grub.cfg', 'new cfg')
	_write(base / 'boot' / 'efi' / 'grub.cfg', 'old cfg')
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	_fake_mounts(hpcp_mod, monkeypatch, (target, 'ext4'), (target / 'boot' / 'efi', 'vfat'))
	applied = []
	real_apply_section = hpcp_mod._apply_section
	def apply_section(plan, *args):
		applied.append(plan.name)
		summary = real_apply_section(plan, *args)
		if plan.name == 'root':
			summary['failed'] = 1
		return summary
	monkeypatch.setattr(hpcp_mod, '_apply_section', apply_section)
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert applied == ['root'] and summary['failed'] == 1
	assert (target / 'boot' / 'efi' / 'grub.cfg').read_text() == 'old cfg'
	assert any(error.startswith('Patch verify error: stopping: scope root failed') for error in hpcp_mod.ERRORS)


def _cross_scope_move(hpcp_mod, tmp_path, new, base, into_mount):
	"""The golden moves y/f to m/x/f (into_mount) or back. Returns (volume, the deleted key, make_target):
	make_target(name) clones the node, where y -> m/x makes both names one file. With m mounted below the target,
	the delete applies in one scope and the add is skipped as already present in the other."""
	moved, kept = ('m/x/f', 'y/f') if into_mount else ('y/f', 'm/x/f')
	(new / 'y').mkdir()
	(new / 'm' / 'x').mkdir(parents=True)
	(base / 'y').mkdir()
	(base / 'm' / 'x').mkdir(parents=True)
	_write(new / moved, 'moved')
	_write(base / kept, 'moved')
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	assert _ops(rows) == {('D', kept), ('A', moved)}

	def make_target(name):
		target = tmp_path / name
		_clone(base, target)
		(target / kept).unlink()
		(target / 'y').rmdir()
		_write(target / 'm' / 'x' / 'f', 'moved')
		os.symlink(os.path.join('m', 'x'), str(target / 'y'))
		return target

	return vol, kept, make_target


def _apply_on_scopes(hpcp_mod, monkeypatch, vol, target, mounts, force=False):
	"""Apply vol to target with mountinfo listing the target and each of mounts (keys below it) as ext4 mounts."""
	_fake_mounts(hpcp_mod, monkeypatch, (target, 'ext4'), *[(target / mount, 'ext4') for mount in mounts])
	hpcp_mod.ERRORS.clear()
	return hpcp_mod.apply_patch_tree([str(vol)], str(target), force=force, max_workers=1)


@pytest.mark.parametrize('into_mount', [True, False])
def test_skipped_write_a_delete_in_another_scope_took_away_is_restored(trees, hpcp_mod, tmp_path, reset_hpcp_globals,
		monkeypatch, into_mount):
	"""Final review C1: D y/f in the target's own scope removes m/x/f, which scope m/ skipped as already present (or
	the reverse). The skipped write is restored with its own scope's settings once every scope ran, and the delete,
	verified again after that, fails: the patch is contradictory on this node and stays loud, exactly as when the
	same patch is applied to a single scope."""
	new, base = trees
	vol, kept, make_target = _cross_scope_move(hpcp_mod, tmp_path, new, base, into_mount)
	summaries = []
	for name, mounts in (('target', ['m']), ('single', [])):
		target = make_target(name)
		summaries.append(_apply_on_scopes(hpcp_mod, monkeypatch, vol, target, mounts))
		assert (target / 'm' / 'x' / 'f').read_text() == 'moved' and os.path.islink(str(target / 'y'))
		assert len(hpcp_mod.ERRORS) == 1 and hpcp_mod.ERRORS[0].startswith(f'Patch verify error: D {kept}: expected -')
	assert summaries[0] == summaries[1]
	assert (summaries[0]['deleted'], summaries[0]['restored'], summaries[0]['skipped'], summaries[0]['failed']) == (1, 1, 0, 1)


def test_cross_scope_restore_runs_after_a_scope_that_failed(trees, hpcp_mod, tmp_path, reset_hpcp_globals, monkeypatch):
	"""Scope m/ deletes m/x/f, which the target's own scope skipped as y/f, and then fails: the loop stops, but the
	skipped write is still restored and the delete verified again."""
	new, base = trees
	vol, _, make_target = _cross_scope_move(hpcp_mod, tmp_path, new, base, into_mount=False)
	real_apply_section = hpcp_mod._apply_section
	def apply_section(plan, *args):
		summary = real_apply_section(plan, *args)
		if plan.name == 'm/':
			hpcp_mod.eprint('Patch verify error: simulated failure')
			summary['failed'] = 1
		return summary
	monkeypatch.setattr(hpcp_mod, '_apply_section', apply_section)
	target = make_target('target')
	summary = _apply_on_scopes(hpcp_mod, monkeypatch, vol, target, ['m'])
	assert (summary['restored'], summary['failed']) == (1, 2)
	assert (target / 'm' / 'x' / 'f').read_text() == 'moved'
	assert hpcp_mod.ERRORS[0] == 'Patch verify error: simulated failure'
	assert hpcp_mod.ERRORS[1].startswith('Patch verify error: D m/x/f: expected -') and len(hpcp_mod.ERRORS) == 2


def test_writes_in_two_scopes_that_reach_one_entry_fail_at_verify(trees, hpcp_mod, tmp_path, reset_hpcp_globals, monkeypatch):
	"""The golden holds y/f and m/x/f with different content; on the node y -> m/x makes them one file. The scope
	written first was verified before the other replaced its file; the final pass sees that the entry changed,
	verifies it again and fails, as on a single scope."""
	new, base = trees
	_write(new / 'y' / 'f', 'from the root scope')
	_write(new / 'm' / 'x' / 'f', 'from the mount scope')
	(base / 'y').mkdir()
	(base / 'm' / 'x').mkdir(parents=True)
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	assert _ops(rows) == {('A', 'y/f'), ('A', 'm/x/f')}
	summaries = []
	for name, mounts in (('target', ['m']), ('single', [])):
		target = tmp_path / name
		_clone(base, target)
		(target / 'y').rmdir()
		os.symlink(os.path.join('m', 'x'), str(target / 'y'))
		summaries.append(_apply_on_scopes(hpcp_mod, monkeypatch, vol, target, mounts))
		assert len(hpcp_mod.ERRORS) == 1 and hpcp_mod.ERRORS[0].startswith('Patch verify error: A ')
	assert summaries[0] == summaries[1]
	assert (summaries[0]['added'], summaries[0]['failed']) == (2, 1)


@pytest.mark.parametrize('op', ['A', 'M'])
def test_directory_rows_in_two_scopes_that_reach_one_entry_fail_at_verify(trees, hpcp_mod, tmp_path, reset_hpcp_globals,
		monkeypatch, op):
	"""The golden's y/d/ and m/x/d/ differ in mode; on the node y -> m/x makes them one directory. A directory is
	written in place (its inode stays), so the final pass verifies every directory row again, whatever its entry's
	identity: the scope written first fails, as one of the two does on a single scope."""
	new, base = trees
	for tree in (new, base):
		(tree / 'y').mkdir()
		(tree / 'm' / 'x').mkdir(parents=True)
	(new / 'y' / 'd').mkdir()
	(new / 'm' / 'x' / 'd').mkdir()
	os.chmod(str(new / 'y' / 'd'), 0o755 if op == 'A' else 0o750)
	os.chmod(str(new / 'm' / 'x' / 'd'), 0o700)
	if op == 'M':
		for path in (base / 'y' / 'd', base / 'm' / 'x' / 'd'):
			path.mkdir()
			os.chmod(str(path), 0o755)
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	assert _ops(rows) == {(op, 'y/d/'), (op, 'm/x/d/')}
	summaries = []
	for name, mounts in (('target', ['m']), ('single', [])):
		target = tmp_path / name
		_clone(base, target)
		shutil.rmtree(str(target / 'y'))
		os.symlink(os.path.join('m', 'x'), str(target / 'y'))
		summaries.append(_apply_on_scopes(hpcp_mod, monkeypatch, vol, target, mounts))
		assert len(hpcp_mod.ERRORS) == 1 and hpcp_mod.ERRORS[0].startswith(f'Patch verify error: {op} ')
	assert summaries[0] == summaries[1]
	assert summaries[0]['failed'] == 1


def test_a_row_failing_after_the_restore_across_scopes_is_counted_once(trees, hpcp_mod, tmp_path, reset_hpcp_globals,
		monkeypatch):
	"""The target's own scope writes y/F and verifies its skipped case sibling y/f (passes); scope m/ replaces m/x/f
	(the same entry, through y -> m/x) with a directory. The restore across scopes finds y/f of another type and
	fails its verify; the final pass must not count it again."""
	new, base = trees
	_write(new / 'y' / 'F', 'upper')
	_write(new / 'y' / 'f', 'lower')
	(new / 'm' / 'x' / 'f').mkdir(parents=True)
	(base / 'y').mkdir()
	_write(base / 'm' / 'x' / 'f', 'lower')
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	assert _ops(rows) == {('A', 'y/F'), ('A', 'y/f'), ('D', 'm/x/f'), ('A', 'm/x/f/')}
	summaries = []
	for name, mounts in (('target', ['m']), ('single', [])):
		target = tmp_path / name
		_clone(base, target)
		(target / 'y').rmdir()
		os.symlink(os.path.join('m', 'x'), str(target / 'y'))
		summaries.append(_apply_on_scopes(hpcp_mod, monkeypatch, vol, target, mounts))
		assert len([error for error in hpcp_mod.ERRORS if error.startswith('Patch verify error: A y/f:')]) == 1
	assert summaries[0]['failed'] == summaries[1]['failed'] == 1


def test_ordinary_multi_scope_apply_passes_the_final_verify(trees, hpcp_mod, tmp_path, reset_hpcp_globals, monkeypatch):
	"""Deletes, adds and changes in the target's own scope and a vfat mount below it, without aliases: the final
	pass finds nothing to report."""
	new, base = trees
	for prefix in ('', 'boot/efi/'):
		_write(new / prefix / 'changed', 'new')
		_write(base / prefix / 'changed', 'old')
		_write(base / prefix / 'gone' / 'f', 'bye')
		_write(new / prefix / 'added' / 'f', 'hello')
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	_fake_mounts(hpcp_mod, monkeypatch, (target, 'ext4'), (target / 'boot' / 'efi', 'vfat'))
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert (summary['deleted'], summary['added'], summary['modified'], summary['failed']) == (4, 4, 2, 0)
	assert hpcp_mod.ERRORS == []
	for prefix in ('', 'boot/efi/'):
		assert (target / prefix / 'changed').read_text() == 'new' and (target / prefix / 'added' / 'f').read_text() == 'hello'
		assert not (target / prefix / 'gone').exists()


@pytest.mark.parametrize('new_type, base_type, rows', [('vfat', 'ext4', 0), ('ext4', 'exfat', 0), ('ext4', 'ext4', 1)])
def test_process_get_diff_image_compares_content_only_when_a_side_is_fat(trees, hpcp_mod, tmp_path, reset_hpcp_globals,
		monkeypatch, capsys, new_type, base_type, rows):
	"""Two trees that differ only in a file's mode: no rows when either side is FAT / exFAT, an M row otherwise."""
	new, base = trees
	_write(new / 'f', 'same')
	_write(base / 'f', 'same')
	os.chmod(str(new / 'f'), 0o755)
	os.chmod(str(base / 'f'), 0o644)
	_fake_mounts(hpcp_mod, monkeypatch, (new, new_type), (base, base_type))
	seen = []
	real_compute = hpcp_mod.compute_changes
	def compute(*args, **kwargs):
		seen.append(real_compute(*args, **kwargs))
		return seen[-1]
	monkeypatch.setattr(hpcp_mod, 'compute_changes', compute)
	class Stop(Exception):
		pass
	def stop(*args, **kwargs):
		raise Stop()
	monkeypatch.setattr(hpcp_mod, 'create_image', stop)
	monkeypatch.setattr(hpcp_mod, '_plan_dest_images', lambda init_size, dest_image_size=0: (1, 1 << 20))
	try:
		result = hpcp_mod.process_get_diff_image([str(new)], None, [str(base)], None, str(tmp_path / 'p.img'), [], [], max_workers=1)
		assert result is None
	except Stop:
		pass
	assert [len(r) for r in seen] == [rows]


def test_rows_by_scope_uses_the_deepest_mount(hpcp_mod):
	R = hpcp_mod.ChangeRow
	rows = [R('M', './', 'd:40755:0:0:-', 'd:40755:0:0:-'), R('A', 'etc/x', '-', 'f:1:1:100644:0:0:-:-'),
		R('M', 'boot/efi/', 'd:40755:0:0:-', 'd:40755:0:0:-'), R('A', 'boot/efi/EFI/a', '-', 'f:1:1:100644:0:0:-:-'),
		R('A', 'boot/efix', '-', 'f:1:1:100644:0:0:-:-')]
	scopes = [('', 'ext4', ''), ('boot/efi', 'vfat', '')]
	grouped = hpcp_mod._rows_by_scope(rows, scopes)
	assert {scope: [row.path for row in scoped] for scope, scoped in grouped.items()} == {
		'': ['./', 'etc/x', 'boot/efix'], 'boot/efi': ['boot/efi/', 'boot/efi/EFI/a']}


def test_rows_by_scope_scales_with_many_mounts(hpcp_mod):
	"""Final review M1: each row looks up its own ancestors, not every mount (20000 kubelet-style mounts)."""
	import time
	R = hpcp_mod.ChangeRow
	scopes = [('', 'ext4', '')] + [(f'var/lib/kubelet/pods/p{i}/volumes/v', 'tmpfs', '') for i in range(20000)]
	rows = [R('A', f'var/lib/kubelet/pods/p{i}/volumes/v/f', '-', 'f:1:1:100644:0:0:-:-') for i in range(20000)]
	rows.append(R('A', 'var/lib/kubelet/pods/p1/volumes/vx', '-', 'f:1:1:100644:0:0:-:-'))
	start = time.time()
	grouped = hpcp_mod._rows_by_scope(rows, scopes)
	assert time.time() - start < 2
	assert len(grouped) == 20001 and [row.path for row in grouped['var/lib/kubelet/pods/p7/volumes/v']] == [
		'var/lib/kubelet/pods/p7/volumes/v/f']
	assert [row.path for row in grouped['']] == ['var/lib/kubelet/pods/p1/volumes/vx']


def test_rows_under_a_content_only_mount_below_the_target_are_content_only(trees, hpcp_mod, tmp_path, reset_hpcp_globals, monkeypatch):
	"""`-siff store-patch.img -d /`: rows under boot/efi land on the vfat ESP mounted there, whose modes come from
	its mount options (here not those of the store's copy): planned and applied by content only, case-insensitively."""
	new, base = trees
	_write(new / 'etc' / 'conf', 'new conf')
	_write(base / 'etc' / 'conf', 'old conf')
	_write(new / 'boot' / 'efi' / 'EFI' / 'grub.cfg', 'new cfg')
	_write(base / 'boot' / 'efi' / 'EFI' / 'grub.cfg', 'old cfg')
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	for path in (target / 'boot' / 'efi' / 'EFI', target / 'boot' / 'efi' / 'EFI' / 'grub.cfg'):
		os.chmod(str(path), 0o700)
	_fake_mounts(hpcp_mod, monkeypatch, (target, 'ext4'), (target / 'boot' / 'efi', 'vfat'))
	planned = []
	real_plan_section = hpcp_mod._plan_section
	def plan_section(*args, **kwargs):
		planned.append(real_plan_section(*args, **kwargs))
		return planned[-1]
	monkeypatch.setattr(hpcp_mod, '_plan_section', plan_section)
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert (summary['modified'], summary['forced'], summary['failed']) == (2, 0, 0)
	assert [(plan.name, plan.metadata, plan.case_insensitive) for plan in planned] == [('root', True, False), ('boot/efi/', False, True)]
	assert (target / 'boot' / 'efi' / 'EFI' / 'grub.cfg').read_text() == 'new cfg'
	assert (target / 'etc' / 'conf').read_text() == 'new conf'


def test_contradictory_alias_patch_writes_and_then_fails_at_verify(trees, hpcp_mod, tmp_path):
	"""D x/f + A y/f on a target where y is a symlink to x: not refused; the write is restored after the delete and
	the delete's verify fails (non-zero exit). Where y is a real directory the patch applies cleanly."""
	new, base = trees
	_moved_file(new, base)
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	assert _ops(rows) == {('D', 'x/f'), ('A', 'y/f')}
	target = tmp_path / 'target'
	_clone(base, target)
	(target / 'y').rmdir()
	os.symlink('x', str(target / 'y'))
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), force=True, max_workers=1)
	assert summary['restored'] == 1 and summary['failed'] == 1
	assert (target / 'x' / 'f').read_text() == 'moved'
	plain = tmp_path / 'plain'
	_clone(base, plain)
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(plain), max_workers=1)
	assert (summary['deleted'], summary['added'], summary['failed']) == (1, 1, 0)
	assert _changes(hpcp_mod, new, plain) == []


def test_refusals_come_before_the_conflict_prompt(trees, hpcp_mod, tmp_path, reset_hpcp_globals, monkeypatch):
	"""A patch with a conflict and a path that escapes the target through a symlink: refused without asking first."""
	new, base = trees
	_write(new / 'conflict.txt', 'new')
	_write(base / 'conflict.txt', 'old')
	_write(new / 'sub' / 'f', 'f')
	(base / 'sub').mkdir()
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	_write(target / 'conflict.txt', 'local edit')
	outside = tmp_path / 'outside'
	outside.mkdir()
	(target / 'sub').rmdir()
	os.symlink(str(outside), str(target / 'sub'))
	monkeypatch.setattr(sys, 'stdin', _FakeStdin(True))
	def prompt(*args, **kwargs):
		raise AssertionError('asked before refusing')
	monkeypatch.setattr(hpcp_mod.multiCMD, 'input_with_timeout_and_countdown', prompt)
	with pytest.raises(RuntimeError, match='resolve outside the target through symlinks'):
		hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert list(outside.iterdir()) == []


def test_single_fs_generation_compares_content_only_on_fat(tmp_path, hpcp_mod):
	new, base = tmp_path / 'new', tmp_path / 'base'
	new.mkdir()
	base.mkdir()
	compare = hpcp_mod._compare_metadata
	assert compare((str(new), str(base)), [_mount_entry('/', 'ext4')]) is True
	assert compare((str(new), str(base)), [_mount_entry('/', 'ext4'), _mount_entry(new, 'vfat')]) is False
	assert compare((str(new), str(base)), [_mount_entry('/', 'ext4'), _mount_entry(base, 'exfat')]) is False


def _vfat_rename_images(hpcp_mod, tmp_path):
	"""golden.img holds EFI/Boot/BOOTX64.EFI, node.img EFI/BOOT/BOOTX64.EFI (vfat); returns (node, patch) for
	-si golden.img -di node.img."""
	golden, node, out = str(tmp_path / 'golden.img'), str(tmp_path / 'node.img'), str(tmp_path / 'patch.img')
	for image, name in ((golden, 'Boot'), (node, 'BOOT')):
		with open(image, 'wb') as f:
			f.truncate(32 << 20)
		subprocess.run(['mkfs.vfat', image], check=True, stdout=subprocess.DEVNULL)
		_change_image(hpcp_mod, image, lambda root, name=name: _write(os.path.join(root, 'EFI', name, 'BOOTX64.EFI'), 'loader'))
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		assert hpcp_mod.process_get_diff_image([], [golden], [], node, out, mount_points, loop_devices, max_workers=1) == out
	return node, out


def _vfat_boot_listing(hpcp_mod, image):
	"""(names under EFI, content of EFI/<boot dir>/BOOTX64.EFI) of a vfat image."""
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		root = hpcp_mod._mount_single_fs_image(image, True, mount_points, loop_devices)
		names = os.listdir(os.path.join(root, 'EFI'))
		with open(os.path.join(root, 'EFI', names[0], 'BOOTX64.EFI')) as f:
			return names, f.read()


@requires_root_and_image_tools
@pytest.mark.skipif(shutil.which('mkfs.vfat') is None, reason='needs mkfs.vfat')
@pytest.mark.root
@pytest.mark.loop
def test_case_only_rename_patch_applies_on_a_vfat_image(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Review Focus 1: the rename applies with the new spelling, and a re-run skips every row."""
	hpcp_mod.HASH_SIZE = 65536
	node, out = _vfat_rename_images(hpcp_mod, tmp_path)
	for run in (1, 2):
		with _mount_session(hpcp_mod) as (mount_points, loop_devices):
			summary = hpcp_mod.process_load_diff_image([out], [], [], node, mount_points, loop_devices, max_workers=1)
		assert summary['failed'] == 0 and summary['restored'] == 0 and summary['forced'] == 0
		assert (summary['deleted'], summary['added']) == ((2, 2) if run == 1 else (0, 0))
	assert _vfat_boot_listing(hpcp_mod, node) == (['Boot'], 'loader')


@requires_root_and_image_tools
@pytest.mark.skipif(shutil.which('mkfs.vfat') is None, reason='needs mkfs.vfat')
@pytest.mark.root
@pytest.mark.loop
def test_case_only_rename_is_restored_on_a_vfat_image_reported_as_nfs4(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Review Focus 5: a filesystem type can hide case-insensitivity (NFS from a case-insensitive server). Planned
	case-sensitively, the adds are skipped as present and the deletes remove those files; the repair step writes them
	again and verify accepts the deleted names (same entries). A re-run stays correct."""
	hpcp_mod.HASH_SIZE = 65536
	node, out = _vfat_rename_images(hpcp_mod, tmp_path)
	monkeypatch.setattr(hpcp_mod, '_mount_scopes', lambda path, entries: [('', 'nfs4', '')])
	for run in (1, 2):
		with _mount_session(hpcp_mod) as (mount_points, loop_devices):
			summary = hpcp_mod.process_load_diff_image([out], [], [], node, mount_points, loop_devices, max_workers=1)
		assert summary['failed'] == 0 and hpcp_mod.ERRORS == []
		if run == 1:
			assert summary['restored'] == 2
		assert _vfat_boot_listing(hpcp_mod, node) == (['Boot'], 'loader')


def _fat_attrs(path, value=None):
	"""Read (and first set, when value is given) the FAT attribute byte of a file on vfat."""
	import fcntl
	fd = os.open(str(path), os.O_RDONLY)
	try:
		if value is not None:
			fcntl.ioctl(fd, 0x40047211, struct.pack('I', value))
		return struct.unpack('I', fcntl.ioctl(fd, 0x80047210, b'\0' * 4))[0]
	finally:
		os.close(fd)


@requires_root_and_image_tools
@pytest.mark.skipif(shutil.which('mkfs.vfat') is None, reason='needs mkfs.vfat')
@pytest.mark.root
@pytest.mark.loop
def test_replacing_a_file_on_vfat_keeps_its_fat_attributes(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Review Focus 4: a read-only + hidden + system file on the node's FAT target is replaced (new content) and keeps
	those attributes."""
	hpcp_mod.HASH_SIZE = 65536
	golden, node, out = str(tmp_path / 'golden.img'), str(tmp_path / 'node.img'), str(tmp_path / 'patch.img')
	for image, data in ((golden, 'new loader'), (node, 'old loader')):
		with open(image, 'wb') as f:
			f.truncate(32 << 20)
		subprocess.run(['mkfs.vfat', image], check=True, stdout=subprocess.DEVNULL)
		_change_image(hpcp_mod, image, lambda root, data=data: _write(os.path.join(root, 'EFI', 'x.efi'), data))
	# read-only (0x01) | hidden (0x02) | system (0x04) | archive (0x20)
	_change_image(hpcp_mod, node, lambda root: _fat_attrs(os.path.join(root, 'EFI', 'x.efi'), 0x27))
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		assert hpcp_mod.process_get_diff_image([], [golden], [], node, out, mount_points, loop_devices, max_workers=1) == out
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		summary = hpcp_mod.process_load_diff_image([out], [], [], node, mount_points, loop_devices, max_workers=1)
	assert summary['failed'] == 0 and summary['modified'] == 1
	seen = {}

	def read(root):
		path = os.path.join(root, 'EFI', 'x.efi')
		seen['attrs'] = _fat_attrs(path)
		with open(path) as f:
			seen['data'] = f.read()

	_change_image(hpcp_mod, node, read)
	assert seen['data'] == 'new loader' and seen['attrs'] & 0x07 == 0x07


def _case_siblings(hpcp_mod, tmp_path, new, base):
	"""The golden (a case-sensitive tree) holds EFI/X = NEW and EFI/x = old; the base EFI/X = old: rows M EFI/X and
	A EFI/x. Returns the patch volume."""
	_write(new / 'EFI' / 'X', 'NEW loader')
	_write(new / 'EFI' / 'x', 'old loader')
	_write(base / 'EFI' / 'X', 'old loader')
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	assert {(op, path) for op, path in _ops(rows) if path.startswith('EFI/')} == {('M', 'EFI/X'), ('A', 'EFI/x')}
	return vol


def test_skipped_case_siblings_on_a_case_sensitive_target_are_not_verified(trees, hpcp_mod, tmp_path, monkeypatch):
	"""On a case-sensitive target EFI/X and EFI/x are separate files: writing EFI/X leaves EFI/x's entry unchanged, so
	the skipped EFI/x is not verified again, and verify passes. (On vfat the write lands on EFI/x's entry, which then
	fails loudly: test_write_over_a_skipped_case_sibling_on_vfat_fails_at_verify.)"""
	new, base = trees
	_write(new / 'other', 'same')
	vol = _case_siblings(hpcp_mod, tmp_path, new, base)
	target = tmp_path / 'target'
	_clone(base, target)
	_write(target / 'EFI' / 'x', 'old loader')
	shutil.copy2(str(new / 'other'), str(target / 'other'))
	verified = []
	real_verify_rows = hpcp_mod._verify_rows

	def verify_rows(rows, *args, **kwargs):
		verified.extend(row.path for row in rows)
		return real_verify_rows(rows, *args, **kwargs)

	monkeypatch.setattr(hpcp_mod, '_verify_rows', verify_rows)
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert (summary['modified'], summary['skipped'], summary['failed']) == (1, 2, 0)
	assert verified == ['EFI/X']
	assert (target / 'EFI' / 'X').read_text() == 'NEW loader' and (target / 'EFI' / 'x').read_text() == 'old loader'


def _alias_overwrite(hpcp_mod, tmp_path, new, base):
	"""The golden adds y/f with the content base's m/x/f has, and changes m/x/f. On a node where y -> m/x, A y/f is
	skipped (already there) and M m/x/f then rewrites that same file. Returns (volume, make_target)."""
	(new / 'm' / 'x').mkdir(parents=True)
	(base / 'm' / 'x').mkdir(parents=True)
	(base / 'y').mkdir()
	(new / 'y').mkdir()
	_write(base / 'm' / 'x' / 'f', 'old')
	_write(new / 'y' / 'f', 'old')
	_write(new / 'm' / 'x' / 'f', 'new')
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	assert _ops(rows) == {('A', 'y/f'), ('M', 'm/x/f')}

	def make_target(name):
		target = tmp_path / name
		_clone(base, target)
		(target / 'y').rmdir()
		os.symlink(os.path.join('m', 'x'), str(target / 'y'))
		return target

	return vol, make_target


def test_skipped_row_overwritten_through_an_alias_fails_at_verify(trees, hpcp_mod, tmp_path, reset_hpcp_globals, monkeypatch):
	"""Spec Part 1 item 2 / Review Focus 1: A y/f is skipped, M m/x/f rewrites the same entry through y -> m/x. The
	skipped row's entry changed, so it is verified and fails, on one scope and across two (m mounted below)."""
	new, base = trees
	vol, make_target = _alias_overwrite(hpcp_mod, tmp_path, new, base)
	summaries = []
	for name, mounts in (('single', []), ('scoped', ['m'])):
		target = make_target(name)
		summaries.append(_apply_on_scopes(hpcp_mod, monkeypatch, vol, target, mounts))
		assert (target / 'm' / 'x' / 'f').read_text() == 'new'
		assert [error for error in hpcp_mod.ERRORS if error.startswith('Patch verify error: A y/f:')]
	assert summaries[0] == summaries[1] and summaries[0]['failed'] == 1


def test_skipped_directory_changed_through_an_alias_fails_at_verify(trees, hpcp_mod, tmp_path, reset_hpcp_globals, monkeypatch):
	"""A skipped directory row keeps its inode when another row changes its mode through an alias: skipped
	directories are always verified once something was written."""
	new, base = trees
	for tree in (new, base):
		(tree / 'm' / 'x' / 'd').mkdir(parents=True)
		(tree / 'y').mkdir()
	(new / 'y' / 'd').mkdir()
	os.chmod(str(new / 'y' / 'd'), 0o755)
	os.chmod(str(base / 'm' / 'x' / 'd'), 0o755)
	os.chmod(str(new / 'm' / 'x' / 'd'), 0o700)
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	assert {('A', 'y/d/'), ('M', 'm/x/d/')} <= _ops(rows)
	target = tmp_path / 'target'
	_clone(base, target)
	(target / 'y').rmdir()
	os.symlink(os.path.join('m', 'x'), str(target / 'y'))
	summary = _apply_on_scopes(hpcp_mod, monkeypatch, vol, target, [])
	assert summary['failed'] == 1
	assert [error for error in hpcp_mod.ERRORS if error.startswith('Patch verify error: A y/d/:')]


def _later_scope_writes(new, base):
	"""Scope m/ also writes m/other and scope n/ writes n/late: a scope that fails stops n/ from being applied."""
	_write(new / 'm' / 'other', 'other')
	_write(new / 'n' / 'late', 'late')
	(base / 'n').mkdir()


def _mkdir(path, mode):
	os.makedirs(str(path), exist_ok=True)
	os.chmod(str(path), mode)


def _alias_node(tmp_path, base, prepare):
	"""make_target(name): a clone of base where y -> m/x makes y/... and m/x/... one entry, then prepare(target)."""
	def make_target(name):
		target = tmp_path / name
		_clone(base, target)
		shutil.rmtree(str(target / 'y'))
		os.symlink(os.path.join('m', 'x'), str(target / 'y'))
		prepare(target)
		return target
	return make_target


def _apply_on_one_and_three_scopes(hpcp_mod, monkeypatch, vol, make_target, force=False):
	"""Group A's invariant: the patch ends alike on one scope and on scopes m/ and n/ below the target (equal summaries
	and errors), and every scope is applied. Returns (summary, errors)."""
	results = []
	for name, mounts in (('single', []), ('scoped', ['m', 'n'])):
		target = make_target(name)
		summary = _apply_on_scopes(hpcp_mod, monkeypatch, vol, target, mounts, force=force)
		assert (target / 'm' / 'other').read_text() == 'other' and (target / 'n' / 'late').read_text() == 'late'
		results.append((summary, list(hpcp_mod.ERRORS)))
	assert results[0] == results[1]
	return results[0]


@pytest.mark.parametrize('kind', ['file', 'directory'])
def test_a_delete_through_an_alias_in_an_earlier_scope_fails_once_and_later_scopes_apply(trees, hpcp_mod, tmp_path,
		reset_hpcp_globals, monkeypatch, kind):
	"""Final review I1, probe 1 (and its directory form): the root's D y/f removes m/x/f through y -> m/x, which scope
	m/ skipped as already present. m/ writes too, but its own verify sees only what its own writes did: the missing
	entry is restored once every scope ran, the delete fails at the final pass, and n/ is applied, as on one scope."""
	new, base = trees
	(new / 'y').mkdir()
	(base / 'm' / 'x').mkdir(parents=True)
	if kind == 'file':
		gone, added = 'y/f', 'm/x/f'
		_write(new / 'm' / 'x' / 'f', 'moved')
		_write(base / 'y' / 'f', 'moved')
		prepare = lambda target: _write(target / 'm' / 'x' / 'f', 'moved')
	else:
		gone, added = 'y/d/', 'm/x/d/'
		_mkdir(new / 'm' / 'x' / 'd', 0o755)
		_mkdir(base / 'y' / 'd', 0o755)
		prepare = lambda target: _mkdir(target / 'm' / 'x' / 'd', 0o755)
	_later_scope_writes(new, base)
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	assert _ops(rows) == {('D', gone), ('A', added), ('A', 'm/other'), ('A', 'n/late')}
	summary, errors = _apply_on_one_and_three_scopes(hpcp_mod, monkeypatch, vol, _alias_node(tmp_path, base, prepare))
	assert (summary['deleted'], summary['added'], summary['restored'], summary['skipped'], summary['failed']) == (1, 2, 1, 0, 1)
	assert len(errors) == 1 and errors[0].startswith(f'Patch verify error: D {gone}: expected -')


@pytest.mark.parametrize('kind', ['file', 'directory'])
def test_a_change_through_an_alias_in_an_earlier_scope_fails_once_and_later_scopes_apply(trees, hpcp_mod, tmp_path,
		reset_hpcp_globals, monkeypatch, kind):
	"""Final review I1, probe 2 (and its directory form): the root's M y/f rewrites m/x/f (M y/d/ changes m/x/d's
	mode) through y -> m/x, which scope m/ skipped as already done. m/ writes too; the skipped row is not failed by m/'s
	own verify (its own writes did not change it) but by the final pass, and n/ is applied, as on one scope."""
	new, base = trees
	(base / 'm' / 'x').mkdir(parents=True)
	if kind == 'file':
		changed, skipped = 'y/f', 'm/x/f'
		_write(new / 'y' / 'f', 'new')
		_write(new / 'm' / 'x' / 'f', 'old')
		_write(base / 'y' / 'f', 'old')
		prepare = lambda target: _write(target / 'm' / 'x' / 'f', 'old')
	else:
		changed, skipped = 'y/d/', 'm/x/d/'
		_mkdir(new / 'y' / 'd', 0o700)
		_mkdir(new / 'm' / 'x' / 'd', 0o755)
		_mkdir(base / 'y' / 'd', 0o755)
		prepare = lambda target: _mkdir(target / 'm' / 'x' / 'd', 0o755)
	_later_scope_writes(new, base)
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	assert _ops(rows) == {('M', changed), ('A', skipped), ('A', 'm/other'), ('A', 'n/late')}
	summary, errors = _apply_on_one_and_three_scopes(hpcp_mod, monkeypatch, vol, _alias_node(tmp_path, base, prepare))
	assert (summary['modified'], summary['added'], summary['skipped'], summary['failed']) == (1, 2, 1, 1)
	assert len(errors) == 1 and errors[0].startswith(f'Patch verify error: A {skipped}: expected ')


def test_a_skipped_row_a_forced_write_replaced_is_counted_once(trees, hpcp_mod, tmp_path, reset_hpcp_globals, monkeypatch):
	"""A forced A y/f (a symlink) replaces the skipped m/x/f through y -> m/x, and D z removes something. The skipped
	row fails once: the repair step across scopes does not verify (and count) again a row that failed its section's
	verify. One scope and scopes m/ and n/ end alike."""
	new, base = trees
	(new / 'y').mkdir()
	os.symlink('elsewhere', str(new / 'y' / 'f'))
	_write(new / 'm' / 'x' / 'f', 'data')
	(base / 'y').mkdir()
	(base / 'm' / 'x').mkdir(parents=True)
	_write(base / 'z', 'z')
	_later_scope_writes(new, base)
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	assert _ops(rows) == {('A', 'y/f'), ('A', 'm/x/f'), ('D', 'z'), ('A', 'm/other'), ('A', 'n/late')}
	summary, errors = _apply_on_one_and_three_scopes(hpcp_mod, monkeypatch, vol,
		_alias_node(tmp_path, base, lambda target: _write(target / 'm' / 'x' / 'f', 'data')), force=True)
	assert (summary['forced'], summary['skipped'], summary['failed']) == (1, 1, 1)
	assert len(errors) == 1 and errors[0].startswith('Patch verify error: A m/x/f: expected ') and errors[0].endswith('found l:elsewhere')


def test_unchanged_skipped_files_are_not_verified_again(trees, hpcp_mod, tmp_path, monkeypatch):
	"""An ordinary apply: skipped files whose entry did not change are not verified again (no extra hashing); the
	written rows are, and the apply ends failed 0."""
	new, base = trees
	for index in range(20):
		_write(new / f'same{index}', 's')
	_write(new / 'changed', 'new')
	_write(base / 'changed', 'old')
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(new, target)
	_write(target / 'changed', 'old')
	verified = []
	real_verify_rows = hpcp_mod._verify_rows

	def verify_rows(rows, *args, **kwargs):
		verified.extend(row.path for row in rows)
		return real_verify_rows(rows, *args, **kwargs)

	monkeypatch.setattr(hpcp_mod, '_verify_rows', verify_rows)
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert summary['failed'] == 0 and summary['skipped'] == 20
	assert verified == ['changed']


@requires_root_and_image_tools
@pytest.mark.skipif(shutil.which('mkfs.vfat') is None, reason='needs mkfs.vfat')
@pytest.mark.root
@pytest.mark.loop
def test_write_over_a_skipped_case_sibling_on_vfat_fails_at_verify(trees, hpcp_mod, tmp_path):
	"""Final review I1: on vfat, A EFI/x is skipped (the lookup finds EFI/X, still old), then M EFI/X writes NEW into
	that one entry. The golden's two files cannot both exist on FAT: verify reports EFI/x instead of a silent success."""
	new, base = trees
	vol = _case_siblings(hpcp_mod, tmp_path, new, base)
	node = str(tmp_path / 'node.img')
	with open(node, 'wb') as f:
		f.truncate(32 << 20)
	subprocess.run(['mkfs.vfat', node], check=True, stdout=subprocess.DEVNULL)
	_change_image(hpcp_mod, node, lambda root: _write(os.path.join(root, 'EFI', 'X'), 'old loader'))
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		root = hpcp_mod._mount_single_fs_image(node, False, mount_points, loop_devices)
		summary = hpcp_mod.apply_patch_tree([str(vol)], root, max_workers=1)
		names = os.listdir(os.path.join(root, 'EFI'))
	assert summary['modified'] == 1 and summary['failed'] >= 1
	assert any(error.startswith('Patch verify error: A EFI/x:') for error in hpcp_mod.ERRORS)
	assert names == ['X']


requires_root = pytest.mark.skipif(os.name == 'nt' or os.geteuid() != 0 or shutil.which('mount') is None,
	reason='needs root and mount')


def test_remove_tree_around_mounts_keeps_mount_points(tmp_path, hpcp_mod):
	"""The walk removes everything on the tree's own filesystem and never enters a directory listed as a mount
	point; the directories above a kept mount stay. Symlinks are unlinked, never followed."""
	top = tmp_path / 'app'
	_write(top / 'a' / 'f', 'x')
	_write(top / 'b' / 'data' / 'keep', 'mounted')
	_write(tmp_path / 'outside' / 'target', 'outside')
	os.symlink(str(tmp_path / 'outside'), str(top / 'link'))
	mount = os.path.realpath(str(top / 'b' / 'data'))
	kept, errors = hpcp_mod._remove_tree_around_mounts(str(top), {mount})
	assert kept == ['b/data'] and errors == []
	assert (top / 'b' / 'data' / 'keep').read_text() == 'mounted'
	assert not (top / 'a').exists() and not os.path.lexists(str(top / 'link'))
	assert (tmp_path / 'outside' / 'target').read_text() == 'outside'
	assert sorted(os.listdir(str(top))) == ['b'] and os.listdir(str(top / 'b')) == ['data']


def test_remove_tree_around_mounts_leaves_a_mount_point_root_alone(tmp_path, hpcp_mod):
	top = tmp_path / 'esp'
	_write(top / 'EFI' / 'x', 'x')
	kept, errors = hpcp_mod._remove_tree_around_mounts(str(top), {os.path.realpath(str(top))})
	assert (kept, errors) == (['.'], []) and (top / 'EFI' / 'x').read_text() == 'x'


def test_remove_tree_around_mounts_stops_at_another_device(tmp_path, hpcp_mod, monkeypatch):
	"""Review Focus 2: a mount made after mountinfo was read is still not entered: its directory's st_dev differs."""
	top = tmp_path / 'app'
	_write(top / 'late' / 'keep', 'k')
	_write(top / 'f', 'x')
	real_stat = os.stat

	def fake_stat(path, *args, **kwargs):
		info = real_stat(path, *args, **kwargs)
		if os.fspath(path) == 'late' and kwargs.get('dir_fd') is not None:
			values = list(info)
			values[stat.ST_DEV] = info.st_dev + 1
			return os.stat_result(values)
		return info

	monkeypatch.setattr(os, 'stat', fake_stat)
	kept, errors = hpcp_mod._remove_tree_around_mounts(str(top), set())
	monkeypatch.undo()
	assert kept == ['late'] and errors == [] and (top / 'late' / 'keep').exists() and not (top / 'f').exists()


@pytest.mark.parametrize('fd_walk', [True, False])
def test_remove_tree_around_mounts_keeps_a_top_on_another_device_than_its_parent(tmp_path, hpcp_mod, monkeypatch, fd_walk):
	"""Final review M2: a mount made on the path itself after mountinfo was read (or a btrfs subvolume as the path)
	has another st_dev than its parent: it is kept like a path listed in mountinfo, and nothing is removed."""
	monkeypatch.setattr(hpcp_mod, '_DIR_FD_WALK', fd_walk)
	top = tmp_path / 'app'
	_write(top / 'sub' / 'f', 'x')
	_write(top / 'g', 'y')
	parent = os.path.realpath(str(tmp_path))
	real_lstat = os.lstat

	def fake_lstat(path, *args, **kwargs):
		info = real_lstat(path, *args, **kwargs)
		if os.fspath(path) == parent:
			values = list(info)
			values[stat.ST_DEV] = info.st_dev + 1
			return os.stat_result(values)
		return info

	parent_id = (real_lstat(parent).st_dev, real_lstat(parent).st_ino)
	real_fstat = os.fstat

	def fake_fstat(fd):
		info = real_fstat(fd)
		if (info.st_dev, info.st_ino) == parent_id:
			values = list(info)
			values[stat.ST_DEV] = info.st_dev + 1
			return os.stat_result(values)
		return info

	monkeypatch.setattr(os, 'lstat', fake_lstat)
	monkeypatch.setattr(os, 'fstat', fake_fstat)
	kept, errors = hpcp_mod._remove_tree_around_mounts(str(top), set())
	monkeypatch.undo()
	assert (kept, errors) == (['.'], [])
	assert (top / 'sub' / 'f').read_text() == 'x' and (top / 'g').read_text() == 'y'


def test_forced_removal_names_another_filesystem_it_kept(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Final review M3: a nested btrfs subvolume is not in mountinfo, but its st_dev differs, so the forced removal of
	app/ keeps app/data; the error says it is a mount point or another filesystem."""
	new, base = trees
	vol = _app_dropped(hpcp_mod, tmp_path, new, base)
	target = tmp_path / 'target'
	_clone(base, target)
	_write(target / 'app' / 'data' / 'precious', 'keep me')
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda *args, **kwargs: [])
	real_stat = os.stat

	def fake_stat(path, *args, **kwargs):
		info = real_stat(path, *args, **kwargs)
		if os.fspath(path) == 'data' and kwargs.get('dir_fd') is not None:
			values = list(info)
			values[stat.ST_DEV] = info.st_dev + 1
			return os.stat_result(values)
		return info

	monkeypatch.setattr(os, 'stat', fake_stat)
	hpcp_mod.ERRORS.clear()
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), force=True, max_workers=1)
	monkeypatch.undo()
	assert (target / 'app' / 'data' / 'precious').read_text() == 'keep me' and summary['failed'] >= 1
	assert ('Remove directory failed: app/: data is a mount point or another filesystem (a btrfs subvolume, say); left in '
		'place with the directories above it') in hpcp_mod.ERRORS


def test_remove_tree_around_mounts_refuses_a_directory_swapped_for_a_symlink(tmp_path, hpcp_mod, monkeypatch):
	"""Security: sub is a real directory when the walk looks at it, a symlink to a directory outside by the time it
	descends. The walk must not follow it; the outside file survives and an error is reported."""
	top = tmp_path / 'app'
	_write(top / 'sub' / 'f', 'x')
	_write(top / 'other', 'y')
	_write(tmp_path / 'outside' / 'precious', 'keep me')
	real_stat = os.stat
	swapped = []

	def racing_stat(path, *args, **kwargs):
		info = real_stat(path, *args, **kwargs)
		if os.fspath(path) == 'sub' and not swapped:
			swapped.append(True)
			os.rename(str(top / 'sub'), str(tmp_path / 'moved'))
			os.symlink(str(tmp_path / 'outside'), str(top / 'sub'))
		return info

	monkeypatch.setattr(os, 'stat', racing_stat)
	kept, errors = hpcp_mod._remove_tree_around_mounts(str(top), set())
	monkeypatch.undo()
	assert swapped and kept == [] and len(errors) == 1 and errors[0].startswith('sub:')
	assert (tmp_path / 'outside' / 'precious').read_text() == 'keep me'
	assert not (top / 'other').exists()


def test_remove_tree_around_mounts_collects_errors_and_keeps_going(tmp_path, hpcp_mod, monkeypatch):
	top = tmp_path / 'app'
	_write(top / 'a' / 'stuck', 'x')
	_write(top / 'a' / 'free', 'x')
	_write(top / 'b', 'y')
	real_unlink = os.unlink

	def unlink(path, *args, **kwargs):
		if os.fspath(path) == 'stuck':
			raise PermissionError(errno.EPERM, 'Operation not permitted')
		return real_unlink(path, *args, **kwargs)

	monkeypatch.setattr(os, 'unlink', unlink)
	kept, errors = hpcp_mod._remove_tree_around_mounts(str(top), set())
	monkeypatch.undo()
	assert kept == [] and errors == ['a/stuck: Operation not permitted']
	assert sorted(os.listdir(str(top))) == ['a'] and os.listdir(str(top / 'a')) == ['stuck']


@pytest.mark.parametrize('fd_walk', [True, False])
def test_remove_tree_around_mounts_refuses_a_symlink_as_the_top(tmp_path, hpcp_mod, monkeypatch, fd_walk):
	"""Security: the path itself is a symlink to a directory outside; nothing is removed and an error comes back."""
	monkeypatch.setattr(hpcp_mod, '_DIR_FD_WALK', fd_walk)
	_write(tmp_path / 'scratch' / 'precious', 'keep me')
	os.symlink(str(tmp_path / 'scratch'), str(tmp_path / 'app'))
	for path in (str(tmp_path / 'app'), str(tmp_path / 'app') + os.sep):
		kept, errors = hpcp_mod._remove_tree_around_mounts(path, set())
		assert kept == [] and errors
	assert (tmp_path / 'scratch' / 'precious').read_text() == 'keep me' and os.path.islink(str(tmp_path / 'app'))


def test_remove_tree_around_mounts_path_fallback_matches_the_fd_walk(tmp_path, hpcp_mod, monkeypatch):
	monkeypatch.setattr(hpcp_mod, '_DIR_FD_WALK', False)
	top = tmp_path / 'app'
	_write(top / 'a' / 'f', 'x')
	_write(top / 'b' / 'data' / 'keep', 'mounted')
	_write(tmp_path / 'outside' / 'target', 'outside')
	os.symlink(str(tmp_path / 'outside'), str(top / 'link'))
	kept, errors = hpcp_mod._remove_tree_around_mounts(str(top), {os.path.realpath(str(top / 'b' / 'data'))})
	assert kept == ['b/data'] and errors == []
	assert (top / 'b' / 'data' / 'keep').read_text() == 'mounted'
	assert not (top / 'a').exists() and not os.path.lexists(str(top / 'link'))
	assert (tmp_path / 'outside' / 'target').read_text() == 'outside'
	assert sorted(os.listdir(str(top))) == ['b'] and os.listdir(str(top / 'b')) == ['data']


def _victim_tree(tmp_path):
	"""target/a/b/victim/ with a file, and outside/a/b/victim/ with a file the removal must never touch."""
	for top in ('target', 'outside'):
		_write(tmp_path / top / 'a' / 'b' / 'victim' / 'f.txt', top)
	return tmp_path / 'target', tmp_path / 'outside'


def test_forced_removal_refuses_a_parent_resolving_outside_the_target(tmp_path, hpcp_mod):
	"""Spec Part 2: a directory above the top swapped for a symlink to outside before the removal resolves its parent:
	the parent's real path lies outside the target, so nothing is removed."""
	target, outside = _victim_tree(tmp_path)
	shutil.rmtree(str(target / 'a'))
	os.symlink(str(outside / 'a'), str(target / 'a'))
	kept, errors = hpcp_mod._remove_tree_around_mounts(str(target / 'a' / 'b' / 'victim'), set(), target_root=str(target))
	assert kept == [] and errors and 'outside' in errors[0]
	assert (outside / 'a' / 'b' / 'victim' / 'f.txt').exists()


def test_forced_removal_refuses_a_parent_swapped_after_it_was_resolved(tmp_path, hpcp_mod, monkeypatch):
	"""Spec Part 2: the swap comes right after the removal resolved its parent; the descriptor walk (O_NOFOLLOW at every
	component) meets the symlink and refuses, so nothing outside the target is removed."""
	target, outside = _victim_tree(tmp_path)
	parent = os.path.realpath(str(target / 'a' / 'b'))
	real_realpath = os.path.realpath
	def realpath(path, *args, **kwargs):
		result = real_realpath(path, *args, **kwargs)
		if path == str(target / 'a' / 'b') and not os.path.islink(str(target / 'a')):
			os.rename(str(target / 'a'), str(target / 'a-old'))
			os.symlink(str(outside / 'a'), str(target / 'a'))
		return result
	monkeypatch.setattr(hpcp_mod.os.path, 'realpath', realpath)
	kept, errors = hpcp_mod._remove_tree_around_mounts(str(target / 'a' / 'b' / 'victim'), set(), target_root=str(target))
	monkeypatch.setattr(hpcp_mod.os.path, 'realpath', real_realpath)
	assert parent.startswith(real_realpath(str(target)))
	assert kept == [] and errors
	assert (outside / 'a' / 'b' / 'victim' / 'f.txt').exists()
	assert (target / 'a-old' / 'b' / 'victim' / 'f.txt').exists()


def test_forced_removal_follows_a_symlinked_directory_inside_the_target(tmp_path, hpcp_mod):
	target = tmp_path / 'target'
	_write(target / 'real' / 'sub' / 'victim' / 'f.txt', 'x')
	os.symlink('real', str(target / 'link'))
	kept, errors = hpcp_mod._remove_tree_around_mounts(str(target / 'link' / 'sub' / 'victim'), set(), target_root=str(target))
	assert (kept, errors) == ([], [])
	assert not (target / 'real' / 'sub' / 'victim').exists() and (target / 'real' / 'sub').is_dir()


@pytest.mark.skipif(not os.path.isdir('/proc/self/fd'), reason='needs /proc/self/fd')
def test_forced_removal_refusal_leaks_no_descriptors(tmp_path, hpcp_mod):
	"""Review Focus 4: every descriptor the walk opened is closed, on refusals too."""
	target, outside = _victim_tree(tmp_path)
	before = len(os.listdir('/proc/self/fd'))
	(target / 'a' / 'b' / 'victim2').mkdir()
	hpcp_mod._remove_tree_around_mounts(str(target / 'a' / 'b' / 'victim2'), set(), target_root=str(target))
	shutil.rmtree(str(target / 'a'))
	os.symlink(str(outside / 'a'), str(target / 'a'))
	hpcp_mod._remove_tree_around_mounts(str(target / 'a' / 'b' / 'victim'), set(), target_root=str(target))
	hpcp_mod._remove_tree_around_mounts(str(target / 'missing' / 'x'), set(), target_root=str(target))
	assert len(os.listdir('/proc/self/fd')) == before


def test_mounts_below(tmp_path, hpcp_mod):
	top = os.path.realpath(str(tmp_path / 'app'))
	points = {top + '/data', top + '/x/y', top + 'z', '/elsewhere', top}
	assert hpcp_mod._mounts_below(str(tmp_path / 'app'), points) == ['.', 'data', 'x/y']


@contextlib.contextmanager
def _tmpfs_at(path):
	os.makedirs(str(path), exist_ok=True)
	subprocess.run(['mount', '-t', 'tmpfs', '-o', 'size=4m', 'tmpfs', str(path)], check=True)
	try:
		yield path
	finally:
		subprocess.run(['umount', str(path)], check=False)


@contextlib.contextmanager
def _bind_at(source, path):
	os.makedirs(str(path), exist_ok=True)
	subprocess.run(['mount', '--bind', str(source), str(path)], check=True)
	try:
		yield path
	finally:
		subprocess.run(['umount', str(path)], check=False)


def _app_dropped(hpcp_mod, tmp_path, new, base):
	"""The golden drops app/ (two files) and adds other.txt. Returns the patch volume."""
	_write(base / 'app' / 'one', '1')
	_write(base / 'app' / 'two', '2')
	_write(new / 'other.txt', 'added')
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	assert ('D', 'app/') in _ops(rows)
	return vol


@requires_root
@pytest.mark.root
@pytest.mark.parametrize('kind', ['tmpfs', 'bind'])
def test_forced_removal_stops_at_a_mount(trees, hpcp_mod, tmp_path, capsys, kind):
	"""Spec Part 1 item 1: -fa removes app/ around the filesystem mounted at app/data (a tmpfs, or a bind of a
	directory of the same filesystem): the mounted files survive, the D row fails at verify, the rest applies, and
	a note names the mount before anything changes."""
	new, base = trees
	vol = _app_dropped(hpcp_mod, tmp_path, new, base)
	target = tmp_path / 'target'
	_clone(base, target)
	outside = tmp_path / 'outside'
	_write(outside / 'precious', 'keep me')
	mount = _tmpfs_at(target / 'app' / 'data') if kind == 'tmpfs' else _bind_at(outside, target / 'app' / 'data')
	with mount:
		if kind == 'tmpfs':
			_write(target / 'app' / 'data' / 'precious', 'keep me')
		hpcp_mod.ERRORS.clear()
		summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), force=True, max_workers=1)
		assert (target / 'app' / 'data' / 'precious').read_text() == 'keep me'
		assert os.path.ismount(str(target / 'app' / 'data')) or kind == 'bind'
	out = capsys.readouterr().out
	assert 'Note: forced removal of app/ stops at the mount at app/data; that row will fail verify' in out
	assert summary['failed'] >= 1 and (target / 'other.txt').read_text() == 'added'
	assert not (target / 'app' / 'one').exists() and not (target / 'app' / 'two').exists()
	assert any(error.startswith('Patch verify error: D app/') for error in hpcp_mod.ERRORS)
	assert (outside / 'precious').read_text() == 'keep me'


@requires_root
@pytest.mark.root
def test_forced_type_change_stops_at_a_mount(trees, hpcp_mod, tmp_path, capsys):
	"""A forced directory-to-file replacement over a directory holding a mount leaves the mount and fails."""
	new, base = trees
	_write(base / 'app' / 'one', '1')
	_write(new / 'app', 'now a file')
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	with _tmpfs_at(target / 'app' / 'data'):
		_write(target / 'app' / 'data' / 'precious', 'keep me')
		hpcp_mod.ERRORS.clear()
		summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), force=True, max_workers=1)
		assert (target / 'app' / 'data' / 'precious').read_text() == 'keep me'
	assert 'Note: forced removal of app/ stops at the mount at app/data; that row will fail verify' in capsys.readouterr().out
	assert summary['failed'] >= 1 and os.path.isdir(str(target / 'app'))


@pytest.mark.parametrize('fs_type, conflict', [('vfat', False), ('exfat', True)])
def test_timezone_tolerance_applies_on_fat_scopes_only(trees, hpcp_mod, tmp_path, reset_hpcp_globals, monkeypatch, fs_type, conflict):
	"""B1 decision 1: the node's copy of a changed file is the base's content with its mtime an hour later (a FAT time
	read under another timezone). On a vfat scope the M row applies; on exFAT (which records UTC offsets) the shift is
	a real difference: a conflict."""
	new, base = trees
	_write(new / 'boot.cfg', 'new')
	_write(base / 'boot.cfg', 'old')
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	os.utime(str(target / 'boot.cfg'), (T0 + 3600, T0 + 3600))
	_fake_mounts(hpcp_mod, monkeypatch, (target, fs_type))
	if conflict:
		with pytest.raises(RuntimeError, match='^Patch conflict'):
			hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	else:
		summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
		assert summary['modified'] == 1 and summary['failed'] == 0
		assert (target / 'boot.cfg').read_text() == 'new'


def test_fat_paths_are_hashed_in_full_and_compared_like_fat(trees, hpcp_mod, reset_hpcp_globals):
	"""Spec Part 3 item 1: a key whose path on either side is on FAT is hashed in full on both sides (marked), compared
	by content only and with the timezone tolerance; other keys keep the tail hash and the plain comparison."""
	hpcp_mod.HASH_SIZE = 16
	new, base = trees
	body = b'h' * 64
	for tree in (new, base):
		_write(tree / 'esp' / 'shifted', body)
		_write(tree / 'root' / 'shifted', body)
	os.utime(str(new / 'esp' / 'shifted'), (T0 + 3600, T0 + 3600))
	os.utime(str(new / 'root' / 'shifted'), (T0 + 3600, T0 + 3600))
	_write(new / 'esp' / 'edited', b'EDIT' + body[4:])
	_write(base / 'esp' / 'edited', body)
	new_map, base_map = hpcp_mod.build_root_maps(str(new), str(base), max_workers=1)
	fat = {path for path in list(new_map.values()) + list(base_map.values()) if os.sep + 'esp' + os.sep in path}
	rows = hpcp_mod.compute_changes(new_map, base_map, max_workers=1, fat_paths=fat)
	by_path = {row.path: row for row in rows}
	assert set(by_path) == {'esp/edited', 'root/shifted'}
	assert hpcp_mod._full_marked(by_path['esp/edited'].old) and hpcp_mod._full_marked(by_path['esp/edited'].new)
	assert not hpcp_mod._full_marked(by_path['root/shifted'].new)


def test_paths_on_fat_follow_the_mount_below_the_root(tmp_path, hpcp_mod, monkeypatch):
	root = tmp_path / 'tree'
	paths = [str(root / 'etc' / 'x'), str(root / 'boot' / 'efi' / 'EFI' / 'x'), str(root / 'boot' / 'efix')]
	entries = [_mount_entry(tmp_path, 'ext4'), _mount_entry(root / 'boot' / 'efi', 'vfat')]
	for path in paths:
		os.makedirs(os.path.dirname(path), exist_ok=True)
	assert hpcp_mod._paths_on_fat(paths, str(root), entries) == {paths[1]}
	assert hpcp_mod._paths_on_fat(paths, str(root), [_mount_entry(tmp_path, 'ext4')]) == set()


def test_paths_on_fat_follow_a_symlinked_root(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Final review M5: build_entry_map lists a symlinked source root from its real path; the FAT mount below the real
	tree still classifies the paths under it (and only those)."""
	real = tmp_path / 'real'
	_write(real / 'etc' / 'x', 'x')
	_write(real / 'boot' / 'efi' / 'EFI' / 'x', 'x')
	_write(real / 'boot' / 'efix', 'x')
	link = tmp_path / 'link'
	os.symlink('real', str(link))
	paths = list(hpcp_mod.build_entry_map(str(link), max_workers=1).values())
	efi = os.path.realpath(str(real / 'boot' / 'efi'))
	expected = {path for path in paths if path == efi or path.startswith(efi + os.sep)}
	assert os.path.join(efi, 'EFI', 'x') in expected and len(expected) == 3
	entries = [_mount_entry(tmp_path, 'ext4'), _mount_entry(real / 'boot' / 'efi', 'vfat')]
	assert hpcp_mod._paths_on_fat(paths, str(link), entries) == expected


@pytest.mark.parametrize('compare_hash', [True, False])
def test_marked_rows_are_hashed_in_full_only_when_hashes_are_compared(trees, hpcp_mod, tmp_path, monkeypatch, compare_hash):
	"""Final review M6: with the patch hashed by another algorithm than the node's (compare_hash False) no hash is
	compared, so checking and verifying do not hash marked rows' files in full."""
	new, base = trees
	_write(new / 'f', 'new')
	_write(base / 'f', 'old')
	new_map, base_map = hpcp_mod.build_root_maps(str(new), str(base), max_workers=1)
	rows = hpcp_mod.compute_changes(new_map, base_map, max_workers=1, fat_paths=set(new_map.values()) | set(base_map.values()))
	assert _ops(rows) == {('M', 'f')} and hpcp_mod._full_marked(rows[0].new)
	target = tmp_path / 'target'
	_clone(base, target)
	seen = []
	real_states_for_paths = hpcp_mod._states_for_paths

	def states_for_paths(paths, *args, **kwargs):
		seen.append(set(kwargs.get('marked', ())))
		return real_states_for_paths(paths, *args, **kwargs)

	monkeypatch.setattr(hpcp_mod, '_states_for_paths', states_for_paths)
	hpcp_mod.check_patch_target(rows, str(target), compare_hash=compare_hash, metadata=False, tz_tolerant=True)
	hpcp_mod._verify_rows(rows, str(target), compare_hash=compare_hash, metadata=False, tz_tolerant=True)
	marked = {str(target / 'f')} if compare_hash else set()
	assert seen == [marked, marked]


def test_format_version_follows_the_full_hash_mark(hpcp_mod):
	R = hpcp_mod.ChangeRow
	plain = [R('A', 'a', '-', 'f:5:1:100644:0:0:aa:-')]
	marked = [R('A', 'a', '-', 'f:5:1:100644:0:0:Faa:-')]
	assert hpcp_mod.make_patch_meta(plain, 'n', 'b', 'root')['format_version'] == 1
	assert hpcp_mod.make_patch_meta(marked, 'n', 'b', 'root')['format_version'] == 3
	assert hpcp_mod.make_disk_patch_meta([(1, plain)], [], 'n', 'b')['format_version'] == 2
	assert hpcp_mod.make_disk_patch_meta([(1, marked)], [], 'n', 'b')['format_version'] == 3


def test_a_newer_reader_accepts_version_3_and_an_older_one_refuses_it(tmp_path, hpcp_mod, monkeypatch):
	path = tmp_path / 'manifest'
	path.write_text('format\thpcp-diff\nformat_version\t3\n')
	assert hpcp_mod.read_manifest(str(path))['format_version'] == '3'
	monkeypatch.setattr(hpcp_mod, 'DIFF_FORMAT_VERSION', 2)
	with pytest.raises(RuntimeError, match='uses format version 3, newer than supported 2'):
		hpcp_mod.read_manifest(str(path))


def test_rooted_excludes_match_like_the_rooted_path(hpcp_mod):
	"""Spec Part 1 / Review Focus 4: a pattern rooted at a mount matches a real path exactly when it matches '/' + the
	path inside; the mount's own path never matches a pattern's '*' part, and is escaped."""
	import fnmatch
	for root in ('/tmp/tmpab12', '/m/[x]*?', '/m/[x]*?/'):
		rooted = hpcp_mod._rooted_excludes(['/var/log/*', '*/tmp/*', '*/cache'], root)
		real = os.path.abspath(root).rstrip(os.sep)
		for inside, excluded in (('/var/log/a', True), ('/var/logs', False), ('/tmp/b', True), ('/a/cache', True),
				('/a', False), ('/var/log', False)):
			assert any(fnmatch.fnmatch(real + inside, pattern) for pattern in rooted) is excluded, (root, inside)
	assert hpcp_mod._rooted_excludes(['/var/log/*', '*/tmp/*'], '/') == frozenset(['/var/log/*', '*/tmp/*'])


def test_inside_excludes_key_paths_by_their_place_in_a_root(tmp_path, hpcp_mod):
	root, other = tmp_path / 'root', tmp_path / 'other'
	(root / 'a' / '[b]').mkdir(parents=True)
	other.mkdir()
	paths = [str(root / 'a' / '[b]'), str(other), str(root)]
	assert hpcp_mod._inside_excludes(paths, [str(root)]) == {'/a/[[]b]'}


def test_image_source_excludes_are_rooted_in_the_image(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Spec Part 1: for -si, '/var/log/*' means var/log inside the image, on both sides; '*/var/log/*' still works; and
	'*/tmp/*' or '/tmp/*' no longer match the (temporary) path the image is mounted at, which excluded the whole image."""
	new, base = trees
	image_root = tmp_path / 'image_root'
	_write(image_root / 'keep.txt', 'k')
	_write(image_root / 'var' / 'log' / 'a.log', 'log')
	_write(image_root / 'tmp' / 'b', 'b')
	_write(base / 'var' / 'log' / 'old.log', 'old')
	_write(base / 'tmp' / 'c', 'c')
	image = tmp_path / 'new.img'
	image.write_bytes(b'')
	_fake_image_mounts(hpcp_mod, monkeypatch, {str(image): image_root})
	for patterns in (['/var/log/*', '*/tmp/*'], ['*/var/log/*', '/tmp/*']):
		rows = _rows_before_mkfs(hpcp_mod, monkeypatch, [], [str(image)], [str(base)], None, str(tmp_path / 'p.img'), [], [],
			exclude=frozenset(patterns), max_workers=1)
		ops = _ops(rows)
		assert ('A', 'keep.txt') in ops, patterns
		assert not [path for _, path in ops if path.startswith(('var/log/', 'tmp/')) and path not in ('var/log/', 'tmp/')], patterns


def test_exclude_file_patterns_are_rooted_like_e(trees, hpcp_mod, tmp_path, monkeypatch):
	"""-x patterns go through format_exclude with -e ones and are rooted the same way."""
	new, base = trees
	image_root = tmp_path / 'image_root'
	_write(image_root / 'keep.txt', 'k')
	_write(image_root / 'var' / 'log' / 'a.log', 'log')
	image = tmp_path / 'new.img'
	image.write_bytes(b'')
	listing = tmp_path / 'exclude.txt'
	listing.write_text('/var/log/*\n')
	_fake_image_mounts(hpcp_mod, monkeypatch, {str(image): image_root})
	rows = _rows_before_mkfs(hpcp_mod, monkeypatch, [], [str(image)], [str(base)], None, str(tmp_path / 'p.img'), [], [],
		exclude=hpcp_mod.format_exclude(None, str(listing)), max_workers=1)
	assert ('A', 'keep.txt') in _ops(rows) and ('A', 'var/log/a.log') not in _ops(rows)


def test_directory_source_excludes_still_match_real_paths(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Spec Part 1 item 4: a directory source keeps copy mode's matching: real paths, not '/' + the path inside."""
	new, base = trees
	_write(new / 'keep.txt', 'k')
	_write(new / 'var' / 'log' / 'a.log', 'log')
	real = _rows_before_mkfs(hpcp_mod, monkeypatch, [str(new)], None, [str(base)], None, str(tmp_path / 'p.img'), [], [],
		exclude=frozenset([str(new / 'var' / 'log') + '/*']), max_workers=1)
	assert ('A', 'keep.txt') in _ops(real) and ('A', 'var/log/a.log') not in _ops(real)
	rooted = _rows_before_mkfs(hpcp_mod, monkeypatch, [str(new)], None, [str(base)], None, str(tmp_path / 'p2.img'), [], [],
		exclude=frozenset(['/var/log/*']), max_workers=1)
	assert ('A', 'var/log/a.log') in _ops(rooted)


def test_rooted_maps_skip_excluded_directories_on_both_sides(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Spec Part 1 item 6: an excluded directory is skipped, not walked, on the new and the base side."""
	new, base = tmp_path / 'mnt_new', tmp_path / 'mnt_base'
	for root in (new, base):
		_write(root / 'var' / 'log' / 'deep' / 'a.log', 'log')
		_write(root / 'keep.txt', 'k')
	walked = []
	real = hpcp_mod.get_file_list_serial
	def spy(root, *args, **kwargs):
		walked.append(root)
		return real(root, *args, **kwargs)
	monkeypatch.setattr(hpcp_mod, 'get_file_list_serial', spy)
	new_map, base_map = hpcp_mod.build_rooted_maps(str(new), str(base), frozenset(['/var/log']), max_workers=1, parallel_file_listing=False)
	for entry_map in (new_map, base_map):
		assert 'var/' in entry_map and 'keep.txt' in entry_map
		assert not [key for key in entry_map if key.startswith('var/log')]
	log_dir = os.path.join('var', 'log')
	assert not [root for root in walked if root.endswith(log_dir) or log_dir + os.sep in root]


def test_di_image_inside_the_new_directory_stays_out(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Spec Part 2: `-diff / -di /root/golden-v1.img` must not install the base image on every node."""
	new, base = trees
	_write(new / 'a.txt', 'a')
	image = new / 'golden-v1.img'
	image.write_bytes(b'image bytes')
	store_root = tmp_path / 'store_root'
	store_root.mkdir()
	_fake_image_mounts(hpcp_mod, monkeypatch, {str(image): store_root})
	rows = _rows_before_mkfs(hpcp_mod, monkeypatch, [str(new)], None, [], str(image), str(tmp_path / 'p.img'), [], [], max_workers=1)
	assert any(row.path.endswith('/a.txt') for row in rows)
	assert not [row for row in rows if 'golden-v1.img' in row.path]


def test_si_image_inside_the_base_directory_stays_out(trees, hpcp_mod, tmp_path, monkeypatch):
	"""`-diff -si /new.img -d /`: the image file on the base side is no D row."""
	new, base = trees
	image_root = tmp_path / 'image_root'
	_write(image_root / 'a.txt', 'a')
	image = base / 'new.img'
	image.write_bytes(b'image bytes')
	_fake_image_mounts(hpcp_mod, monkeypatch, {str(image): image_root})
	rows = _rows_before_mkfs(hpcp_mod, monkeypatch, [], [str(image)], [str(base)], None, str(tmp_path / 'p.img'), [], [], max_workers=1)
	assert ('A', 'a.txt') in _ops(rows)
	assert not [row for row in rows if 'new.img' in row.path]


def test_base_directory_inside_the_new_directory_stays_out(trees, hpcp_mod, tmp_path, monkeypatch):
	"""`-diff / -d /mnt/oldroot` must not add the whole old tree under mnt/oldroot/."""
	new, base = trees
	_write(new / 'a.txt', 'a')
	oldroot = new / 'mnt' / 'oldroot'
	_write(oldroot / 'old.txt', 'old')
	rows = _rows_before_mkfs(hpcp_mod, monkeypatch, [str(new)], None, [str(oldroot)], None, str(tmp_path / 'p.img'), [], [], max_workers=1)
	assert ('A', 'a.txt') in _ops(rows) and ('A', 'mnt/') in _ops(rows)
	assert not [row for row in rows if row.path.startswith('mnt/oldroot')]


def test_new_directory_inside_the_base_directory_stays_out(trees, hpcp_mod, tmp_path, monkeypatch):
	"""The base listing must not see the new tree inside it and delete it on every node."""
	new, base = trees
	inner = base / 'srv' / 'golden'
	_write(inner / 'a.txt', 'a')
	_write(base / 'keep.txt', 'k')
	rows = _rows_before_mkfs(hpcp_mod, monkeypatch, [str(inner)], None, [str(base)], None, str(tmp_path / 'p.img'), [], [], max_workers=1)
	assert ('A', 'a.txt') in _ops(rows) and ('D', 'keep.txt') in _ops(rows)
	assert not [row for row in rows if row.path.startswith('srv/golden')]


def test_base_directory_given_through_a_symlink_inside_the_new_directory_stays_out(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Review Focus 5: inputs are compared by real path, so a symlinked spelling of the base is still found."""
	new, base = trees
	_write(new / 'a.txt', 'a')
	oldroot = new / 'mnt' / 'oldroot'
	_write(oldroot / 'old.txt', 'old')
	link = tmp_path / 'oldroot-link'
	os.symlink(str(oldroot), str(link))
	rows = _rows_before_mkfs(hpcp_mod, monkeypatch, [str(new)], None, [str(link)], None, str(tmp_path / 'p.img'), [], [], max_workers=1)
	assert ('A', 'a.txt') in _ops(rows)
	assert not [row for row in rows if row.path.startswith('mnt/oldroot')]


def _one_mount(hpcp_mod, monkeypatch, fs_type='ext4', options=('rw',), super_options=('rw',)):
	"""mountinfo holding every path on one mount at / (no ids: a test double is never hidden)."""
	entry = {'devnum': (0, 99), 'root': '/', 'mount_point': '/', 'options': list(options), 'fstype': fs_type,
		'source': '/dev/x', 'super_options': list(super_options)}
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda *args, **kwargs: [dict(entry)])


def _scan_and_build(hpcp_mod, new, base, tmp_path, monkeypatch, change=lambda: None, max_workers=1, scan=True, volume='vol'):
	"""Scan new vs base (keeping the scan stats unless scan is False), change() the sources, then build the patch.
	Returns the payload copies _check_payload hashed again, relative to the volume."""
	new_map, base_map = hpcp_mod.build_root_maps(str(new), str(base), max_workers=1)
	kept = {} if scan else None
	rows = hpcp_mod.compute_changes(new_map, base_map, max_workers=max_workers, scan=kept)
	change()
	hashed = []
	real = hpcp_mod._states_for_paths
	def spy(paths, *args, **kwargs):
		paths = list(paths)
		hashed.extend(paths)
		return real(paths, *args, **kwargs)
	monkeypatch.setattr(hpcp_mod, '_states_for_paths', spy)
	vol = tmp_path / volume
	vol.mkdir()
	hpcp_mod.build_patch_tree(rows, str(new), [str(vol)], hpcp_mod.make_patch_meta(rows, 'n', 'b', 'root'),
		max_workers=max_workers, scan=kept)
	monkeypatch.setattr(hpcp_mod, '_states_for_paths', real)
	return sorted(os.path.relpath(path, str(vol)).replace(os.sep, '/') for path in hashed)


def test_mountinfo_entries_keep_super_options(tmp_path, hpcp_mod):
	path = tmp_path / 'mountinfo'
	path.write_text('36 35 98:0 / /mnt/a rw,noatime master:1 - ext4 /dev/sda1 ro,errors=continue\n'
		'37 35 98:1 / /mnt/b ro - vfat /dev/sda2\n')
	entries = hpcp_mod._mountinfo_entries(str(path))
	assert entries[0]['options'] == ['rw', 'noatime'] and entries[0]['super_options'] == ['ro', 'errors=continue']
	assert entries[1]['super_options'] == []


def test_compute_changes_keeps_the_scan_stats_of_new_files(trees, hpcp_mod):
	new, base = trees
	_write(new / 'a.txt', 'a')
	_write(base / 'b.txt', 'b')
	before = int(time.time() * 1000000000)
	scan = {}
	hpcp_mod.compute_changes(*hpcp_mod.build_root_maps(str(new), str(base), max_workers=1), max_workers=1, scan=scan)
	st = os.lstat(str(new / 'a.txt'))
	assert scan['stats'] == {'a.txt': (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)}
	assert before <= scan['started_ns'] <= int(time.time() * 1000000000)


def test_compute_changes_keeps_scan_stats_only_for_added_and_modified_files(trees, hpcp_mod):
	new, base = trees
	_write(new / 'a.txt', 'a')
	_write(new / 'same.txt', 'same')
	_write(base / 'same.txt', 'same')
	os.utime(str(base / 'same.txt'), ns=(os.lstat(str(new / 'same.txt')).st_atime_ns, os.lstat(str(new / 'same.txt')).st_mtime_ns))
	_write(new / 'm.txt', 'new content')
	_write(base / 'm.txt', 'old')
	_write(base / 'gone.txt', 'gone')
	scan = {}
	rows = hpcp_mod.compute_changes(*hpcp_mod.build_root_maps(str(new), str(base), max_workers=1), max_workers=1, scan=scan)
	assert {(row.op, row.path) for row in rows} == {('A', 'a.txt'), ('M', 'm.txt'), ('D', 'gone.txt')}
	assert sorted(scan['stats']) == ['a.txt', 'm.txt']


def test_compute_changes_parallel_scan_stats_cover_only_changed_new_files(trees, hpcp_mod):
	new, base = trees
	for index in range(300):
		_write(new / 'many' / f'f{index:03}.txt', f'file {index:03}')
		_write(base / 'many' / f'f{index:03}.txt', f'file {index:03}')
		_write(base / 'only' / f'g{index:03}.txt', f'base {index:03}')
	_write(new / 'many' / 'f007.txt', 'FILE 007')
	_write(new / 'added.txt', 'added')
	scan = {}
	hpcp_mod.compute_changes(*hpcp_mod.build_root_maps(str(new), str(base), max_workers=1), max_workers=2, scan=scan)
	assert 'added.txt' in scan['stats'] and 'many/f007.txt' in scan['stats']
	assert not [key for key in scan['stats'] if key.startswith('only/')]
	assert len(scan['stats']) < 10


@pytest.mark.parametrize('workers', [1, 2])
def test_states_for_paths_collects_stats_only_for_the_wanted_paths(tmp_path, hpcp_mod, reset_hpcp_globals, workers):
	paths = []
	for index in range(300):
		path = tmp_path / f'f{index:03}.txt'
		_write(path, f'file {index:03}')
		paths.append(str(path))
	wanted = {paths[3], paths[290]}
	stats = {}
	hpcp_mod._states_for_paths(paths, max_workers=workers, stats=stats, stat_paths=wanted)
	assert set(stats) == wanted
	assert stats[paths[3]] == hpcp_mod._scan_stat(os.lstat(paths[3]))


def _ro_recheck(hpcp_mod, monkeypatch, tmp_path, entries, stat_delta, with_stat=True):
	(tmp_path / 'f').write_text('x')
	source = os.path.realpath(str(tmp_path / 'f'))
	st = os.lstat(source)
	key = os.path.relpath(source, '/')
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda *args, **kwargs: [dict(entry) for entry in entries])
	monkeypatch.setattr(hpcp_mod, '_racy_window_ns', lambda: 0)
	row = hpcp_mod.ChangeRow('A', key, '-', hpcp_mod.get_entry_state(source))
	scan = {'started_ns': st.st_ctime_ns + 10 ** 9,
		'stats': {key: (st.st_dev, st.st_ino, st.st_size + stat_delta, st.st_mtime_ns, st.st_ctime_ns)} if with_stat else {}}
	return hpcp_mod._rows_to_recheck([row], '/', scan), row


def test_recheck_of_a_read_only_row_still_compares_the_scan_stat(tmp_path, hpcp_mod, monkeypatch):
	"""A ro mount replaced at the same path, or a remount-ro after a write: the stat moved, so the copy is read again."""
	picked, row = _ro_recheck(hpcp_mod, monkeypatch, tmp_path, [_entry('/', 'ext4', ['ro'])], 1)
	assert picked == [row]


def test_recheck_of_a_read_only_row_with_a_matching_stat_is_skipped(tmp_path, hpcp_mod, monkeypatch):
	picked, row = _ro_recheck(hpcp_mod, monkeypatch, tmp_path, [_entry('/', 'ext4', ['ro'])], 0)
	assert picked == []


def test_recheck_of_a_read_only_row_without_a_scan_stat_is_skipped(tmp_path, hpcp_mod, monkeypatch):
	picked, row = _ro_recheck(hpcp_mod, monkeypatch, tmp_path, [_entry('/', 'ext4', ['ro'])], 0, with_stat=False)
	assert picked == []


def test_recheck_without_a_mountinfo_entry_rechecks_every_live_row(tmp_path, hpcp_mod, monkeypatch):
	picked, row = _ro_recheck(hpcp_mod, monkeypatch, tmp_path, [], 0)
	assert picked == [row]


def test_payload_recheck_skips_a_read_only_filesystem(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Spec Part 3 item 1: a source on a read-only superblock cannot change; its copies are not read again."""
	new, base = trees
	_write(new / 'a.txt', 'a')
	_one_mount(hpcp_mod, monkeypatch, options=('ro',), super_options=('ro',))
	assert _scan_and_build(hpcp_mod, new, base, tmp_path, monkeypatch) == []


@pytest.mark.parametrize('options', [('rw',), ('ro',)])
def test_payload_recheck_catches_a_rewrite_that_keeps_size_and_mtime(trees, hpcp_mod, tmp_path, monkeypatch, options):
	"""Only ctime shows this rewrite; a read-only bind of a writable filesystem (mount ro, superblock rw) is live."""
	new, base = trees
	_write(new / 'live.log', 'version 1')
	_one_mount(hpcp_mod, monkeypatch, options=options)
	monkeypatch.setattr(hpcp_mod, '_racy_window_ns', lambda: 0)
	with pytest.raises(RuntimeError, match=_CHANGED_WHILE_BUILT):
		_scan_and_build(hpcp_mod, new, base, tmp_path, monkeypatch, lambda: _write(new / 'live.log', 'version 2'))


def test_payload_recheck_trusts_an_unchanged_file_outside_the_racy_window(trees, hpcp_mod, tmp_path, monkeypatch):
	new, base = trees
	_write(new / 'a.txt', 'a')
	_write(new / 'sub' / 'b.txt', 'b')
	_one_mount(hpcp_mod, monkeypatch)
	monkeypatch.setattr(hpcp_mod, '_racy_window_ns', lambda: 0)
	assert _scan_and_build(hpcp_mod, new, base, tmp_path, monkeypatch) == []


def test_payload_recheck_hashes_a_file_changed_within_the_racy_window(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Written a moment before the scan: a second write in the same timestamp tick would keep its ctime."""
	new, base = trees
	_write(new / 'a.txt', 'a')
	_one_mount(hpcp_mod, monkeypatch)
	assert _scan_and_build(hpcp_mod, new, base, tmp_path, monkeypatch) == ['a.txt']


@pytest.mark.parametrize('fs_type', ['vfat', 'exfat', 'ntfs3', 'fuseblk', 'nfs4', 'cifs', 'zfs', 'tmpfs'])
def test_payload_recheck_hashes_files_on_an_untrusted_filesystem(trees, hpcp_mod, tmp_path, monkeypatch, fs_type):
	new, base = trees
	_write(new / 'a.txt', 'a')
	_one_mount(hpcp_mod, monkeypatch, fs_type=fs_type)
	monkeypatch.setattr(hpcp_mod, '_racy_window_ns', lambda: 0)
	assert _scan_and_build(hpcp_mod, new, base, tmp_path, monkeypatch) == ['a.txt']


def test_payload_recheck_without_scan_stats_hashes_every_copy(trees, hpcp_mod, tmp_path, monkeypatch):
	new, base = trees
	_write(new / 'a.txt', 'a')
	_one_mount(hpcp_mod, monkeypatch)
	monkeypatch.setattr(hpcp_mod, '_racy_window_ns', lambda: 0)
	assert _scan_and_build(hpcp_mod, new, base, tmp_path, monkeypatch, scan=False) == ['a.txt']


def test_payload_recheck_hashes_a_metadata_change_and_keeps_the_copy(trees, hpcp_mod, tmp_path, monkeypatch):
	"""A chmod moves ctime: the copy is hashed again, matches its row, and the patch is written."""
	new, base = trees
	_write(new / 'a.txt', 'a')
	_one_mount(hpcp_mod, monkeypatch)
	monkeypatch.setattr(hpcp_mod, '_racy_window_ns', lambda: 0)
	assert _scan_and_build(hpcp_mod, new, base, tmp_path, monkeypatch, lambda: os.chmod(str(new / 'a.txt'), 0o600)) == ['a.txt']
	assert (tmp_path / 'vol' / '.hpcp_diff' / 'manifest').exists()


def test_payload_recheck_hashes_files_on_an_untrusted_mount_below_the_source(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Review Focus 3: `-diff /` with a FAT ESP at /boot/efi: only the ESP's files are read again."""
	new, base = trees
	_write(new / 'root.txt', 'r')
	_write(new / 'esp' / 'boot.efi', 'e')
	entries = [{'devnum': (0, 1), 'root': '/', 'mount_point': '/', 'options': ['rw'], 'fstype': 'ext4', 'source': '/dev/a',
			'super_options': ['rw']},
		{'devnum': (0, 2), 'root': '/', 'mount_point': os.path.realpath(str(new / 'esp')), 'options': ['rw'], 'fstype': 'vfat',
			'source': '/dev/b', 'super_options': ['rw']}]
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda *args, **kwargs: [dict(entry) for entry in entries])
	monkeypatch.setattr(hpcp_mod, '_racy_window_ns', lambda: 0)
	assert _scan_and_build(hpcp_mod, new, base, tmp_path, monkeypatch) == ['esp/boot.efi']


def test_payload_recheck_uses_scan_stats_from_worker_processes(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Review Focus 1: with >= 256 paths the scan runs in a process pool; its stats must come back from the workers."""
	new, base = trees
	for index in range(300):
		_write(new / 'many' / f'f{index:03}.txt', f'file {index:03}')
	_one_mount(hpcp_mod, monkeypatch)
	monkeypatch.setattr(hpcp_mod, '_racy_window_ns', lambda: 0)
	assert _scan_and_build(hpcp_mod, new, base, tmp_path, monkeypatch, max_workers=2) == []
	with pytest.raises(RuntimeError, match=r'^Diff image error: 1 file\(s\) changed while the patch was built \(e\.g\. many/f007\.txt\)'):
		_scan_and_build(hpcp_mod, new, base, tmp_path, monkeypatch, lambda: _write(new / 'many' / 'f007.txt', 'FILE 007'),
			max_workers=2, volume='vol2')


def test_payload_recheck_through_a_symlinked_new_root(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Review Focus 2: the scan lists the link's target, the copy goes through the link; stats are keyed by row."""
	new, base = trees
	_write(new / 'live.log', 'version 1')
	link = tmp_path / 'link'
	os.symlink(str(new), str(link))
	_one_mount(hpcp_mod, monkeypatch)
	monkeypatch.setattr(hpcp_mod, '_racy_window_ns', lambda: 0)
	assert _scan_and_build(hpcp_mod, link, base, tmp_path, monkeypatch) == []
	with pytest.raises(RuntimeError, match=_CHANGED_WHILE_BUILT):
		_scan_and_build(hpcp_mod, link, base, tmp_path, monkeypatch, lambda: _write(new / 'live.log', 'version 2'), volume='vol2')


def test_directory_source_generation_passes_the_scan_to_the_payload_check(trees, hpcp_mod, tmp_path, monkeypatch):
	"""process_get_diff_image hands compute_changes' scan to build_patch_tree (without it every copy is read again)."""
	new, base = trees
	_write(new / 'a.txt', 'a')
	vol = tmp_path / 'vol'
	vol.mkdir()
	monkeypatch.setattr('tempfile.tempdir', str(tmp_path))
	monkeypatch.setattr(hpcp_mod, '_claim_volume_names', lambda output, count: [output])
	monkeypatch.setattr(hpcp_mod, 'create_image', lambda *args, **kwargs: [str(vol)])
	_one_mount(hpcp_mod, monkeypatch)
	monkeypatch.setattr(hpcp_mod, '_racy_window_ns', lambda: 0)
	rechecked = []
	real = hpcp_mod._states_for_paths
	def spy(paths, *args, **kwargs):
		paths = list(paths)
		if kwargs.get('read_xattrs', True) is False:
			rechecked.extend(paths)
		return real(paths, *args, **kwargs)
	monkeypatch.setattr(hpcp_mod, '_states_for_paths', spy)
	out = str(tmp_path / 'p.img')
	assert hpcp_mod.process_get_diff_image([str(new)], None, [str(base)], None, out, [], [], max_workers=1) == out
	assert rechecked == [] and (vol / '.hpcp_diff' / 'manifest').exists()


@requires_root_and_image_tools
@pytest.mark.root
@pytest.mark.loop
def test_image_source_payload_is_not_read_again(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Spec Part 3 item 1 on a real read-only loop mount: a -si image's payload copies are not hashed a second time."""
	new, base = trees
	_write(new / 'a.txt', 'a' * 1000)
	_write(new / 'sub' / 'b.txt', 'b')
	image = str(tmp_path / 'new.img')
	rc = hpcp_mod.hpcp([str(new)], dest_image=image, single_thread=True, max_workers=1, batch=True, do_not_remove_files_while_listing=True)
	assert rc in (None, 0)
	rechecked = []
	real = hpcp_mod._states_for_paths
	def spy(paths, *args, **kwargs):
		paths = list(paths)
		if kwargs.get('read_xattrs', True) is False:
			rechecked.extend(paths)
		return real(paths, *args, **kwargs)
	monkeypatch.setattr(hpcp_mod, '_states_for_paths', spy)
	out = str(tmp_path / 'patch.img')
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		assert hpcp_mod.process_get_diff_image([], [image], [str(base)], None, out, mount_points, loop_devices, max_workers=1) == out
	assert rechecked == []


def _vm_files(hpcp_mod, monkeypatch, tmp_path, expire, writeback):
	"""Point _read_vm_centisecs at fake /proc/sys/vm files (None: file missing)."""
	for name, value in (('dirty_expire_centisecs', expire), ('dirty_writeback_centisecs', writeback)):
		if value is not None:
			(tmp_path / name).write_text(f'{value}\n')
	real = open
	monkeypatch.setattr('builtins.open', lambda path, *args, **kwargs: real(str(tmp_path / os.path.basename(path)), *args, **kwargs)
		if str(path).startswith('/proc/sys/vm/') else real(path, *args, **kwargs))


@pytest.mark.parametrize('expire, writeback, expected', [
	(3000, 500, 2 * 3500 * 10000000),
	(10, 10, 2000000000),
	(3000, 0, None),
	(None, 500, 120000000000),
	(3000, None, 120000000000),
	('x', 500, 120000000000),
])
def test_racy_window_follows_the_writeback_horizon(hpcp_mod, tmp_path, monkeypatch, expire, writeback, expected):
	_vm_files(hpcp_mod, monkeypatch, tmp_path, expire, writeback)
	assert hpcp_mod._racy_window_ns() == expected


def test_recheck_without_a_window_hashes_every_live_row(trees, hpcp_mod, tmp_path, monkeypatch):
	new, base = trees
	_write(new / 'a.txt', 'a')
	_one_mount(hpcp_mod, monkeypatch)
	monkeypatch.setattr(hpcp_mod, '_racy_window_ns', lambda: None)
	assert _scan_and_build(hpcp_mod, new, base, tmp_path, monkeypatch) == ['a.txt']


def _store_layout_recheck(hpcp_mod, monkeypatch, tmp_path, entries, fresh=False):
	"""A row of a store-layout scan ('/' as new_base) whose source is spelled through a symlinked directory, with a stale scan stat.
	Returns the rows _rows_to_recheck picks."""
	real = tmp_path / 'real' / 'u'
	real.mkdir(parents=True)
	(real / 'f').write_text('x')
	link = tmp_path / 'home'
	os.symlink(str(tmp_path / 'real'), str(link))
	key = os.path.relpath(str(link / 'u' / 'f'), '/')
	st = os.lstat(str(real / 'f'))
	stale = (st.st_dev, st.st_ino, st.st_size + (0 if fresh else 1), st.st_mtime_ns, st.st_ctime_ns)
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda *args, **kwargs: [dict(entry) for entry in entries])
	monkeypatch.setattr(hpcp_mod, '_racy_window_ns', lambda: 0)
	row = hpcp_mod.ChangeRow('A', key, '-', hpcp_mod.get_entry_state(str(real / 'f')))
	scan = {'started_ns': st.st_ctime_ns + 10 ** 9, 'stats': {key: stale}}
	return hpcp_mod._rows_to_recheck([row], '/', scan), row


def _entry(mount_point, fstype, super_options):
	return {'devnum': (0, 1), 'root': '/', 'mount_point': mount_point, 'options': ['rw'], 'fstype': fstype, 'source': '/dev/x',
		'super_options': super_options}


def test_recheck_judges_a_store_layout_row_by_the_filesystem_holding_its_file_not_its_link(tmp_path, hpcp_mod, monkeypatch):
	"""A read-only root with a writable ext4 mounted where the symlink points: the row is not skipped as read-only."""
	entries = [_entry('/', 'erofs', ['ro']), _entry(os.path.realpath(str(tmp_path / 'real')), 'ext4', ['rw'])]
	picked, row = _store_layout_recheck(hpcp_mod, monkeypatch, tmp_path, entries)
	assert picked == [row]


def test_recheck_hashes_a_store_layout_row_whose_symlink_target_is_on_fat(tmp_path, hpcp_mod, monkeypatch):
	entries = [_entry('/', 'ext4', ['rw']), _entry(os.path.realpath(str(tmp_path / 'real')), 'vfat', ['rw'])]
	picked, row = _store_layout_recheck(hpcp_mod, monkeypatch, tmp_path, entries, fresh=True)
	assert picked == [row]


@pytest.mark.parametrize('root, file_mount, fresh', [
	(_entry('/', 'erofs', ['ro']), ('ext4', ['rw']), False),
	(_entry('/', 'ext4', ['rw']), ('tmpfs', ['rw']), True),
])
def test_recheck_judges_a_row_by_a_mount_on_the_file_itself(tmp_path, hpcp_mod, monkeypatch, root, file_mount, fresh):
	"""A bind mount of one file (/etc/machine-id from a tmpfs, /etc/hosts of a container) holds that file."""
	(tmp_path / 'etc').mkdir()
	(tmp_path / 'etc' / 'machine-id').write_text('x')
	source = os.path.join(os.path.realpath(str(tmp_path / 'etc')), 'machine-id')
	st = os.lstat(source)
	key = os.path.relpath(source, '/')
	entries = [dict(root), _entry(source, *file_mount)]
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda *args, **kwargs: [dict(entry) for entry in entries])
	monkeypatch.setattr(hpcp_mod, '_racy_window_ns', lambda: 0)
	row = hpcp_mod.ChangeRow('A', key, '-', hpcp_mod.get_entry_state(source))
	scan = {'started_ns': st.st_ctime_ns + 10 ** 9,
		'stats': {key: (st.st_dev, st.st_ino, st.st_size + (0 if fresh else 1), st.st_mtime_ns, st.st_ctime_ns)}}
	assert hpcp_mod._rows_to_recheck([row], '/', scan) == [row]


def _automount_entries(target, triggered, net=False):
	"""mountinfo for an ext4 / with an automounted ESP at target/boot/efi (autofs; vfat stacked on it once triggered),
	and optionally an untouched automount at target/net."""
	real = os.path.realpath(str(target))
	esp = os.path.join(real, 'boot', 'efi')
	entries = [
		{'mount_id': 20, 'parent_id': 1, 'devnum': (8, 2), 'root': '/', 'mount_point': '/', 'options': ['rw'],
			'fstype': 'ext4', 'source': '/dev/sda2', 'super_options': ['rw']},
		{'mount_id': 30, 'parent_id': 20, 'devnum': (0, 50), 'root': '/', 'mount_point': esp, 'options': ['rw'],
			'fstype': 'autofs', 'source': 'systemd-1', 'super_options': ['rw', 'fd=40', 'direct']},
	]
	if net:
		entries.append({'mount_id': 40, 'parent_id': 20, 'devnum': (0, 51), 'root': '/', 'mount_point': os.path.join(real, 'net'),
			'options': ['rw'], 'fstype': 'autofs', 'source': 'auto.net', 'super_options': ['rw', 'direct']})
	if triggered:
		entries.append({'mount_id': 31, 'parent_id': 30, 'devnum': (8, 1), 'root': '/', 'mount_point': esp, 'options': ['rw'],
			'fstype': 'vfat', 'source': '/dev/sda1', 'super_options': ['rw']})
	return entries


def _esp_patch(hpcp_mod, tmp_path):
	"""A patch adding boot/efi/EFI/x.efi; returns (volume, target), the target holding boot/efi/ and net/."""
	new, base = tmp_path / 'new', tmp_path / 'base'
	_write(new / 'boot' / 'efi' / 'EFI' / 'x.efi', 'efi')
	(base / 'boot' / 'efi').mkdir(parents=True)
	for tree in (new, base):
		(tree / 'net').mkdir()
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	return str(vol), target


def _apply_until_preflight(hpcp_mod, monkeypatch, volume, target, state, net=False):
	"""apply_patch_tree with faked mountinfo (see _automount_entries) up to the preflight; returns [(scope name,
	metadata, tz_tolerant)] of the planned sections and the automounts triggered."""
	triggered, planned = [], []
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda *args, **kwargs: _automount_entries(target, state['mounted'], net))
	def trigger(mount_point):
		triggered.append(mount_point)
		state['mounted'] = state['mounts_when_triggered']
	monkeypatch.setattr(hpcp_mod, '_trigger_automount', trigger)
	real_plan = hpcp_mod._plan_section
	def plan(name, rows, volume_roots, target_root, compare_hash, full_hash, max_workers, metadata=True, case_insensitive=False, tz_tolerant=False):
		planned.append((name, metadata, tz_tolerant))
		return real_plan(name, rows, volume_roots, target_root, compare_hash, full_hash, max_workers, metadata, case_insensitive, tz_tolerant)
	monkeypatch.setattr(hpcp_mod, '_plan_section', plan)
	class Stop(Exception):
		pass
	def stop(plan):
		raise Stop()
	monkeypatch.setattr(hpcp_mod, '_preflight_section', stop)
	with pytest.raises(Stop):
		hpcp_mod.apply_patch_tree([volume], str(target), max_workers=1)
	return planned, triggered


def test_apply_triggers_the_automounted_esp_it_writes_into(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Spec Part 1 item 3: the ESP at boot/efi is an untriggered automount (autofs); once triggered it is planned as
	the FAT filesystem it holds (content only, timezone tolerance)."""
	volume, target = _esp_patch(hpcp_mod, tmp_path)
	state = {'mounted': False, 'mounts_when_triggered': True}
	planned, triggered = _apply_until_preflight(hpcp_mod, monkeypatch, volume, target, state)
	assert triggered == [os.path.join(os.path.realpath(str(target)), 'boot', 'efi')]
	assert ('boot/efi/', False, True) in planned


def test_apply_refuses_an_automount_that_cannot_be_mounted(trees, hpcp_mod, tmp_path, monkeypatch):
	volume, target = _esp_patch(hpcp_mod, tmp_path)
	state = {'mounted': False, 'mounts_when_triggered': False}
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda *args, **kwargs: _automount_entries(target, False))
	monkeypatch.setattr(hpcp_mod, '_trigger_automount', lambda mount_point: None)
	with pytest.raises(RuntimeError, match=r'^Patch error: .*boot/efi is an automount \(autofs\) that could not be mounted; mount it and re-run$'):
		hpcp_mod.apply_patch_tree([volume], str(target), max_workers=1)
	assert not (target / 'boot' / 'efi' / 'EFI').exists()


def test_apply_leaves_unrelated_automounts_alone(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Review Focus 5: an automount the patch does not write into (net/) is never triggered."""
	volume, target = _esp_patch(hpcp_mod, tmp_path)
	state = {'mounted': False, 'mounts_when_triggered': True}
	planned, triggered = _apply_until_preflight(hpcp_mod, monkeypatch, volume, target, state, net=True)
	assert triggered == [os.path.join(os.path.realpath(str(target)), 'boot', 'efi')]


def test_trigger_automount_enters_the_mount_point(tmp_path, hpcp_mod, monkeypatch):
	"""A plain stat of an automount point does not trigger it; a lookup through it (mount_point/.) does."""
	seen = []
	real_stat = os.stat
	monkeypatch.setattr(hpcp_mod.os, 'stat', lambda path, *args, **kwargs: seen.append(path) or real_stat(path, *args, **kwargs))
	hpcp_mod._trigger_automount(str(tmp_path))
	hpcp_mod._trigger_automount(str(tmp_path / 'missing'))
	assert seen == [os.path.join(str(tmp_path), '.'), os.path.join(str(tmp_path / 'missing'), '.')]


def test_generation_reads_mountinfo_after_its_listing_entered_any_automount(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Spec Part 1 item 3: generation needs no trigger, since its listing walks into an automount before mountinfo is
	read (FAT detection then sees the mounted ESP). Pin that order."""
	new, base = trees
	_write(new / 'a.txt', 'a')
	order = []
	real_maps, real_entries = hpcp_mod.build_root_maps, hpcp_mod._mountinfo_entries
	monkeypatch.setattr(hpcp_mod, 'build_root_maps', lambda *args, **kwargs: order.append('listing') or real_maps(*args, **kwargs))
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda *args, **kwargs: order.append('mountinfo') or real_entries(*args, **kwargs))
	_rows_before_mkfs(hpcp_mod, monkeypatch, [str(new)], None, [str(base)], None, str(tmp_path / 'p.img'), [], [], max_workers=1)
	assert 'mountinfo' in order and order.index('listing') < order.index('mountinfo')


@pytest.mark.parametrize('machine, numbers', [
	('x86_64', (0x80047210, 0x40047211)), ('i686', (0x80047210, 0x40047211)), ('aarch64', (0x80047210, 0x40047211)),
	('armv7l', (0x80047210, 0x40047211)), ('riscv64', (0x80047210, 0x40047211)), ('s390x', (0x80047210, 0x40047211)),
	('loongarch64', (0x80047210, 0x40047211)),
	('ppc64le', (0x40047210, 0x80047211)), ('ppc64', (0x40047210, 0x80047211)), ('ppc', (0x40047210, 0x80047211)),
	('mips', (0x40047210, 0x80047211)), ('mips64', (0x40047210, 0x80047211)), ('mipsel', (0x40047210, 0x80047211)),
	('sparc64', (0x40047210, 0x80047211)), ('alpha', (0x40047210, 0x80047211)), ('parisc64', (0x40047210, 0x80047211)),
])
def test_fat_ioctl_numbers_follow_the_architecture(hpcp_mod, machine, numbers):
	"""Spec Part 1 item 4: _IOR('r', 0x10, __u32) / _IOW('r', 0x11, __u32). alpha, mips, powerpc and sparc put
	_IOC_READ (2) and _IOC_WRITE (4) at bit 29, and parisc swaps the generic read and write bits: either way the two
	numbers come out the other way round."""
	assert hpcp_mod._fat_ioctl_numbers(machine) == numbers
	assert (hpcp_mod._FAT_IOCTL_GET_ATTRIBUTES, hpcp_mod._FAT_IOCTL_SET_ATTRIBUTES) == hpcp_mod._fat_ioctl_numbers()


def _compare_rows(hpcp_mod, monkeypatch, *args, **kwargs):
	captured = {}
	monkeypatch.setattr(hpcp_mod, 'report_changes', lambda rows, *rest, **options: captured.setdefault('rows', rows))
	hpcp_mod.process_compare_file_list(*args, **kwargs)
	return captured['rows']


def test_compare_mode_roots_exclude_patterns_for_image_sources(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Spec Part 3 item 1: -si image sources (and a -di image base) are listed with rooted matching in compare mode,
	as in -diff -si; '*/tmp/*' no longer matches the temporary mount path."""
	image_root, store_root = tmp_path / 'image_root', tmp_path / 'store_root'
	_write(image_root / 'changed.txt', 'new content')
	_write(store_root / 'changed.txt', 'old')
	_write(image_root / 'var' / 'log' / 'a.log', 'new log')
	_write(store_root / 'var' / 'log' / 'old.log', 'old log')
	rows = _compare_rows(hpcp_mod, monkeypatch, [str(image_root)], [str(store_root)], max_workers=1,
		exclude=frozenset(['/var/log/*', '*/tmp/*']), dest_image='store.img', src_image_roots=[str(image_root)])
	# Something is compared at all (before 9.72, '*/tmp/*' matched both temp mount paths and excluded everything) ...
	assert ('M', 'changed.txt') in _ops(rows)
	# ... and '/var/log/*' left out what is under var/log on both sides.
	assert not [row for row in rows if row.path.startswith('var/log/') and row.path != 'var/log/']


def test_compare_mode_roots_exclude_patterns_against_directory_bases(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Without -di, an image source's patterns are rooted on both sides, as -diff -si -d DIR does (C1 Part 1 item 3)."""
	image_root, base = tmp_path / 'image_root', tmp_path / 'base_dir'
	_write(image_root / 'var' / 'log' / 'a.log', 'new log')
	_write(base / 'var' / 'log' / 'old.log', 'old log')
	_write(image_root / 'changed.txt', 'new content')
	_write(base / 'changed.txt', 'old')
	rows = _compare_rows(hpcp_mod, monkeypatch, [str(image_root) + os.sep], [str(base) + os.sep], max_workers=1,
		exclude=frozenset(['/var/log/*']), src_image_roots=[str(image_root) + os.sep])
	assert any(row.op == 'M' and row.path.endswith('changed.txt') for row in rows)
	assert not [row for row in rows if 'var/log/' in row.path and not row.path.endswith('var/log/')]


def test_compare_mode_keeps_real_path_matching_for_directory_sources(trees, hpcp_mod, tmp_path, monkeypatch):
	new, base = trees
	_write(new / 'var' / 'log' / 'a.log', 'log')
	rows = _compare_rows(hpcp_mod, monkeypatch, [str(new) + os.sep], [str(base) + os.sep], max_workers=1,
		exclude=frozenset(['/var/log/*']))
	assert any(row.path.endswith('var/log/a.log') for row in rows)


def test_format_exclude_skips_blank_values(tmp_path, hpcp_mod):
	"""Spec Part 3 item 3: a blank -e value or -x line is ignored (it would become '*/', matching every root)."""
	xfile = tmp_path / 'x.txt'
	xfile.write_text('/b\n\n  \n')
	assert hpcp_mod.format_exclude(['', '  ', '/a'], str(xfile)) == frozenset(['/a', '/b'])


@pytest.mark.parametrize('pattern', ['/', '*/'])
def test_mounted_source_refuses_a_root_matching_exclude(trees, hpcp_mod, tmp_path, monkeypatch, pattern):
	new, base = trees
	_write(new / 'a.txt', 'a')
	img = tmp_path / 'new.img'
	img.write_bytes(b'')
	_fake_image_mounts(hpcp_mod, monkeypatch, {str(img): new})
	def listed(*args, **kwargs):
		raise AssertionError('nothing may be listed before the refusal')
	monkeypatch.setattr(hpcp_mod, 'build_rooted_maps', listed)
	with pytest.raises(RuntimeError, match=r"^Invalid arguments: exclude pattern %s matches the root" % re.escape(repr(pattern))):
		hpcp_mod.process_get_diff_image([], [str(img)], [str(base)], None, str(tmp_path / 'p.img'), [], [],
			exclude=frozenset([pattern]), max_workers=1)


def test_directory_source_refuses_an_exclude_matching_its_root(trees, hpcp_mod, tmp_path):
	new, base = trees
	_write(new / 'a.txt', 'a')
	with pytest.raises(RuntimeError, match=r'^Invalid arguments: exclude pattern .* matches the root of '):
		hpcp_mod.process_get_diff_image([str(new)], None, [str(base)], None, str(tmp_path / 'p.img'), [], [],
			exclude=frozenset([str(new)]), max_workers=1)


def test_refuse_root_exclude_helper(hpcp_mod):
	for pattern in ('/', '*/'):
		with pytest.raises(RuntimeError, match='matches the root of /:'):
			hpcp_mod._refuse_root_exclude(frozenset([pattern]), root='/')
		with pytest.raises(RuntimeError, match='matches the root of the filesystem:'):
			hpcp_mod._refuse_root_exclude(frozenset([pattern]))
	hpcp_mod._refuse_root_exclude(frozenset(['/var/log/*', '*/cache']), root='/')
	hpcp_mod._refuse_root_exclude(frozenset(['/var/log/*', '*/cache']))
	hpcp_mod._refuse_root_exclude(None)


class _CountingDict(dict):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.lookups = 0
	def __contains__(self, key):
		self.lookups += 1
		return super().__contains__(key)
	def _scan(self, *args, **kwargs):
		raise AssertionError('the holding mount must be found by walking up the path, not by scanning the mounts')
	__iter__ = keys = values = items = _scan


def test_holding_mount_walks_up_the_path_not_the_mounts(hpcp_mod):
	"""Spec Part 3 item 2: the deepest mount holding a directory is found with at most one lookup per path component
	(plus '/'), however many mounts there are."""
	points = _CountingDict({'': 'root', '/nx/a': 'a', '/nx/a/b': 'b'})
	points.update({f'/nx-mnt/m{index}': index for index in range(5000)})
	for directory, holder in (('/nx/a/b/c/d', 'b'), ('/nx/a/x', 'a'), ('/nx/other', 'root'), ('/', 'root'), ('/nx/a/b', 'b')):
		points.lookups = 0
		assert hpcp_mod._holding_mount(points, directory) == holder
		assert points.lookups <= directory.count('/') + 1
	assert hpcp_mod._holding_mount(_CountingDict({'/nx/a': 'a'}), '/nx/other') is None


def test_compute_changes_collects_scan_stats_for_new_side_paths_only(trees, hpcp_mod, monkeypatch):
	"""C1 parked gap: base-side paths are never stat-collected (memory grows with the changes, not the tree)."""
	new, base = trees
	_write(new / 'a.txt', 'a')
	_write(base / 'b.txt', 'b')
	new_map, base_map = hpcp_mod.build_root_maps(str(new), str(base), max_workers=1)
	asked = []
	real = hpcp_mod._states_for_paths
	def spy(paths, *args, **kwargs):
		asked.append(set(kwargs.get('stat_paths') or ()))
		return real(paths, *args, **kwargs)
	monkeypatch.setattr(hpcp_mod, '_states_for_paths', spy)
	hpcp_mod.compute_changes(new_map, base_map, max_workers=1, scan={})
	assert asked == [set(new_map.values())]


def test_full_hash_patches_hash_every_payload_copy_again(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Spec Part 3 item 5: with -fh every copy is hashed again (as up to 9.70), even from a read-only source."""
	new, base = trees
	_write(new / 'a.txt', 'a')
	_write(new / 'sub' / 'b.txt', 'b')
	_one_mount(hpcp_mod, monkeypatch, options=('ro',), super_options=('ro',))
	new_map, base_map = hpcp_mod.build_root_maps(str(new), str(base), max_workers=1)
	scan = {}
	rows = hpcp_mod.compute_changes(new_map, base_map, full_hash=True, max_workers=1, scan=scan)
	vol = tmp_path / 'vol'
	vol.mkdir()
	hashed = []
	real = hpcp_mod._states_for_paths
	def spy(paths, *args, **kwargs):
		paths = list(paths)
		hashed.extend(paths)
		return real(paths, *args, **kwargs)
	monkeypatch.setattr(hpcp_mod, '_states_for_paths', spy)
	hpcp_mod.build_patch_tree(rows, str(new), [str(vol)], hpcp_mod.make_patch_meta(rows, 'n', 'b', 'root', full_hash=True),
		max_workers=1, scan=scan)
	assert sorted(os.path.relpath(path, str(vol)).replace(os.sep, '/') for path in hashed) == ['a.txt', 'sub/b.txt']


def test_full_hash_patches_catch_a_copy_damaged_after_the_copy(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Spec Part 3 item 5: a copy that came out wrong with the right size (and the source's mtime) stops a -fh
	generation before any manifest is written, even from a read-only source."""
	new, base = trees
	source = _write(new / 'a.txt', 'aaaa')
	_one_mount(hpcp_mod, monkeypatch, options=('ro',), super_options=('ro',))
	new_map, base_map = hpcp_mod.build_root_maps(str(new), str(base), max_workers=1)
	scan = {}
	rows = hpcp_mod.compute_changes(new_map, base_map, full_hash=True, max_workers=1, scan=scan)
	vol = tmp_path / 'vol'
	vol.mkdir()
	real_copy = hpcp_mod.copy_file_list_parallel
	def damaging_copy(*args, **kwargs):
		result = real_copy(*args, **kwargs)
		with open(str(vol / 'a.txt'), 'r+b') as f:
			f.write(b'X')
		st = os.stat(source)
		os.utime(str(vol / 'a.txt'), ns=(st.st_atime_ns, st.st_mtime_ns))
		return result
	monkeypatch.setattr(hpcp_mod, 'copy_file_list_parallel', damaging_copy)
	with pytest.raises(RuntimeError, match=r'^Diff image error: 1 file\(s\) changed while the patch was built \(e\.g\. a\.txt\)'):
		hpcp_mod.build_patch_tree(rows, str(new), [str(vol)], hpcp_mod.make_patch_meta(rows, 'n', 'b', 'root', full_hash=True),
			max_workers=1, scan=scan)
	assert not (vol / '.hpcp_diff' / 'manifest').exists()


def test_apply_refuses_an_indirect_automount_map_without_triggering_it(trees, hpcp_mod, tmp_path, monkeypatch):
	"""An indirect map mounts only through its keys: a patch writing under it is refused with the leave-it-out message."""
	new, base = trees
	_write(new / 'home' / 'u' / 'f.txt', 'f')
	(base / 'home').mkdir()
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	real = os.path.realpath(str(target))
	entries = [
		{'mount_id': 20, 'parent_id': 1, 'devnum': (8, 2), 'root': '/', 'mount_point': '/', 'options': ['rw'],
			'fstype': 'ext4', 'source': '/dev/sda2', 'super_options': ['rw']},
		{'mount_id': 30, 'parent_id': 20, 'devnum': (0, 50), 'root': '/', 'mount_point': os.path.join(real, 'home'),
			'options': ['rw'], 'fstype': 'autofs', 'source': 'auto.home', 'super_options': ['rw', 'indirect']},
	]
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda *args, **kwargs: [dict(entry) for entry in entries])
	def trigger(mount_point):
		raise AssertionError('an indirect map must not be triggered')
	monkeypatch.setattr(hpcp_mod, '_trigger_automount', trigger)
	with pytest.raises(RuntimeError, match=r"^Patch error: .*home is an indirect automount map \(autofs\); hpcp does not write into one: "
			r"leave it out of the patch \(-e '/home/\*' when generating\) or mount what it needs and re-run$"):
		hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert not (target / 'home' / 'u').exists()


def test_full_marked_copies_are_hashed_again_without_fh(trees, hpcp_mod, tmp_path, monkeypatch):
	"""A FAT file (full-hash mark) is hashed again whatever its source; a plain file on a read-only source is not."""
	new, base = trees
	_write(new / 'fat.txt', 'fat')
	_write(new / 'plain.txt', 'plain')
	_one_mount(hpcp_mod, monkeypatch, options=('ro',), super_options=('ro',))
	new_map, base_map = hpcp_mod.build_root_maps(str(new), str(base), max_workers=1)
	scan = {}
	rows = hpcp_mod.compute_changes(new_map, base_map, max_workers=1, scan=scan, fat_paths={new_map['fat.txt']})
	assert hpcp_mod._full_marked(next(row for row in rows if row.path == 'fat.txt').new)
	assert not hpcp_mod._full_marked(next(row for row in rows if row.path == 'plain.txt').new)
	vol = tmp_path / 'vol'
	vol.mkdir()
	hashed = []
	real = hpcp_mod._states_for_paths
	def spy(paths, *args, **kwargs):
		paths = list(paths)
		hashed.extend(paths)
		return real(paths, *args, **kwargs)
	monkeypatch.setattr(hpcp_mod, '_states_for_paths', spy)
	hpcp_mod.build_patch_tree(rows, str(new), [str(vol)], hpcp_mod.make_patch_meta(rows, 'n', 'b', 'root'), max_workers=1, scan=scan)
	assert sorted(os.path.relpath(path, str(vol)).replace(os.sep, '/') for path in hashed) == ['fat.txt']


@pytest.mark.parametrize('last', ['..', '.'])
def test_remove_tree_around_mounts_refuses_dot_components(tmp_path, hpcp_mod, last):
	target = tmp_path / 'target'
	_write(target / 'f', 'x')
	_write(tmp_path / 'sibling' / 'keep', 'k')
	kept, errors = hpcp_mod._remove_tree_around_mounts(str(target) + '/' + last, set(), target_root=str(target))
	assert errors and 'refusing to remove . or ..' in errors[0]
	assert (target / 'f').read_text() == 'x' and (tmp_path / 'sibling' / 'keep').read_text() == 'k'


# ---- 9.73: undecodable names never abort an apply halfway (review D3) ----

def _strict_utf8_stdout(monkeypatch):
	"""stdout as under LANG=en_US.UTF-8: encoding errors are fatal."""
	import io
	stream = io.TextIOWrapper(io.BytesIO(), encoding='utf-8', errors='strict')
	monkeypatch.setattr(sys, 'stdout', stream)
	return stream


def _undecodable_restore_scenario(hpcp_mod, tmp_path):
	"""_restore_scenario with a file name that is not valid UTF-8."""
	name = os.fsdecode(b'x\xff.efi')
	target, vol = tmp_path / 'target', tmp_path / 'vol'
	_write(target / 'EFI' / 'BOOT' / name, 'loader')
	os.symlink('BOOT', str(target / 'EFI' / 'Boot'))
	_write(vol / 'EFI' / 'Boot' / name, 'loader')
	state = hpcp_mod.get_entry_state
	R = hpcp_mod.ChangeRow
	rows = [R('D', f'EFI/BOOT/{name}', state(str(target / 'EFI' / 'BOOT' / name)), '-'),
		R('A', f'EFI/Boot/{name}', '-', state(str(vol / 'EFI' / 'Boot' / name)))]
	return target, vol, rows


def test_apply_restore_note_prints_an_undecodable_name(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	hpcp_mod.HASH_SIZE = 65536
	target, vol, rows = _undecodable_restore_scenario(hpcp_mod, tmp_path)
	plan = hpcp_mod._plan_section('root', rows, [str(vol)], str(target), True, False, 1)
	stream = _strict_utf8_stdout(monkeypatch)
	summary = hpcp_mod._apply_section(plan, 1, False, False, True)
	assert summary['restored'] == 1 and summary['failed'] == 0
	stream.flush()
	assert b'x\\xff.efi' in stream.buffer.getvalue()


def test_restore_across_sections_note_prints_an_undecodable_name(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	hpcp_mod.HASH_SIZE = 65536
	target, vol, rows = _undecodable_restore_scenario(hpcp_mod, tmp_path)
	os.unlink(str(vol / 'EFI' / 'Boot' / rows[1].path.rpartition('/')[2]))
	plan = hpcp_mod._plan_section('root', rows, [str(vol)], str(target), True, False, 1)
	outcome = {}
	_strict_utf8_stdout(monkeypatch)
	summary = hpcp_mod._apply_section(plan, 1, False, False, True, outcome)
	hpcp_mod._restore_across_sections([(plan, summary, outcome)], 1, False, False, True)
	assert summary['failed'] == 1


def test_conflict_list_prints_undecodable_states(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	name = os.fsdecode(b'l\xfe')
	row = hpcp_mod.ChangeRow('M', name, 'l:' + os.fsdecode(b'old\xff'), 'l:new')
	check = hpcp_mod.PatchCheck(row, 'conflict', 'l:' + os.fsdecode(b'other\xfd'), 'differs')
	stream = _strict_utf8_stdout(monkeypatch)
	hpcp_mod._resolve_conflict_lines(hpcp_mod._conflict_lines([check]), True)
	stream.flush()
	assert b'other\\xfd' in stream.buffer.getvalue()


def test_payload_source_deleted_before_the_copy_is_diagnosed(trees, hpcp_mod, tmp_path):
	"""Review D6: a live file that went away after the scan made the payload copy's failure escape before _check_payload
	could name it; now the copy records it and the check reports the file as changed while the patch was built."""
	new, base = trees
	_write(new / 'stays.txt', 'stays')
	_write(new / 'rotated.log', 'gone soon')
	vol = tmp_path / 'vol'
	vol.mkdir()
	rows = _changes(hpcp_mod, new, base)
	os.unlink(str(new / 'rotated.log'))
	with pytest.raises(RuntimeError, match=r'^Diff image error: 1 file\(s\) changed while the patch was built \(e\.g\. rotated\.log\)'):
		hpcp_mod.build_patch_tree(rows, str(new), [str(vol)], hpcp_mod.make_patch_meta(rows, 'n', 'b', 'root'), max_workers=2)
	assert (vol / 'stays.txt').read_text() == 'stays'
	assert not (vol / '.hpcp_diff').exists()


@requires_root_and_image_tools
@pytest.mark.skipif(shutil.which('mkfs.xfs') is None, reason='needs mkfs.xfs')
@pytest.mark.root
@pytest.mark.loop
def test_clone_of_a_mounted_xfs_image_mounts_read_write(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Review D1: -siff PATCH -di CLONE.img mounts the target read-write; a clone of a mounted xfs has the same UUID."""
	golden, clone = str(tmp_path / 'golden.img'), str(tmp_path / 'clone.img')
	_make_xfs_image(hpcp_mod, golden, lambda root: _write(os.path.join(root, 'f.txt'), 'f'))
	subprocess.run(['cp', '--sparse=always', golden, clone], check=True)
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		hpcp_mod._mount_single_fs_image(golden, False, mount_points, loop_devices)
		root = hpcp_mod._mount_single_fs_image(clone, False, mount_points, loop_devices)
		_write(os.path.join(root, 'written.txt'), 'w')
		assert open(os.path.join(root, 'f.txt')).read() == 'f'


# ---- 9.73 ultrareview of the fix branch ----

def _fail_scandir_at(hpcp_mod, monkeypatch, failing):
	real = os.scandir
	def scandir(path='.'):
		if os.path.abspath(os.fspath(path)) == str(failing):
			raise PermissionError(errno.EACCES, 'Permission denied', str(path))
		return real(path)
	monkeypatch.setattr(hpcp_mod.os, 'scandir', scandir)


def test_build_entry_map_refuses_a_partial_listing(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""A directory that could not be listed would show its entries as deleted: no map, and so no patch, from it."""
	_write(tmp_path / 'new' / 'sub' / 'f.txt', 'f')
	_fail_scandir_at(hpcp_mod, monkeypatch, tmp_path / 'new' / 'sub')
	with pytest.raises(RuntimeError, match='^Listing error'):
		hpcp_mod.build_entry_map(str(tmp_path / 'new'), max_workers=1)


def test_build_entry_map_refuses_a_listing_whose_task_failed(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""The parallel listing records a crashed task and returns what it has: the same refusal."""
	_write(tmp_path / 'new' / 'f.txt', 'f')
	real = hpcp_mod.get_file_list
	def partial(*args, **kwargs):
		result = real(*args, **kwargs)
		hpcp_mod.eprint('Listing error: a task crashed')
		return result
	monkeypatch.setattr(hpcp_mod, 'get_file_list', partial)
	with pytest.raises(RuntimeError, match='^Listing error'):
		hpcp_mod.build_entry_map(str(tmp_path / 'new'), max_workers=2, parallel_file_listing=True)


def test_diff_with_an_unreadable_new_directory_writes_no_patch(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Ultrareview: base/sub/f.txt exists, new/sub cannot be listed. A patch would delete sub/f.txt on apply."""
	for side in ('new', 'base'):
		_write(tmp_path / side / 'sub' / 'f.txt', 'f')
	_fail_scandir_at(hpcp_mod, monkeypatch, tmp_path / 'new' / 'sub')
	out = tmp_path / 'p.img'
	assert hpcp_mod.hpcp([str(tmp_path / 'new')], dest_paths=[str(tmp_path / 'base')], get_diff_image=str(out),
		max_workers=1, single_thread=True, batch=True) == 209
	assert not out.exists()


def test_build_entry_map_ignores_a_listing_error_inside_a_logically_excluded_subtree(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""The base side lists everything and filters by the new tree's excludes afterwards: an unreadable directory the
	user excluded with -e must not refuse the diff (-e is the advice the refusal gives)."""
	_write(tmp_path / 'base' / 'cache' / 'c.bin', 'c')
	_write(tmp_path / 'base' / 'keep.txt', 'k')
	_fail_scandir_at(hpcp_mod, monkeypatch, tmp_path / 'base' / 'cache')
	entries = hpcp_mod.build_entry_map(str(tmp_path / 'base'), exclude=hpcp_mod.format_exclude(['*/cache']),
		logical_root=str(tmp_path / 'new'), max_workers=1)
	assert './' in entries and 'keep.txt' in entries and not any(key.startswith('cache') for key in entries)
