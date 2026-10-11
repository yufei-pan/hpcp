import contextlib
import hashlib
import os
import re
import shutil
import stat
import struct
import subprocess
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
	path.write_text('format\thpcp-diff\nformat_version\t3\n')
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
	assert hpcp_mod.compute_changes(*hpcp_mod.build_store_maps(str(new), str(store), max_workers=1), max_workers=1) == []


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


def _case_conflict(fs_type, a, b, where):
	return (rf'^Patch conflict: {re.escape(repr(a))} and {re.escape(repr(b))} differ only by case and {re.escape(str(where))} '
		rf'is a case-insensitive {fs_type} filesystem; applying would delete what it writes; nothing applied '
		r'\(regenerate with -dd from whole disks, or remove the case-only rename\)$')


def _unknown_case_conflict(fs_type, a, b, where):
	return (rf'^Patch conflict: {re.escape(repr(a))} and {re.escape(repr(b))} differ only by case and the filesystem holding '
		rf'{re.escape(str(where))} \({re.escape(fs_type or "unknown")}\) is not known to be case-sensitive; '
		r'applying could delete what it writes; nothing applied$')


@pytest.mark.parametrize('fs_type, refused', [('vfat', 'insensitive'), ('msdos', 'insensitive'), ('exfat', 'insensitive'),
	('hfsplus', 'insensitive'), ('fuseblk', 'unknown'), ('fuseblk.exfat', 'unknown'), ('cifs', 'unknown'), ('smb3', 'unknown'),
	('vboxsf', 'unknown'), ('ntfs3', 'unknown'), ('', 'unknown'),
	('ext4', None), ('xfs', None), ('btrfs', None), ('tmpfs', None), ('zfs', None), ('nfs4', None), ('overlay', None)])
def test_single_fs_apply_refuses_case_only_rename_on_case_insensitive_target(trees, hpcp_mod, tmp_path, reset_hpcp_globals,
		monkeypatch, fs_type, refused):
	"""A case-only rename in a single-filesystem patch is a D + A pair. On a FAT, exFAT or HFS+ target the A rows
	find the old entries and are skipped as done, then the D rows delete them: refused before anything changes,
	even with -fa. So it is on a filesystem not known to be case-sensitive (exFAT through FUSE, CIFS, ntfs3 with
	nocase, or a type mountinfo does not show): it fails closed. On a known case-sensitive one the pair applies."""
	new, base = trees
	_case_only_rename(new, base)
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	assert {(row.op, row.path) for row in rows} == {('D', 'EFI/BOOT/'), ('D', 'EFI/BOOT/BOOTX64.EFI'),
		('A', 'EFI/Boot/'), ('A', 'EFI/Boot/BOOTX64.EFI'), ('M', 'EFI/grub.cfg')}
	target = tmp_path / 'target'
	_clone(base, target)
	_fake_mounts(hpcp_mod, monkeypatch, (target, fs_type))
	if refused:
		before = _snapshot(target)
		conflict = _case_conflict if refused == 'insensitive' else _unknown_case_conflict
		with pytest.raises(RuntimeError, match=conflict(fs_type, 'EFI/BOOT/', 'EFI/Boot/', target)):
			hpcp_mod.apply_patch_tree([str(vol)], str(target), force=True, max_workers=1)
		assert _snapshot(target) == before and (target / 'EFI' / 'grub.cfg').read_text() == 'old cfg'
	else:
		summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
		assert (summary['deleted'], summary['added'], summary['modified'], summary['failed']) == (2, 2, 1, 0)
		assert _changes(hpcp_mod, new, target) == []


def test_single_fs_apply_finds_the_target_mount_by_longest_prefix(trees, hpcp_mod, tmp_path, reset_hpcp_globals, monkeypatch):
	"""The target's filesystem is the deepest mount holding its real path; of two on one mount point, the later
	(which is on top) counts. A target reached through a symlink is looked up by its real path."""
	new, base = trees
	_case_only_rename(new, base)
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'mnt' / 'esp'
	_clone(base, target)
	link = tmp_path / 'link'
	os.symlink(str(target), str(link))
	_fake_mounts(hpcp_mod, monkeypatch, ('/', 'ext4'), (target, 'ext4'), (tmp_path / 'mnt', 'ext4'), (target, 'vfat'))
	with pytest.raises(RuntimeError, match=_case_conflict('vfat', 'EFI/BOOT/', 'EFI/Boot/', str(link) + os.sep)):
		hpcp_mod.apply_patch_tree([str(vol)], str(link) + os.sep, max_workers=1)
	_fake_mounts(hpcp_mod, monkeypatch, ('/', 'ext4'), (target, 'vfat'), (tmp_path / 'mnt', 'vfat'), (target, 'ext4'))
	assert hpcp_mod.apply_patch_tree([str(vol)], str(link) + os.sep, max_workers=1)['failed'] == 0
	assert _changes(hpcp_mod, new, target) == []


@pytest.mark.parametrize('below_type', ['vfat', 'fuseblk'])
def test_single_fs_apply_refuses_case_only_rename_under_a_case_insensitive_mount_below_the_target(trees, hpcp_mod, tmp_path,
		reset_hpcp_globals, monkeypatch, below_type):
	"""`-siff store-patch.img -d /`: the target is ext4, but rows under boot/efi land on the vfat ESP mounted there
	(or on an exFAT ESP through FUSE, not known to be case-sensitive). A case-only rename elsewhere on the
	case-sensitive target still applies."""
	new, base = trees
	_case_only_rename(new, base, prefix='boot/efi')
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	_fake_mounts(hpcp_mod, monkeypatch, (target, 'ext4'), (target / 'boot' / 'efi', below_type))
	before = _snapshot(target)
	conflict = _case_conflict if below_type == 'vfat' else _unknown_case_conflict
	with pytest.raises(RuntimeError, match=conflict(below_type, 'boot/efi/EFI/BOOT/', 'boot/efi/EFI/Boot/',
			os.path.realpath(str(target / 'boot' / 'efi')))):
		hpcp_mod.apply_patch_tree([str(vol)], str(target), force=True, max_workers=1)
	assert _snapshot(target) == before
	# The same rename on the ext4 part of the target (boot/EFI is not under the vfat mount at boot/efi).
	new2, base2, vol2, target2 = (tmp_path / name for name in ('new2', 'base2', 'vol2', 'target2'))
	_case_only_rename(new2, base2, prefix='boot')
	_patch_from(hpcp_mod, new2, base2, [vol2])
	_clone(base2, target2)
	(target2 / 'boot' / 'efi').mkdir()
	(new2 / 'boot' / 'efi').mkdir()
	_fake_mounts(hpcp_mod, monkeypatch, (target2, 'ext4'), (target2 / 'boot' / 'efi', below_type))
	assert hpcp_mod.apply_patch_tree([str(vol2)], str(target2), max_workers=1)['failed'] == 0
	assert _changes(hpcp_mod, new2, target2) == []


def _same_entry_conflict(deleted, written):
	return (rf'^Patch conflict: {re.escape(repr(deleted))} \(deleted\) and {re.escape(repr(written))} \(written\) are the same '
		r'entry on the target - a case-insensitive or aliased filesystem; applying would delete what it writes; nothing applied$')


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


@requires_root_and_image_tools
@pytest.mark.skipif(shutil.which('mkfs.vfat') is None, reason='needs mkfs.vfat')
@pytest.mark.root
@pytest.mark.loop
def test_case_only_rename_patch_is_refused_on_a_vfat_image(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""-si golden_esp.img -di node_esp.img where golden renamed EFI/BOOT to EFI/Boot, then -siff onto node_esp.img:
	applied, the A rows would be skipped as present and the D rows delete the boot loader."""
	hpcp_mod.HASH_SIZE = 65536
	golden, node, out = str(tmp_path / 'golden.img'), str(tmp_path / 'node.img'), str(tmp_path / 'patch.img')
	for image, name in ((golden, 'Boot'), (node, 'BOOT')):
		with open(image, 'wb') as f:
			f.truncate(32 << 20)
		subprocess.run(['mkfs.vfat', image], check=True, stdout=subprocess.DEVNULL)
		_change_image(hpcp_mod, image, lambda root, name=name: _write(os.path.join(root, 'EFI', name, 'BOOTX64.EFI'), 'loader'))
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		assert hpcp_mod.process_get_diff_image([], [golden], [], node, out, mount_points, loop_devices, max_workers=1) == out
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		with pytest.raises(RuntimeError, match=r"^Patch conflict: 'EFI/BOOT/' and 'EFI/Boot/' differ only by case and /\S+ "
				r'is a case-insensitive vfat filesystem; applying would delete what it writes; nothing applied'):
			hpcp_mod.process_load_diff_image([out], [], [], node, mount_points, loop_devices, force=True, max_workers=1)
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		root = hpcp_mod._mount_single_fs_image(node, True, mount_points, loop_devices)
		assert os.listdir(os.path.join(root, 'EFI')) == ['BOOT']
		with open(os.path.join(root, 'EFI', 'BOOT', 'BOOTX64.EFI')) as f:
			assert f.read() == 'loader'


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
	def partial_rmtree(path, *args, **kwargs):
		os.unlink(str(target / 'keep.txt'))
		raise OSError('boom')
	monkeypatch.setattr(hpcp_mod.shutil, 'rmtree', partial_rmtree)
	restored = hpcp_mod._apply_rows([row], [str(vol)], str(target), forced_rows=[row], skipped=[skipped])
	assert restored == [skipped] and (target / 'keep.txt').read_text() == 'k'
