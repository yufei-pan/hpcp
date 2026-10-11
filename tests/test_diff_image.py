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
	hpcp_mod.HASH_SIZE = 65536
	vol, target = _conflicted(hpcp_mod, new, base, tmp_path)
	hpcp_mod.HASH_SIZE = 1234
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
	real = hpcp_mod.copy_file_list_parallel
	calls = []

	def flaky(*args, **kwargs):
		calls.append(1)
		if len(calls) == 1:
			raise RuntimeError('simulated copy failure')
		return real(*args, **kwargs)

	monkeypatch.setattr(hpcp_mod, 'copy_file_list_parallel', flaky)
	monkeypatch.setattr(sys, 'stdin', _FakeStdin(False))
	first = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert first['failed'] > 0
	assert (target / 'mod.txt').read_text() == 'old'
	monkeypatch.setattr(hpcp_mod, 'copy_file_list_parallel', real)
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


def test_apply_full_hash_same_size_same_mtime_change(trees, hpcp_mod, tmp_path):
	new, base = trees
	hpcp_mod.HASH_SIZE = 4
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


def _fail_first_copy(hpcp_mod, monkeypatch):
	real = hpcp_mod.copy_file_list_parallel
	calls = []

	def flaky(*args, **kwargs):
		calls.append(1)
		if len(calls) == 1:
			raise RuntimeError('simulated copy failure')
		return real(*args, **kwargs)

	monkeypatch.setattr(hpcp_mod, 'copy_file_list_parallel', flaky)
	return real


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
	real = _fail_first_copy(hpcp_mod, monkeypatch)
	first = hpcp_mod.apply_patch_tree([str(vol)], str(target), force=True, max_workers=1)
	assert first['failed'] > 0
	assert 'user.tag' in os.listxattr(str(target / 'm.txt'))
	monkeypatch.setattr(hpcp_mod, 'copy_file_list_parallel', real)
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
	real = hpcp_mod.copy_file_list_parallel
	modes = []

	def recording(*args, **kwargs):
		modes.append(stat.S_IMODE(os.stat(target / 'secret').st_mode))
		return real(*args, **kwargs)

	monkeypatch.setattr(hpcp_mod, 'copy_file_list_parallel', recording)
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
