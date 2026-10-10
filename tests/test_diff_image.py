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
