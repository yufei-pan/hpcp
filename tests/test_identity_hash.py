import os
import time
import pytest


def test_hash_file_stable_for_same_bytes(tmp_tree, hpcp_mod, reset_hpcp_globals):
	path = tmp_tree.add_file('x.bin', b'abcdefghijklmnopqrstuvwxyz')
	hpcp_mod.HASH_SIZE = 65536
	h1 = hpcp_mod.hash_file(path, os.path.getsize(path), full_hash=False)
	h2 = hpcp_mod.hash_file(path, os.path.getsize(path), full_hash=False)
	assert h1 == h2
	assert isinstance(h1, str) and len(h1) > 0


def test_hash_file_disabled_when_hash_size_zero(tmp_tree, hpcp_mod, reset_hpcp_globals):
	path = tmp_tree.add_file('x.bin', b'abc')
	hpcp_mod.HASH_SIZE = 0
	# Clear lru_cache on hash_file if present
	if hasattr(hpcp_mod.hash_file, 'cache_clear'):
		hpcp_mod.hash_file.cache_clear()
	assert hpcp_mod.hash_file(path, os.path.getsize(path)) == ''


def test_is_file_identical_true_for_same_mtime_size_hash(tmp_tree, hpcp_mod, reset_hpcp_globals):
	src = tmp_tree.add_file('a.txt', 'same', under='src')
	dst = tmp_tree.add_file('a.txt', 'same', under='dst')
	t = time.time() - 1000
	os.utime(src, (t, t))
	os.utime(dst, (t, t))
	hpcp_mod.HASH_SIZE = 65536
	if hasattr(hpcp_mod.hash_file, 'cache_clear'):
		hpcp_mod.hash_file.cache_clear()
	assert hpcp_mod.is_file_identical(src, dst, os.path.getsize(src), full_hash=False) is True


def test_is_file_identical_false_for_different_content(tmp_tree, hpcp_mod, reset_hpcp_globals):
	src = tmp_tree.add_file('a.txt', 'one', under='src')
	dst = tmp_tree.add_file('a.txt', 'two', under='dst')
	t = time.time() - 1000
	os.utime(src, (t, t))
	os.utime(dst, (t, t))
	hpcp_mod.HASH_SIZE = 65536
	if hasattr(hpcp_mod.hash_file, 'cache_clear'):
		hpcp_mod.hash_file.cache_clear()
	assert hpcp_mod.is_file_identical(src, dst, os.path.getsize(src), full_hash=True) is False


def test_full_hash_sees_difference_partial_hash_misses(tmp_tree, hpcp_mod, reset_hpcp_globals):
	hpcp_mod.HASH_SIZE = 4
	if hasattr(hpcp_mod.hash_file, 'cache_clear'):
		hpcp_mod.hash_file.cache_clear()
	src = tmp_tree.add_file('a.bin', b'XXXXYYYY', under='src')
	dst = tmp_tree.add_file('a.bin', b'ZZZZYYYY', under='dst')
	t = time.time() - 1000
	os.utime(src, (t, t))
	os.utime(dst, (t, t))
	assert hpcp_mod.is_file_identical(src, dst, os.path.getsize(src), full_hash=False) is True
	assert hpcp_mod.is_file_identical(src, dst, os.path.getsize(src), full_hash=True) is False


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
