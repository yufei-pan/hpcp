import os
import pytest


def test_get_last_existing_parent_finds_root(tmp_path, hpcp_mod, reset_hpcp_globals):
	missing = tmp_path / 'a' / 'b' / 'c'
	parent = hpcp_mod.get_last_existing_parent(str(missing))
	assert os.path.isdir(parent)
	assert os.path.samefile(parent, tmp_path)


def test_get_file_size_missing_returns_zero(hpcp_mod, reset_hpcp_globals, tmp_path):
	missing = str(tmp_path / 'nope')
	assert hpcp_mod.get_file_size(missing) == 0


def test_natural_sort_orders_numeric_suffixes(hpcp_mod):
	assert hpcp_mod.natural_sort(['file10.txt', 'file2.txt', 'file1.txt']) == [
		'file1.txt', 'file2.txt', 'file10.txt',
	]


def test_format_time_examples(hpcp_mod):
	assert hpcp_mod.format_time(0) == '0s'
	assert hpcp_mod.format_time(65) == '1m5s'
	assert hpcp_mod.format_time(3661) == '1h1m1s'


def test_is_excluded_globs(hpcp_mod):
	assert hpcp_mod.is_excluded('/data/tmp/cache', ['*/cache']) is True
	assert hpcp_mod.is_excluded('/data/tmp/cache', ['*/logs']) is False
	assert hpcp_mod.is_excluded('/data/tmp/cache', None) is False


def test_get_file_size_reports_allocated_size_of_sparse_file(hpcp_mod, tmp_path):
	path = tmp_path / 'sparse.img'
	with open(path, 'wb') as f:
		f.truncate(64 << 20)
		f.write(b'x')
	st = os.stat(path)
	if not hasattr(st, 'st_blocks') or st.st_blocks * 512 >= st.st_size:
		pytest.skip('filesystem does not report sparse allocation')
	assert hpcp_mod.get_file_size(str(path)) == st.st_blocks * 512
	assert hpcp_mod.get_file_size(str(path)) < st.st_size
