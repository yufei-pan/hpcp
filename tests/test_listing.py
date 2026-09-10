import os
import pytest


def test_get_file_list_serial_lists_files(tmp_tree, hpcp_mod, reset_hpcp_globals):
	tmp_tree.add_file('a.txt', 'A')
	tmp_tree.add_file('sub/b.txt', 'B')
	# Returns (file_list, links, size, folders)
	files, links, size, folders = hpcp_mod.get_file_list(
		str(tmp_tree.src),
		parallel_file_listing=False,
		drop_cache=True,
		remove_files_while_listing=False,
	)
	joined = ' '.join(map(str, files))
	assert 'a.txt' in joined
	assert 'b.txt' in joined
	assert size >= 2
	assert isinstance(links, (set, frozenset))
	assert isinstance(folders, (set, frozenset))


def test_get_file_list_respects_exclude(tmp_tree, hpcp_mod, reset_hpcp_globals):
	tmp_tree.add_file('keep.txt', 'k')
	tmp_tree.add_file('skip.dat', 's')
	exclude = hpcp_mod.format_exclude(['*.dat'])
	files, links, size, folders = hpcp_mod.get_file_list(
		str(tmp_tree.src),
		exclude=exclude,
		parallel_file_listing=False,
		drop_cache=True,
		remove_files_while_listing=False,
	)
	joined = ' '.join(map(str, files))
	assert 'keep.txt' in joined
	assert 'skip.dat' not in joined


def test_get_file_list_parallel_matches_serial(tmp_tree, hpcp_mod, reset_hpcp_globals):
	tmp_tree.add_file('a.txt', 'A')
	tmp_tree.add_file('sub/b.txt', 'B')
	serial, _, _, _ = hpcp_mod.get_file_list(
		str(tmp_tree.src), parallel_file_listing=False, drop_cache=True, remove_files_while_listing=False,
	)
	parallel, _, _, _ = hpcp_mod.get_file_list(
		str(tmp_tree.src), parallel_file_listing=True, drop_cache=True, max_workers=2, remove_files_while_listing=False,
	)
	assert set(serial) == set(parallel)


def test_get_file_list_puts_symlinks_in_links(tmp_tree, hpcp_mod, reset_hpcp_globals):
	tmp_tree.add_file('target.txt', 't')
	tmp_tree.add_symlink('link', 'target.txt')
	files, links, _, _ = hpcp_mod.get_file_list(
		str(tmp_tree.src), parallel_file_listing=False, drop_cache=True, remove_files_while_listing=False,
	)
	joined_files = ' '.join(map(str, files))
	joined_links = ' '.join(map(str, links))
	assert 'target.txt' in joined_files
	assert 'link' in joined_links
	assert 'link' not in joined_files


def test_format_exclude_reads_exclude_file(tmp_tree, hpcp_mod, reset_hpcp_globals):
	ex = tmp_tree.root / 'exclude.txt'
	ex.write_text('*.tmp\nbuild\n')
	got = hpcp_mod.format_exclude(exclude_file=str(ex))
	assert any(p.endswith('*.tmp') or p.endswith('/.tmp') or '*.tmp' in p for p in got)
	assert any('build' in p for p in got)


def test_format_exclude_missing_file_is_skipped(hpcp_mod, reset_hpcp_globals, tmp_path):
	got = hpcp_mod.format_exclude(['cache'], exclude_file=str(tmp_path / 'nope'))
	assert '*/cache' in got


def test_get_file_list_append_hash(tmp_tree, hpcp_mod, reset_hpcp_globals):
	tmp_tree.add_file('a.txt', 'payload')
	files, _, _, _ = hpcp_mod.get_file_list(
		str(tmp_tree.src),
		append_hash=True,
		parallel_file_listing=False,
		drop_cache=True,
		remove_files_while_listing=False,
	)
	assert any(':' in str(f) for f in files)


def test_trim_paths_relative(hpcp_mod):
	base = '/home/user/project/main.py'
	got = hpcp_mod.trim_paths({'/home/user/project/file1.py', '/home/user/project/file2.py'}, base)
	assert got == {'file1.py', 'file2.py'}
