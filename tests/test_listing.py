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


def test_trim_paths_relative(hpcp_mod):
	base = '/home/user/project/main.py'
	got = hpcp_mod.trim_paths({'/home/user/project/file1.py', '/home/user/project/file2.py'}, base)
	assert got == {'file1.py', 'file2.py'}
