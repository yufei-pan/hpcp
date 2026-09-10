import os
import pytest


def test_expand_paths_glob(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	tmp_tree.add_file('a.txt', 'a')
	tmp_tree.add_file('b.txt', 'b')
	pattern = str(tmp_tree.src / '*.txt')
	got = hpcp_mod.expand_paths([pattern])
	assert any(p.endswith('a.txt') for p in got)
	assert any(p.endswith('b.txt') for p in got)


def test_get_free_space_bytes_positive(tmp_path, hpcp_mod, reset_hpcp_globals):
	free = hpcp_mod.get_free_space_bytes(str(tmp_path))
	assert isinstance(free, int)
	assert free > 0


def test_get_last_existing_parent_and_is_device(tmp_tree, hpcp_mod, reset_hpcp_globals):
	path = tmp_tree.add_file('a.txt', 'x')
	assert hpcp_mod.is_device(path) is False
	missing = str(tmp_tree.root / 'no' / 'such' / 'dir')
	parent = hpcp_mod.get_last_existing_parent(missing)
	assert os.path.isdir(parent)


def test_src_to_dest_map_trailing_sep(tmp_tree, hpcp_mod, reset_hpcp_globals):
	# Public-ish helper used by remove_extra; documents dest layout contract
	src = str(tmp_tree.src) + os.sep
	dst = str(tmp_tree.dst) + os.sep
	mapping = hpcp_mod._src_to_dest_map([src], [dst])
	assert len(mapping) == 1
	src_abs, dests = mapping[0]
	assert os.path.abspath(str(tmp_tree.dst)) in [os.path.abspath(d) for d in dests]
