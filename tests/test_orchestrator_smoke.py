import os
import pytest


@pytest.mark.linux
def test_hpcp_copy_smoke_single_thread(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	tmp_tree.add_file('a.txt', 'orch')
	# Trailing seps => contents of src land directly under dst (see _src_to_dest_map)
	src = str(tmp_tree.src) + os.sep
	dst = str(tmp_tree.dst) + os.sep
	rc = hpcp_mod.hpcp(
		[src],
		dest_paths=[dst],
		single_thread=True,
		max_workers=1,
		verbose=False,
		batch=True,
		do_not_remove_files_while_listing=True,
	)
	assert rc in (None, 0)
	assert (tmp_tree.dst / 'a.txt').is_file()
	assert (tmp_tree.dst / 'a.txt').read_text() == 'orch'


def test_compare_file_list_identical_sets(tmp_tree, hpcp_mod, reset_hpcp_globals):
	a = {'file1:hash', 'file2:hash'}
	b = {'file1:hash', 'file2:hash'}
	# No exception; function prints summary to stdout
	hpcp_mod.compare_file_list(a, b, diff_file_list=None, tar_diff_file_list=False)
