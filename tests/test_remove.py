import os
import sys
import pytest


def test_delete_file_bulk_removes_files(tmp_tree, hpcp_mod, reset_hpcp_globals):
	p1 = tmp_tree.add_file('a.txt', 'a', under='dst')
	p2 = tmp_tree.add_file('b.txt', 'b', under='dst')
	hpcp_mod.delete_file_bulk([p1, p2])
	assert not os.path.exists(p1)
	assert not os.path.exists(p2)


def test_remove_extra_files_deletes_only_extras(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch):
	# Mirror process_copy mapping: src path WITHOUT trailing sep => dest/src_basename/...
	# Use trailing sep on both so files land directly under dst (see _src_to_dest_map).
	src_keep = tmp_tree.add_file('keep.txt', 'k', under='src')
	dst_keep = tmp_tree.add_file('keep.txt', 'k', under='dst')
	dst_extra = tmp_tree.add_file('extra.txt', 'e', under='dst')
	src_paths = [str(tmp_tree.src) + os.sep]
	dests = [str(tmp_tree.dst) + os.sep]
	# remove_extra_files requires a TTY + interactive 'y' before deleting (safety gate).
	monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
	monkeypatch.setattr('builtins.input', lambda: 'y')
	hpcp_mod.remove_extra_files(
		[src_keep],
		dests,
		max_workers=2,
		verbose=False,
		files_per_job=1,
		single_thread=True,
		exclude=None,
		parallel_file_listing=False,
		src_paths=src_paths,
		sym_links=None,
	)
	assert os.path.exists(dst_keep)
	assert not os.path.exists(dst_extra)


def test_process_remove_force_deletes_tree(tmp_tree, hpcp_mod, reset_hpcp_globals):
	gone = tmp_tree.add_file('a.txt', 'x')
	hpcp_mod.process_remove([str(tmp_tree.src)], single_thread=True, remove_force=True)
	assert not os.path.exists(gone)
	assert not tmp_tree.src.exists()


def test_process_remove_without_force_requires_tty(tmp_tree, hpcp_mod, reset_hpcp_globals, monkeypatch):
	tmp_tree.add_file('a.txt', 'x')
	monkeypatch.setattr(sys.stdin, 'isatty', lambda: False)
	with pytest.raises(RuntimeError, match='not a tty'):
		hpcp_mod.process_remove([str(tmp_tree.src)], single_thread=True, remove_force=False)


def test_remove_files_while_listing_deletes_entries(tmp_tree, hpcp_mod, reset_hpcp_globals):
	p = tmp_tree.add_file('gone.txt', 'x')
	hpcp_mod.get_file_list(
		str(tmp_tree.src), parallel_file_listing=False, drop_cache=True, remove_files_while_listing=True,
	)
	assert not os.path.exists(p)


def test_rwloff_listing_leaves_files(tmp_tree, hpcp_mod, reset_hpcp_globals):
	p = tmp_tree.add_file('stay.txt', 'x')
	hpcp_mod.get_file_list(
		str(tmp_tree.src), parallel_file_listing=False, drop_cache=True, remove_files_while_listing=False,
	)
	assert os.path.exists(p)


def test_remove_extra_dirs_drops_empty_extra(tmp_tree, hpcp_mod, reset_hpcp_globals):
	extra = tmp_tree.dst / 'extra_dir'
	extra.mkdir()
	hpcp_mod.remove_extra_dirs([str(tmp_tree.src) + os.sep], [str(tmp_tree.dst) + os.sep])
	assert not extra.exists()


@pytest.mark.xfail(reason='BUGS.md#1 README: -rf implies --remove, but hpcp() only removes when remove=True', strict=False)
def test_remove_force_implies_remove(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, copy_args, monkeypatch):
	called = []
	monkeypatch.setattr(hpcp_mod, 'process_remove', lambda *a, **k: called.append(True))
	tmp_tree.add_file('a.txt', 'x')
	srcs, opts = copy_args(remove=False, remove_force=True)
	hpcp_mod.hpcp(srcs, **opts)
	assert called


def test_delete_files_parallel_counts_files_removed_while_listing(
	tmp_tree, hpcp_mod, reset_hpcp_globals, capsys,
):
	"""Files unlinked during the scan must still be counted in the summary."""
	hpcp_mod.REMOVE_FILES_WHILE_LISTING = True
	for name in ('a.txt', 'b.txt', 'sub/c.txt'):
		tmp_tree.add_file(name, 'x')
	tmp_tree.add_symlink('l.lnk', 'a.txt')
	count, _ = hpcp_mod.delete_files_parallel(
		str(tmp_tree.src), max_workers=2, parallel_file_listing=False,
	)
	assert 'Number of files: 4' in capsys.readouterr().out
	# 4 entries unlinked during the scan, +1 for the directory structure itself.
	assert count == 5
	assert not tmp_tree.src.exists()


def test_delete_files_parallel_init_size_spans_all_paths(
	tmp_tree, hpcp_mod, reset_hpcp_globals, monkeypatch,
):
	"""The progress-bar baseline must be the total over every path, not the last one."""
	hpcp_mod.REMOVE_FILES_WHILE_LISTING = False
	tmp_tree.add_file('one/a.txt', b'a' * 100)
	tmp_tree.add_file('two/b.txt', b'b' * 5)
	seen = {}

	def fake_delete(file_list, max_workers, *a, **k):
		seen['init_size'] = k.get('init_size')
		return len(file_list), 0

	monkeypatch.setattr(hpcp_mod, 'delete_file_list_parallel', fake_delete)
	hpcp_mod.delete_files_parallel(
		[str(tmp_tree.src / 'one'), str(tmp_tree.src / 'two')],
		max_workers=2, batch=True, parallel_file_listing=False,
	)
	assert seen['init_size'] == 105


def test_cached_listing_still_removes_files(tmp_tree, hpcp_mod, reset_hpcp_globals):
	"""A warm cache entry must not let remove-while-listing skip the actual unlink."""
	p = tmp_tree.add_file('gone.txt', 'x')
	hpcp_mod.get_file_list(str(tmp_tree.src), parallel_file_listing=False)
	assert os.path.exists(p)
	hpcp_mod.get_file_list(
		str(tmp_tree.src), parallel_file_listing=False, remove_files_while_listing=True,
	)
	assert not os.path.exists(p)
