"""9.74: plain -rm and -rme remove by directory descriptors, never through a symlink swapped into the tree."""
import os
import shutil
import sys

import pytest


def _swap_for_a_symlink(directory, target):
	"""Replace directory (and what it holds) with a symlink to target, as another user writing the tree could."""
	shutil.rmtree(str(directory))
	os.symlink(str(target), str(directory))


def _victim(tmp_path):
	victim = tmp_path / 'victim'
	(victim / 'sub').mkdir(parents=True)
	(victim / 'f.txt').write_text('not yours')
	(victim / 'sub' / 'g.txt').write_text('not yours')
	(victim / 'empty').mkdir()
	return victim


def _victim_intact(victim):
	return ((victim / 'f.txt').read_text() == 'not yours' and (victim / 'sub' / 'g.txt').read_text() == 'not yours'
		and (victim / 'empty').is_dir())


def _tree(root):
	(root / 'sub').mkdir(parents=True)
	(root / 'f.txt').write_text('remove')
	(root / 'sub' / 'g.txt').write_text('remove')
	(root / 'empty').mkdir()


MODES = [
	pytest.param(dict(single_thread=True), True, id='serial'),
	pytest.param(dict(single_thread=False, max_workers=2, parallel_file_listing=True), True, id='parallel-listing'),
	pytest.param(dict(single_thread=False, max_workers=2), False, id='rwloff'),
]


@pytest.mark.parametrize('opts, remove_while_listing', MODES)
def test_plain_remove_never_follows_a_root_replaced_while_asking(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, opts, remove_while_listing):
	"""`hpcp -rm /scratch/u/tree` as root: while the y/n prompt waits, the user renames tree away and leaves a symlink to
	/etc. 9.73 removed that symlink and said nothing. The root is pinned before the prompt: a replaced root is reported
	and nothing is removed through it."""
	src = tmp_path / 'src'
	_tree(src)
	victim = _victim(tmp_path)
	monkeypatch.setattr(hpcp_mod, 'REMOVE_FILES_WHILE_LISTING', remove_while_listing)
	monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
	def answer():
		os.rename(str(src), str(tmp_path / 'moved-away'))
		os.symlink(str(victim), str(src))
		return 'y'
	monkeypatch.setattr('builtins.input', answer)
	hpcp_mod.process_remove([str(src)], remove_force=False, **opts)
	assert _victim_intact(victim)
	assert (tmp_path / 'moved-away' / 'f.txt').read_text() == 'remove'
	assert any(error.startswith('Remove process exception') for error in hpcp_mod.ERRORS)


def test_serial_remove_never_enters_a_directory_swapped_after_its_parent_was_read(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""The walk lists src, then src/sub is swapped for a symlink before the walk enters it: O_NOFOLLOW stops it."""
	src = tmp_path / 'src'
	_tree(src)
	victim = _victim(tmp_path)
	real = hpcp_mod._dir_entries
	calls = []
	def dir_entries(fd):
		entries = real(fd)
		calls.append(fd)
		if len(calls) == 1:
			_swap_for_a_symlink(src / 'sub', victim)
		return entries
	monkeypatch.setattr(hpcp_mod, '_dir_entries', dir_entries)
	hpcp_mod.process_remove([str(src)], single_thread=True, remove_force=True)
	assert _victim_intact(victim)


def test_parallel_remove_refuses_a_directory_whose_path_now_leads_elsewhere(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""-pfl: a worker opens its directory by path; one reached through a swapped symlink has another identity."""
	src = tmp_path / 'src'
	_tree(src)
	victim = _victim(tmp_path)
	sub_id = hpcp_mod._path_identity(str(src / 'sub'))
	_swap_for_a_symlink(src / 'sub', victim / 'sub')
	removed, subdirectories = hpcp_mod._remove_directory_entries(str(src / 'sub'), sub_id)
	assert (removed, subdirectories) == (0, [])
	assert _victim_intact(victim)
	assert any(error.startswith('Remove process exception') for error in hpcp_mod.ERRORS)


def test_rwloff_remove_never_follows_a_directory_swapped_after_the_listing(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	src = tmp_path / 'src'
	_tree(src)
	victim = _victim(tmp_path)
	real = hpcp_mod.get_file_list
	def get_file_list(*args, **kwargs):
		result = real(*args, **kwargs)
		_swap_for_a_symlink(src / 'sub', victim / 'sub')
		return result
	monkeypatch.setattr(hpcp_mod, 'get_file_list', get_file_list)
	monkeypatch.setattr(hpcp_mod, 'REMOVE_FILES_WHILE_LISTING', False)
	hpcp_mod.process_remove([str(src)], single_thread=False, max_workers=2, remove_force=True)
	assert _victim_intact(victim)


@pytest.mark.parametrize('single_thread', [True, False])
def test_remove_extra_files_never_follows_a_directory_swapped_while_asking(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, single_thread):
	"""-rme lists the destination, asks y/n, then deletes: a destination directory swapped for a symlink meanwhile
	led the deletion outside the destination (by path, os.remove / shutil.rmtree)."""
	src, dst = tmp_path / 'src', tmp_path / 'dst'
	src.mkdir()
	(dst / 'sub').mkdir(parents=True)
	(dst / 'sub' / 'g.txt').write_text('extra')
	victim = _victim(tmp_path)
	monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
	def answer():
		_swap_for_a_symlink(dst / 'sub', victim / 'sub')
		return 'y'
	monkeypatch.setattr('builtins.input', answer)
	hpcp_mod.remove_extra_files([], [str(dst) + os.sep], 2, False, 1, single_thread=single_thread, src_paths=[str(src) + os.sep])
	assert _victim_intact(victim)


def test_remove_extra_dirs_never_follows_a_directory_swapped_after_the_walk(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	src, dst = tmp_path / 'src', tmp_path / 'dst'
	src.mkdir()
	(dst / 'extra' / 'empty').mkdir(parents=True)
	victim = _victim(tmp_path)
	real_walk = os.walk
	def walk(*args, **kwargs):
		yield from real_walk(*args, **kwargs)
		_swap_for_a_symlink(dst / 'extra', victim)
	monkeypatch.setattr(hpcp_mod.os, 'walk', walk)
	hpcp_mod.remove_extra_dirs([str(src) + os.sep], [str(dst) + os.sep])
	assert _victim_intact(victim)


def test_plain_remove_opens_each_directory_once(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""No slower than 9.73: one open per directory and no stat per file (a server round trip each on Lustre)."""
	src = tmp_path / 'src'
	directories = 0
	for top in range(3):
		path = src / ('t%d' % top)
		for level in range(4):
			path = path / ('l%d' % level)
			path.mkdir(parents=True)
			directories += 1
			for index in range(5):
				(path / ('f%d' % index)).write_text('x')
	directories += 3 + 1
	opens, stats = [], []
	real_open, real_stat = os.open, os.stat
	monkeypatch.setattr(hpcp_mod.os, 'open', lambda *a, **k: opens.append(a[0]) or real_open(*a, **k))
	monkeypatch.setattr(hpcp_mod.os, 'stat', lambda *a, **k: stats.append(a[0]) or real_stat(*a, **k))
	hpcp_mod.process_remove([str(src)], single_thread=True, remove_force=True)
	assert not src.exists()
	assert len(opens) <= directories + 2
	# A few checks of the root itself (exists, isdir, its identity); none per file (60 here).
	assert len(stats) <= 6


@pytest.mark.parametrize('opts, remove_while_listing', MODES)
def test_plain_remove_counts_what_it_removed(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, capsys, opts, remove_while_listing):
	src = tmp_path / 'src'
	_tree(src)
	os.symlink('f.txt', str(src / 'ln'))
	monkeypatch.setattr(hpcp_mod, 'REMOVE_FILES_WHILE_LISTING', remove_while_listing)
	hpcp_mod.process_remove([str(src)], remove_force=True, **opts)
	assert not src.exists()
	assert 'Removed 3 files and links' in capsys.readouterr().out
	# The final walk would remove what a crashed pool job left: no failure may be recorded.
	assert [error for error in hpcp_mod.ERRORS if hpcp_mod._is_failure(error)] == []


def test_plain_remove_of_a_symlink_root_removes_the_link_only(tmp_path, hpcp_mod, reset_hpcp_globals):
	real = tmp_path / 'real'
	_tree(real)
	os.symlink(str(real), str(tmp_path / 'ln'))
	hpcp_mod.process_remove([str(tmp_path / 'ln') + os.sep], single_thread=True, remove_force=True)
	assert not os.path.lexists(tmp_path / 'ln')
	assert (real / 'f.txt').read_text() == 'remove'


def test_plain_remove_after_a_cached_listing_still_removes_everything(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	src = tmp_path / 'src'
	_tree(src)
	hpcp_mod.get_file_list(str(src), parallel_file_listing=False)
	(src / 'late.txt').write_text('after the cached listing')
	monkeypatch.setattr(hpcp_mod, 'REMOVE_FILES_WHILE_LISTING', False)
	hpcp_mod.process_remove([str(src)], single_thread=True, remove_force=True)
	assert not src.exists()
