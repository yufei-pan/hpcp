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


@pytest.mark.parametrize('root', ['/', '//', '/.', '/tmp/..'])
def test_plain_remove_refuses_the_filesystem_root(hpcp_mod, reset_hpcp_globals, monkeypatch, root):
	"""As rm's --preserve-root: `hpcp -rm /` is refused (9.73 hung in a loop on it). Every removal is stubbed to fail
	the test first, so a missing guard cannot remove anything here."""
	def refuse(*args, **kwargs):
		raise AssertionError('a removal ran for the filesystem root')
	for name in ('_remove_tree_fd', '_remove_one_listed', '_remove_trees_parallel', 'delete_file_list_parallel', '_remove_listed_files'):
		monkeypatch.setattr(hpcp_mod, name, refuse)
	with pytest.raises(RuntimeError, match='^Invalid arguments'):
		hpcp_mod.process_remove([root], single_thread=True, remove_force=True)


def test_plain_remove_of_a_symlink_to_the_root_removes_the_link(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	def refuse(*args, **kwargs):
		raise AssertionError('a tree removal ran for a link to /')
	for name in ('_remove_tree_fd', '_remove_trees_parallel', 'delete_file_list_parallel'):
		monkeypatch.setattr(hpcp_mod, name, refuse)
	os.symlink(os.sep, str(tmp_path / 'to-root'))
	hpcp_mod.process_remove([str(tmp_path / 'to-root')], single_thread=True, remove_force=True)
	assert not os.path.lexists(tmp_path / 'to-root')


# ---- 9.74: review of main..HEAD ----

def _answer_yes(monkeypatch):
	monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
	monkeypatch.setattr('builtins.input', lambda: 'y')


def test_remove_extra_into_a_symlinked_destination_removes_the_extras(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""`hpcp src/ /scratch/link/ -rme`, link -> a directory: the destination as given is trusted (9.73 removed there)."""
	src, real = tmp_path / 'src', tmp_path / 'real'
	src.mkdir()
	(real / 'extra-dir').mkdir(parents=True)
	(real / 'extra.txt').write_text('extra')
	os.symlink(str(real), str(tmp_path / 'link'))
	_answer_yes(monkeypatch)
	dest = str(tmp_path / 'link') + os.sep
	hpcp_mod.remove_extra_files([], [dest], 2, False, 1, single_thread=True, src_paths=[str(src) + os.sep])
	hpcp_mod.remove_extra_dirs([str(src) + os.sep], [dest])
	assert not (real / 'extra.txt').exists() and not (real / 'extra-dir').exists()
	assert [error for error in hpcp_mod.ERRORS if hpcp_mod._is_failure(error)] == []


def test_legacy_remove_extra_with_a_relative_destination_removes_the_extras(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""-fl list -rme data/ (no src_paths): the extras are built relative to the CWD; they were dropped silently."""
	(tmp_path / 'data').mkdir()
	(tmp_path / 'data' / 'extra').write_text('extra')
	monkeypatch.chdir(tmp_path)
	_answer_yes(monkeypatch)
	hpcp_mod.remove_extra_files(set(), ['data/'], 2, False, 1, single_thread=True)
	assert not (tmp_path / 'data' / 'extra').exists()


def test_parallel_remove_of_a_tree_deeper_than_path_max_records_no_error(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""-pfl jobs open their directory by path; one too deep for that is left to the descriptor walk, which reaches it
	relative to its parent: removed, and no false error (9.73 recorded none)."""
	root = tmp_path / 'deep'
	root.mkdir()
	monkeypatch.chdir(root)
	for _ in range(30):
		os.mkdir('d' * 200)
		os.chdir('d' * 200)
	with open('bottom.txt', 'w') as f:
		f.write('x')
	os.chdir(str(tmp_path))
	hpcp_mod.process_remove([str(root)], single_thread=False, max_workers=2, parallel_file_listing=True, remove_force=True)
	assert not root.exists()
	assert [error for error in hpcp_mod.ERRORS if hpcp_mod._is_failure(error)] == []


def test_serial_plain_remove_honours_the_file_rate_limit(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	src = tmp_path / 'src'
	_tree(src)
	calls = []
	monkeypatch.setattr(hpcp_mod, 'FILES_RATE_LIMIT', 1000)
	monkeypatch.setattr(hpcp_mod, '_file_list_rate_limit', lambda state, increment=1: calls.append(increment))
	hpcp_mod.process_remove([str(src)], single_thread=True, remove_force=True)
	assert not src.exists()
	assert len(calls) == 2


def test_parallel_plain_remove_honours_the_file_rate_limit(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""-pfl -F 50: 60 files in 6 directories of 10. Jobs are queued only while under the limit, up to 4 at a time (as
	9.73's paced listing): 40 go at once, the last 20 only at 40 / 50 = 0.8 s."""
	import time
	src = tmp_path / 'src'
	for index in range(6):
		(src / ('d%d' % index)).mkdir(parents=True)
		for number in range(10):
			(src / ('d%d' % index) / ('f%d' % number)).write_text('x')
	monkeypatch.setattr(hpcp_mod, 'FILES_RATE_LIMIT', 50)
	start = time.monotonic()
	hpcp_mod.process_remove([str(src)], single_thread=False, max_workers=2, parallel_file_listing=True, remove_force=True)
	assert not src.exists()
	assert time.monotonic() - start >= 0.7


def test_fallback_walk_removes_a_symlink_to_a_directory(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Without descriptor support os.walk lists a link to a directory among the directories: it is a link, removed."""
	src, other = tmp_path / 'src', tmp_path / 'other'
	(src / 'sub').mkdir(parents=True)
	other.mkdir()
	(other / 'keep.txt').write_text('keep')
	os.symlink('../../other', str(src / 'sub' / 'ln'))
	monkeypatch.setattr(hpcp_mod, '_DIR_FD_WALK', False)
	hpcp_mod.process_remove([str(src)], single_thread=True, remove_force=True)
	assert not src.exists()
	assert (other / 'keep.txt').read_text() == 'keep'


def test_serial_walk_removes_an_entry_that_became_a_directory(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Read as a file, a directory (with a file) by the time it is unlinked: removed as a directory, no error."""
	src = tmp_path / 'src'
	(src / 'x').mkdir(parents=True)
	(src / 'x' / 'inside.txt').write_text('x')
	real = hpcp_mod._dir_entries
	def dir_entries(fd):
		return [(name, False if name == 'x' else is_dir) for name, is_dir in real(fd)]
	monkeypatch.setattr(hpcp_mod, '_dir_entries', dir_entries)
	hpcp_mod.process_remove([str(src)], single_thread=True, remove_force=True)
	assert not src.exists()
	assert [error for error in hpcp_mod.ERRORS if hpcp_mod._is_failure(error)] == []


@pytest.mark.parametrize('remove_while_listing', [True, False])
def test_plain_remove_leaves_no_cached_listing_of_what_it_removed(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, remove_while_listing):
	src = tmp_path / 'src'
	_tree(src)
	monkeypatch.setattr(hpcp_mod, 'REMOVE_FILES_WHILE_LISTING', remove_while_listing)
	hpcp_mod.process_remove([str(src)], single_thread=False, max_workers=2, remove_force=True)
	assert not any(str(src) in str(key) for key in hpcp_mod._get_file_list_cache)


def test_remove_extra_leaves_no_cached_listing_of_the_destination(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	src, dst = tmp_path / 'src', tmp_path / 'dst'
	src.mkdir()
	dst.mkdir()
	(dst / 'extra.txt').write_text('extra')
	_answer_yes(monkeypatch)
	hpcp_mod.remove_extra_files([], [str(dst) + os.sep], 2, False, 1, single_thread=True, src_paths=[str(src) + os.sep])
	assert not (dst / 'extra.txt').exists()
	assert not any(str(dst) in str(key) for key in hpcp_mod._get_file_list_cache)
