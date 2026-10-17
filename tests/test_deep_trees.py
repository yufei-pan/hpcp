"""9.76: a tree deeper than Python's recursion limit (1000 frames) is listed, copied, removed and cleaned (-rme)."""
import os
import sys

import pytest

DEPTH = 1500


@pytest.fixture(autouse=True)
def _remove_deep_trees(tmp_path, hpcp_mod):
	"""pytest removes tmp_path with shutil.rmtree, which recurses once per level before Python 3.13: the deep trees go
	first, through hpcp's own iterative _rmtree."""
	yield
	for entry in os.listdir(str(tmp_path)):
		path = os.path.join(str(tmp_path), entry)
		if os.path.isdir(path) and not os.path.islink(path):
			hpcp_mod._rmtree(path)


def _deep(root, depth=DEPTH, name='a'):
	"""root/a/a/.../a (depth levels, made one at a time: os.makedirs recurses per missing level), a file at the bottom
	and one halfway. Returns the bottom directory."""
	os.makedirs(str(root), exist_ok=True)
	path = str(root)
	for level in range(depth):
		path = os.path.join(path, name)
		os.mkdir(path)
		if level == depth // 2:
			with open(os.path.join(path, 'middle.txt'), 'w') as f:
				f.write('middle')
	with open(os.path.join(path, 'bottom.txt'), 'w') as f:
		f.write('bottom')
	return path


def _relative_bottom(depth=DEPTH, name='a'):
	return os.path.join(*([name] * depth), 'bottom.txt')


def _failures(hpcp_mod):
	return [error for error in hpcp_mod.ERRORS if hpcp_mod._is_failure(error)]


@pytest.mark.parametrize('parallel', [False, True])
def test_a_deep_tree_is_listed(tmp_path, hpcp_mod, reset_hpcp_globals, parallel):
	"""get_file_list_serial recursed once per directory level: about 1000 levels raised RecursionError (a copy too)."""
	bottom = _deep(tmp_path / 'src')
	files, links, size, folders = hpcp_mod.get_file_list(str(tmp_path / 'src'), max_workers=2, parallel_file_listing=parallel)
	assert os.path.join(bottom, 'bottom.txt') in files
	assert len(folders) == DEPTH + 1
	assert _failures(hpcp_mod) == []


@pytest.mark.parametrize('opts', [dict(single_thread=True), dict(single_thread=False, max_workers=2),
	dict(single_thread=True, no_directory_sync=True)], ids=['serial', 'parallel', 'no-directory-sync'])
def test_a_deep_tree_is_copied(tmp_path, hpcp_mod, reset_hpcp_globals, require_linux, opts):
	_deep(tmp_path / 'src')
	rc = hpcp_mod.hpcp([str(tmp_path / 'src') + os.sep], dest_paths=[str(tmp_path / 'dst') + os.sep], batch=True, **opts)
	assert rc == 0
	with open(os.path.join(str(tmp_path / 'dst'), _relative_bottom())) as f:
		assert f.read() == 'bottom'


def test_a_deep_tree_is_cleaned_by_remove_extra(tmp_path, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch):
	(tmp_path / 'src').mkdir()
	(tmp_path / 'src' / 'keep.txt').write_text('keep')
	bottom = _deep(tmp_path / 'dst')
	monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
	monkeypatch.setattr('builtins.input', lambda: 'y')
	rc = hpcp_mod.hpcp([str(tmp_path / 'src') + os.sep], dest_paths=[str(tmp_path / 'dst') + os.sep], remove_extra=True,
		single_thread=True, batch=True)
	assert rc == 0
	assert not os.path.exists(bottom)
	assert not (tmp_path / 'dst' / 'a').exists()
	assert (tmp_path / 'dst' / 'keep.txt').read_text() == 'keep'


@pytest.mark.parametrize('opts', [dict(single_thread=True), dict(single_thread=False, max_workers=2, parallel_file_listing=True)],
	ids=['serial', 'parallel-listing'])
def test_a_deep_tree_is_removed_and_moved(tmp_path, hpcp_mod, reset_hpcp_globals, require_linux, opts):
	_deep(tmp_path / 'gone')
	hpcp_mod.process_remove([str(tmp_path / 'gone')], remove_force=True, **opts)
	assert not (tmp_path / 'gone').exists()
	_deep(tmp_path / 'moved')
	rc = hpcp_mod.hpcp([str(tmp_path / 'moved') + os.sep], dest_paths=[str(tmp_path / 'to') + os.sep], remove_force=True,
		batch=True, **{key: value for key, value in opts.items() if key != 'parallel_file_listing'})
	assert rc == 0
	assert not (tmp_path / 'moved').exists()
	with open(os.path.join(str(tmp_path / 'to'), _relative_bottom())) as f:
		assert f.read() == 'bottom'
	assert _failures(hpcp_mod) == []


def test_a_deep_tree_is_mapped_for_a_diff(tmp_path, hpcp_mod, reset_hpcp_globals):
	_deep(tmp_path / 'new')
	entries = hpcp_mod.build_entry_map(str(tmp_path / 'new'), max_workers=1)
	assert '/'.join(['a'] * DEPTH) + '/bottom.txt' in entries


def test_iterative_makedirs_makes_every_missing_level(tmp_path, hpcp_mod):
	"""os.makedirs recurses once per missing level; _makedirs does not."""
	path = os.path.join(str(tmp_path), *(['a'] * DEPTH))
	hpcp_mod._makedirs(path)
	assert os.path.isdir(path)
	# The same contract as os.makedirs: an existing path is fine only with exist_ok.
	hpcp_mod._makedirs(path, exist_ok=True)
	with pytest.raises(FileExistsError):
		hpcp_mod._makedirs(path)
	with open(os.path.join(path, 'file'), 'w'):
		pass
	with pytest.raises(FileExistsError):
		hpcp_mod._makedirs(os.path.join(path, 'file'), exist_ok=True)
	hpcp_mod._makedirs(os.path.join(str(tmp_path), 'leaf') + os.sep, mode=0o700)
	assert os.stat(os.path.join(str(tmp_path), 'leaf')).st_mode & 0o777 == 0o700 & ~_umask()


def _umask():
	mask = os.umask(0)
	os.umask(mask)
	return mask
