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


# ---- 9.76: review of main..HEAD ----

import contextlib
import shutil
import subprocess


@contextlib.contextmanager
def _open_files_limit(soft):
	"""The soft RLIMIT_NOFILE at soft (a common default is 1024; macOS's is 256), restored after."""
	resource = pytest.importorskip('resource')
	old_soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
	resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
	try:
		yield
	finally:
		resource.setrlimit(resource.RLIMIT_NOFILE, (old_soft, hard))


def test_a_deep_tree_is_removed_under_a_low_open_files_limit(tmp_path, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch):
	"""Review: the descriptor walks held one descriptor per level: a tree deeper than the open-files limit could not be
	removed (EMFILE). They hold a bounded number now."""
	for name in ('rmtree', 'plain', 'move-src'):
		_deep(tmp_path / name)
	(tmp_path / 'src').mkdir()
	_deep(tmp_path / 'dst')
	monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
	monkeypatch.setattr('builtins.input', lambda: 'y')
	with _open_files_limit(256):
		hpcp_mod._rmtree(str(tmp_path / 'rmtree'))
		hpcp_mod.process_remove([str(tmp_path / 'plain')], single_thread=True, remove_force=True)
		files, links, _, _ = hpcp_mod.get_file_list(str(tmp_path / 'move-src'), parallel_file_listing=False)
		hpcp_mod.remove_moved_sources([str(tmp_path / 'move-src')], files, {}, single_thread=True, remove_force=True,
			root_ids=hpcp_mod._source_identities([str(tmp_path / 'move-src')]))
		hpcp_mod.remove_extra_files([], [str(tmp_path / 'dst') + os.sep], 1, False, 1, single_thread=True, src_paths=[str(tmp_path / 'src') + os.sep])
		hpcp_mod.remove_extra_dirs([str(tmp_path / 'src') + os.sep], [str(tmp_path / 'dst') + os.sep])
	for name in ('rmtree', 'plain', 'move-src'):
		assert not (tmp_path / name).exists(), name
	assert os.listdir(str(tmp_path / 'dst')) == []
	assert _failures(hpcp_mod) == []


def test_delete_file_bulk_matches_excludes_as_the_paths_were_given(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Review: delete_file_bulk(['ex/d'], exclude=['ex/d/keep']) kept keep/ before, and removed it once the walk tested
	absolute paths."""
	(tmp_path / 'ex' / 'd' / 'keep').mkdir(parents=True)
	(tmp_path / 'ex' / 'd' / 'keep' / 'k.txt').write_text('keep')
	(tmp_path / 'ex' / 'd' / 'gone.txt').write_text('gone')
	monkeypatch.chdir(tmp_path)
	hpcp_mod.delete_file_bulk(['ex/d'], exclude=['ex/d/keep'])
	assert (tmp_path / 'ex' / 'd' / 'keep' / 'k.txt').read_text() == 'keep'
	assert not (tmp_path / 'ex' / 'd' / 'gone.txt').exists()


def test_delete_file_bulk_reports_every_failure(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	(tmp_path / 'd').mkdir()
	for name in ('a', 'b', 'c'):
		(tmp_path / 'd' / name).write_text(name)
	real_unlink = os.unlink
	def unlink(path, *args, **kwargs):
		if os.fspath(path) in ('a', 'b'):
			raise PermissionError(1, 'Operation not permitted', os.fspath(path))
		return real_unlink(path, *args, **kwargs)
	monkeypatch.setattr(hpcp_mod.os, 'unlink', unlink)
	hpcp_mod.delete_file_bulk([str(tmp_path / 'd')])
	messages = [error for error in hpcp_mod.ERRORS if error.startswith('Remove process exception')]
	assert len(messages) == 2 and any(os.path.join(str(tmp_path / 'd'), 'a') in m for m in messages)


def test_rmtree_keeps_shutil_rmtree_s_contract(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Review: a missing root raises FileNotFoundError; a tree that survives raises, naming the full path."""
	with pytest.raises(FileNotFoundError):
		hpcp_mod._rmtree(str(tmp_path / 'missing'))
	(tmp_path / 'd' / 'sub').mkdir(parents=True)
	(tmp_path / 'd' / 'sub' / 'stuck').write_text('x')
	real_unlink = os.unlink
	monkeypatch.setattr(hpcp_mod.os, 'unlink', lambda path, *a, **k: (_ for _ in ()).throw(PermissionError(1, 'Operation not permitted', os.fspath(path)))
		if os.fspath(path) == 'stuck' else real_unlink(path, *a, **k))
	with pytest.raises(OSError) as raised:
		hpcp_mod._rmtree(str(tmp_path / 'd'))
	assert raised.value.filename == os.path.join(str(tmp_path / 'd'), 'sub', 'stuck')


def test_rmtree_needs_no_read_permission_on_the_parent(tmp_path, hpcp_mod):
	"""Review: shutil.rmtree needs search and write on the parent, not read; _rmtree opened the parent to read it."""
	if os.geteuid() != 0 or shutil.which('setpriv') is None:
		pytest.skip('needs root and setpriv to run as nobody')
	import pathlib
	import tempfile
	# nobody cannot pass /tmp/pytest-of-root (0700): a directory of its own under /tmp, removed after.
	outer = pathlib.Path(tempfile.mkdtemp(prefix='hpcp-dropbox-', dir='/tmp'))
	try:
		_run_dropbox_case(outer, hpcp_mod)
	finally:
		subprocess.run(['chmod', '-R', 'u+rwx', str(outer)], capture_output=True)
		shutil.rmtree(str(outer))


def _run_dropbox_case(outer, hpcp_mod):
	tmp_path = outer
	work = outer / 'work'
	(work / 'dropbox' / 'tree' / 'sub').mkdir(parents=True)
	(work / 'dropbox' / 'tree' / 'sub' / 'f').write_text('x')
	shutil.copy(hpcp_mod.__file__, str(work / 'hpcp.py'))
	for path in (work, work / 'dropbox' / 'tree', work / 'dropbox' / 'tree' / 'sub'):
		os.chmod(str(path), 0o755)
	os.chmod(str(work / 'hpcp.py'), 0o644)
	os.chmod(str(tmp_path), 0o755)
	subprocess.run(['chown', '-R', 'nobody', str(work / 'dropbox')], check=True)
	os.chmod(str(work / 'dropbox'), 0o333)
	result = subprocess.run(['setpriv', '--reuid=65534', '--regid=65534', '--clear-groups', sys.executable, '-c',
		f'import sys; sys.path.insert(0, {str(work)!r}); import hpcp; hpcp._rmtree({str(work / "dropbox" / "tree")!r})'],
		capture_output=True, text=True)
	os.chmod(str(work / 'dropbox'), 0o755)
	assert result.returncode == 0, result.stderr[-500:]
	assert not (work / 'dropbox' / 'tree').exists()


@pytest.mark.parametrize('relative', ['x/y/.', 'x/y/', 'x//y', 'a/./b/c'])
def test_makedirs_matches_os_makedirs(tmp_path, hpcp_mod, monkeypatch, relative):
	"""Review: '/.'-ended paths, '' and a dangling symlink on the way behaved unlike os.makedirs."""
	for side, make in (('ours', hpcp_mod._makedirs), ('os', os.makedirs)):
		(tmp_path / side).mkdir()
		monkeypatch.chdir(tmp_path / side)
		make(relative)
		with pytest.raises(FileExistsError):
			make(relative.rstrip('/.') or relative)
	assert sorted(str(p.relative_to(tmp_path / 'ours')) for p in (tmp_path / 'ours').rglob('*')) == \
		sorted(str(p.relative_to(tmp_path / 'os')) for p in (tmp_path / 'os').rglob('*'))
	with pytest.raises(FileNotFoundError):
		hpcp_mod._makedirs('')
	os.symlink('nowhere', str(tmp_path / 'dangling'))
	with pytest.raises(OSError) as ours:
		hpcp_mod._makedirs(str(tmp_path / 'dangling' / 'child'))
	with pytest.raises(OSError) as theirs:
		os.makedirs(str(tmp_path / 'dangling' / 'child'))
	assert type(ours.value) is type(theirs.value)


def test_listing_stats_each_subdirectory_once(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Review: each subdirectory got islink, isfile and isdir (three stats; a server round trip each on Lustre) though
	its parent's entry already said it is a directory. One lstat catches a swap since then."""
	root = tmp_path / 'tree'
	for index in range(20):
		(root / ('d%d' % index) / 'e').mkdir(parents=True)
	calls = []
	real_stat, real_lstat = os.stat, os.lstat
	monkeypatch.setattr(hpcp_mod.os, 'stat', lambda *a, **k: calls.append(a[0]) or real_stat(*a, **k))
	monkeypatch.setattr(hpcp_mod.os, 'lstat', lambda *a, **k: calls.append(a[0]) or real_lstat(*a, **k))
	files, links, size, folders = hpcp_mod.get_file_list_serial(str(root))
	assert len(folders) == 41
	# The root's own checks, then one lstat per subdirectory (40).
	assert len(calls) <= 40 + 3


# ---- 9.76: re-review of c4d89da..HEAD ----

def _count_opens(monkeypatch, hpcp_mod):
	opens = []
	real_open = os.open
	monkeypatch.setattr(hpcp_mod.os, 'open', lambda *a, **k: opens.append(a[0]) or real_open(*a, **k))
	return opens


def test_remove_extra_dirs_of_a_deep_tree_opens_each_level_about_once(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Re-review: the cursor opened every released directory again from the root: 540257 opens for a 1500-level -rme
	directory pass (9.75: about 1500). Going back up is one '..' a level now."""
	(tmp_path / 'src').mkdir()
	(tmp_path / 'dst').mkdir()
	_deep(tmp_path / 'dst' / 'x', name='d')
	for path in ('middle.txt', 'bottom.txt'):
		pass
	for dirpath, dirnames, filenames in hpcp_mod._walk(str(tmp_path / 'dst')):
		for name in filenames:
			os.unlink(os.path.join(dirpath, name))
	opens = _count_opens(monkeypatch, hpcp_mod)
	hpcp_mod.remove_extra_dirs([str(tmp_path / 'src') + os.sep], [str(tmp_path / 'dst') + os.sep])
	assert os.listdir(str(tmp_path / 'dst')) == []
	assert len(opens) <= 4 * DEPTH


def test_a_move_of_a_deep_chain_with_a_file_per_level_opens_each_level_about_once(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	root = tmp_path / 'chain'
	path = str(root)
	os.mkdir(path)
	files = []
	for level in range(800):
		path = os.path.join(path, 'l%03d' % level)
		os.mkdir(path)
		# 'z' sorts after the next level's name: the deeper paths come first, so the cursor climbs one level per file.
		files.append(os.path.join(path, 'z'))
		open(files[-1], 'w').close()
	opens = _count_opens(monkeypatch, hpcp_mod)
	hpcp_mod.remove_moved_sources([str(root)], files, {}, single_thread=True, remove_force=True, root_ids=hpcp_mod._source_identities([str(root)]))
	assert not root.exists()
	assert len(opens) <= 4 * 800


def test_a_released_directory_mounted_on_meanwhile_is_not_entered(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Re-review: a directory the cursor let go and opened again by name crossed into a filesystem mounted on it since."""
	if os.geteuid() != 0:
		pytest.skip('needs root to mount')
	root = tmp_path / 'dst'
	path = str(root)
	os.mkdir(path)
	chain = []
	for level in range(60):
		path = os.path.join(path, 'c%02d' % level)
		os.mkdir(path)
		chain.append(path)
	open(os.path.join(chain[3], 'x'), 'w').close()
	open(os.path.join(chain[-1], 'deep'), 'w').close()
	root_fd = hpcp_mod._open_source_root(str(root), hpcp_mod._path_identity(str(root)))
	held = hpcp_mod._HeldDirs(root_fd)
	try:
		for level in range(60):
			held.push('c%02d' % level, os.open('c%02d' % level, hpcp_mod._BELOW_FLAGS, dir_fd=held.top()))
		assert subprocess.run(['mount', '-t', 'tmpfs', 'hpcp-test', chain[3]], capture_output=True).returncode == 0
		try:
			open(os.path.join(chain[3], 'mounted'), 'w').close()
			try:
				held.truncate(4)
				os.unlink('x', dir_fd=held.top())
			except OSError as e:
				assert e.errno == 116
			assert os.path.exists(os.path.join(chain[3], 'mounted'))
		finally:
			subprocess.run(['umount', '-l', chain[3]], capture_output=True)
	finally:
		held.close_all()


def test_walk_goes_on_when_a_released_parent_s_child_is_removed_meanwhile(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Re-review: '..' of a directory removed by another process is ENOENT; the walk stopped and left every sibling."""
	root = tmp_path / 'tree'
	_deep(root, depth=80)
	for index in range(5):
		(root / ('sibling%d' % index)).mkdir()
		(root / ('sibling%d' % index) / 'f').write_text('x')
	real_open = os.open
	failed = []
	def flaky_open(path, *args, **kwargs):
		if path == os.pardir and not failed:
			failed.append(True)
			raise FileNotFoundError(2, 'No such file or directory', path)
		return real_open(path, *args, **kwargs)
	monkeypatch.setattr(hpcp_mod.os, 'open', flaky_open)
	hpcp_mod._rmtree(str(root))
	assert failed and not root.exists()


def test_rmtree_of_an_empty_path_removes_nothing(tmp_path, hpcp_mod, monkeypatch):
	"""Re-review: _rmtree('') removed the whole working directory; shutil.rmtree('') raises and touches nothing."""
	(tmp_path / 'work' / 'keep').mkdir(parents=True)
	monkeypatch.chdir(tmp_path / 'work')
	with pytest.raises(FileNotFoundError):
		hpcp_mod._rmtree('')
	assert (tmp_path / 'work' / 'keep').is_dir()


def test_rmtree_resolves_the_path_as_the_kernel_does(tmp_path, hpcp_mod, monkeypatch):
	"""'link/..' is the link target's parent, as shutil.rmtree sees it, not the lexical one."""
	(tmp_path / 'far' / 'target').mkdir(parents=True)
	(tmp_path / 'near').mkdir()
	(tmp_path / 'near' / 'keep').write_text('keep')
	os.symlink(str(tmp_path / 'far' / 'target'), str(tmp_path / 'near' / 'link'))
	monkeypatch.chdir(tmp_path / 'near')
	try:
		hpcp_mod._rmtree('link/..')
	except OSError:
		# shutil.rmtree raises here once 'link' dangles (its final rmdir); either way the same entries go.
		pass
	# What shutil.rmtree('link/..') does: far's contents go, near (the lexical 'link/..') is untouched.
	assert (tmp_path / 'near' / 'keep').read_text() == 'keep'
	assert not (tmp_path / 'far' / 'target').exists()


@pytest.mark.parametrize('name, exist_ok', [('x/.', False), ('.', False), ('file/.', True)])
def test_makedirs_raises_where_os_makedirs_does(tmp_path, hpcp_mod, monkeypatch, name, exist_ok):
	monkeypatch.chdir(tmp_path)
	(tmp_path / 'x').mkdir()
	(tmp_path / 'file').write_text('f')
	with pytest.raises(OSError) as theirs:
		os.makedirs(name, exist_ok=exist_ok)
	with pytest.raises(OSError) as ours:
		hpcp_mod._makedirs(name, exist_ok=exist_ok)
	assert type(ours.value) is type(theirs.value)


def test_a_failing_fstat_on_the_way_back_is_recorded_not_raised(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	root = tmp_path / 'tree'
	_deep(root, depth=60)
	real_fstat = os.fstat
	calls = []
	def fstat(fd):
		calls.append(fd)
		if len(calls) == 40:
			raise OSError(5, 'Input/output error')
		return real_fstat(fd)
	before = len(os.listdir('/proc/self/fd'))
	monkeypatch.setattr(hpcp_mod.os, 'fstat', fstat)
	removed, left, errors = hpcp_mod._remove_tree_fd(str(root), hpcp_mod._path_identity(str(root)), remove_files=True)
	monkeypatch.setattr(hpcp_mod.os, 'fstat', real_fstat)
	assert len(os.listdir('/proc/self/fd')) == before
	assert isinstance(errors, list)


def test_a_parallel_listing_folder_gets_one_lstat(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Re-review: -pfl lists each folder as a root (recurse=False): islink, isfile, isdir again. A folder from a parent's
	listing (known_dir) gets one lstat."""
	(tmp_path / 'folder' / 'sub').mkdir(parents=True)
	(tmp_path / 'folder' / 'f').write_text('x')
	calls = []
	real_stat, real_lstat = os.stat, os.lstat
	monkeypatch.setattr(hpcp_mod.os, 'stat', lambda *a, **k: calls.append(('stat', a[0])) or real_stat(*a, **k))
	monkeypatch.setattr(hpcp_mod.os, 'lstat', lambda *a, **k: calls.append(('lstat', a[0])) or real_lstat(*a, **k))
	files, links, size, folders = hpcp_mod.get_file_list_serial(str(tmp_path / 'folder'), recurse=False, known_dir=True)
	assert str(tmp_path / 'folder' / 'sub') in folders
	assert [call for call in calls if call[1] == str(tmp_path / 'folder')] == [('lstat', str(tmp_path / 'folder'))]
