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


def test_iterative_makedirs_makes_every_missing_level(tmp_path, hpcp_mod, monkeypatch):
	"""os.makedirs recurses once per missing level; _makedirs does not, and keeps the contract hpcp's callers use."""
	path = os.path.join(str(tmp_path), *(['a'] * DEPTH))
	hpcp_mod._makedirs(path)
	assert os.path.isdir(path)
	# An existing path is fine only with exist_ok, and only when it is a directory.
	hpcp_mod._makedirs(path, exist_ok=True)
	with pytest.raises(FileExistsError):
		hpcp_mod._makedirs(path)
	with open(os.path.join(path, 'file'), 'w'):
		pass
	with pytest.raises(FileExistsError):
		hpcp_mod._makedirs(os.path.join(path, 'file'), exist_ok=True)
	# The mode is name's alone; a destination as typed ('x/', 'x//y', 'x/.', relative) works.
	hpcp_mod._makedirs(os.path.join(str(tmp_path), 'leaf', 'deeper') + os.sep, mode=0o700)
	assert os.stat(os.path.join(str(tmp_path), 'leaf', 'deeper')).st_mode & 0o777 == 0o700 & ~_umask()
	assert os.stat(os.path.join(str(tmp_path), 'leaf')).st_mode & 0o777 == 0o777 & ~_umask()
	monkeypatch.chdir(tmp_path)
	for typed in ('x//y', 'z/.', os.path.join('w', '.', 'v')):
		hpcp_mod._makedirs(typed, exist_ok=True)
		assert os.path.isdir(typed)
	# 'dangling/./' (a dangling symlink) returns as in 9.79: the copy then fails file by file, not all at once.
	os.symlink('nowhere', 'dangling')
	hpcp_mod._makedirs(os.path.join('dangling', '.', ''), exist_ok=True)
	# A parent made meanwhile (another worker) is fine.
	real_mkdir = os.mkdir
	def racing_mkdir(directory, *args):
		if directory == 'p':
			real_mkdir(directory)
		return real_mkdir(directory, *args)
	monkeypatch.setattr(os, 'mkdir', racing_mkdir)
	hpcp_mod._makedirs(os.path.join('p', 'q'), exist_ok=True)
	assert os.path.isdir(os.path.join('p', 'q'))


def _umask():
	mask = os.umask(0)
	os.umask(mask)
	return mask


# ---- 9.76: review of main..HEAD ----

import shutil
import subprocess


_LOW_LIMIT_REMOVALS = """if 1:
	import builtins, os, resource, sys
	sys.path.insert(0, sys.argv[1])
	import hpcp
	assert resource.getrlimit(resource.RLIMIT_NOFILE)[0] > 256, 'the soft limit was not raised'
	t = sys.argv[2]
	sys.stdin.isatty, builtins.input = (lambda: True), (lambda: 'y')
	hpcp._rmtree(os.path.join(t, 'rmtree'))
	hpcp.process_remove([os.path.join(t, 'plain')], single_thread=True, remove_force=True)
	files, links, _, _ = hpcp.get_file_list(os.path.join(t, 'move-src'), parallel_file_listing=False)
	hpcp.remove_moved_sources([os.path.join(t, 'move-src')], files, {}, single_thread=True, remove_force=True,
		root_ids=hpcp._source_identities([os.path.join(t, 'move-src')]))
	hpcp.remove_extra_files([], [os.path.join(t, 'dst') + os.sep], 1, False, 1, single_thread=True, src_paths=[os.path.join(t, 'src') + os.sep])
	hpcp.remove_extra_dirs([os.path.join(t, 'src') + os.sep], [os.path.join(t, 'dst') + os.sep])
	print([error for error in hpcp.ERRORS if hpcp._is_failure(error)])
"""


def test_a_deep_tree_is_removed_under_a_low_open_files_limit(tmp_path, hpcp_mod, require_linux):
	"""Review: the descriptor walks hold one descriptor per level, so a tree deeper than the soft open-files limit (a
	common default is 1024; macOS's is 256) failed with EMFILE. hpcp raises its soft limit to the hard one at import."""
	resource = pytest.importorskip('resource')
	if resource.getrlimit(resource.RLIMIT_NOFILE)[1] < 4 * DEPTH:
		pytest.skip('the hard open-files limit is below what a %d-level removal holds' % DEPTH)
	for name in ('rmtree', 'plain', 'move-src'):
		_deep(tmp_path / name)
	(tmp_path / 'src').mkdir()
	_deep(tmp_path / 'dst')
	result = subprocess.run(['sh', '-c', 'ulimit -Sn 256 && exec "$0" -c "$1" "$2" "$3"', sys.executable, _LOW_LIMIT_REMOVALS,
		os.path.dirname(os.path.abspath(hpcp_mod.__file__)), str(tmp_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
		universal_newlines=True)
	assert result.returncode == 0, result.stderr[-2000:]
	assert result.stdout.splitlines()[-1] == '[]'
	for name in ('rmtree', 'plain', 'move-src'):
		assert not (tmp_path / name).exists(), name
	assert os.listdir(str(tmp_path / 'dst')) == []


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
	directory pass (9.75: about 1500). Every level stays held now."""
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


def test_a_held_directory_mounted_on_meanwhile_is_not_entered(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Re-review: a directory the cursor let go and opened again by name crossed into a filesystem mounted on it since.
	Every level stays held now: the removal goes on in the directory held, below the mount, never in it."""
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
			held.truncate(4)
			os.unlink('x', dir_fd=held.top())
			assert os.path.exists(os.path.join(chain[3], 'mounted'))
		finally:
			subprocess.run(['umount', '-l', chain[3]], capture_output=True)
		assert not os.path.exists(os.path.join(chain[3], 'x'))
	finally:
		held.close_all()


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
