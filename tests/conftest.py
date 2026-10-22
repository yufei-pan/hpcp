import os
import sys
import shutil
import subprocess
import tempfile
import pytest

# Repo root on path for `import hpcp` if pythonpath not honored
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
	sys.path.insert(0, _ROOT)

import hpcp

if sys.version_info < (3, 7):
	# The tests' own subprocess calls use capture_output and text (Python 3.7). Translated for calls from the tests
	# only, so one in hpcp would still fail here, as it would for a user.
	import subprocess
	_TESTS_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep
	_subprocess_run = subprocess.run

	def _run_for_python_36(*args, **kwargs):
		if sys._getframe(1).f_code.co_filename.startswith(_TESTS_DIR):
			if kwargs.pop('capture_output', False):
				kwargs.setdefault('stdout', subprocess.PIPE)
				kwargs.setdefault('stderr', subprocess.PIPE)
			if 'text' in kwargs:
				kwargs['universal_newlines'] = kwargs.pop('text')
		return _subprocess_run(*args, **kwargs)

	subprocess.run = _run_for_python_36


_REAL_SCANDIR_FD = hpcp._SCANDIR_FD


@pytest.fixture(params=['fd', 'proc', 'path'])
def scan_mode(request, monkeypatch):
	"""How hpcp reads a directory it holds open: scandir on the descriptor (Python 3.7+), /proc/self/fd/N (3.6 on
	Linux), or by its path with the descriptor held (3.6 elsewhere)."""
	mode = request.param
	if mode == 'fd' and not _REAL_SCANDIR_FD:
		pytest.skip('this Python has no scandir on a descriptor')
	if mode == 'proc' and not os.path.isdir('/proc/self/fd'):
		pytest.skip('no /proc/self/fd')
	monkeypatch.setattr(hpcp, '_SCANDIR_FD', mode == 'fd')
	monkeypatch.setattr(hpcp, '_PROC_FD', mode == 'proc', raising=False)
	return mode


@pytest.fixture
def hpcp_mod():
	return hpcp


@pytest.fixture
def restore_argv():
	old = sys.argv[:]
	try:
		yield
	finally:
		sys.argv = old


def _clear_hpcp_transient_state():
	"""Drop cross-test pollution from caches and the global ERRORS list."""
	if hasattr(hpcp, '_get_file_list_cache'):
		hpcp._get_file_list_cache.clear()
	if hasattr(hpcp, 'ERRORS'):
		hpcp.ERRORS.clear()
	if hasattr(hpcp, '_EXT_MKFS_FEATURE_KNOWN'):
		hpcp._EXT_MKFS_FEATURE_KNOWN.clear()
	hash_file = getattr(hpcp, 'hash_file', None)
	if hash_file is not None and hasattr(hash_file, 'cache_clear'):
		hash_file.cache_clear()


@pytest.fixture
def reset_hpcp_globals():
	"""Snapshot and restore module globals tests commonly mutate."""
	snap = {
		'CONTENT_ONLY': hpcp.CONTENT_ONLY,
		'NO_CREATE_DIR': hpcp.NO_CREATE_DIR,
		'HASH_SIZE': hpcp.HASH_SIZE,
		'BYTES_RATE_LIMIT': hpcp.BYTES_RATE_LIMIT,
		'FILES_RATE_LIMIT': hpcp.FILES_RATE_LIMIT,
		'COMMAND_TIMEOUT': hpcp.COMMAND_TIMEOUT,
		'REMOVE_FILES_WHILE_LISTING': hpcp.REMOVE_FILES_WHILE_LISTING,
		'RANDOM_DESTINATION_SELECTION': hpcp.RANDOM_DESTINATION_SELECTION,
		'MIRROR_FS_PARAMS': hpcp.MIRROR_FS_PARAMS,
	}
	_clear_hpcp_transient_state()
	try:
		yield hpcp
	finally:
		for k, v in snap.items():
			setattr(hpcp, k, v)
		_clear_hpcp_transient_state()


@pytest.fixture
def tmp_tree(tmp_path):
	"""Build a small src/dst tree under tmp_path. Returns paths namespace."""
	class Tree:
		def __init__(self):
			self.root = tmp_path
			self.src = tmp_path / 'src'
			self.dst = tmp_path / 'dst'
			self.src.mkdir()
			self.dst.mkdir()

		def add_file(self, rel, content=b'hello', under='src'):
			base = self.src if under == 'src' else self.dst
			path = base / rel
			path.parent.mkdir(parents=True, exist_ok=True)
			path.write_bytes(content if isinstance(content, (bytes, bytearray)) else content.encode())
			return str(path)

		def add_dir(self, rel, under='src'):
			base = self.src if under == 'src' else self.dst
			path = base / rel
			path.mkdir(parents=True, exist_ok=True)
			return str(path)

		def add_symlink(self, rel, target, under='src'):
			base = self.src if under == 'src' else self.dst
			path = base / rel
			path.parent.mkdir(parents=True, exist_ok=True)
			os.symlink(target, path)
			return str(path)

	return Tree()


def _have_losetup():
	return shutil.which('losetup') is not None


def _can_use_loop():
	if os.name == 'nt':
		return False
	if not _have_losetup():
		return False
	# Non-root often fails; detect by dry permission check later in fixture
	return True


@pytest.fixture
def loop_image(tmp_path, reset_hpcp_globals):
	"""Create a small sparse file and attach via create_loop_device; detach on teardown.

	Skips when losetup is missing or loop attach fails (typically needs root).
	"""
	if not _can_use_loop():
		pytest.skip('losetup / loop devices not available')
	img = tmp_path / 'test.img'
	# 8 MiB empty image is enough for attach smoke
	with open(img, 'wb') as f:
		f.truncate(8 * 1024 * 1024)
	loop_dev = None
	try:
		loop_dev = hpcp.create_loop_device(str(img), read_only=True)
	except Exception as e:
		pytest.skip(f'cannot create loop device (need privileges?): {e}')
	try:
		yield {'image': str(img), 'loop': loop_dev}
	finally:
		if loop_dev:
			hpcp.detach_loop_device(loop_dev)


@pytest.fixture
def require_linux():
	if os.name == 'nt':
		pytest.skip('Linux-only test')


@pytest.fixture
def require_windows():
	if os.name != 'nt':
		pytest.skip('Windows-only test')


def trailing_copy_args(tmp_tree, **kwargs):
	"""Build hpcp() kwargs that land src contents directly under dest."""
	opts = dict(
		dest_paths=[str(tmp_tree.dst) + os.sep],
		single_thread=True,
		max_workers=1,
		verbose=False,
		batch=True,
		do_not_remove_files_while_listing=True,
	)
	opts.update(kwargs)
	return [str(tmp_tree.src) + os.sep], opts


@pytest.fixture
def copy_args(tmp_tree):
	"""Return (src_paths, kwargs) for an hpcp() copy that lands under dest."""
	def _args(**kwargs):
		return trailing_copy_args(tmp_tree, **kwargs)
	return _args


def _runs_scripts(directory):
	"""True if a script written to `directory` can be executed there (it is not a noexec mount)."""
	try:
		path = os.path.join(directory, 'hpcp_exec_probe')
		with open(path, 'w') as f:
			f.write('#!/bin/sh\nexit 0\n')
		os.chmod(path, 0o755)
		try:
			return subprocess.call([path]) == 0
		finally:
			os.unlink(path)
	except OSError:
		return False


@pytest.fixture
def exec_dir(tmp_path):
	"""A directory that allows executing the scripts a test writes as fake tools: tmp_path unless it is on a
	noexec mount (a hardened /tmp), else the first of the usual alternatives that is not one; skip when none."""
	if os.name != 'posix':
		pytest.skip('runs shell scripts')
	if _runs_scripts(str(tmp_path)):
		yield tmp_path
		return
	for base in ('/dev/shm', '/var/tmp', os.path.expanduser('~'), os.getcwd()):
		if not os.path.isdir(base) or not os.access(base, os.W_OK):
			continue
		try:
			directory = tempfile.mkdtemp(prefix='hpcp-test-', dir=base)
		except OSError:
			continue  # writable by access(2) but no room or not allowed (ENOSPC, SELinux): try the next
		if _runs_scripts(directory):
			try:
				yield type(tmp_path)(directory)
			finally:
				shutil.rmtree(directory, ignore_errors=True)
			return
		shutil.rmtree(directory, ignore_errors=True)
	pytest.skip('no writable directory allows executing scripts (every candidate is mounted noexec)')
