import os
import random
import stat
import time
import pytest


def test_copy_file_writes_content(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	src = tmp_tree.add_file('a.txt', 'payload-copy')
	dst = str(tmp_tree.dst / 'a.txt')
	hpcp_mod.CONTENT_ONLY = False
	hpcp_mod.NO_CREATE_DIR = False
	size, _, _ = hpcp_mod.copy_file(src, [dst])
	assert os.path.isfile(dst)
	assert open(dst).read() == 'payload-copy'
	assert size > 0


def test_copy_files_serial_smoke(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	tmp_tree.add_file('a.txt', 'A')
	tmp_tree.add_file('b.txt', 'B')
	hpcp_mod.CONTENT_ONLY = False
	# copy_files_serial(src_path, dest_paths, full_hash=False, verbose=False, exclude=None)
	# places files at dest/<relpath> when src is a directory
	hpcp_mod.copy_files_serial(str(tmp_tree.src), [str(tmp_tree.dst)], full_hash=False, verbose=False, exclude=None)
	assert (tmp_tree.dst / 'a.txt').read_text() == 'A'
	assert (tmp_tree.dst / 'b.txt').read_text() == 'B'


def test_sync_directory_metadata_default_applies_mtime(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	# Mirrors intent of test_default_dir_sync_still_applies_metadata but uses shared fixtures
	src_dir = tmp_tree.add_dir('sub')
	dst_dir = str(tmp_tree.dst / 'sub')
	os.chmod(src_dir, 0o750)
	t = __import__('time').time() - 86400
	os.utime(src_dir, (t, t))
	hpcp_mod.CONTENT_ONLY = False
	hpcp_mod.sync_directory_metadata(src_dir, [dst_dir])
	assert os.path.isdir(dst_dir)
	assert abs(os.stat(dst_dir).st_mtime - os.stat(src_dir).st_mtime) < 2


def test_copy_file_skips_identical_content(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	src = tmp_tree.add_file('a.txt', 'same-bytes', under='src')
	dst = tmp_tree.add_file('a.txt', 'same-bytes', under='dst')
	t = time.time() - 1000
	os.utime(src, (t, t))
	os.utime(dst, (t, t))
	hpcp_mod.HASH_SIZE = 65536
	if hasattr(hpcp_mod.hash_file, 'cache_clear'):
		hpcp_mod.hash_file.cache_clear()
	size, _, _ = hpcp_mod.copy_file(src, [dst])
	assert size == 0


def test_no_create_dir_skips_missing_parent(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	src = tmp_tree.add_file('a.txt', 'x')
	dst = str(tmp_tree.dst / 'missing' / 'a.txt')
	hpcp_mod.NO_CREATE_DIR = True
	size, _, _ = hpcp_mod.copy_file(src, [dst])
	assert not os.path.exists(dst)
	assert size == 0


def test_hpcp_no_create_dir_skips_missing_dest(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, copy_args):
	tmp_tree.add_file('a.txt', 'x')
	missing = str(tmp_tree.root / 'nope') + os.sep
	srcs, opts = copy_args(dest_paths=[missing], no_create_dir=True)
	hpcp_mod.hpcp(srcs, **opts)
	assert not os.path.exists(missing.rstrip(os.sep))


def test_copy_files_parallel_smoke(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	tmp_tree.add_file('a.txt', 'A')
	tmp_tree.add_file('nested/b.txt', 'B')
	hpcp_mod.copy_files_parallel(
		str(tmp_tree.src),
		[str(tmp_tree.dst)],
		max_workers=2,
		parallel_file_listing=False,
	)
	assert (tmp_tree.dst / 'a.txt').read_text() == 'A'
	assert (tmp_tree.dst / 'nested' / 'b.txt').read_text() == 'B'


def test_hpcp_parallel_copy(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, copy_args):
	tmp_tree.add_file('a.txt', 'par')
	srcs, opts = copy_args(single_thread=False, max_workers=2)
	hpcp_mod.hpcp(srcs, **opts)
	assert (tmp_tree.dst / 'a.txt').read_text() == 'par'


def test_directory_only_creates_dirs_not_files(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, copy_args):
	tmp_tree.add_file('sub/a.txt', 'secret')
	srcs, opts = copy_args(directory_only=True)
	hpcp_mod.hpcp(srcs, **opts)
	assert (tmp_tree.dst / 'sub').is_dir()
	assert not (tmp_tree.dst / 'sub' / 'a.txt').exists()


def test_no_directory_sync_skips_nested_dir_mtime(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, copy_args):
	tmp_tree.add_file('sub/a.txt', 'x')
	src_sub = tmp_tree.src / 'sub'
	t = time.time() - 86400
	os.utime(src_sub, (t, t))
	srcs, opts = copy_args(no_directory_sync=True)
	hpcp_mod.hpcp(srcs, **opts)
	assert (tmp_tree.dst / 'sub' / 'a.txt').read_text() == 'x'
	assert abs(os.stat(tmp_tree.dst / 'sub').st_mtime - t) > 5


def test_copy_file_uses_first_dest_without_random(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	src = tmp_tree.add_file('a.txt', 'one-dest')
	alt = tmp_tree.root / 'dst2'
	alt.mkdir()
	d1 = str(tmp_tree.dst / 'a.txt')
	d2 = str(alt / 'a.txt')
	hpcp_mod.RANDOM_DESTINATION_SELECTION = False
	hpcp_mod.copy_file(src, [d1, d2])
	assert os.path.isfile(d1)
	assert not os.path.exists(d2)


def test_copy_file_random_dest_picks_one(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	src = tmp_tree.add_file('a.txt', 'rand')
	alt = tmp_tree.root / 'dst2'
	alt.mkdir()
	d1 = str(tmp_tree.dst / 'a.txt')
	d2 = str(alt / 'a.txt')
	hpcp_mod.RANDOM_DESTINATION_SELECTION = True
	random.seed(1)
	hpcp_mod.copy_file(src, [d1, d2])
	assert os.path.isfile(d1) ^ os.path.isfile(d2)


def test_symlink_is_recreated_on_dest(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, copy_args):
	tmp_tree.add_file('target.txt', 'data')
	tmp_tree.add_symlink('link', 'target.txt')
	srcs, opts = copy_args()
	hpcp_mod.hpcp(srcs, **opts)
	assert (tmp_tree.dst / 'target.txt').read_text() == 'data'
	assert (tmp_tree.dst / 'link').is_symlink()
	assert os.readlink(tmp_tree.dst / 'link') == 'target.txt'


def test_no_link_tracking_still_creates_symlink(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, copy_args):
	tmp_tree.add_file('target.txt', 'data')
	tmp_tree.add_symlink('link', 'target.txt')
	srcs, opts = copy_args(no_link_tracking=True)
	hpcp_mod.hpcp(srcs, **opts)
	assert (tmp_tree.dst / 'link').is_symlink()
	assert os.readlink(tmp_tree.dst / 'link') == 'target.txt'


def _fail_sparse_cp(real_run):
	"""Wrap the command runner so `cp --sparse=...` fails like BSD / macOS cp."""
	calls = []
	def fake_run(command, *args, **kwargs):
		calls.append(list(command))
		if any(str(arg).startswith('--sparse') for arg in command):
			raise RuntimeError("Task return code error: cp: illegal option -- -")
		return real_run(command, *args, **kwargs)
	return fake_run, calls


def test_copy_file_reports_size_when_sparse_cp_fails(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch):
	src = tmp_tree.add_file('a.bin', os.urandom(300000))
	dst = str(tmp_tree.dst / 'a.bin')
	fake_run, calls = _fail_sparse_cp(hpcp_mod.run_command_in_multicmd_with_path_check)
	monkeypatch.setattr(hpcp_mod, 'cp_supports_sparse', lambda: True)
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', fake_run)
	size, _, _ = hpcp_mod.copy_file(src, [dst])
	assert open(dst, 'rb').read() == open(src, 'rb').read()
	assert size == hpcp_mod.get_file_size(dst) > 0
	assert any('--sparse=always' in c for c in calls)
	assert calls[-1][-2:] == [src, dst] and '--sparse=always' not in calls[-1]


def test_copy_file_sparse_fallback_writes_only_one_dest(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch):
	src = tmp_tree.add_file('a.bin', os.urandom(4096))
	alt = tmp_tree.root / 'dst2'
	alt.mkdir()
	d1 = str(tmp_tree.dst / 'a.bin')
	d2 = str(alt / 'a.bin')
	hpcp_mod.RANDOM_DESTINATION_SELECTION = False
	fake_run, _ = _fail_sparse_cp(hpcp_mod.run_command_in_multicmd_with_path_check)
	monkeypatch.setattr(hpcp_mod, 'cp_supports_sparse', lambda: True)
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', fake_run)
	size, _, _ = hpcp_mod.copy_file(src, [d1, d2])
	assert size > 0
	assert os.path.isfile(d1)
	assert not os.path.exists(d2)


def test_copy_file_omits_sparse_flag_when_cp_lacks_it(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch):
	src = tmp_tree.add_file('a.bin', os.urandom(4096))
	dst = str(tmp_tree.dst / 'a.bin')
	fake_run, calls = _fail_sparse_cp(hpcp_mod.run_command_in_multicmd_with_path_check)
	monkeypatch.setattr(hpcp_mod, 'cp_supports_sparse', lambda: False)
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', fake_run)
	size, _, _ = hpcp_mod.copy_file(src, [dst])
	assert size > 0
	assert len(calls) == 1 and '--sparse=always' not in calls[0]


def test_copy_file_skips_identical_small_file_on_rerun(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	# get_file_size reports allocated blocks; the identity check must still use the apparent size
	src = tmp_tree.add_file('small.txt', 'tiny')
	dst = str(tmp_tree.dst / 'small.txt')
	hpcp_mod.HASH_SIZE = 65536
	assert hpcp_mod.copy_file(src, [dst])[0] > 0
	assert hpcp_mod.copy_file(src, [dst])[0] == 0


def test_no_directory_sync_preserves_root_metadata(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	os.chmod(tmp_tree.src, 0o750)
	os.chmod(tmp_tree.dst, 0o700)
	os.utime(tmp_tree.src, (1234567890, 1234567890))
	os.utime(tmp_tree.dst, (1600000000, 1600000000))
	rc = hpcp_mod.hpcp(
		[str(tmp_tree.src) + os.sep], dest_paths=[str(tmp_tree.dst) + os.sep],
		single_thread=True, no_directory_sync=True,
	)
	assert rc == 0
	assert os.stat(tmp_tree.dst).st_mode & 0o777 == 0o700
	assert os.stat(tmp_tree.dst).st_mtime == 1600000000


@pytest.mark.parametrize('no_create_dir', [True, False])
def test_no_directory_sync_creates_root_only_when_allowed(
	tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, no_create_dir,
):
	tmp_tree.add_file('a.txt', 'copy me')
	rc = hpcp_mod.hpcp(
		[str(tmp_tree.src)], dest_paths=[str(tmp_tree.dst) + os.sep],
		single_thread=True, no_directory_sync=True, no_create_dir=no_create_dir,
	)
	new_root = tmp_tree.dst / 'src'
	if no_create_dir:
		assert not new_root.exists()
		assert rc != 0
	else:
		assert rc == 0
		assert (new_root / 'a.txt').read_text() == 'copy me'


@pytest.mark.parametrize('single_thread', [True, False])
def test_successful_destination_retry_continues_copying(
	tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, single_thread,
):
	# The first destination cannot hold sub/*; the second is usable.
	(tmp_tree.dst / 'sub').write_text('not a directory')
	alt = tmp_tree.root / 'alt'
	(alt / 'sub').mkdir(parents=True)
	for i in range(6):
		tmp_tree.add_file(f'sub/{i}.txt', f'payload {i}')
	rc = hpcp_mod.hpcp(
		[str(tmp_tree.src) + os.sep],
		dest_paths=[str(tmp_tree.dst) + os.sep, str(alt) + os.sep],
		single_thread=single_thread, max_workers=2, files_per_job=1,
		no_directory_sync=True, no_create_dir=True, verbose=True,
	)
	assert rc == 0
	for i in range(6):
		assert (alt / 'sub' / f'{i}.txt').read_text() == f'payload {i}'


def test_successful_retry_preserves_prior_errors(
	tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux,
):
	src = tmp_tree.add_file('a.txt', 'payload')
	blocked = tmp_tree.dst / 'blocked'
	blocked.write_text('not a directory')
	hpcp_mod.NO_CREATE_DIR = True
	hpcp_mod.ERRORS.append('Copy failed: earlier file')
	size, _, _ = hpcp_mod.copy_file(src, [str(blocked / 'a.txt'), str(tmp_tree.dst / 'a.txt')])
	assert size > 0
	assert (tmp_tree.dst / 'a.txt').read_text() == 'payload'
	assert hpcp_mod.ERRORS == ['Copy failed: earlier file']


def _set_xattr_or_skip(path, name, value=b'1'):
	try:
		os.setxattr(path, name, value, follow_symlinks=False)
	except (OSError, AttributeError) as e:
		pytest.skip(f'user xattrs unsupported here: {e}')


def _xattr_names(path):
	return set(os.listxattr(path, follow_symlinks=False))


def _identical_files(tmp_tree):
	src = tmp_tree.add_file('x.txt', 'same', under='src')
	dst = tmp_tree.add_file('x.txt', 'same', under='dst')
	t = time.time() - 1000
	os.utime(src, (t, t))
	os.utime(dst, (t, t))
	return src, dst


def test_copy_file_removes_stale_xattr_on_identical_file(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	src, dst = _identical_files(tmp_tree)
	_set_xattr_or_skip(dst, 'user.stale')
	_set_xattr_or_skip(src, 'user.keep')
	hpcp_mod.CONTENT_ONLY = False
	hpcp_mod.HASH_SIZE = 65536
	hpcp_mod.copy_file(src, [dst])
	names = _xattr_names(dst)
	assert 'user.stale' not in names
	assert 'user.keep' in names


def test_copy_file_keeps_stale_xattr_under_content_only(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	src, dst = _identical_files(tmp_tree)
	_set_xattr_or_skip(dst, 'user.stale')
	hpcp_mod.CONTENT_ONLY = True
	hpcp_mod.HASH_SIZE = 65536
	hpcp_mod.copy_file(src, [dst])
	assert 'user.stale' in _xattr_names(dst)


def test_sync_directory_metadata_removes_stale_xattr(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	src = tmp_tree.add_dir('d', under='src')
	dst = tmp_tree.add_dir('d', under='dst')
	_set_xattr_or_skip(dst, 'user.stale')
	hpcp_mod.CONTENT_ONLY = False
	hpcp_mod.sync_directory_metadata(src, [dst])
	assert 'user.stale' not in _xattr_names(dst)


# ---- 9.73: one failed pool task must not stop the others (review C2) ----

def _tree_with_one_unwritable_dest(tmp_tree, count=12):
	"""Files under src/ok/ plus one under src/missing/; dst has ok/ only, so with NO_CREATE_DIR the missing/ copy fails."""
	for index in range(count):
		tmp_tree.add_file(f'ok/f{index}.txt', f'data {index}')
	tmp_tree.add_file('missing/lost.txt', 'lost')
	(tmp_tree.dst / 'ok').mkdir()


def test_parallel_copy_keeps_going_after_a_failed_file(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	_tree_with_one_unwritable_dest(tmp_tree)
	hpcp_mod.NO_CREATE_DIR = True
	files, links, size, _ = hpcp_mod.get_file_list(str(tmp_tree.src), parallel_file_listing=False, drop_cache=True)
	hpcp_mod.copy_file_list_parallel(sorted(files), links, str(tmp_tree.src), [str(tmp_tree.dst)], max_workers=2, files_per_job=1)
	assert sorted(os.listdir(tmp_tree.dst / 'ok')) == sorted(f'f{index}.txt' for index in range(12))
	assert any(error.startswith('Copy failed') for error in hpcp_mod.ERRORS)


def test_parallel_batch_copy_keeps_going_after_a_failed_file(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	_tree_with_one_unwritable_dest(tmp_tree)
	hpcp_mod.NO_CREATE_DIR = True
	hpcp_mod.copy_files_parallel_batch([(str(tmp_tree.src) + os.sep, [str(tmp_tree.dst) + os.sep])], max_workers=2)
	assert sorted(os.listdir(tmp_tree.dst / 'ok')) == sorted(f'f{index}.txt' for index in range(12))
	assert hpcp_mod.get_rc_from_error() == 190


def test_hpcp_parallel_copy_reports_a_failed_file_and_copies_the_rest(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, copy_args):
	_tree_with_one_unwritable_dest(tmp_tree)
	srcs, opts = copy_args(single_thread=False, max_workers=2, no_create_dir=True, no_directory_sync=True)
	assert hpcp_mod.hpcp(srcs, **opts) == 190
	assert len(os.listdir(tmp_tree.dst / 'ok')) == 12


def _tree_with_one_blocked_dir(tmp_tree, count=12):
	"""Directories src/d<i>/ plus src/blocked/inner/; dst/blocked is a regular file, so syncing blocked/inner fails."""
	for index in range(count):
		tmp_tree.add_dir(f'd{index}/sub')
	tmp_tree.add_dir('blocked/inner')
	(tmp_tree.dst / 'blocked').write_text('in the way')


def test_parallel_dir_sync_keeps_going_after_a_failed_directory(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	_tree_with_one_blocked_dir(tmp_tree)
	hpcp_mod.sync_directories_parallel_batch([(str(tmp_tree.src) + os.sep, [str(tmp_tree.dst) + os.sep])], max_workers=8, folder_per_job=1)
	assert all((tmp_tree.dst / f'd{index}' / 'sub').is_dir() for index in range(12))
	assert hpcp_mod.ERRORS


def test_parallel_single_dir_sync_keeps_going_after_a_failed_directory(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	_tree_with_one_blocked_dir(tmp_tree)
	hpcp_mod.sync_directories_parallel(str(tmp_tree.src) + os.sep, [str(tmp_tree.dst) + os.sep], max_workers=8, folder_per_job=1)
	assert all((tmp_tree.dst / f'd{index}' / 'sub').is_dir() for index in range(12))
	assert hpcp_mod.ERRORS


def test_dir_sync_job_keeps_going_after_a_failed_directory(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	"""A failed directory in a pool job must not cost the job's other directories."""
	_tree_with_one_blocked_dir(tmp_tree, count=2)
	src = str(tmp_tree.src)
	folders = [os.path.join(src, 'blocked', 'inner'), os.path.join(src, 'd0', 'sub'), os.path.join(src, 'd1', 'sub')]
	hpcp_mod.sync_directory_metadata_bulk(folders, [str(tmp_tree.dst)], src)
	assert (tmp_tree.dst / 'd0' / 'sub').is_dir() and (tmp_tree.dst / 'd1' / 'sub').is_dir()
	assert [error.partition(':')[0] for error in hpcp_mod.ERRORS] == ['Directory sync error']


def test_serial_dir_sync_records_a_failed_directory_and_goes_on(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	_tree_with_one_blocked_dir(tmp_tree, count=3)
	hpcp_mod.sync_directories_serial_batch([(str(tmp_tree.src) + os.sep, [str(tmp_tree.dst) + os.sep])])
	assert all((tmp_tree.dst / f'd{index}' / 'sub').is_dir() for index in range(3))
	assert hpcp_mod.get_rc_from_error() == 193


def test_dir_sync_ignores_chown_permission_error_without_privileges(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch):
	"""As cp -a: a non-root copy cannot keep another user's ownership, and that is not an error."""
	src = tmp_tree.add_dir('d')
	def chown(*args, **kwargs):
		raise PermissionError(1, 'Operation not permitted')
	monkeypatch.setattr(hpcp_mod.os, 'chown', chown)
	monkeypatch.setattr(hpcp_mod.os, 'geteuid', lambda: 1000)
	hpcp_mod.sync_directory_metadata(src, [str(tmp_tree.dst / 'd')])
	assert (tmp_tree.dst / 'd').is_dir()
	assert hpcp_mod.ERRORS == []
	monkeypatch.setattr(hpcp_mod.os, 'geteuid', lambda: 0)
	hpcp_mod.sync_directory_metadata(src, [str(tmp_tree.dst / 'd')])
	assert [error.partition(':')[0] for error in hpcp_mod.ERRORS] == ['Directory sync error']


def test_spawned_workers_get_the_run_settings(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch):
	"""Review C5: a worker that did not fork (spawn: Windows, macOS, Linux on Python 3.14+) must see -co and -ncd."""
	import concurrent.futures
	import functools
	import multiprocessing
	monkeypatch.setattr(hpcp_mod, 'ProcessPoolExecutor', functools.partial(
		concurrent.futures.ProcessPoolExecutor, mp_context=multiprocessing.get_context('spawn')))
	old = time.time() - 10 * 365 * 86400
	for name in ('a.txt', 'b.txt'):
		os.utime(tmp_tree.add_file(name, name), (old, old))
	tmp_tree.add_file('nodir/c.txt', 'c')
	hpcp_mod.CONTENT_ONLY = True
	hpcp_mod.NO_CREATE_DIR = True
	files, links, _, _ = hpcp_mod.get_file_list(str(tmp_tree.src), parallel_file_listing=False, drop_cache=True)
	hpcp_mod.copy_file_list_parallel(sorted(files), links, str(tmp_tree.src), [str(tmp_tree.dst)], max_workers=2)
	# -co: cp -f, so the copies get fresh mtimes; -ncd: the missing nodir/ is not created.
	assert all(os.stat(tmp_tree.dst / name).st_mtime > old + 86400 for name in ('a.txt', 'b.txt'))
	assert not (tmp_tree.dst / 'nodir').exists()


# ---- 9.73: symlinks (review C8, C9) ----

@pytest.mark.parametrize('single_thread', [True, False])
def test_directory_of_only_symlinks_is_copied(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, copy_args, single_thread):
	"""Review C9: with -s the early return for 'no files' dropped the symlinks the listing found."""
	tmp_tree.add_symlink('links/abs', '/etc/hostname')
	tmp_tree.add_symlink('links/rel', 'abs')
	srcs, opts = copy_args(single_thread=single_thread, max_workers=1 if single_thread else 2)
	assert hpcp_mod.hpcp(srcs, **opts) == 0
	assert os.readlink(tmp_tree.dst / 'links' / 'abs') == '/etc/hostname'
	assert os.readlink(tmp_tree.dst / 'links' / 'rel') == 'abs'


def _relative_link_outside_the_copy(tmp_tree):
	"""src/proj/lib -> ../shared/lib.txt; only src/proj/ is copied."""
	tmp_tree.add_file('shared/lib.txt', 'library')
	tmp_tree.add_symlink('proj/lib', '../shared/lib.txt')
	tmp_tree.add_file('proj/main.txt', 'main')
	return [str(tmp_tree.src / 'proj') + os.sep], [str(tmp_tree.dst / 'proj') + os.sep]


@pytest.mark.parametrize('single_thread', [True, False])
def test_relative_link_target_outside_the_copy_is_tracked(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, single_thread):
	"""Review C8: without -nlt a relative link's missing target is copied next to it, so the link resolves."""
	srcs, dests = _relative_link_outside_the_copy(tmp_tree)
	assert hpcp_mod.hpcp(srcs, dest_paths=dests, single_thread=single_thread, max_workers=1 if single_thread else 2,
		batch=True, do_not_remove_files_while_listing=True) == 0
	assert os.readlink(tmp_tree.dst / 'proj' / 'lib') == '../shared/lib.txt'
	assert (tmp_tree.dst / 'proj' / 'lib').read_text() == 'library'


def test_no_link_tracking_leaves_the_target_alone(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	srcs, dests = _relative_link_outside_the_copy(tmp_tree)
	assert hpcp_mod.hpcp(srcs, dest_paths=dests, single_thread=True, max_workers=1, no_link_tracking=True,
		batch=True, do_not_remove_files_while_listing=True) == 0
	assert os.readlink(tmp_tree.dst / 'proj' / 'lib') == '../shared/lib.txt'
	assert not (tmp_tree.dst / 'shared').exists()


def test_link_goes_to_the_destination_that_holds_its_target(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	"""Files are spread over the destinations: a relative link belongs where its target landed, not in the last one."""
	src_link = tmp_tree.add_symlink('ln', 'target.txt')
	first, second = tmp_tree.root / 'd1', tmp_tree.root / 'd2'
	first.mkdir()
	second.mkdir()
	(first / 'target.txt').write_text('t')
	hpcp_mod.create_sym_links({src_link: [str(first / 'ln'), str(second / 'ln')]}, no_link_tracking=True)
	assert os.readlink(first / 'ln') == 'target.txt'
	assert not os.path.lexists(second / 'ln')


def test_link_is_skipped_when_a_destination_holds_a_real_entry_there(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	src_link = tmp_tree.add_symlink('ln', '/etc/hostname')
	first, second = tmp_tree.root / 'd1', tmp_tree.root / 'd2'
	first.mkdir()
	second.mkdir()
	(second / 'ln').write_text('a real file')
	hpcp_mod.create_sym_links({src_link: [str(first / 'ln'), str(second / 'ln')]})
	assert not os.path.lexists(first / 'ln')
	assert (second / 'ln').read_text() == 'a real file'
	assert [error.partition(':')[0] for error in hpcp_mod.ERRORS] == ['Failed to create symbolic link']


def test_dir_sync_records_a_source_directory_that_went_away(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch):
	"""A listed directory removed before its sync (a live tree) is recorded; the job's other directories still sync."""
	src = str(tmp_tree.src)
	tmp_tree.add_dir('d0')
	folders = [os.path.join(src, 'vanished'), os.path.join(src, 'd0')]
	tmp_tree.add_dir('vanished', under='dst')  # synced on an earlier run
	hpcp_mod.sync_directory_metadata_bulk(folders, [str(tmp_tree.dst)], src)
	assert (tmp_tree.dst / 'd0').is_dir() and (tmp_tree.dst / 'vanished').is_dir()
	assert [error.partition(':')[0] for error in hpcp_mod.ERRORS] == ['Directory sync error']


def test_identical_file_skip_ignores_chown_permission_error_without_privileges(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch):
	"""As cp -a and the directory sync: a non-root re-run over unchanged files cannot keep another user's ownership."""
	src, dst = _identical_files(tmp_tree)
	hpcp_mod.HASH_SIZE = 65536
	def chown(*args, **kwargs):
		raise PermissionError(1, 'Operation not permitted')
	monkeypatch.setattr(hpcp_mod.os, 'chown', chown)
	monkeypatch.setattr(hpcp_mod.os, 'geteuid', lambda: 1000)
	assert hpcp_mod.copy_file(src, [dst])[0] == 0
	assert hpcp_mod.ERRORS == []


def test_link_tracking_never_copies_an_excluded_target(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	"""-e secrets keeps src/secrets/ out of the copy; tracking a link into it must not bring it back."""
	tmp_tree.add_file('secrets/key.pem', 'private')
	tmp_tree.add_symlink('k', 'secrets/key.pem')
	assert hpcp_mod.hpcp([str(tmp_tree.src) + os.sep], dest_paths=[str(tmp_tree.dst) + os.sep], exclude=['secrets'],
		single_thread=True, batch=True, do_not_remove_files_while_listing=True) == 0
	assert os.readlink(tmp_tree.dst / 'k') == 'secrets/key.pem'
	assert not (tmp_tree.dst / 'secrets').exists()


@pytest.mark.parametrize('group_allowed', [True, False])
def test_non_root_ownership_falls_back_to_the_group_like_cp(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch, group_allowed):
	"""cp -a: when the owner cannot be kept, still keep the group (a group the user belongs to); neither is an error."""
	src = tmp_tree.add_dir('d')
	calls = []
	def chown(path, uid, gid, **kwargs):
		calls.append((uid, gid))
		if uid != -1 or not group_allowed:
			raise PermissionError(1, 'Operation not permitted')
	monkeypatch.setattr(hpcp_mod.os, 'chown', chown)
	monkeypatch.setattr(hpcp_mod.os, 'geteuid', lambda: 1000)
	hpcp_mod.sync_directory_metadata(src, [str(tmp_tree.dst / 'd')])
	st = os.stat(src)
	assert calls == [(st.st_uid, st.st_gid), (-1, st.st_gid)]
	assert hpcp_mod.ERRORS == []


@pytest.mark.parametrize('group_allowed, expected', [(True, 0o2755), (False, 0o755)])
def test_non_root_skip_drops_setid_bits_it_cannot_keep_the_ids_for(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch, group_allowed, expected):
	"""As cp -a: a set-user-ID file whose owner could not be kept must not become a setuid file of the copier."""
	src, dst = _identical_files(tmp_tree)
	os.chmod(src, 0o6755)
	hpcp_mod.HASH_SIZE = 65536
	real_chown = os.chown
	def chown(path, uid, gid, **kwargs):
		if uid != -1 or not group_allowed:
			raise PermissionError(1, 'Operation not permitted')
		real_chown(path, uid, gid, **kwargs)
	monkeypatch.setattr(hpcp_mod.os, 'chown', chown)
	monkeypatch.setattr(hpcp_mod.os, 'geteuid', lambda: 1000)
	hpcp_mod.copy_file(src, [dst])
	assert stat.S_IMODE(os.stat(dst).st_mode) == expected
	assert hpcp_mod.ERRORS == []


def test_link_tracking_honours_absolute_excludes_inside_the_tracked_tree(tmp_path, hpcp_mod, reset_hpcp_globals, require_linux):
	"""src/lnk -> ../other: the tracked tree is listed by its real path, so -e /…/other/private leaves that part out."""
	data = tmp_path / 'data'
	(data / 'src').mkdir(parents=True)
	(data / 'other' / 'pub').mkdir(parents=True)
	(data / 'other' / 'private').mkdir()
	(data / 'other' / 'pub' / 'x').write_text('x')
	(data / 'other' / 'private' / 'y').write_text('secret')
	os.symlink('../other', str(data / 'src' / 'lnk'))
	backup = tmp_path / 'backup'
	backup.mkdir()
	assert hpcp_mod.hpcp([str(data / 'src')], dest_paths=[str(backup) + os.sep], exclude=[str(data / 'other' / 'private')],
		single_thread=True, batch=True, do_not_remove_files_while_listing=True) == 0
	assert (backup / 'other' / 'pub' / 'x').read_text() == 'x'
	assert not (backup / 'other' / 'private').exists()


def test_link_through_a_symlinked_directory_and_dotdot_is_not_tracked(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	"""src/l -> sub/../t with sub a symlink: the kernel reaches elsewhere/t, the spelling says src/t. Not tracked."""
	elsewhere = tmp_tree.root / 'elsewhere'
	(elsewhere / 'deep').mkdir(parents=True)
	(elsewhere / 't').write_text('kernel target')
	tmp_tree.add_file('t', 'lexical twin')
	tmp_tree.add_symlink('sub', str(elsewhere / 'deep'))
	link = tmp_tree.add_symlink('l', 'sub/../t')
	tracked = hpcp_mod.create_sym_links({link: [str(tmp_tree.dst / 'l')]})
	assert tracked == set()
	assert os.readlink(tmp_tree.dst / 'l') == 'sub/../t'
	assert not (tmp_tree.dst / 't').exists()
