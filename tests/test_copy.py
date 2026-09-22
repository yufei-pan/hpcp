import os
import random
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
