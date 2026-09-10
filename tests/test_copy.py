import os
import stat
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
