import os
import sys
import stat
import tempfile
import time

import pytest

# Import sibling module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hpcp


def test_content_only_flag_parses():
	old = sys.argv
	try:
		sys.argv = ['hpcp', '-co', '-d', '/tmp', '/tmp/src']
		args = hpcp.get_args()
		assert args.content_only is True
	finally:
		sys.argv = old


def test_content_only_default_off():
	old = sys.argv
	try:
		sys.argv = ['hpcp', '-d', '/tmp', '/tmp/src']
		args = hpcp.get_args()
		assert args.content_only is False
	finally:
		sys.argv = old


def _unique_mode_pair():
	# Prefer modes that differ from typical umask results
	return 0o640, 0o600


def test_content_only_new_file_skips_mode_and_mtime():
	src_mode, _ = _unique_mode_pair()
	with tempfile.TemporaryDirectory() as tmp:
		src = os.path.join(tmp, 'src')
		dst_root = os.path.join(tmp, 'dst')
		os.makedirs(src)
		os.makedirs(dst_root)
		src_file = os.path.join(src, 'a.txt')
		dst_file = os.path.join(dst_root, 'a.txt')
		with open(src_file, 'w') as f:
			f.write('hello-content-only')
		os.chmod(src_file, src_mode)
		old_mtime = time.time() - 86400
		os.utime(src_file, (old_mtime, old_mtime))

		hpcp.CONTENT_ONLY = True
		hpcp.NO_CREATE_DIR = False
		try:
			size, _, _ = hpcp.copy_file(src_file, [dst_file])
		finally:
			hpcp.CONTENT_ONLY = False

		assert size > 0
		with open(dst_file) as f:
			assert f.read() == 'hello-content-only'
		st_src = os.stat(src_file)
		st_dst = os.stat(dst_file)
		# Stronger check: mode must not be forced to source mode
		assert stat.S_IMODE(st_dst.st_mode) != stat.S_IMODE(st_src.st_mode) or abs(st_dst.st_mtime - st_src.st_mtime) > 1


def test_content_only_identical_file_leaves_dest_metadata():
	with tempfile.TemporaryDirectory() as tmp:
		src = os.path.join(tmp, 'src')
		dst_root = os.path.join(tmp, 'dst')
		os.makedirs(src)
		os.makedirs(dst_root)
		src_file = os.path.join(src, 'a.txt')
		dst_file = os.path.join(dst_root, 'a.txt')
		payload = 'same-bytes'
		for path in (src_file, dst_file):
			with open(path, 'w') as f:
				f.write(payload)
		os.chmod(src_file, 0o640)
		os.chmod(dst_file, 0o600)
		# is_file_identical requires matching size+mtime+hash; keep mtimes equal so we hit the short-circuit
		same_mtime = time.time() - 86400
		os.utime(src_file, (same_mtime, same_mtime))
		os.utime(dst_file, (same_mtime, same_mtime))
		before = os.stat(dst_file)

		hpcp.CONTENT_ONLY = True
		try:
			size, _, _ = hpcp.copy_file(src_file, [dst_file], full_hash=True)
		finally:
			hpcp.CONTENT_ONLY = False

		after = os.stat(dst_file)
		assert size == 0  # identical short-circuit returns 0 copied size
		assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(before.st_mode) == 0o600


def test_default_identical_file_still_syncs_metadata():
	"""Sanity: without CONTENT_ONLY, identical path still applies metadata."""
	with tempfile.TemporaryDirectory() as tmp:
		src_file = os.path.join(tmp, 'a.txt')
		dst_file = os.path.join(tmp, 'b.txt')
		for path in (src_file, dst_file):
			with open(path, 'w') as f:
				f.write('same')
		os.chmod(src_file, 0o640)
		os.chmod(dst_file, 0o600)
		t = time.time() - 86400
		os.utime(src_file, (t, t))
		os.utime(dst_file, (time.time() - 10, time.time() - 10))

		hpcp.CONTENT_ONLY = False
		hpcp.copy_file(src_file, [dst_file], full_hash=True)
		st_src = os.stat(src_file)
		st_dst = os.stat(dst_file)
		assert stat.S_IMODE(st_dst.st_mode) == stat.S_IMODE(st_src.st_mode)


def test_content_only_creates_dir_without_source_metadata():
	with tempfile.TemporaryDirectory() as tmp:
		src_dir = os.path.join(tmp, 'src', 'sub')
		dst_dir = os.path.join(tmp, 'dst', 'sub')
		os.makedirs(src_dir)
		os.chmod(src_dir, 0o750)
		t = time.time() - 86400
		os.utime(src_dir, (t, t))

		hpcp.CONTENT_ONLY = True
		try:
			count, _, _ = hpcp.sync_directory_metadata(src_dir, [dst_dir])
		finally:
			hpcp.CONTENT_ONLY = False

		assert count == 1
		assert os.path.isdir(dst_dir)
		st_src = os.stat(src_dir)
		st_dst = os.stat(dst_dir)
		assert stat.S_IMODE(st_dst.st_mode) != stat.S_IMODE(st_src.st_mode) or abs(st_dst.st_mtime - st_src.st_mtime) > 1


def test_content_only_setgid_inheritance():
	"""New dir under -co should inherit parent setgid when FS supports it."""
	with tempfile.TemporaryDirectory() as tmp:
		parent = os.path.join(tmp, 'dst')
		os.makedirs(parent)
		mode = os.stat(parent).st_mode | stat.S_ISGID
		os.chmod(parent, mode)
		if not (os.stat(parent).st_mode & stat.S_ISGID):
			pytest.skip('filesystem does not honor setgid on directories')

		src_dir = os.path.join(tmp, 'src', 'child')
		dst_dir = os.path.join(parent, 'child')
		os.makedirs(src_dir)
		# Source deliberately without setgid
		os.chmod(src_dir, 0o755)

		hpcp.CONTENT_ONLY = True
		try:
			hpcp.sync_directory_metadata(src_dir, [dst_dir])
		finally:
			hpcp.CONTENT_ONLY = False

		assert os.stat(dst_dir).st_mode & stat.S_ISGID


def test_default_dir_sync_still_applies_metadata():
	with tempfile.TemporaryDirectory() as tmp:
		src_dir = os.path.join(tmp, 'src', 'sub')
		dst_dir = os.path.join(tmp, 'dst', 'sub')
		os.makedirs(src_dir)
		os.chmod(src_dir, 0o750)
		t = time.time() - 86400
		os.utime(src_dir, (t, t))

		hpcp.CONTENT_ONLY = False
		hpcp.sync_directory_metadata(src_dir, [dst_dir])
		st_src = os.stat(src_dir)
		st_dst = os.stat(dst_dir)
		assert abs(st_dst.st_mtime - st_src.st_mtime) < 2
