'''9.87 group 4: a rerun's looks (R1), the identical file's metadata (R2), the copy check by directory (R3). In process, as the
plan's probes count (probe/p1_rerun_calls.py, p3_first_and_dirs.py).'''
import collections
import contextlib
import errno
import io
import os
import shutil
import stat
import subprocess
import sys
import threading
import time

import pytest

import hpcp

posix_only = pytest.mark.skipif(os.name != 'posix', reason='POSIX metadata')
linux_only = pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux: O_PATH and /proc/self/fd')
root_only = pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() != 0, reason='needs root')

_COUNTED = ('lstat', 'stat', 'open', 'chmod', 'chown', 'utime', 'listxattr', 'getxattr', 'setxattr', 'removexattr', 'statvfs',
	'readlink', 'access', 'scandir', 'mkdir', 'unlink', 'rename', 'fstat')
_SETS = ('chmod', 'chown', 'utime', 'setxattr', 'removexattr')


@contextlib.contextmanager
def _counting(monkeypatch, names=_COUNTED):
	'''A Counter of the os calls this thread makes inside the block.'''
	counts, me = collections.Counter(), threading.get_ident()
	with monkeypatch.context() as m:
		for name in names:
			real = getattr(os, name, None)
			if real is None:
				continue
			def counted(*args, _real=real, _name=name, **kwargs):
				if threading.get_ident() == me:
					counts[_name] += 1
				return _real(*args, **kwargs)
			m.setattr(os, name, counted)
		yield counts


def _now_ns():
	return int(time.time() * 10 ** 9)  # 3.6 has no time.time_ns


def _tree(root):
	'''3 directories of 5 files each. Their read times are an hour ahead, after their change times, so that a read moves no
	read time (relatime), on the copies either (cp -a keeps them): what a rerun then finds to set is only what differs.'''
	files = []
	now = _now_ns() + 3600 * 10 ** 9
	for d in range(3):
		os.makedirs(os.path.join(root, 'd%d' % d, 'e'))
		for f in range(5):
			path = os.path.join(root, 'd%d' % d, 'e', 'f%d' % f)
			with open(path, 'w') as handle:
				handle.write('x' * 100)
			os.utime(path, ns=(now, now - 100 * 10 ** 9))
			files.append(path)
	return files


def _first_copy(src, out):
	with contextlib.redirect_stdout(io.StringIO()):
		assert hpcp.hpcp([src + '/'], dest_paths=[out + '/'], max_workers=1, single_thread=True) == 0


def _copy_each(files, src, out, label):
	hpcp.NAMED_DESTS = frozenset([os.path.abspath(out)])
	hpcp.RUN_ID = label
	with contextlib.redirect_stdout(io.StringIO()):
		for path in files:
			hpcp.copy_file(path, [os.path.join(out, os.path.relpath(path, src))])


@pytest.fixture
def copied(tmp_path, reset_hpcp_globals):
	src, out = str(tmp_path / 'src'), str(tmp_path / 'out')
	files = _tree(src)
	_first_copy(src, out)
	probe = os.path.join(out, 'd0', 'e', 'f0')
	before = os.stat(probe).st_atime_ns
	with open(probe, 'rb') as handle:
		handle.read()
	if os.stat(probe).st_atime_ns != before:
		pytest.skip('a read moves the read time here (strictatime)')
	return src, out, files


# ---- R1: one look per landing

@posix_only
def test_an_identical_file_costs_few_calls(copied, monkeypatch):
	# 9.86: 22.5 os calls a file (lstat 7.4, stat 7.1, listxattr 3, chmod 2, chown 1, utime 2), besides the two hash reads.
	src, out, files = copied
	with _counting(monkeypatch) as counts:
		_copy_each(files, src, out, 'skip')
	assert sum(counts.values()) / len(files) <= 8, counts


@posix_only
def test_a_rewrite_in_place_costs_few_calls(copied, monkeypatch):
	# 9.86: 25.6 a file.
	src, out, files = copied
	for path in files:
		with open(path, 'a') as handle:
			handle.write('y')
	with _counting(monkeypatch) as counts:
		_copy_each(files, src, out, 'rewrite')
	assert sum(counts.values()) / len(files) <= 14, counts
	assert all(open(path).read() == open(os.path.join(out, os.path.relpath(path, src))).read() for path in files)


@posix_only
def test_a_first_copy_costs_few_calls(tmp_path, reset_hpcp_globals, monkeypatch):
	# 9.86: 23.5 a file, closes counted (p3_first_and_dirs.py).
	src, out = str(tmp_path / 'src'), str(tmp_path / 'out')
	files = _tree(src)
	for path in files:
		os.makedirs(os.path.dirname(os.path.join(out, os.path.relpath(path, src))), exist_ok=True)
	with _counting(monkeypatch, _COUNTED + ('close',)) as counts:
		_copy_each(files, src, out, 'first')
	assert sum(counts.values()) / len(files) <= 23.5, counts


# ---- R2: the identical file through one checked descriptor, setting only what differs

@posix_only
def test_a_clean_rerun_sets_no_metadata(copied, monkeypatch):
	src, out, files = copied
	landings = [os.path.join(out, os.path.relpath(path, src)) for path in files]
	ctimes = [os.stat(landing).st_ctime_ns for landing in landings]
	time.sleep(0.01)
	with _counting(monkeypatch, _SETS) as counts:
		_copy_each(files, src, out, 'clean')
	assert not counts, counts
	assert [os.stat(landing).st_ctime_ns for landing in landings] == ctimes


def _xattrs(path):
	if not hasattr(os, 'listxattr'):
		return {}
	try:
		return {name: os.getxattr(path, name, follow_symlinks=False) for name in os.listxattr(path, follow_symlinks=False)}
	except OSError as e:
		if e.errno in (errno.ENOTSUP, errno.EOPNOTSUPP):
			return {}
		raise


def _user_xattrs_work(path):
	try:
		os.setxattr(path, 'user.probe', b'1', follow_symlinks=False)
		os.removexattr(path, 'user.probe', follow_symlinks=False)
		return True
	except (OSError, AttributeError):
		return False


def _skip_986(src, dest):
	'''9.86's skip branch, the reference.'''
	st = os.stat(src, follow_symlinks=False)
	hpcp._remove_stale_xattrs(src, dest)
	shutil.copystat(src, dest, follow_symlinks=False)
	owner_kept, group_kept = hpcp._chown_like_cp(dest, st)
	os.chmod(dest, st.st_mode & ~((0 if owner_kept else stat.S_ISUID) | (0 if group_kept else stat.S_ISGID)))
	os.utime(dest, ns=(st.st_atime_ns, st.st_mtime_ns), follow_symlinks=False)


def _metadata(path):
	st = os.lstat(path)
	return stat.S_IMODE(st.st_mode), st.st_uid, st.st_gid, st.st_atime_ns, st.st_mtime_ns, _xattrs(path)


_CASES = ['mode', 'setuid', 'times', 'xattr-added', 'xattr-changed', 'xattr-removed', 'owner']


@posix_only
@pytest.mark.parametrize('case', _CASES)
def test_the_identical_file_s_metadata_matches_9_86(tmp_path, reset_hpcp_globals, case):
	src = str(tmp_path / 'src')
	with open(src, 'w') as handle:
		handle.write('same')
	ours, theirs = str(tmp_path / 'ours'), str(tmp_path / 'theirs')
	if case.startswith('xattr') and not _user_xattrs_work(src):
		pytest.skip('no user xattrs here')
	if case == 'owner' and os.geteuid() != 0:
		pytest.skip('only root gives a file another owner')
	now = _now_ns()
	os.utime(src, ns=(now, now - 50 * 10 ** 9))
	for landing in (ours, theirs):
		shutil.copyfile(src, landing)
		os.utime(landing, ns=(now, now - 50 * 10 ** 9))
		if case == 'mode':
			os.chmod(landing, 0o600)
		elif case == 'times':
			os.utime(landing, ns=(now - 10 ** 9, now - 70 * 10 ** 9))
		elif case == 'xattr-changed':
			os.setxattr(landing, 'user.a', b'old')
		elif case == 'xattr-removed':
			os.setxattr(landing, 'user.gone', b'1')
		elif case == 'owner':
			os.chown(landing, 1000, 1000)
	if case == 'setuid':
		os.chmod(src, 0o4755)
	elif case in ('xattr-added', 'xattr-changed'):
		os.setxattr(src, 'user.a', b'new')
	src_st = os.lstat(src)
	_skip_986(src, theirs)
	assert hpcp._settle_identical(src, src_st, ours, os.lstat(ours)) is True
	assert _metadata(ours) == _metadata(theirs)


@root_only
@linux_only
@pytest.mark.parametrize('swap', ['link', 'second-name'])
def test_a_landing_swapped_after_the_look_gets_no_metadata(tmp_path, reset_hpcp_globals, monkeypatch, swap):
	# Between the skip check and the metadata, a user who can write the destination puts a link to another file, or another
	# file's second name, at the landing. 9.86 set the source's owner and mode (and times, xattrs) on that file through it.
	victim = str(tmp_path / 'victim')
	with open(victim, 'w') as handle:
		handle.write('victim')
	os.chmod(victim, 0o600)
	if _user_xattrs_work(victim):
		os.setxattr(victim, 'user.keep', b'1')
	src_dir, out = str(tmp_path / 'src'), str(tmp_path / 'out')
	os.makedirs(src_dir)
	os.makedirs(out)
	src, landing = os.path.join(src_dir, 'f'), os.path.join(out, 'f')
	for path in (src, landing):
		with open(path, 'w') as handle:
			handle.write('same')
	os.chown(src, 1000, 1000)
	os.chmod(src, 0o4755)  # after the chown, which clears set-user-ID
	st = os.lstat(src)
	os.utime(landing, ns=(st.st_atime_ns, st.st_mtime_ns))  # identical: the skip check passes
	os.chown(out, 1000, 1000)
	before = _metadata(victim)
	real = hpcp._known_identical
	def checked_then_swapped(*args, **kwargs):
		same = real(*args, **kwargs)
		if same and not os.path.lexists(landing + '.away'):
			os.rename(landing, landing + '.away')
			if swap == 'link':
				os.symlink(victim, landing)
			else:
				os.link(victim, landing)
		return same
	monkeypatch.setattr(hpcp, '_known_identical', checked_then_swapped)
	_copy_each([src], src_dir, out, 'swap')
	assert os.path.lexists(landing + '.away')  # the skip check passed, and the swap was made
	assert _metadata(victim) == before
	with open(victim) as handle:
		assert handle.read() == 'victim'
	# The landing is the source's copy now: a link below a destination is replaced, a file of two names written beside.
	assert not os.path.islink(landing) and os.lstat(landing).st_ino != os.lstat(victim).st_ino
	with open(landing) as handle:
		assert handle.read() == 'same'
