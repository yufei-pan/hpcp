'''9.87 group 2: a directory's metadata goes on after what goes in it (M1), and a move writes its directories to disk (M2,
M4). Subprocess runs use the CLI as a person does; the call counts and orders are taken in process.'''
import collections
import errno
import functools
import os
import pathlib
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid

import pytest

import hpcp

HPCP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hpcp.py')
posix_only = pytest.mark.skipif(os.name != 'posix', reason='POSIX modes and times')
not_root = pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() == 0, reason='root writes every directory')
DAY = 86400 * 10**9


def run(args, cwd, env=None, timeout=120):
	'''The CLI in a session of its own: (return code, output as text)'''
	task = subprocess.run([sys.executable, HPCP] + args, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT, env=dict(os.environ, **(env or {})), timeout=timeout, start_new_session=True)
	return task.returncode, task.stdout.decode(errors='replace')


def _set_times(root, back_ns):
	'''Every directory below root (root too) dated back_ns ago, deepest first so no later change moves a parent's time'''
	now = time.time_ns() if hasattr(time, 'time_ns') else int(time.time() * 10**9)
	dirs = [r for r, _, _ in os.walk(str(root))]
	for depth, path in sorted(((p.count(os.sep), p) for p in dirs), reverse=True):
		os.utime(path, ns=(now - back_ns - depth, now - back_ns - depth * 1000))


def _dir_times(root):
	return {os.path.relpath(r, str(root)): os.lstat(r).st_mtime_ns for r, _, _ in os.walk(str(root))}


def _tree(src, links=True):
	for d in range(3):
		for e in range(2):
			path = src / ('d%d' % d) / ('e%d' % e)
			path.mkdir(parents=True)
			for f in range(3):
				(path / ('f%d' % f)).write_text('x' * 100)
		(src / ('d%d' % d) / 'empty').mkdir()
	if links and hasattr(os, 'symlink'):
		os.symlink('d0/e0/f0', str(src / 'link'))
		os.symlink('../e0', str(src / 'd1' / 'e1' / 'up'))


# ---- M1: directory metadata after what goes in the directory

@posix_only
@pytest.mark.parametrize('workers', [['-s'], ['-j', '4']], ids=['serial', 'pool'])
def test_directory_times_follow_the_source_after_a_first_copy(tmp_path, workers):
	# 9.86 set a directory's times before its files went in: 3 to 5 of 7 directories ended with the copy's time
	src = tmp_path / 'src'
	_tree(src)
	# an empty extra directory: -rme removes it without asking, and its parent's time moves
	(tmp_path / 'out' / 'd2' / 'extra').mkdir(parents=True)
	_set_times(src, 30 * DAY)
	rc, output = run(['src/', '-d', 'out/', '-rme'] + workers, tmp_path)
	assert rc == 0, output[-3000:]
	assert not (tmp_path / 'out' / 'd2' / 'extra').exists()
	assert _dir_times(tmp_path / 'out') == _dir_times(src), output[-2000:]
	# a rerun changes none of them
	rc, output = run(['src/', '-d', 'out/'] + workers, tmp_path)
	assert rc == 0 and _dir_times(tmp_path / 'out') == _dir_times(src)


@posix_only
@not_root
@pytest.mark.parametrize('workers', [['-s'], ['-j', '4']], ids=['serial', 'pool'])
def test_a_non_root_copy_of_read_only_directories_holding_files(tmp_path, workers):
	# 9.86 made out/ro 0555 before its files: they were refused (190)
	src = tmp_path / 'src'
	(src / 'ro' / 'deep').mkdir(parents=True)
	(src / 'ro' / 'f').write_text('in a read-only directory')
	(src / 'ro' / 'deep' / 'g').write_text('deeper')
	(src / 'shut').mkdir()
	modes = {'ro/deep': 0o555, 'ro': 0o555, 'shut': 0o444}
	try:
		for name, mode in modes.items():
			os.chmod(str(src / name), mode)
		for attempt in ('first', 'rerun'):
			rc, output = run(['src/', '-d', 'out/'] + workers, tmp_path)
			assert rc == 0, (attempt, output[-3000:])
			assert (tmp_path / 'out' / 'ro' / 'f').read_text() == 'in a read-only directory'
			assert (tmp_path / 'out' / 'ro' / 'deep' / 'g').read_text() == 'deeper'
			for name, mode in modes.items():
				assert stat.S_IMODE(os.lstat(str(tmp_path / 'out' / name)).st_mode) == mode, (attempt, name)
	finally:
		for root in (src, tmp_path / 'out'):
			for name in modes:
				if os.path.lexists(str(root / name)):
					os.chmod(str(root / name), 0o755)


@posix_only
@not_root
def test_a_stopped_copy_leaves_restrictive_directories_open_to_their_owner_until_a_rerun(tmp_path):
	# Not an issue (the plan's section 1): the finish never ran, so a restrictive directory keeps its owner's rwx until a rerun
	src = tmp_path / 'src'
	(src / 'ro').mkdir(parents=True)
	for f in range(40):
		(src / 'ro' / ('f%02d' % f)).write_text('x')
	slow = tmp_path / 'slowcp'
	slow.write_text('#!/bin/sh\ncase "$1" in --version) exec cp "$@";; esac\nsleep 0.2\nexec cp "$@"\n')
	slow.chmod(0o755)
	os.chmod(str(src / 'ro'), 0o555)
	try:
		task = subprocess.Popen([sys.executable, HPCP, 'src/', '-d', 'out/', '-s'], cwd=str(tmp_path), stdin=subprocess.DEVNULL,
			stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=dict(os.environ, CP_PATH=str(slow)), start_new_session=True)
		try:
			deadline = time.monotonic() + 60
			while not (tmp_path / 'out' / 'ro' / 'f01').exists() and time.monotonic() < deadline:
				time.sleep(0.05)
			os.killpg(task.pid, signal.SIGINT)
			output = task.communicate(timeout=60)[0].decode(errors='replace')
		finally:
			if task.poll() is None:
				os.killpg(task.pid, signal.SIGKILL)
				task.wait()
		assert task.returncode == 130, output[-2000:]
		assert stat.S_IMODE(os.lstat(str(tmp_path / 'out' / 'ro')).st_mode) == 0o755
		rc, output = run(['src/', '-d', 'out/'], tmp_path)
		assert rc == 0, output[-2000:]
		assert stat.S_IMODE(os.lstat(str(tmp_path / 'out' / 'ro')).st_mode) == 0o555
	finally:
		for root in (src, tmp_path / 'out'):
			if os.path.lexists(str(root / 'ro')):
				os.chmod(str(root / 'ro'), 0o755)


def _record(path):
	return ('src/' + os.path.basename(path), path, 0o755, 0, 0, None)


def test_the_finish_goes_deepest_first(tmp_path, monkeypatch):
	order = []
	monkeypatch.setattr(hpcp, '_finish_directory', lambda record, sync=False: order.append(record[1]))
	paths = [str(tmp_path / 'a'), str(tmp_path / 'a' / 'b' / 'c'), str(tmp_path / 'x'), str(tmp_path / 'a' / 'b')]
	hpcp.finish_directories([_record(p) for p in paths], 1)
	depths = [p.count(os.sep) for p in order]
	assert sorted(order) == sorted(paths) and depths == sorted(depths, reverse=True)


_LOG = None


def _logging_finish(record, sync=False):
	with open(_LOG, 'a') as f:
		f.write('%d %r\n' % (record[1].count(os.sep), time.monotonic()))


@posix_only
@pytest.mark.skipif('fork' not in __import__('multiprocessing').get_all_start_methods(), reason='forked workers see the patch')
def test_a_pooled_finish_ends_each_depth_before_the_next(tmp_path, monkeypatch, reset_hpcp_globals):
	# one pool for the pass, a barrier per depth: a child's times go on while its parent still lets its owner search it
	import multiprocessing
	global _LOG
	_LOG = str(tmp_path / 'log')
	monkeypatch.setattr(hpcp, '_finish_directory', _logging_finish)
	records = [_record(str(tmp_path.joinpath(*(['d'] * depth)) / str(i))) for depth in (1, 2, 3) for i in range(70)]
	method = multiprocessing.get_start_method()
	multiprocessing.set_start_method('fork', force=True)
	try:
		hpcp.finish_directories(records, 4)
	finally:
		multiprocessing.set_start_method(method, force=True)
	lines = [line.split() for line in open(_LOG)]
	assert len(lines) == len(records)
	ends = {}
	for depth, when in lines:
		ends.setdefault(int(depth), []).append(float(when))
	depths = sorted(ends, reverse=True)
	for deeper, shallower in zip(depths, depths[1:]):
		assert max(ends[deeper]) <= min(ends[shallower])


_SETTERS = ('chmod', 'fchmod', 'utime', 'setxattr', 'removexattr', 'chown', 'fchown', 'chflags')


def _count_setters(monkeypatch, counts):
	'''Count each call that sets a directory's metadata (by path or descriptor); a file's are group 4's'''
	real_fstat, real_isdir = os.fstat, os.path.isdir
	for name in _SETTERS:
		real = getattr(hpcp.os, name, None)
		if real is None:
			continue
		def counted(target, *args, _name=name, _real=real, **kwargs):
			if real_isdir(target) if not isinstance(target, int) else stat.S_ISDIR(real_fstat(target).st_mode):
				counts[_name] += 1
			return _real(target, *args, **kwargs)
		monkeypatch.setattr(hpcp.os, name, counted)


@posix_only
def test_a_rerun_sets_no_directory_metadata(tmp_path, monkeypatch, reset_hpcp_globals):
	src = tmp_path / 'src'
	src.mkdir()
	# no links: a rerun remakes each link (as 9.86 did), and the finish then sets its directory's times again
	_tree(src, links=False)
	(tmp_path / 'dst').mkdir()
	_set_times(src, 30 * DAY)
	restrictive = os.geteuid() != 0
	if restrictive:
		os.chmod(str(src / 'd0' / 'e0'), 0o555)
	try:
		srcs = [str(src) + os.sep]
		opts = dict(dest_paths=[str(tmp_path / 'dst') + os.sep], single_thread=True, max_workers=1, verbose=False, batch=True)
		assert hpcp.hpcp(srcs, **opts) == 0
		counts = collections.Counter()
		_count_setters(monkeypatch, counts)
		assert hpcp.hpcp(srcs, **opts) == 0
		# a restrictive directory: opened at the sync (chmod), closed at the finish (fchmod); nothing else sets anything
		assert dict(counts) == ({'chmod': 1, 'fchmod': 1} if restrictive else {})
	finally:
		for root in (src, tmp_path / 'dst'):
			if os.path.lexists(str(root / 'd0' / 'e0')):
				os.chmod(str(root / 'd0' / 'e0'), 0o755)


@posix_only
def test_a_rerun_leaves_a_directory_read_time_that_alone_differs(tmp_path, monkeypatch, reset_hpcp_globals):
	# EL8 (relatime): test_a_rerun_sets_no_directory_metadata's rerun set 2 directories' times. A read moves a read time that
	# is not after the change time, and the finish's own set makes the change time now: chasing the read time alone, every
	# rerun set it again (R2's files did the same, 6d96a27). Here the landings' read times are moved as such a read moves them.
	src = tmp_path / 'src'
	src.mkdir()
	_tree(src, links=False)
	(tmp_path / 'dst').mkdir()
	_set_times(src, 30 * DAY)
	srcs = [str(src) + os.sep]
	opts = dict(dest_paths=[str(tmp_path / 'dst') + os.sep], single_thread=True, max_workers=1, verbose=False, batch=True)
	assert hpcp.hpcp(srcs, **opts) == 0
	read = time.time_ns() if hasattr(time, 'time_ns') else int(time.time() * 10**9)
	landings = [r for r, _, _ in os.walk(str(tmp_path / 'dst'))]
	for landing in landings:
		os.utime(landing, ns=(read, os.lstat(landing).st_mtime_ns))
	counts = collections.Counter()
	_count_setters(monkeypatch, counts)
	assert hpcp.hpcp(srcs, **opts) == 0
	assert dict(counts) == {}
	assert _dir_times(tmp_path / 'dst') == _dir_times(src)


@posix_only
def test_a_move_keeps_its_sources_when_a_directory_cannot_be_finished(tmp_path, monkeypatch, reset_hpcp_globals):
	# 9.86 set a directory's metadata before the copy's failures were counted, so one it could not set kept the sources; the
	# finish comes later, and its failures still stop the move's removal
	src = tmp_path / 'src'
	src.mkdir()
	_tree(src, links=False)
	_set_times(src, 30 * DAY)
	real_utime, real_isdir, real_fstat = os.utime, os.path.isdir, os.fstat
	def utime(target, *args, **kwargs):
		if stat.S_ISDIR(real_fstat(target).st_mode) if isinstance(target, int) else real_isdir(target):
			raise PermissionError(1, 'Operation not permitted')
		return real_utime(target, *args, **kwargs)
	monkeypatch.setattr(hpcp.os, 'utime', utime)
	rc = hpcp.hpcp([str(src) + os.sep], dest_paths=[str(tmp_path / 'out') + os.sep], single_thread=True, max_workers=1, remove=True)
	assert rc != 0
	assert (src / 'd0' / 'e0' / 'f0').exists() and (tmp_path / 'out' / 'd0' / 'e0' / 'f0').exists()


def _attribute_count(path):
	'''The extended attributes path carries (none where the filesystem has none)'''
	try:
		return len(os.listxattr(path, follow_symlinks=False))
	except (AttributeError, OSError):
		return 0


def _calls_per_directory(tmp_path, monkeypatch, label):
	'''os calls per directory of sync_directory_metadata (the sync and its finish at once), as the plan's probe counts 9.86's'''
	src, out = tmp_path / 'src', tmp_path / 'out'
	folders = sorted(r for r, _, _ in os.walk(str(src)) if r != str(src))
	names = ['lstat', 'stat', 'open', 'close', 'chmod', 'chown', 'utime', 'listxattr', 'getxattr', 'setxattr', 'removexattr',
		'statvfs', 'readlink', 'access', 'scandir', 'mkdir', 'unlink', 'rename', 'fstat', 'fchmod', 'fchown']
	counts = collections.Counter()
	# this thread's calls only: under pytest-xdist the worker's I/O thread calls os too (the rerun's count wandered by 0.25)
	me = threading.get_ident()
	with monkeypatch.context() as patch:
		for name in names:
			real = getattr(os, name, None)
			if real is None:
				continue
			@functools.wraps(real)
			def counted(*args, _name=name, _real=real, **kwargs):
				if threading.get_ident() == me:
					counts[_name] += 1
				return _real(*args, **kwargs)
			patch.setattr(os, name, counted)
		for folder in folders:
			hpcp.sync_directory_metadata(folder, [os.path.join(str(out), os.path.relpath(folder, str(src)))])
	return sum(counts.values()) / len(folders)


@posix_only
def test_directory_calls_stay_within_9_86_s(monkeypatch, reset_hpcp_globals):
	# 9.86 (the plan's probe p3): 17.1 os calls per directory on a first copy, 18.1 on a rerun (23.0 on Python 3.6). Counted
	# where p3 counted, in a directory of the temporary directory: a rerun's count grows with the path's depth (one more
	# lstat a directory for each level, as in 9.86), and pytest's tmp_path is deeper.
	import shutil, tempfile
	base = pathlib.Path(tempfile.mkdtemp(prefix='p3-'))
	try:
		src = base / 'src'
		for d in range(4):
			for e in range(5):
				(src / ('d%d' % d) / ('e%d' % e)).mkdir(parents=True)
		(base / 'out').mkdir()
		monkeypatch.setattr(hpcp, 'NAMED_DESTS', frozenset([str(base / 'out')]))
		# p3's tree carried no extended attributes; each one a directory carries is read on both sides, in 9.86 too (SELinux
		# labels every directory: EL8 counts 18.1 a first copy, the 2 reads of the label over Docker's 16.1)
		labels = 2 * _attribute_count(str(src / 'd0'))
		# 9.88's door (W1) holds each directory on the way by descriptor: here 2.34 more calls a directory than 9.87 (18.46
		# against 16.12 on 3.6, 3.12 and 3.14; its two door results' closes, the levels it opens and closes, less the lstats
		# of 9.87's door and mount check; a result's dup is not counted here)
		assert _calls_per_directory(base, monkeypatch, 'first') <= 17.1 + 2.34 + labels
		# 3.7 to 3.12 were not in p3: 9.86 counts 21.8 a rerun there, in this test's tree (Docker; 9.87 20.0), and 3.13 and
		# later make fewer lstats a path, in both
		rerun = 23.0 if sys.version_info < (3, 7) else 21.8 if sys.version_info < (3, 13) else 18.1
		assert _calls_per_directory(base, monkeypatch, 'rerun') <= rerun + labels
	finally:
		shutil.rmtree(str(base))


# ---- M2: a move fsyncs its directories

proc_fd = pytest.mark.skipif(not os.path.isdir('/proc/self/fd'), reason='names a descriptor by /proc/self/fd')


def _move_tree(src):
	(src / 'empty').mkdir(parents=True)
	(src / 'm').mkdir()
	(src / 'm' / 'f').write_text('m')
	# its last change at the destination is its mode, at the finish
	os.chmod(str(src / 'm'), 0o700)
	(src / 'l').mkdir()
	if hasattr(os, 'symlink'):
		os.symlink('../top', str(src / 'l' / 'link'))
	(src / 'top').write_text('top')


def _log_fsyncs(monkeypatch, events, src, fail=None):
	'''events gets ('fsync', directory) for each directory fsynced and ('remove', path) for each removal below src; fail: an
	errno each directory fsync raises. os.sync and syncfs fail the test.'''
	real_fsync, real_fstat = os.fsync, os.fstat
	def fsync(fd):
		if stat.S_ISDIR(real_fstat(fd).st_mode):
			events.append(('fsync', os.readlink('/proc/self/fd/%d' % fd)))
			if fail:
				raise OSError(fail, os.strerror(fail))
		return real_fsync(fd)
	monkeypatch.setattr(hpcp.os, 'fsync', fsync)
	for name in ('remove', 'unlink', 'rmdir'):
		real = getattr(os, name)
		def removing(path, *args, _real=real, **kwargs):
			where = str(path)
			if kwargs.get('dir_fd') is not None:
				where = os.path.join(os.readlink('/proc/self/fd/%d' % kwargs['dir_fd']), where)
			if os.path.abspath(where).startswith(os.path.realpath(str(src))):
				events.append(('remove', where))
			return _real(path, *args, **kwargs)
		monkeypatch.setattr(hpcp.os, name, removing)
	def no_sync(*args):
		raise AssertionError('a move syncs no whole filesystem')
	monkeypatch.setattr(hpcp.os, 'sync', no_sync, raising=False)
	monkeypatch.setattr(hpcp, '_syncfs', no_sync)


@posix_only
@proc_fd
@pytest.mark.parametrize('nds', [False, True], ids=['synced', 'nds'])
def test_a_move_fsyncs_each_directory_once(tmp_path, monkeypatch, reset_hpcp_globals, nds):
	# 9.86 fsynced only the directories a move made links in: an empty directory, a directory's mode, owner or times, and the
	# name of a file outside ext4 and XFS could be lost to a power cut with the source gone
	src = tmp_path / 'src'
	_move_tree(src)
	events = []
	_log_fsyncs(monkeypatch, events, src)
	# each directory's times set at the finish: every fsync comes after the last of them (one journal commit, not one each)
	real_utime, real_fstat = os.utime, os.fstat
	def utime(target, *args, **kwargs):
		# by path: patched, os.utime is no longer in os.supports_fd
		if not isinstance(target, int) and os.path.isdir(target):
			events.append(('set', str(target)))
		return real_utime(target, *args, **kwargs)
	monkeypatch.setattr(hpcp.os, 'utime', utime)
	out = tmp_path / 'out'
	rc = hpcp.hpcp([str(src) + os.sep], dest_paths=[str(out) + os.sep], single_thread=True, max_workers=1, remove=True, remove_force=True,
		no_directory_sync=nds)
	assert rc == 0
	assert (out / 'm' / 'f').read_text() == 'm' and not (src / 'm' / 'f').exists()
	synced = [path for kind, path in events if kind == 'fsync']
	expected = [str(tmp_path), str(out), str(out / 'l'), str(out / 'm')] + ([] if nds else [str(out / 'empty')])
	assert sorted(synced) == sorted(os.path.realpath(path) for path in expected)
	removals = [i for i, (kind, _) in enumerate(events) if kind == 'remove']
	assert removals and all(i < removals[0] for i, (kind, _) in enumerate(events) if kind == 'fsync')
	sets = [i for i, (kind, _) in enumerate(events) if kind == 'set']
	assert bool(sets) != nds and all(i > max(sets or [-1]) for i, (kind, _) in enumerate(events) if kind == 'fsync')
	if not nds:
		assert stat.S_IMODE(os.lstat(str(out / 'm')).st_mode) == 0o700


@posix_only
@proc_fd
@pytest.mark.parametrize('nds', [False, True], ids=['synced', 'nds'])
def test_a_move_whose_directory_cannot_be_written_to_disk_keeps_every_source(tmp_path, monkeypatch, reset_hpcp_globals, nds):
	src = tmp_path / 'src'
	_move_tree(src)
	events = []
	_log_fsyncs(monkeypatch, events, src, fail=errno.EIO)
	rc = hpcp.hpcp([str(src) + os.sep], dest_paths=[str(tmp_path / 'out') + os.sep], single_thread=True, max_workers=1,
		remove=True, remove_force=True, no_directory_sync=nds)
	assert rc == 212
	assert (src / 'm' / 'f').exists() and (src / 'top').exists() and not [kind for kind, _ in events if kind == 'remove']


@posix_only
@proc_fd
def test_the_finish_fsyncs_a_directory_through_the_descriptor_it_opened_before_closing_it(tmp_path, monkeypatch):
	# a directory whose final mode lacks owner read (0333) cannot be opened once it has it: the finish opens it first
	dest = tmp_path / 'd'
	dest.mkdir()
	events = []
	_log_fsyncs(monkeypatch, events, tmp_path / 'src')
	try:
		hpcp.finish_directories([(str(tmp_path / 'src'), str(dest), 0o333, 10**18, 10**18, None)], 1, sync=True)
		# root opens it again whatever its mode: the move's pass fsyncs it with the others (_sync_moved_directories)
		assert events == ([('fsync', str(dest))] if os.geteuid() != 0 else []) and stat.S_IMODE(os.lstat(str(dest)).st_mode) == 0o333
		assert os.lstat(str(dest)).st_mtime_ns == 10**18
	finally:
		os.chmod(str(dest), 0o755)


@posix_only
@not_root
@proc_fd
@pytest.mark.skipif(not os.environ.get('HPCP_TEST_DROP_BOX'), reason='needs a directory of another user, mode 0733 (the Docker harness mounts one)')
def test_a_move_into_another_users_write_only_directory_passes_it_over(tmp_path, monkeypatch, reset_hpcp_globals):
	# the known limitation: such a directory cannot be opened, so it is not fsynced; the move goes on, every other directory
	# fsynced, and no whole filesystem is synced
	box = os.environ['HPCP_TEST_DROP_BOX']
	assert os.stat(box).st_uid != os.geteuid() and not os.access(box, os.R_OK) and os.access(box, os.W_OK | os.X_OK)
	name = 'm2-' + uuid.uuid4().hex
	src = tmp_path / name
	_move_tree(src)
	note = tmp_path / (name + '.note')
	note.write_text('note')
	events = []
	_log_fsyncs(monkeypatch, events, tmp_path)
	try:
		rc = hpcp.hpcp([str(src), str(note)], dest_paths=[box + os.sep], single_thread=True, max_workers=1, remove=True, remove_force=True)
		assert rc == 0
		landing = os.path.join(box, name)
		assert open(os.path.join(landing, 'm', 'f')).read() == 'm' and open(os.path.join(box, name + '.note')).read() == 'note'
		assert not src.exists() and not note.exists()
		synced = sorted(path for kind, path in events if kind == 'fsync')
		assert synced == sorted([landing] + [os.path.join(landing, d) for d in ('empty', 'l', 'm')])
	finally:
		if os.path.isdir(os.path.join(box, name)):
			for root, dirs, _ in os.walk(os.path.join(box, name)):
				for d in dirs:
					os.chmod(os.path.join(root, d), 0o755)
		import shutil
		shutil.rmtree(os.path.join(box, name), ignore_errors=True)
		with __import__('contextlib').suppress(OSError):
			os.unlink(os.path.join(box, name + '.note'))


# ---- M4: one full flush per destination filesystem in a move (macOS's F_FULLFSYNC, a fake here)

SMALL_FS = os.environ.get('HPCP_TEST_SMALL_FS')


class _FakeFcntl:
	'''fcntl with F_FULLFSYNC (macOS's 51): each call logged as ('flush', st_dev, path); fail: the errno it raises.'''
	F_FULLFSYNC = 51

	def __init__(self, real, events, fail=None):
		self._real, self.events, self.fail = real, events, fail

	def __getattr__(self, name):
		return getattr(self._real, name)

	def fcntl(self, fd, command, *args):
		if command != self.F_FULLFSYNC:
			return self._real.fcntl(fd, command, *args)
		self.events.append(('flush', os.fstat(fd).st_dev, os.readlink('/proc/self/fd/%d' % fd)))
		if self.fail:
			raise OSError(self.fail, os.strerror(self.fail))
		return 0


def _log_all(monkeypatch, events, fail=None):
	'''events gets ('fsync', path) for every fsync and ('remove', path) for every removal; F_FULLFSYNC is faked (_FakeFcntl).'''
	import fcntl
	real_fsync = os.fsync
	def fsync(fd):
		events.append(('fsync', os.readlink('/proc/self/fd/%d' % fd)))
		return real_fsync(fd)
	monkeypatch.setattr(hpcp.os, 'fsync', fsync)
	for name in ('remove', 'unlink', 'rmdir'):
		real = getattr(os, name)
		def removing(path, *args, _real=real, **kwargs):
			where = str(path)
			if kwargs.get('dir_fd') is not None:
				where = os.path.join(os.readlink('/proc/self/fd/%d' % kwargs['dir_fd']), where)
			events.append(('remove', where))
			return _real(path, *args, **kwargs)
		monkeypatch.setattr(hpcp.os, name, removing)
	monkeypatch.setitem(sys.modules, 'fcntl', _FakeFcntl(fcntl, events, fail))


@posix_only
@proc_fd
@pytest.mark.skipif(not SMALL_FS, reason='HPCP_TEST_SMALL_FS: a second filesystem')
def test_a_move_asks_each_destination_filesystem_once_for_a_full_flush(tmp_path, monkeypatch, reset_hpcp_globals):
	src = tmp_path / 'src'
	_move_tree(src)
	second = os.path.join(SMALL_FS, 'm4-' + uuid.uuid4().hex)
	events = []
	_log_all(monkeypatch, events)
	try:
		rc = hpcp.hpcp([str(src) + os.sep], dest_paths=[str(tmp_path / 'out') + os.sep, second + os.sep], single_thread=True,
			max_workers=1, remove=True, remove_force=True)
		assert rc == 0
		flushes = [event for event in events if event[0] == 'flush']
		# the files all land in the first destination (no -rds): a landing file there, a directory on the second filesystem
		assert sorted(dev for _, dev, _ in flushes) == sorted({os.stat(str(tmp_path)).st_dev, os.stat(SMALL_FS).st_dev})
		assert any(os.path.isfile(path) for _, _, path in flushes) and any(os.path.isdir(path) for _, _, path in flushes)
		kinds = [event[0] for event in events]
		last_fsync = max(i for i, kind in enumerate(kinds) if kind == 'fsync')
		first_flush = kinds.index('flush')
		removals = [i for i, kind in enumerate(kinds) if kind == 'remove' and events[i][1].startswith(str(src))]
		assert last_fsync < first_flush and removals and max(i for i, kind in enumerate(kinds) if kind == 'flush') < removals[0]
	finally:
		import shutil
		shutil.rmtree(second, ignore_errors=True)


@posix_only
@proc_fd
@pytest.mark.parametrize('fail', [errno.ENOTSUP, errno.EIO], ids=['ENOTSUP', 'EIO'])
def test_a_drive_that_cannot_be_asked_is_noted_and_one_that_fails_keeps_every_source(tmp_path, monkeypatch, capsys,
		reset_hpcp_globals, fail):
	src = tmp_path / 'src'
	_move_tree(src)
	events = []
	_log_all(monkeypatch, events, fail=fail)
	rc = hpcp.hpcp([str(src) + os.sep], dest_paths=[str(tmp_path / 'out') + os.sep], single_thread=True, max_workers=1,
		remove=True, remove_force=True)
	err = capsys.readouterr().err
	assert len([event for event in events if event[0] == 'flush']) == 1
	if fail == errno.ENOTSUP:
		assert rc == 0 and not (src / 'm' / 'f').exists()
		assert 'Note: the drive under %s' % (tmp_path / 'out') in err and 'could not be asked to flush its cache' in err
	else:
		assert rc == 212 and (src / 'm' / 'f').exists() and (src / 'top').exists()
		assert 'Copy failed: the drive under %s' % (tmp_path / 'out') in err and 'could not flush its cache' in err


@posix_only
@proc_fd
@pytest.mark.parametrize('beside', [False, True], ids=['copy', 'wb'])
def test_a_plain_copy_and_wb_make_no_full_flush(tmp_path, monkeypatch, reset_hpcp_globals, beside):
	src = tmp_path / 'src'
	_move_tree(src)
	opts = dict(dest_paths=[str(tmp_path / 'out') + os.sep], single_thread=True, max_workers=1, write_beside=beside)
	assert hpcp.hpcp([str(src) + os.sep], **opts) == 0
	(src / 'm' / 'f').write_text('changed')
	events = []
	_log_all(monkeypatch, events)
	assert hpcp.hpcp([str(src) + os.sep], **opts) == 0
	assert (tmp_path / 'out' / 'm' / 'f').read_text() == 'changed'
	assert not [event for event in events if event[0] == 'flush']


# ---- M3: a move's fsync waits overlap (more workers by default), and a failed fsync removes only its own landing

@posix_only
def test_a_move_runs_at_least_16_workers_by_default(tmp_path, monkeypatch, reset_hpcp_globals):
	# each worker waits on its files' fsyncs: the 9.85 review's -j 16 on 4 vCPUs took 8.2 to 11.8 s against 11.9 to 15.8 s
	seen = []
	real = hpcp.process_copy
	def process_copy(*args, **kwargs):
		seen.append(kwargs['max_workers'])
		return real(*args, **kwargs)
	monkeypatch.setattr(hpcp, 'process_copy', process_copy)
	(tmp_path / 'a').mkdir()
	(tmp_path / 'a' / 'f').write_text('f')
	(tmp_path / 'b').mkdir()
	(tmp_path / 'b' / 'f').write_text('f')
	assert hpcp.hpcp([str(tmp_path / 'a') + os.sep], dest_paths=[str(tmp_path / 'out') + os.sep]) == 0
	assert hpcp.hpcp([str(tmp_path / 'b') + os.sep], dest_paths=[str(tmp_path / 'moved') + os.sep], remove=True, remove_force=True) == 0
	assert hpcp.hpcp([str(tmp_path / 'out') + os.sep], dest_paths=[str(tmp_path / 'again') + os.sep], remove=True,
		remove_force=True, max_workers=3) == 0
	cpus = __import__('multiprocessing').cpu_count()
	assert seen == [cpus, max(16, cpus), 3]


@posix_only
def test_a_failed_fsync_removes_only_the_landing_it_wrote(tmp_path, monkeypatch, reset_hpcp_globals):
	# the landing renamed away between the copy and its failed fsync, another file put at its name: that file is kept
	(tmp_path / 'src').mkdir()
	(tmp_path / 'src' / 'f').write_text('source')
	real = hpcp._fsync_file
	def fsync_file(path, *args):
		if os.path.basename(path) == 'f' and os.path.isfile(path):
			os.rename(path, path + '.away')
			with open(path, 'w') as other:
				other.write('someone else')
			raise OSError(errno.EIO, os.strerror(errno.EIO))
		return real(path, *args)
	monkeypatch.setattr(hpcp, '_fsync_file', fsync_file)
	rc = hpcp.hpcp([str(tmp_path / 'src') + os.sep], dest_paths=[str(tmp_path / 'out') + os.sep], single_thread=True,
		max_workers=1, remove=True, remove_force=True)
	assert rc != 0
	assert (tmp_path / 'src' / 'f').read_text() == 'source'
	assert (tmp_path / 'out' / 'f').read_text() == 'someone else'


@posix_only
@proc_fd
def test_an_nds_move_fsyncs_the_directories_on_the_way_to_a_landing(tmp_path, monkeypatch, reset_hpcp_globals):
	# under -nds no folder sync makes out/a and out/a/b records: each holds a new name all the same
	src = tmp_path / 'src'
	(src / 'a' / 'b' / 'c').mkdir(parents=True)
	(src / 'a' / 'b' / 'c' / 'f').write_text('f')
	events = []
	_log_fsyncs(monkeypatch, events, src)
	out = tmp_path / 'out'
	rc = hpcp.hpcp([str(src) + os.sep], dest_paths=[str(out) + os.sep], single_thread=True, max_workers=1, remove=True,
		remove_force=True, no_directory_sync=True)
	assert rc == 0 and (out / 'a' / 'b' / 'c' / 'f').read_text() == 'f'
	synced = sorted(path for kind, path in events if kind == 'fsync')
	assert synced == sorted(os.path.realpath(str(path)) for path in (tmp_path, out, out / 'a', out / 'a' / 'b', out / 'a' / 'b' / 'c'))


@posix_only
@proc_fd
def test_a_landing_its_owner_may_not_read_is_flushed_through_its_directory(tmp_path, monkeypatch, reset_hpcp_globals):
	# a copy of another user's 0044 file is the runner's own and unreadable to it: 9.87's first build failed the move (212)
	src, out = tmp_path / 'src', tmp_path / 'out'
	(src / 'a').mkdir(parents=True)
	(src / 'a' / 'f').write_text('f')
	(out / 'a').mkdir(parents=True)
	(out / 'a' / 'f').write_text('f')
	os.chmod(str(out / 'a' / 'f'), 0o044)
	events = []
	_log_all(monkeypatch, events)
	try:
		hpcp._flush_drives([str(src / 'a' / 'f')], set(), [str(src) + os.sep], [str(out) + os.sep], False)
	finally:
		os.chmod(str(out / 'a' / 'f'), 0o644)
	flushes = [event[2] for event in events if event[0] == 'flush']
	# root opens the landing itself; anyone else its directory, on the same filesystem
	assert flushes == [str(out / 'a' / 'f') if os.geteuid() == 0 else str(out / 'a')]
	assert not [error for error in hpcp.ERRORS if hpcp._is_failure(error)]


@posix_only
@not_root
def test_a_rerun_gives_a_restrictive_directory_its_source_s_changed_xattr(tmp_path):
	src = tmp_path / 'src'
	(src / 'ro').mkdir(parents=True)
	(src / 'ro' / 'f').write_text('f')
	try:
		os.setxattr(str(src / 'ro'), 'user.tag', b'one')
	except (AttributeError, OSError):
		pytest.skip('no user xattrs here')
	os.chmod(str(src / 'ro'), 0o555)
	try:
		assert run(['src/', '-d', 'out/', '-s'], tmp_path)[0] == 0
		assert os.getxattr(str(tmp_path / 'out' / 'ro'), 'user.tag') == b'one'
		os.chmod(str(src / 'ro'), 0o755)
		os.setxattr(str(src / 'ro'), 'user.tag', b'two')
		os.chmod(str(src / 'ro'), 0o555)
		rc, output = run(['src/', '-d', 'out/', '-s'], tmp_path)
		assert rc == 0, output[-2000:]
		assert os.getxattr(str(tmp_path / 'out' / 'ro'), 'user.tag') == b'two'
		assert stat.S_IMODE(os.lstat(str(tmp_path / 'out' / 'ro')).st_mode) == 0o555
	finally:
		for root in (src, tmp_path / 'out'):
			if os.path.isdir(str(root / 'ro')):
				os.chmod(str(root / 'ro'), 0o755)


# ---- 9.88 D3: a destination directory several sources share is synced once, from the later source

T1, T2 = 1600000000 * 10**9, 1650000000 * 10**9


def _shared(path, name, mode, mtime_ns):
	'''path/name, a directory holding a file of its own (name.f), of mode and modification time mtime_ns.'''
	path.mkdir(parents=True)
	(path / (name + '.f')).write_text(name)
	os.chmod(str(path), mode)
	os.utime(str(path), ns=(mtime_ns, mtime_ns))


@posix_only
@pytest.mark.parametrize('workers', [['-s'], ['-j', '4']], ids=['serial', 'pool'])
def test_merged_sources_sync_a_shared_directory_once_from_the_later(tmp_path, workers):
	# 9.87: out/x got a sync and a finish record from each source, so every rerun set its mode and times twice (its ctime moved
	# each run), and in a pool which source's metadata won was not fixed. The later source's, as merged files go.
	_shared(tmp_path / 'a' / 'x', 'a', 0o750, T1)
	_shared(tmp_path / 'b' / 'x', 'b', 0o700, T2)
	rc, output = run(['a/', 'b/', '-d', 'out/'] + workers, tmp_path)
	landed = os.lstat(str(tmp_path / 'out' / 'x'))
	assert rc == 0 and (stat.S_IMODE(landed.st_mode), landed.st_mtime_ns) == (0o700, T2), output[-3000:]
	rc, output = run(['a/', 'b/', '-d', 'out/'] + workers, tmp_path)
	again = os.lstat(str(tmp_path / 'out' / 'x'))
	assert rc == 0 and again.st_ctime_ns == landed.st_ctime_ns and again.st_mtime_ns == T2, output[-3000:]


@posix_only
@pytest.mark.parametrize('workers', [['-s'], ['-j', '4']], ids=['serial', 'pool'])
def test_a_merged_source_s_root_wins_over_an_earlier_source_s_folder(tmp_path, workers):
	# hpcp a/ b -d out/ with a/b/ there: out/b is a/b's folder and b's root. 9.87 synced b's root first and a/b in the pool
	# after it, so the earlier source's metadata won, and a rerun set it twice.
	_shared(tmp_path / 'a' / 'b', 'a', 0o750, T1)
	_shared(tmp_path / 'b', 'b', 0o700, T2)
	rc, output = run(['a/', 'b', '-d', 'out/'] + workers, tmp_path)
	landed = os.lstat(str(tmp_path / 'out' / 'b'))
	assert rc == 0 and (stat.S_IMODE(landed.st_mode), landed.st_mtime_ns) == (0o700, T2), output[-3000:]
	rc, output = run(['a/', 'b', '-d', 'out/'] + workers, tmp_path)
	again = os.lstat(str(tmp_path / 'out' / 'b'))
	assert rc == 0 and again.st_ctime_ns == landed.st_ctime_ns, output[-3000:]


@posix_only
@pytest.mark.parametrize('batch', ['parallel', 'serial'])
def test_the_folder_sync_batches_give_one_record_per_destination(tmp_path, monkeypatch, reset_hpcp_globals, batch):
	# The public batches: a destination directory two jobs share, and each job's root (also in its own listing), synced once,
	# from the last job that has it; one record each. 9.87 returned one per job and the roots twice (parallel).
	_shared(tmp_path / 'a' / 'x', 'a', 0o750, T1)
	_shared(tmp_path / 'b' / 'x', 'b', 0o700, T2)
	out = str(tmp_path / 'out')
	jobs = [(str(tmp_path / 'a'), [out]), (str(tmp_path / 'b'), [out])]
	syncs = []
	real = hpcp._sync_directory
	def sync(src_path, dest_paths):
		syncs.append((src_path, tuple(dest_paths)))
		return real(src_path, dest_paths)
	monkeypatch.setattr(hpcp, '_sync_directory', sync)
	if batch == 'parallel':
		_, records = hpcp.sync_directories_parallel_batch(jobs, 2)
	else:
		_, records = hpcp.sync_directories_serial_batch(jobs)
	destinations = sorted(record[1] for record in records)
	assert destinations == [out, os.path.join(out, 'x')], records
	assert {record[1]: record[0] for record in records} == {out: str(tmp_path / 'b'), os.path.join(out, 'x'): str(tmp_path / 'b' / 'x')}
	if batch == 'serial':  # the pool's syncs run in its workers
		assert len(syncs) == 2, syncs


@posix_only
def test_a_directory_finish_takes_times_within_the_filesystem_step(tmp_path, monkeypatch, reset_hpcp_globals):
	# A directory on vfat keeps its time in 2 s steps: the finish compared the source's time exactly and set it again on each
	# rerun. Within the step the destination keeps (_time_stored_as), nothing is set.
	directory = tmp_path / 'd'
	directory.mkdir()
	on_disk = 1600000000 * 10**9  # a 2 s step's time, as vfat keeps it
	os.utime(str(directory), ns=(on_disk, on_disk))
	mode = stat.S_IMODE(os.lstat(str(directory)).st_mode)
	set_times = []
	real = os.utime
	monkeypatch.setattr(os, 'utime', lambda *args, **kwargs: set_times.append(args) or real(*args, **kwargs))
	hpcp._finish_directory((str(tmp_path / 'src'), str(directory), mode, on_disk, on_disk + 10**9 + 7, None))
	assert set_times == [] and not hpcp.ERRORS, (set_times, hpcp.ERRORS)
	hpcp._finish_directory((str(tmp_path / 'src'), str(directory), mode, on_disk, on_disk + 3 * 10**9, None))
	assert len(set_times) == 1, set_times


def test_a_non_root_run_sets_only_the_user_bsd_flags(monkeypatch):
	# macOS gives /Users, /private and /var/folders sunlnk (SF_NOUNLINK), a system flag only root can set: a non-root list-mode
	# copy of a path below /Users synced /Users to out/Users, and chflags with the source's flags was EPERM for the whole call
	# (exit 193; a move never ended). A non-root run sets the user flags (UF_*, the lower 16 bits) and keeps the system ones
	# the destination has, as an owner it cannot keep is no error; root sets them all.
	hidden, nodump, sunlnk, archived = 0x8000, 0x1, 0x100000, 0x10000
	monkeypatch.setattr(os, 'geteuid', lambda: 1000, raising=False)
	assert hpcp._flags_to_set(sunlnk | hidden, 0) == hidden
	assert hpcp._flags_to_set(sunlnk, archived | nodump) == archived
	assert hpcp._flags_to_set(0, archived) == archived
	monkeypatch.setattr(os, 'geteuid', lambda: 0, raising=False)
	assert hpcp._flags_to_set(sunlnk | hidden, archived | nodump) == sunlnk | hidden


@posix_only
def test_a_directory_finish_sets_an_even_second_source_time_exactly_on_a_fine_filesystem(tmp_path, reset_hpcp_globals):
	# The tolerance is the step the destination keeps, not the source's: a source time on an even second (vfat's, or a
	# whole-second tool's) onto a directory this filesystem (tmpfs, ext4, XFS: 1 ns) last wrote 1.5 s later. 9.88's first
	# build took the coarser step of the two (2 s) and left it 1.5 s off; 9.87 set it exactly.
	directory = tmp_path / 'd'
	directory.mkdir()
	source_ns = 1700000000 * 10**9
	os.utime(str(directory), ns=(source_ns + 1523456789, source_ns + 1523456789))
	mode = stat.S_IMODE(os.lstat(str(directory)).st_mode)
	hpcp._finish_directory((str(tmp_path / 'src'), str(directory), mode, source_ns, source_ns, None))
	assert os.lstat(str(directory)).st_mtime_ns == source_ns and not hpcp.ERRORS, hpcp.ERRORS


@posix_only
@pytest.mark.skipif(not sys.platform.startswith('linux') or not hasattr(os, 'geteuid') or os.geteuid() != 0,
	reason='needs root to mount a vfat image (VM 901)')
def test_a_merged_rerun_on_vfat_sets_no_directory_again(tmp_path, monkeypatch, reset_hpcp_globals):
	# vfat keeps times in 2 s steps: 9.87's finish compared them exactly, and with merged sources gave a shared directory two
	# records, so every rerun set its times again (twice). The modes are ones vfat can hold (its default dmask, 022). The
	# copy goes below the mount: vfat's root directory keeps no times (it reads 1970 after a touch: VM 901, Linux 6.1), so
	# a run sets them on every rerun, as 9.87 did.
	mkfs = shutil.which('mkfs.vfat') or shutil.which('/usr/sbin/mkfs.vfat') or shutil.which('/sbin/mkfs.vfat')
	if not mkfs:
		pytest.skip('needs mkfs.vfat')
	image, mount = tmp_path / 'fat.img', tmp_path / 'fat'
	image.write_bytes(b'')
	os.truncate(str(image), 32 << 20)
	if subprocess.run([mkfs, str(image)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).returncode:
		pytest.skip('mkfs.vfat refused')
	mount.mkdir()
	if subprocess.run(['mount', '-o', 'loop', str(image), str(mount)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).returncode:
		pytest.skip('this kernel does not mount vfat')
	try:
		_shared(tmp_path / 'a' / 'x', 'a', 0o755, T1 + 1)
		_shared(tmp_path / 'b' / 'x', 'b', 0o755, T2 + 1)
		_shared(tmp_path / 'b' / 'y', 'b', 0o755, T2 + 3)
		sources = [str(tmp_path / 'a') + os.sep, str(tmp_path / 'b') + os.sep]
		dest = str(mount / 'out') + os.sep
		assert hpcp.hpcp(sources, dest_paths=[dest], single_thread=True, batch=True) == 0, hpcp.ERRORS
		assert abs(os.lstat(str(mount / 'out' / 'x')).st_mtime_ns - T2) <= 2 * 10**9
		changes = []
		for name in ('utime', 'chmod', 'fchmod'):
			real = getattr(os, name)
			def spy(target, *args, _real=real, _name=name, **kwargs):
				where = os.readlink('/proc/self/fd/%d' % target) if isinstance(target, int) else os.path.abspath(target)
				if os.path.isdir(where) and where.startswith(str(mount)):
					changes.append((_name, where))
				return _real(target, *args, **kwargs)
			monkeypatch.setattr(os, name, spy)
		assert hpcp.hpcp(sources, dest_paths=[dest], single_thread=True, batch=True) == 0, hpcp.ERRORS
		assert changes == [], changes
	finally:
		subprocess.run(['umount', '-l', str(mount)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


# ---- 9.88 D2: a non-root move of a directory whose mode shuts its owner out

root_only = pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() != 0, reason='needs root to make another user\'s tree')
NOBODY = 65534


def _shut_out_tree():
	'''f5's tree, made as root in a work directory nobody can use (0777, not sticky, so a move may remove the source root):
	src/d of mode 0607 owned by root (no search for its owner; nobody, the runner, reaches it as "other", and owns the copy),
	holding sub/f and g; out owned by nobody.'''
	import tempfile
	work = tempfile.mkdtemp(prefix='hpcp-d2-')
	os.chmod(work, 0o777)
	src = os.path.join(work, 'src')
	os.makedirs(os.path.join(src, 'd', 'sub'))
	for path, text in ((os.path.join(src, 'd', 'sub', 'f'), 'hi'), (os.path.join(src, 'd', 'g'), 'top')):
		with open(path, 'w') as handle:
			handle.write(text)
		os.chmod(path, 0o666)
	os.chmod(src, 0o777)
	os.chmod(os.path.join(src, 'd', 'sub'), 0o777)
	os.chmod(os.path.join(src, 'd'), 0o607)
	os.mkdir(os.path.join(work, 'out'))
	os.chown(os.path.join(work, 'out'), NOBODY, NOBODY)
	return work


def _as_nobody(args, cwd, script=None):
	'''hpcp (or script, a wrapper that runs hpcp.main()) as nobody: (return code, output). Skips where nobody cannot reach
	Python or hpcp.py (the EL VMs keep the checkout and the venv in a home directory of mode 0700); a venv's Python is
	run by its real path, which hpcp needs nothing beyond.'''
	python = os.path.realpath(sys.executable)
	for path in (python, HPCP, script or HPCP):
		above = os.path.dirname(path)
		while above != os.path.dirname(above):
			if not os.stat(above).st_mode & stat.S_IXOTH:
				pytest.skip(f'nobody cannot reach {path}: {above} is not searchable by others')
			above = os.path.dirname(above)
	def demote():
		os.setgroups([])
		os.setgid(NOBODY)
		os.setuid(NOBODY)
	task = subprocess.run([python, script or HPCP] + args, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT, preexec_fn=demote, timeout=120, env=dict(os.environ, HOME=cwd))
	return task.returncode, task.stdout.decode(errors='replace')


@posix_only
@root_only
def test_a_non_root_move_of_a_directory_its_owner_cannot_search_completes(tmp_path):
	# 9.87: exit 212, 'Copy failed: the directory OUT/d/sub cannot be written to disk (Permission denied)', every source kept:
	# the finish gave out/d 0607, and the move's fsync of d/sub (by path) met EACCES. Its mode is now set after the removal.
	work = _shut_out_tree()
	try:
		rc, output = _as_nobody(['src/', '-d', 'out/', '-rf'], work)
		out = os.path.join(work, 'out', 'd')
		assert rc == 0 and not os.path.exists(os.path.join(work, 'src', 'd')), output[-3000:]
		assert stat.S_IMODE(os.lstat(out).st_mode) == 0o607, oct(os.lstat(out).st_mode)
		with open(os.path.join(out, 'sub', 'f')) as f, open(os.path.join(out, 'g')) as g:
			assert (f.read(), g.read()) == ('hi', 'top')
	finally:
		shutil.rmtree(work, ignore_errors=True)


_STOPPED_AT_REMOVAL = '''import sys
sys.path.insert(0, %(root)r)
import hpcp
def stopped(*args, **kwargs):
	raise KeyboardInterrupt  # a stop as the move's removal starts: the late pass never runs
hpcp.remove_moved_sources = stopped
if __name__ == '__main__':  # forkserver and spawn workers import this file as __mp_main__
	sys.argv = [%(hpcp)r] + sys.argv[1:]
	hpcp.main()
'''


@posix_only
@root_only
def test_a_stopped_move_leaves_owner_search_until_a_rerun(tmp_path):
	# A stop before the late pass (here as the removal starts) leaves such a directory with owner read and search (0707): a
	# rerun of the move sets its exact mode.
	work = _shut_out_tree()
	try:
		script = os.path.join(work, 'stopped.py')
		with open(script, 'w') as handle:
			handle.write(_STOPPED_AT_REMOVAL % {'root': os.path.dirname(HPCP), 'hpcp': HPCP})
		out = os.path.join(work, 'out', 'd')
		rc, output = _as_nobody(['src/', '-d', 'out/', '-rf'], work, script)
		assert rc == 130 and stat.S_IMODE(os.lstat(out).st_mode) == 0o707, (oct(os.lstat(out).st_mode), output[-3000:])
		rc, output = _as_nobody(['src/', '-d', 'out/', '-rf'], work)
		assert rc == 0 and stat.S_IMODE(os.lstat(out).st_mode) == 0o607, (oct(os.lstat(out).st_mode), output[-3000:])
		assert not os.path.exists(os.path.join(work, 'src', 'd'))
	finally:
		shutil.rmtree(work, ignore_errors=True)


_STOPPED_AFTER_REMOVAL = '''import sys
sys.path.insert(0, %(root)r)
import hpcp
removal = hpcp.remove_moved_sources
def stopped(*args, **kwargs):
	removal(*args, **kwargs)
	raise KeyboardInterrupt  # a stop as the move's removal ends: every source directory is gone
hpcp.remove_moved_sources = stopped
if __name__ == '__main__':  # forkserver and spawn workers import this file as __mp_main__
	sys.argv = [%(hpcp)r] + sys.argv[1:]
	hpcp.main()
'''


@posix_only
@root_only
def test_a_move_stopped_as_its_removal_ends_still_gets_the_exact_mode(tmp_path):
	# A stop at the end of the removal (a big tree's removal takes long, so a Ctrl-C there is likely): out/d's source is gone,
	# so no rerun reaches it (one exits 240, the source does not exist), and 9.88's first build left it 0707 for good. The late
	# pass now runs however the removal ended, for each directory whose source it removed.
	work = _shut_out_tree()
	try:
		script = os.path.join(work, 'stopped.py')
		with open(script, 'w') as handle:
			handle.write(_STOPPED_AFTER_REMOVAL % {'root': os.path.dirname(HPCP), 'hpcp': HPCP})
		out = os.path.join(work, 'out', 'd')
		rc, output = _as_nobody(['src/', '-d', 'out/', '-rf'], work, script)
		assert rc == 130 and not os.path.exists(os.path.join(work, 'src', 'd')), output[-3000:]
		assert stat.S_IMODE(os.lstat(out).st_mode) == 0o607, (oct(os.lstat(out).st_mode), output[-3000:])
	finally:
		shutil.rmtree(work, ignore_errors=True)


@posix_only
def test_a_move_off_posix_does_not_ask_for_an_euid(tmp_path, monkeypatch, reset_hpcp_globals):
	# Windows has no os.geteuid: 9.87's _shut_to_its_owner called it for every directory a move synced (an AttributeError
	# out of _sync_moved_directories; reasoned, no Windows machine here). Off POSIX the finish sets no mode first.
	out = tmp_path / 'out'
	out.mkdir()
	monkeypatch.setattr(hpcp, 'SYNC_COPIES', True)
	monkeypatch.delattr(os, 'geteuid', raising=False)
	record = (str(tmp_path / 'src'), str(out), 0o600, 0, 0, None)
	hpcp._sync_moved_directories([], set(), [str(tmp_path / 'src') + os.sep], [str(out) + os.sep], False, [record], 1)
	assert hpcp._first_mode(0o600) == 0o600 and hpcp._synced_at_finish(0o600) is False
