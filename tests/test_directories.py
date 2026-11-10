'''9.87 group 2: a directory's metadata goes on after what goes in it (M1), and a move writes its directories to disk (M2,
M4). Subprocess runs use the CLI as a person does; the call counts and orders are taken in process.'''
import collections
import errno
import functools
import os
import signal
import stat
import subprocess
import sys
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


def _calls_per_directory(tmp_path, monkeypatch, label):
	'''os calls per directory of sync_directory_metadata (the sync and its finish at once), as the plan's probe counts 9.86's'''
	src, out = tmp_path / 'src', tmp_path / 'out'
	folders = sorted(r for r, _, _ in os.walk(str(src)) if r != str(src))
	names = ['lstat', 'stat', 'open', 'close', 'chmod', 'chown', 'utime', 'listxattr', 'getxattr', 'setxattr', 'removexattr',
		'statvfs', 'readlink', 'access', 'scandir', 'mkdir', 'unlink', 'rename', 'fstat', 'fchmod', 'fchown']
	counts = collections.Counter()
	with monkeypatch.context() as patch:
		for name in names:
			real = getattr(os, name, None)
			if real is None:
				continue
			@functools.wraps(real)
			def counted(*args, _name=name, _real=real, **kwargs):
				counts[_name] += 1
				return _real(*args, **kwargs)
			patch.setattr(os, name, counted)
		for folder in folders:
			hpcp.sync_directory_metadata(folder, [os.path.join(str(out), os.path.relpath(folder, str(src)))])
	return sum(counts.values()) / len(folders)


@posix_only
def test_directory_calls_stay_within_9_86_s(tmp_path, monkeypatch, reset_hpcp_globals):
	# 9.86 (the plan's probe p3): 17.1 os calls per directory on a first copy, 18.1 on a rerun (23.0 on Python 3.6)
	src = tmp_path / 'src'
	for d in range(4):
		for e in range(5):
			(src / ('d%d' % d) / ('e%d' % e)).mkdir(parents=True)
	(tmp_path / 'out').mkdir()
	monkeypatch.setattr(hpcp, 'NAMED_DESTS', frozenset([str(tmp_path / 'out')]))
	assert _calls_per_directory(tmp_path, monkeypatch, 'first') <= 17.1
	assert _calls_per_directory(tmp_path, monkeypatch, 'rerun') <= (23.0 if sys.version_info < (3, 7) else 18.1)


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
		assert events == [('fsync', str(dest))] and stat.S_IMODE(os.lstat(str(dest)).st_mode) == 0o333
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
