'''hpcp 9.90: the deferred entries of 9.88 and 9.89 (docs/limitations.md), each test failing on 9.89 (ae258b0) unless it
says it guards a case 9.89 already gets right.'''
import contextlib
import errno
import io
import os
import stat
import subprocess
import sys

import pytest

import hpcp
from test_held import _RUNS, _w1_run, fresh, read, write  # noqa: F401 (fresh is a fixture)

HPCP = hpcp.__file__
posix_only = pytest.mark.skipif(os.name != 'posix', reason='POSIX')
linux_only = pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux')
not_root = pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() == 0, reason='root may do it')



def _run_named(monkeypatch, *named):
	'''One run's settings: named (-d each) and their pin, a run of its own.'''
	monkeypatch.setattr(hpcp, 'NAMED_DESTS', frozenset(os.path.abspath(path) for path in named))
	monkeypatch.setattr(hpcp, 'RUN_ID', '990-%d' % next(_RUNS))
	monkeypatch.setattr(hpcp, 'NAMED_IDS', hpcp._named_ids(hpcp.NAMED_DESTS))


def _copy(src, dests):
	with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
		return hpcp.copy_file(str(src), [str(dest) for dest in dests])


def _files(root):
	return {os.path.relpath(os.path.join(d, name), str(root)): read(os.path.join(d, name))
		for d, _, names in os.walk(str(root)) for name in names}


# -- A rerun to several destinations: the door's answers are duplicated only where two landings can share a cursor --------

@posix_only
def test_990_two_destinations_side_by_side_lend_the_doors_answers(tmp_path, fresh, monkeypatch):
	# 9.89 duplicated (and closed) each landing's directory descriptor whenever a file had several landings; two -d side by
	# side never share a door cursor, so 9.90 lends both (no dup), as with one destination.
	write(tmp_path / 'src' / 'a' / 'f', 'F')
	for out in ('out', 'out2'):
		os.makedirs(str(tmp_path / out / 'a'))
	_run_named(monkeypatch, tmp_path / 'out', tmp_path / 'out2')
	dups, real_dup = [], os.dup
	def dup(fd):
		dups.append(fd)
		return real_dup(fd)
	monkeypatch.setattr(os, 'dup', dup)
	try:
		_copy(tmp_path / 'src' / 'a' / 'f', [tmp_path / 'out' / 'a' / 'f', tmp_path / 'out2' / 'a' / 'f'])
	finally:
		monkeypatch.setattr(os, 'dup', real_dup)
		hpcp._close_cursors()
	assert read(tmp_path / 'out' / 'a' / 'f') == 'F' and not hpcp.ERRORS, hpcp.ERRORS
	assert dups == [], dups


@posix_only
def test_990_nested_destinations_still_duplicate(tmp_path, fresh, monkeypatch):
	# Guards what 9.89 gets right: one -d inside another shares a cursor (the 9.89 two-landings test), so the answers are dups.
	write(tmp_path / 'src' / 'x' / 'a' / 'b' / 'f', 'F')
	out = str(tmp_path / 'out')
	for directory in ('x/a/b', 'x/x/a/b'):
		os.makedirs(os.path.join(out, directory))
	_run_named(monkeypatch, out, os.path.join(out, 'x'))
	assert getattr(hpcp, '_nested_named', lambda: True)()
	try:
		_copy(tmp_path / 'src' / 'x' / 'a' / 'b' / 'f', [os.path.join(out, 'x/a/b/f'), os.path.join(out, 'x/x/a/b/f')])
	finally:
		hpcp._close_cursors()
	assert _files(out) == {'x/a/b/f': 'F'} and not hpcp.ERRORS, hpcp.ERRORS


@posix_only
@pytest.mark.parametrize('rerun', [False, True])
def test_990_a_named_link_landing_below_another_destination_keeps_its_directory(tmp_path, fresh, monkeypatch, rerun):
	# The re-review's case: -d out/l -d other/, l -> ../other/a/b/real. Neither -d lies below the other, but the link's landing
	# (named, followed at the copy) lies below other/, whose cursor the second landing walks: lent, the answer for out/l's
	# directory was closed by the second door call, and the file went to other/f (exit 0, no message). Named: dup'd.
	write(tmp_path / 'src' / 'f', 'NEW')
	os.makedirs(str(tmp_path / 'other' / 'x' / 'y'))
	write(tmp_path / 'other' / 'a' / 'b' / 'real', 'NEW' if rerun else 'old')
	if rerun:
		st = os.stat(str(tmp_path / 'src' / 'f'))
		os.utime(str(tmp_path / 'other' / 'a' / 'b' / 'real'), ns=(st.st_atime_ns, st.st_mtime_ns))
	os.makedirs(str(tmp_path / 'out'))
	os.symlink('../other/a/b/real', str(tmp_path / 'out' / 'l'))
	link, other = str(tmp_path / 'out' / 'l'), str(tmp_path / 'other')
	_run_named(monkeypatch, link, other)
	assert not getattr(hpcp, '_nested_named', lambda: False)()
	try:
		hpcp._dest_directory(os.path.join(other, 'x', 'y'), make=False)  # the cursor elsewhere below other/ first
		_copy(tmp_path / 'src' / 'f', [link, os.path.join(other, 'f')])
	finally:
		hpcp._close_cursors()
	assert _files(other) == {'a/b/real': 'NEW'} and not hpcp.ERRORS, (_files(other), hpcp.ERRORS)


# -- A -d link retargeted between a failed first attempt and its retry -----------------------------------------------------

@posix_only
def test_990_a_retry_goes_where_the_first_attempt_went(tmp_path, fresh, monkeypatch):
	# Probe C of the 9.89 final review's leads.py: -d out/l, l -> ../other/real; the first attempt fails and l is pointed at
	# ../third/name2 before the retry. 9.89 worked the landing out again for the retry and wrote other/name2 (the new name in
	# the old directory). The landing is worked out once a file: the retry writes other/real.
	write(tmp_path / 'src' / 'f', 'NEW')
	for directory, name in (('other', 'real'), ('other', 'name2'), ('third', 'name2')):
		write(tmp_path / directory / name, 'old %s/%s' % (directory, name))
	os.makedirs(str(tmp_path / 'out'))
	link = str(tmp_path / 'out' / 'l')
	os.symlink('../other/real', link)
	_run_named(monkeypatch, link)
	real, calls = hpcp._cp_once, []
	def once(*args, **kwargs):
		calls.append(1)
		if len(calls) == 1:
			os.unlink(link)
			os.symlink('../third/name2', link)
			raise OSError(errno.ENOSPC, 'No space left on device', args[1])
		return real(*args, **kwargs)
	monkeypatch.setattr(hpcp, '_cp_once', once)
	try:
		_copy(tmp_path / 'src' / 'f', [link])
	finally:
		hpcp._close_cursors()
	assert len(calls) == 2
	assert read(tmp_path / 'other' / 'real') == 'NEW', hpcp.ERRORS
	assert read(tmp_path / 'other' / 'name2') == 'old other/name2' and read(tmp_path / 'third' / 'name2') == 'old third/name2'


# -- A device node copied by a non-root user onto a -d link ------------------------------------------------------------------

@posix_only
@not_root
def test_990_a_device_node_onto_a_named_link_says_it_cannot_make_the_node(tmp_path):
	# 9.88 and 9.89 looked the landing up by the link's own name in the target's directory, found nothing, did not ask, and
	# left cp to fail on the copy beside ("the new copy beside it failed"); 9.87 said it could not make the node.
	if not os.path.exists('/dev/null') or not stat.S_ISCHR(os.stat('/dev/null').st_mode):
		pytest.skip('no /dev/null device here')
	write(tmp_path / 'other' / 'real', 'keep')
	os.makedirs(str(tmp_path / 'out'))
	os.symlink('../other/real', str(tmp_path / 'out' / 'l'))
	p = subprocess.run([sys.executable, HPCP, '/dev/null', '-d', str(tmp_path / 'out' / 'l')], stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT, universal_newlines=True, cwd=str(tmp_path))
	assert p.returncode == 190, p.stdout
	assert 'cannot make a device node like /dev/null here' in p.stdout, p.stdout
	assert 'beside' not in p.stdout, p.stdout
	assert read(tmp_path / 'other' / 'real') == 'keep' and os.readlink(str(tmp_path / 'out' / 'l')) == '../other/real'
	assert sorted(os.listdir(str(tmp_path / 'other'))) == ['real']


# -- -lt: a link whose target directory holds it, and whose copy fails ---------------------------------------------------------

@posix_only
def test_990_link_tracking_tries_a_failing_target_once(tmp_path):
	# src/k/a/a -> ../a, and out/k/a a file, so the tracked target's copy fails every time: 9.89 found the link nested in the
	# target again after each attempt and tracked it again, until Python's recursion limit (992 'Copy failed' lines here,
	# about 20,000 lines in all). The target is tried once.
	os.makedirs(str(tmp_path / 'src' / 'k' / 'a'))
	os.symlink('../a', str(tmp_path / 'src' / 'k' / 'a' / 'a'))
	write(tmp_path / 'src' / 'k' / 'a' / 'g', 'x')
	write(tmp_path / 'out' / 'k' / 'a', 'file')
	p = subprocess.run([sys.executable, HPCP, 'src/k', '-d', 'out/', '-lt'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
		universal_newlines=True, cwd=str(tmp_path))
	assert p.returncode == 213, p.stdout[-2000:]
	failed = [line for line in p.stdout.splitlines() if line.startswith('Copy failed')]
	assert 1 <= len(failed) <= 3, (len(failed), p.stdout[-2000:])
	assert read(tmp_path / 'out' / 'k' / 'a') == 'file'


# -- A serial run's progress bar ------------------------------------------------------------------------------------------------

@posix_only
def test_990_a_serial_copy_draws_a_frame_at_most_each_refresh_and_the_last_at_the_total(tmp_path, fresh, monkeypatch):
	# 9.89 drew a frame after every file of a serial run (rate_limit), a third of a serial rerun's instructions a file; one
	# every 0.1 s is what a terminal shows anyway. The frame at the total is always drawn: it ends the bar's line.
	count = 300
	for index in range(count):
		write(tmp_path / 'src' / ('f%03d' % index), 'x')
	frames, real = [], hpcp.multiCMD.print_progress_bar
	def frame(*args, **kwargs):
		frames.append(args[:2])
	monkeypatch.setattr(hpcp.multiCMD, 'print_progress_bar', frame)
	with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
		hpcp.copy_files_serial(str(tmp_path / 'src'), [str(tmp_path / 'out')])
	assert len(os.listdir(str(tmp_path / 'out'))) == count
	assert frames and frames[-1] == (count, count), frames[-3:]
	assert len(frames) < count // 4, len(frames)


# -- A pinned destination refused in one process while another destination takes the file ------------------------------------

@posix_only
def test_990_a_pin_refused_where_another_destination_takes_the_file_is_said_once(tmp_path, fresh, monkeypatch):
	# -d B/ -d A/, B swapped for a link to another directory mid-run: a process that opens B anew (forkserver, spawn) finds it
	# leading elsewhere and writes nothing below it (9.88's pin), and the files go to A. 9.89 said nothing then (exit 0 with
	# -nds); 9.90 says it once a process, as a note: the files are whole where they went.
	for name in ('f1', 'f2'):
		write(tmp_path / 'src' / name, name)
	B, A, victim = (str(tmp_path / name) for name in ('B', 'A', 'victim'))
	for directory in (B, A, victim):
		os.makedirs(directory)
	_run_named(monkeypatch, B, A)
	os.rename(B, B + '.away')
	os.symlink(victim, B)
	hpcp._close_cursors()  # a worker that opens the destinations itself
	out = io.StringIO()
	try:
		with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
			for name in ('f1', 'f2'):
				hpcp.copy_file(str(tmp_path / 'src' / name), [os.path.join(B, name), os.path.join(A, name)])
	finally:
		hpcp._close_cursors()
	assert sorted(os.listdir(A)) == ['f1', 'f2'] and os.listdir(victim) == [] and os.listdir(B + '.away') == []
	assert not hpcp.ERRORS, hpcp.ERRORS
	notes = [line for line in out.getvalue().splitlines() if 'now leads to another directory' in line]
	assert len(notes) == 1 and notes[0].startswith('Note: ') and 'goes to another destination' in notes[0], out.getvalue()


# -- _fsync_file's chmod of a file its owner may not read --------------------------------------------------------------------

@posix_only
@not_root
def test_990_the_fsync_of_a_0044_file_never_chmods_through_a_link_put_at_its_name(tmp_path, monkeypatch):
	# A move's fsync of a file of mode 0044 gives its owner read for the open (a chmod by name) and takes it back. 9.89's chmod
	# followed a link put at the name after its lstat: the link's target (the runner's own file) was made 0444, then 0044.
	f, victim = str(tmp_path / 'f'), str(tmp_path / 'victim')
	write(f, 'data', mode=0o044)
	write(victim, 'mine', mode=0o600)
	real_open, swapped = os.open, []
	def opening(name, flags, *args, **kwargs):
		try:
			return real_open(name, flags, *args, **kwargs)
		except PermissionError:
			if name == f and flags & os.O_WRONLY and not swapped:
				swapped.append(True)
				os.unlink(f)
				os.symlink(victim, f)
			raise
	monkeypatch.setattr(os, 'open', opening)
	with pytest.raises(OSError):
		hpcp._fsync_file(f)
	monkeypatch.setattr(os, 'open', real_open)
	assert swapped
	assert stat.S_IMODE(os.stat(victim).st_mode) == 0o600


# -- A move's directory fsyncs ---------------------------------------------------------------------------------------------------

@posix_only
def test_990_a_moves_directory_fsync_never_goes_through_a_link_on_the_way(tmp_path, fresh, monkeypatch):
	# 9.89 fsynced each directory a move wrote into by its path (_fsync_file): out/a swapped for a link to another directory
	# after the copy, the fsync of out/a/b went to the link's target's b. Through the door's look (_dest_directory, nothing
	# made or replaced) a link on the way is never followed, and left as it is.
	os.makedirs(str(tmp_path / 'src'))
	out, victim = str(tmp_path / 'out'), str(tmp_path / 'victim')
	os.makedirs(os.path.join(out, 'a', 'b'))
	os.makedirs(os.path.join(victim, 'b'))
	_run_named(monkeypatch, out)
	os.rename(os.path.join(out, 'a'), os.path.join(out, 'a.away'))
	os.symlink(victim, os.path.join(out, 'a'))
	synced, real = [], os.fsync
	def fsync(fd):
		st = os.fstat(fd)
		synced.append((st.st_dev, st.st_ino))
		return real(fd)
	monkeypatch.setattr(os, 'fsync', fsync)
	try:
		with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
			hpcp._sync_moved_directories([], {os.path.join(out, 'a', 'b')}, [str(tmp_path / 'src') + os.sep], [out + os.sep], False, [], 1)
	finally:
		monkeypatch.setattr(os, 'fsync', real)
		hpcp._close_cursors()
	for directory in (victim, os.path.join(victim, 'b')):
		st = os.stat(directory)
		assert (st.st_dev, st.st_ino) not in synced, (directory, synced)
	assert os.listdir(victim) == ['b'] and os.listdir(os.path.join(victim, 'b')) == []
	assert os.readlink(os.path.join(out, 'a')) == victim
	out_st = os.stat(out)
	assert (out_st.st_dev, out_st.st_ino) in synced


# -- A landing held under a write lease ---------------------------------------------------------------------------------------

_LEASE_HOLDER = r'''
import fcntl, os, signal, sys, time
fd = os.open(sys.argv[1], os.O_RDONLY)
def give_up(signum, frame):
	fcntl.fcntl(fd, fcntl.F_SETLEASE, fcntl.F_UNLCK)
	sys.stdout.write('released\n')
	sys.stdout.flush()
signal.signal(signal.SIGIO, give_up)
try:
	fcntl.fcntl(fd, fcntl.F_SETLEASE, fcntl.F_RDLCK)
except OSError as e:
	sys.stdout.write('no lease: %s\n' % e)
	sys.exit(0)
sys.stdout.write('held\n')
sys.stdout.flush()
time.sleep(30)
'''


@linux_only
def test_990_a_rewrite_of_a_leased_landing_waits_for_the_break_and_keeps_the_inode(tmp_path, fresh, monkeypatch):
	# 9.86 to 9.89 opened the landing non-blocking (never waiting on a FIFO), met EAGAIN where another program held a lease on
	# it (Samba's oplocks, an NFS server's delegation), and rewrote it beside: a new inode, the lease holder keeping the old
	# file. The open now waits for the lease's break and writes in place.
	if not hasattr(__import__('fcntl'), 'F_SETLEASE'):
		pytest.skip('no fcntl leases here')
	write(tmp_path / 'src' / 'f', 'new content')
	landing = write(tmp_path / 'out' / 'f', 'old')
	os.utime(landing, ns=(1_000_000_000, 1_000_000_000))
	before = os.stat(landing)
	holder = subprocess.Popen([sys.executable, '-c', _LEASE_HOLDER, landing], stdout=subprocess.PIPE, universal_newlines=True)
	try:
		first = holder.stdout.readline().strip()
		if first != 'held':
			pytest.skip('a lease cannot be taken here: %s' % first)
		_w1_run(monkeypatch, str(tmp_path / 'out'))
		try:
			_copy(tmp_path / 'src' / 'f', [landing])
		finally:
			hpcp._close_cursors()
		assert holder.stdout.readline().strip() == 'released'
	finally:
		holder.kill()
		holder.wait()
	after = os.stat(landing)
	assert read(landing) == 'new content' and not hpcp.ERRORS, hpcp.ERRORS
	assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)


# -- A pool worker stopped while it reads a queued job --------------------------------------------------------------------------

_READER = r'''
import os, signal, sys
sys.path.insert(0, sys.argv[2])
import hpcp
from multiprocessing.connection import Connection
reader = Connection(int(sys.argv[1]), writable=False)
hpcp._MAIN_PID = -1  # this process is a pool worker to hpcp's handler, outside a task
signal.signal(signal.SIGINT, hpcp._stop)
sys.stdout.write('ready\n')
sys.stdout.flush()
data = reader.recv_bytes()
sys.stdout.write('got %d\n' % len(data))
sys.stdout.flush()
'''


@linux_only
def test_990_a_worker_stopped_after_a_jobs_length_reads_the_rest(tmp_path):
	# A stop's SystemExit landed in an idle worker between a queued job's 4-byte length and its payload (Connection._recv_bytes
	# reads them in two calls), leaving the payload in the pipe; the next worker read its first bytes as a length and printed
	# 'EOFError: Ran out of input' (Python 3.6, forkserver or spawn, about 1 in 1,600 early stops). Past the length, the stop
	# is recorded and the read finishes (the worker then ends at the job's start, _run_task).
	import fcntl
	import struct
	import termios
	import time
	readable, writable = os.pipe()
	reader = subprocess.Popen([sys.executable, '-c', _READER, str(readable), os.path.dirname(HPCP)], pass_fds=(readable,),
		stdout=subprocess.PIPE, universal_newlines=True)
	try:
		assert reader.stdout.readline().strip() == 'ready'
		os.write(writable, struct.pack('!i', 1000))
		deadline = time.monotonic() + 30  # milliseconds on an idle host; the bound is for a loaded one
		while struct.unpack('i', fcntl.ioctl(readable, termios.FIONREAD, b'\0\0\0\0'))[0] and time.monotonic() < deadline:
			time.sleep(0.01)  # until the reader has taken the length
		time.sleep(0.05)
		reader.send_signal(2)
		time.sleep(0.2)
		os.write(writable, b'x' * 1000)
		os.close(writable)
		output = reader.stdout.read()
		reader.wait(30)
	finally:
		os.close(readable)
		if reader.poll() is None:
			reader.kill()
			reader.wait()
	assert 'got 1000' in output and reader.returncode == 0, (output, reader.returncode)


# -- A directory whose mode shuts its owner out (0607) -------------------------------------------------------------------------

@posix_only
@not_root
@pytest.mark.parametrize('route', ['proc', 'read'])
def test_990_the_door_enters_the_runners_own_directory_shut_to_its_owner(tmp_path, fresh, monkeypatch, route):
	# P1: a non-root copy or move gives a directory of mode 0607 that mode at its end; a rerun's folder sync of a subdirectory
	# that ran before the directory's own start met EACCES opening below it (exit 193). The door gives the runner's own such
	# directory owner search where it needs it (the finish sets the exact mode again).
	if route == 'proc' and not hpcp._PATH_FDS:
		pytest.skip('no /proc/self/fd here')
	monkeypatch.setattr(hpcp, '_PATH_FDS', route == 'proc')
	out = str(tmp_path / 'out')
	os.makedirs(os.path.join(out, 'd', 'sub'))
	os.chmod(os.path.join(out, 'd'), 0o607)
	_w1_run(monkeypatch, out)
	try:
		held = hpcp._dest_directory(os.path.join(out, 'd', 'sub'), make=False)
		assert isinstance(held, int) and os.path.samestat(os.fstat(held), os.stat(os.path.join(out, 'd', 'sub')))
	finally:
		hpcp._close_cursors()
		mode = stat.S_IMODE(os.stat(os.path.join(out, 'd')).st_mode)
		os.chmod(os.path.join(out, 'd'), 0o755)
	assert mode == 0o707, oct(mode)


_root_only = pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() != 0, reason='needs root to make another user\'s tree')


@posix_only
@_root_only
@pytest.mark.parametrize('remove', [False, True])
def test_990_a_rerun_over_a_0607_directory_with_many_jobs_below_it_completes(tmp_path, remove):
	# P1 as an operator meets it: src/d 0607, root's, with 200 subdirectories (more than one pool job's 64 folders); a copy as
	# nobody gives out/d 0607, and the rerun's folder sync of a subdirectory before out/d's own start met EACCES (9.87 failed
	# every rerun; 9.88 and 9.89 some, by the pool's order).
	import shutil
	from test_directories import _as_nobody, _shut_out_tree
	work = _shut_out_tree()
	try:
		for index in range(200):
			path = os.path.join(work, 'src', 'd', 's%03d' % index)
			os.mkdir(path)
			os.chmod(path, 0o777)
			write(os.path.join(path, 'f'), 'x', mode=0o666)
		args = ['src/', '-d', 'out/']
		rc, output = _as_nobody(args, work)
		assert rc == 0, output[-3000:]
		out = os.path.join(work, 'out', 'd')
		assert stat.S_IMODE(os.lstat(out).st_mode) == 0o607
		for _ in range(3):
			rc, output = _as_nobody(args + (['-rf'] if remove else []), work)
			assert rc == 0, output[-3000:]
			assert stat.S_IMODE(os.lstat(out).st_mode) == 0o607, oct(os.lstat(out).st_mode)
			if remove:
				break
	finally:
		shutil.rmtree(work, ignore_errors=True)


# -- macOS ------------------------------------------------------------------------------------------------------------------------

darwin_only = pytest.mark.skipif(sys.platform != 'darwin', reason='macOS')


def _hpcp_cli(args, cwd):
	p = subprocess.run([sys.executable, HPCP] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
		cwd=str(cwd))
	return p.returncode, p.stdout


@darwin_only
def test_990_macos_raises_the_soft_open_files_limit():
	# hpcp raises the soft limit to the hard one at import; where macOS refuses that (a hard limit of RLIM_INFINITY, which
	# setrlimit takes for the soft one on macOS 26 but not everywhere), to OPEN_MAX (10240), not 256 (the door went by path
	# below 128 levels). Guards what 9.89 already does on macOS 26.
	p = subprocess.run([sys.executable, '-c', 'import resource, sys; sys.path.insert(0, %r); import hpcp; '
		'print(resource.getrlimit(resource.RLIMIT_NOFILE)[0])' % os.path.dirname(HPCP)], stdout=subprocess.PIPE,
		universal_newlines=True)
	assert int(p.stdout.split()[-1]) >= 10240, p.stdout


@darwin_only
def test_990_macos_copies_a_named_socket(tmp_path):
	# A socket the operator names (a walk skips one: 'Special file skipped'): macOS's cp skips it ("is a socket (not copied)"),
	# and 9.89 reported a failed copy, exit 213; Linux's cp -a makes the node, exit 0. hpcp makes it on macOS.
	import socket
	os.makedirs(str(tmp_path / 'src'))
	write(tmp_path / 'src' / 'f', 'x')
	sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
	try:
		sock.bind(str(tmp_path / 'src' / 's'))
	finally:
		sock.close()
	os.chmod(str(tmp_path / 'src' / 's'), 0o640)
	os.makedirs(str(tmp_path / 'out'))
	for _ in range(2):  # a first copy, then a rerun that finds it identical
		rc, output = _hpcp_cli(['src/s', '-d', 'out/'], tmp_path)
		assert rc == 0, output[-3000:]
		st = os.lstat(str(tmp_path / 'out' / 's'))
		assert stat.S_ISSOCK(st.st_mode) and stat.S_IMODE(st.st_mode) == 0o640, oct(st.st_mode)


@darwin_only
def test_990_macos_a_link_named_with_d_gets_no_bsd_flags_its_directory_does(tmp_path):
	# 9.88 and 9.89 set a directory's BSD flags by name with follow_symlinks=False: -d out/, out a link to real, gave the link
	# the source's nodump and real none. Through the held descriptor (fchflags) they go to real.
	os.makedirs(str(tmp_path / 'src'))
	write(tmp_path / 'src' / 'f', 'x')
	os.chflags(str(tmp_path / 'src'), stat.UF_NODUMP)
	os.makedirs(str(tmp_path / 'real'))
	os.symlink('real', str(tmp_path / 'out'))
	try:
		rc, output = _hpcp_cli(['src/', '-d', 'out/'], tmp_path)
		assert rc == 0, output[-3000:]
		assert os.stat(str(tmp_path / 'real')).st_flags & stat.UF_NODUMP
		assert not os.lstat(str(tmp_path / 'out')).st_flags & stat.UF_NODUMP
	finally:
		os.chflags(str(tmp_path / 'src'), 0)


def _mac_tag(path):
	"""The org.example.tag xattr of path by macOS's xattr tool, or None."""
	p = subprocess.run(['xattr', '-p', 'org.example.tag', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
	return p.stdout.strip() if p.returncode == 0 else None


def _mac_acl(path):
	"""path's ACL entries as ls -led shows them (the lines after the first)."""
	return subprocess.run(['ls', '-led', path], stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout.splitlines()[1:]


@darwin_only
def test_990_macos_a_rerun_carries_changed_xattrs_and_acls(tmp_path):
	# 9.89 read no xattr or ACL on macOS (no os.listxattr there): a rerun that found a landing identical left the source's
	# changed xattrs and ACL behind, and a directory got none. Through libc they reach the file and its directory.
	os.makedirs(str(tmp_path / 'src' / 'd'))
	f = write(tmp_path / 'src' / 'd' / 'f', 'x')
	rc, output = _hpcp_cli(['src/', '-d', 'out/'], tmp_path)
	assert rc == 0, output[-3000:]
	landing, landing_dir = str(tmp_path / 'out' / 'd' / 'f'), str(tmp_path / 'out' / 'd')
	for path in (f, str(tmp_path / 'src' / 'd')):
		subprocess.run(['xattr', '-w', 'org.example.tag', 'one', path], check=True)
		subprocess.run(['chmod', '+a', 'everyone deny delete', path], check=True)
	rc, output = _hpcp_cli(['src/', '-d', 'out/'], tmp_path)
	assert rc == 0, output[-3000:]
	for path in (landing, landing_dir):
		assert _mac_tag(path) == 'one', path
		assert any('deny delete' in line for line in _mac_acl(path)), (path, _mac_acl(path))
	for path in (f, str(tmp_path / 'src' / 'd')):
		subprocess.run(['xattr', '-d', 'org.example.tag', path], check=True)
		subprocess.run(['chmod', '-N', path], check=True)
	rc, output = _hpcp_cli(['src/', '-d', 'out/'], tmp_path)
	assert rc == 0, output[-3000:]
	for path in (landing, landing_dir):
		assert _mac_tag(path) is None, path
		assert _mac_acl(path) == [], (path, _mac_acl(path))


@linux_only
def test_990_a_read_that_returned_bytes_counts_as_part_of_a_job(tmp_path):
	# The window the first 9.90 build missed (2 'EOFError' blocks in 2,400 early stops on Python 3.6 spawn): a stop is
	# handled after a read returns, before its bytes are counted, so a worker that has just read the 4-byte length (or some of
	# the payload) is mid-message although nothing else in its frames says so yet. Asked at every line of the read.
	import multiprocessing.connection
	import struct
	import threading
	readable, writable = multiprocessing.Pipe(duplex=False)
	answers = []
	recv, recv_bytes = hpcp._RECV_CODE, hpcp._RECV_BYTES_CODE
	def tracer(frame, event, arg):
		if frame.f_code in (recv, recv_bytes):
			if event == 'line':
				answers.append((frame.f_code.co_name, frame.f_lineno, sorted(frame.f_locals), hpcp._in_message_read(frame)))
			return tracer
		return None
	writable.send_bytes(b'x' * 100)
	sys.settrace(tracer)
	try:
		data = readable.recv_bytes()
	finally:
		sys.settrace(None)
	assert data == b'x' * 100
	# Before the first read of the length nothing is consumed; from the moment that read returned, everything is.
	first_read = [i for i, (name, _, names, _) in enumerate(answers) if name == '_recv' and 'chunk' in names][0]
	assert not any(answer for _, _, _, answer in answers[:first_read]), answers[:first_read]
	assert all(answer for _, _, _, answer in answers[first_read:]), [a for a in answers[first_read:] if not a[3]]
