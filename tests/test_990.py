'''hpcp 9.90: the deferred entries of 9.88 and 9.89 (docs/limitations.md), each test failing on 9.89 (ae258b0) unless it
says it guards a case 9.89 already gets right.'''
import contextlib
import errno
import io
import os
import signal
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

# A subprocess here ends in under 2 s on an idle host (a copy of a few files); this is as long as test_directories gives a run of
# hpcp as another user, so a hung one fails its test instead of stopping the suite.
RUN_TIMEOUT = 120


def _cli(args, cwd):
	'''hpcp's command line in cwd: (return code, stdout and stderr together).'''
	p = subprocess.run([sys.executable, HPCP] + list(args), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT, universal_newlines=True, cwd=str(cwd), timeout=RUN_TIMEOUT)
	return p.returncode, p.stdout


@contextlib.contextmanager
def _within(seconds, what):
	'''Fail, not hang, where a call waits for something that never comes (a FIFO's writer, a pipe's data): SIGALRM raises.'''
	def expired(signum, frame):
		raise AssertionError(f'{what} did not finish in {seconds} s')
	previous = signal.signal(signal.SIGALRM, expired)
	signal.alarm(seconds)
	try:
		yield
	finally:
		signal.alarm(0)
		signal.signal(signal.SIGALRM, previous)


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
	assert hpcp._nested_named()
	dups, real_dup = [], os.dup
	def dup(fd):
		dups.append(fd)
		return real_dup(fd)
	monkeypatch.setattr(os, 'dup', dup)
	try:
		_copy(tmp_path / 'src' / 'x' / 'a' / 'b' / 'f', [os.path.join(out, 'x/a/b/f'), os.path.join(out, 'x/x/a/b/f')])
	finally:
		monkeypatch.setattr(os, 'dup', real_dup)
		hpcp._close_cursors()
	assert _files(out) == {'x/a/b/f': 'F'} and not hpcp.ERRORS, hpcp.ERRORS
	assert len(dups) == 2, dups  # one a landing


@posix_only
@pytest.mark.parametrize('rerun', [False, True])
def test_990_a_named_link_landing_below_another_destination_keeps_its_directory(tmp_path, fresh, monkeypatch, rerun):
	# Guards what 9.89 gets right (it dups every answer) against the lending of 9.90: -d out/l -d other/, l ->
	# ../other/a/b/real. Neither -d lies below the other, but the link's landing (named, followed at the copy) lies below other/,
	# whose cursor the second landing walks: lent, the answer for out/l's directory was closed by the second door call, and
	# the file went to other/f (exit 0, no message). A landing that is a named destination is dup'd.
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
	assert not hpcp._nested_named()
	try:
		hpcp._dest_directory(os.path.join(other, 'x', 'y'), make=False)  # the cursor elsewhere below other/ first
		_copy(tmp_path / 'src' / 'f', [link, os.path.join(other, 'f')])
	finally:
		hpcp._close_cursors()
	assert _files(other) == {'a/b/real': 'NEW'} and not hpcp.ERRORS, (_files(other), hpcp.ERRORS)


# -- A -d link retargeted between a failed first attempt and its retry -----------------------------------------------------

@posix_only
def test_990_a_retry_goes_where_the_first_attempt_went(tmp_path, fresh, monkeypatch):
	# -d out/l, l -> ../other/real; the first attempt fails and l is pointed at ../third/name2 before the retry. 9.89 worked the
	# landing out again for the retry and wrote other/name2 (the new name in the old directory). The landing is worked out once a
	# file: the retry writes other/real.
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
	rc, output = _cli(['/dev/null', '-d', str(tmp_path / 'out' / 'l')], tmp_path)
	assert rc == 190, output
	assert 'cannot make a device node like /dev/null here' in output, output
	assert 'beside' not in output, output  # the old message: "the new copy beside it failed"
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
	rc, output = _cli(['src/k', '-d', 'out/', '-lt'], tmp_path)
	assert rc == 213, output[-2000:]
	failed = [line for line in output.splitlines() if line.startswith('Copy failed')]
	# The file's failure is said for the walk's copy and for the tracked target's copy of it (2 here, 3 on one other run of the
	# builder's): a bound of 3 tells a few lines from 9.89's recursion (992).
	assert 1 <= len(failed) <= 3, (len(failed), output[-2000:])
	assert read(tmp_path / 'out' / 'k' / 'a') == 'file'


# -- A serial run's progress bar ------------------------------------------------------------------------------------------------

@posix_only
def test_990_a_serial_copy_draws_a_frame_at_most_each_refresh_and_the_last_at_the_total(tmp_path, fresh, monkeypatch):
	# 9.89 drew a frame after every file of a serial run (rate_limit), a third of a serial rerun's instructions a file; one
	# every 0.1 s is what a terminal shows anyway. The frame at the total is always drawn: it ends the bar's line.
	import time
	count = 300
	for index in range(count):
		write(tmp_path / 'src' / ('f%03d' % index), 'x')
	frames = []
	def frame(*args, **kwargs):
		frames.append((time.monotonic(), args[:2]))
	monkeypatch.setattr(hpcp.multiCMD, 'print_progress_bar', frame)
	with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
		hpcp.copy_files_serial(str(tmp_path / 'src'), [str(tmp_path / 'out')])
	assert len(os.listdir(str(tmp_path / 'out'))) == count
	assert frames and frames[-1][1] == (count, count), frames[-3:]
	# Every frame but the last is about a refresh after the one before it (0.1 s: half allowed, for the clock's
	# step); the frame count alone would hold on a fast host whatever the interval, and wall time bounds nothing on a slow one.
	gaps = [later[0] - earlier[0] for earlier, later in zip(frames, frames[1:-1])]
	assert all(gap >= 0.05 for gap in gaps), gaps
	assert len(frames) < count, len(frames)  # 9.89 drew one after every file


# -- A pinned destination refused in one process while another destination takes the file ------------------------------------

@posix_only
def test_990_a_pin_refused_where_another_destination_takes_the_file_is_said_once(tmp_path, fresh, monkeypatch):
	# -d B/ -d A/, B swapped for a link to another directory mid-run: a process that opens B anew (forkserver, spawn) finds it
	# leading elsewhere and writes nothing below it (9.88's pin), and the files go to A. 9.89 said nothing then (exit 0 with
	# -nds); 9.90 says it once a process, as a note of its own: the files are whole where they went.
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
	assert len(notes) == 1 and notes[0].startswith('Note: ') and 'the other destinations take its files' in notes[0], out.getvalue()
	assert 'not copying' not in notes[0], notes[0]  # the refusal's own text is for the 213 line


@posix_only
def test_990_a_refusal_another_destination_took_over_is_not_noted_again(tmp_path):
	# Guards 9.89 (it says nothing more) against 9.90's first build: -ncd with -d A/ -d B/ and B/sub a link: the directory sync refuses B/sub (213, its own line), and the file below it goes to
	# A. 9.90's first build then added a Note built from that refusal's text, "not copying FILE there, which goes to another
	# destination below it", which splits the sentence and says again what the 213 line said. Only a pin's refusal is noted.
	os.makedirs(str(tmp_path / 'src' / 'sub'))
	write(tmp_path / 'src' / 'sub' / 'f', 'x')
	os.makedirs(str(tmp_path / 'A' / 'sub'))
	os.makedirs(str(tmp_path / 'elsewhere'))
	os.makedirs(str(tmp_path / 'B'))
	os.symlink('../elsewhere', str(tmp_path / 'B' / 'sub'))
	rc, output = _cli(['src/', '-d', 'A/', '-d', 'B/', '-ncd', '-s'], tmp_path)
	assert rc == 213, output[-2000:]
	assert 'Note:' not in output and 'which goes to another destination' not in output, output[-2000:]
	assert read(tmp_path / 'A' / 'sub' / 'f') == 'x' and os.listdir(str(tmp_path / 'elsewhere')) == []


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
	holder = subprocess.Popen([sys.executable, '-c', _LEASE_HOLDER, landing], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
		universal_newlines=True)
	try:
		with _within(RUN_TIMEOUT, 'the lease holder'):
			first = holder.stdout.readline().strip()
		if first != 'held':
			pytest.skip('a lease cannot be taken here: %s' % first)
		_w1_run(monkeypatch, str(tmp_path / 'out'))
		try:
			with _within(RUN_TIMEOUT, 'the copy'):
				_copy(tmp_path / 'src' / 'f', [landing])
		finally:
			hpcp._close_cursors()
		with _within(RUN_TIMEOUT, 'the lease holder'):
			assert holder.stdout.readline().strip() == 'released'
	finally:
		holder.kill()
		holder.wait()
	after = os.stat(landing)
	assert read(landing) == 'new content' and not hpcp.ERRORS, hpcp.ERRORS
	assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)


def _lease_retry_with_a_swap(tmp_path):
	'''_open_after_lease on d/f, a regular file, with a FIFO renamed over d/f right after its look at the name (an lstat, or the
	O_PATH open that holds it) returns: (what it returned, d/f's stat as first looked at).'''
	f = write(tmp_path / 'd' / 'f', 'old')
	st = os.lstat(f)
	base = os.open(str(tmp_path / 'd'), os.O_RDONLY | os.O_DIRECTORY)
	real_lstat, real_open, swapped = os.lstat, os.open, []
	def swap():
		if not swapped:
			swapped.append(True)
			os.rename(f, f + '.old')
			os.mkfifo(f)
	def lstat(*args, **kwargs):
		result = real_lstat(*args, **kwargs)
		if sys._getframe(1).f_code.co_name == '_open_after_lease':
			swap()
		return result
	def opening(*args, **kwargs):
		result = real_open(*args, **kwargs)
		if sys._getframe(1).f_code.co_name == '_open_after_lease' and args[1] & getattr(os, 'O_PATH', 0):
			swap()
		return result
	os.lstat, os.open = lstat, opening
	try:
		with _within(10, 'the lease retry (blocked in the open of a FIFO)'):
			return hpcp._open_after_lease(base, 'f', st), st
	finally:
		os.lstat, os.open = real_lstat, real_open
		os.close(base)


@linux_only
def test_990_the_lease_retry_never_waits_on_a_fifo_put_at_the_name(tmp_path):
	# The retry opens without O_NONBLOCK, to wait for the lease's break; 9.86's flag was there so that nothing waits on a FIFO's
	# writer. The first 9.90 build looked at the name with an lstat and then opened it, and a FIFO renamed over it between the
	# two (a lease holder, told of the break by SIGIO, can aim at that moment) made the open wait for a writer for good. The name
	# is held by an O_PATH descriptor now, checked, and the open goes through its /proc/self/fd entry: the FIFO is never opened.
	if not hpcp._PATH_FDS:
		pytest.skip('no /proc/self/fd here')
	fd, st = _lease_retry_with_a_swap(tmp_path)
	try:
		assert fd is not None
		now = os.fstat(fd)
		assert stat.S_ISREG(now.st_mode) and (now.st_dev, now.st_ino) == (st.st_dev, st.st_ino)
	finally:
		if fd is not None:
			os.close(fd)
	assert stat.S_ISFIFO(os.lstat(str(tmp_path / 'd' / 'f')).st_mode)  # what was put there is as it was


@linux_only
def test_990_the_lease_retry_without_proc_goes_beside_as_9_89_did(tmp_path, monkeypatch):
	# Without /proc there is no name for the held file to open through: nothing is opened (the caller goes beside), where a
	# blocking open by name could wait on whatever is put there.
	monkeypatch.setattr(hpcp, '_PATH_FDS', False)
	f = write(tmp_path / 'd' / 'f', 'old')
	base = os.open(str(tmp_path / 'd'), os.O_RDONLY | os.O_DIRECTORY)
	try:
		assert hpcp._open_after_lease(base, 'f', os.lstat(f)) is None
	finally:
		os.close(base)


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
		stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, universal_newlines=True)
	try:
		with _within(RUN_TIMEOUT, 'the reader'):
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
		writable = None
		with _within(RUN_TIMEOUT, 'the reader'):
			output = reader.stdout.read()
			reader.wait(30)
	finally:
		os.close(readable)
		if writable is not None:
			os.close(writable)
		if reader.poll() is None:
			reader.kill()
			reader.wait()
	assert 'got 1000' in output and reader.returncode == 0, (output, reader.returncode)


# -- A socket the operator names, where cp copies none (macOS): skipped with a note, as one met in a tree is ----------------------

def _a_socket(path, mode=0o640):
	'''A socket at path (a short, relative one: a socket address holds about 100 bytes), of mode.'''
	import socket
	sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
	try:
		sock.bind(path)
	finally:
		sock.close()
	os.chmod(path, mode)
	return path


def _cli_as_macos_takes_a_socket(args, cwd, stdin=None):
	'''hpcp's command line in cwd where cp copies no socket: macOS itself, and elsewhere with _CP_SKIPS_SOCKETS set in the
	run's own process (the main process decides what is a source). (return code, stdout and stderr together).'''
	code = ('import sys; sys.path.insert(0, %r); import hpcp; hpcp._CP_SKIPS_SOCKETS = True; sys.argv[0] = "hpcp"; hpcp.main()'
		% os.path.dirname(HPCP))
	p = subprocess.run([sys.executable, '-c', code] + list(args), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
		universal_newlines=True, cwd=str(cwd), timeout=RUN_TIMEOUT, **({'stdin': subprocess.DEVNULL} if stdin is None else {'input': stdin}))
	return p.returncode, p.stdout


@contextlib.contextmanager
def _a_short_tree():
	'''A directory under /tmp to work in, holding src/s (a socket, 0640), src/f and out/; removed afterwards. The socket is bound by a
	relative path (an address holds 103 bytes on macOS, 107 on Linux), so the directory's own length does not matter.'''
	import shutil
	import tempfile
	work = tempfile.mkdtemp(prefix='s', dir='/tmp')
	here = os.getcwd()
	try:
		os.chdir(work)
		os.mkdir('src')
		os.mkdir('out')
		_a_socket('src/s')
		write(os.path.join(work, 'src', 'f'), 'x')
		yield work
	finally:
		os.chdir(here)
		shutil.rmtree(work, ignore_errors=True)


@posix_only
@pytest.mark.parametrize('run', ['copy', 'move', 'list'])
def test_990_a_named_socket_is_skipped_with_a_note_where_cp_copies_none(run):
	# macOS's cp copies no socket ("cp: X is a socket (not copied)", exit 0), and 9.89 then reported a copy that is at none of its
	# landings: exit 213, and a move kept every source (212). 9.90's builds made the socket themselves, by a bind in a directory
	# of their own known by its name, which another user who could write the landing's directory could swap, or enter through an
	# inherited ACL: the reviews found the socket's owner, mode and times given to another file, entries deleted, a socket left
	# elsewhere. Now a socket the operator names is what one met in a tree is on every platform: noted, no error, not copied,
	# and left in place by a move; what is at its landing stays, and nothing is made for it.
	with _a_short_tree() as work:
		landing = 'out/src/s' if run == 'list' else 'out/s'
		if run != 'list':
			write(os.path.join(work, 'out', 's'), 'the old file')
		args = {'copy': ['src/s', 'src/f', '-d', 'out/'], 'move': ['src/s', 'src/f', '-d', 'out/', '-rf'], 'list': ['-d', 'out/']}[run]
		rc, output = _cli_as_macos_takes_a_socket(args, work, stdin='src/s\nsrc/f\n' if run == 'list' else None)
		assert rc == 0, output[-3000:]
		notes = [line for line in output.splitlines() if line.startswith('Special file skipped:')]
		assert len(notes) == 1 and 'src/s is a socket' in notes[0], output[-3000:]
		assert stat.S_ISSOCK(os.lstat('src/s').st_mode)  # a move leaves it
		if run == 'list':
			assert not os.path.lexists(landing) and read(os.path.join(work, 'out', 'src', 'f')) == 'x'
			assert sorted(os.listdir('out')) == ['src'] and os.listdir('out/src') == ['f']
		else:
			assert read(os.path.join(work, landing)) == 'the old file' and read(os.path.join(work, 'out', 'f')) == 'x'
			assert sorted(os.listdir('out')) == ['f', 's']  # nothing made beside them
		assert os.path.lexists('src/f') != (run == 'move')


@posix_only
def test_990_a_run_whose_sources_are_all_sockets_copies_nothing_where_cp_copies_none():
	with _a_short_tree() as work:
		os.unlink('src/f')
		rc, output = _cli_as_macos_takes_a_socket(['src/s', '-d', 'out/', '-rf'], work)
		assert rc == 0 and 'Nothing to copy: every source is a socket' in output, output[-3000:]
		assert os.listdir('out') == [] and stat.S_ISSOCK(os.lstat('src/s').st_mode)


@posix_only
def test_990_a_named_socket_is_still_removed_by_a_delete_where_cp_copies_none():
	# Guards what 9.89 and 9.90's builds do: -rm with no destination deletes what is named, a socket too (only a copy leaves
	# one out).
	with _a_short_tree() as work:
		rc, output = _cli_as_macos_takes_a_socket(['src/s', '-rf'], work)
		assert rc == 0 and not os.path.lexists('src/s') and os.path.lexists('src/f'), output[-3000:]


# -- A directory's xattrs and ACL: every call skips the same errors -------------------------------------------------------------

_SKIPPED_ERRORS = ['EPERM', 'EACCES', 'EINVAL', 'ENODATA', 'ENOTSUP', 'EOPNOTSUPP']


def _xattr_site(tmp_path, monkeypatch, site, code):
	'''One xattr or ACL call of the copy (site) failing with errno code: the call that meets it must go on. 'settle-list' is the
	skip check's listing of an identical landing (_settle_identical), 'settle' its removal.'''
	src, dest = str(write(tmp_path / 'src', 'x')), str(write(tmp_path / 'dest', 'x'))
	def failing(*args, **kwargs):
		raise OSError(code, os.strerror(code))
	def nothing(*args, **kwargs):
		return None
	def listing(path, **kwargs):
		if site in ('list', 'settle-list'):
			failing()
		# 'set': the source has a name the landing lacks; 'remove' and 'settle': the landing has one the source lacks
		if site == 'set':
			return ['user.x'] if path == src else []
		return ['user.x'] if site in ('remove', 'settle') and path != src else []
	monkeypatch.setattr(hpcp, '_mac_listxattr', listing)  # _XATTRS[0] is it: the ACL is asked too
	monkeypatch.setattr(hpcp, '_XATTRS', (listing, lambda path, name, **kwargs: b'v', failing if site == 'set' else nothing,
		failing if site in ('remove', 'settle') else nothing))
	monkeypatch.setattr(hpcp, '_mac_sync_acl', failing if site == 'acl' else lambda src, fd: False)
	if site in ('settle', 'settle-list'):
		hpcp._settle_identical(src, os.lstat(src), dest, os.lstat(dest))
		return
	fd = os.open(dest, os.O_RDONLY)
	try:
		hpcp._sync_xattrs(src, fd if site == 'acl' else dest)
	finally:
		os.close(fd)


@posix_only
@pytest.mark.parametrize('error', _SKIPPED_ERRORS)
@pytest.mark.parametrize('site', ['list', 'set', 'remove', 'acl', 'settle', 'settle-list'])
def test_990_every_xattr_and_acl_call_skips_the_same_errors(tmp_path, monkeypatch, site, error):
	# Five calls each had an errno list of their own, and macOS answers EOPNOTSUPP (102) where ENOTSUP is 45. They share
	# hpcp._XATTR_SKIPPED now. Linux has one number for both, so here it fails on 9.89's and the first 9.90 build's lists only for
	# the errors one lacked (EPERM and EACCES at the listing, ENODATA at the ACL); the rest are guards, and the Mac's run of this
	# file is where EOPNOTSUPP differs. The skip check's listing of an identical landing (settle-list) skipped only ENOTSUP and
	# EOPNOTSUPP until the second fix round: macOS's listxattr answers EACCES for a file its owner may not read, so a rerun under
	# -hs 0 raised an error for it every time, where _sync_xattrs skips the same error for a directory.
	_xattr_site(tmp_path, monkeypatch, site, getattr(errno, error))


@posix_only
@pytest.mark.parametrize('site', ['remove', 'settle'])
def test_990_an_xattr_removal_skips_a_name_that_is_gone_since_the_listing(tmp_path, monkeypatch, site):
	# The removals also meet ENOENT (the shared set _XATTR_NOT_REMOVED); no case had it.
	_xattr_site(tmp_path, monkeypatch, site, errno.ENOENT)


@posix_only
def test_990_an_xattr_removal_also_skips_a_name_that_is_gone():
	assert hpcp._XATTR_NOT_REMOVED == hpcp._XATTR_SKIPPED | {errno.ENOENT}
	assert errno.EOPNOTSUPP in hpcp._XATTR_SKIPPED


@posix_only
def test_990_macos_marks_are_not_portable_names_a_rerun_removes(monkeypatch):
	# Mirroring the source removes a landing's name its source lacks, on macOS every one but the system's own: com.apple.system.*,
	# and the marks it keeps on a file by itself.
	monkeypatch.setattr(hpcp, '_XATTRS', (hpcp._mac_listxattr, None, None, None))
	for name in ('com.apple.quarantine', 'com.apple.lastuseddate#PS', 'com.apple.provenance', 'com.apple.macl',
			'com.apple.system.Security'):
		assert not hpcp._is_portable_xattr(name), name
	for name in ('org.example.mine', 'com.apple.metadata:_kMDItemUserTags', 'com.apple.FinderInfo'):
		assert hpcp._is_portable_xattr(name), name


@posix_only
@pytest.mark.parametrize('kind', ['fifo', 'regular'])
def test_990_an_identical_node_gets_no_xattrs_by_the_landings_full_path_on_macos(tmp_path, monkeypatch, kind):
	# R6: no xattr call on macOS takes a dir_fd, so an identical FIFO, socket or device node's xattrs went by the landing's full
	# path (_io gives dest where /proc is not there), through a directory on the way that another user swapped for a link in a
	# root run: they were removed from, and the source's set on, the file of that name where the link leads, while everything
	# else of its settle goes relative to the held directory. 9.89 carried no xattrs on macOS. A node gets none; a regular file
	# keeps getting them, through the descriptor hpcp checked. (macOS's shape here: _PATH_FDS off, macOS's listing.)
	src, dest = str(tmp_path / 'src'), str(tmp_path / 'dest')
	for path in (src, dest):
		if kind == 'fifo':
			os.mkfifo(path, 0o600)
		else:
			write(path, 'x', mode=0o600)
		os.utime(path, ns=(1_000_000_000, 2_000_000_000))
	listed = []
	def listing(path, **kwargs):
		listed.append(path)
		return []
	monkeypatch.setattr(hpcp, '_PATH_FDS', False)
	monkeypatch.setattr(hpcp, '_mac_listxattr', listing)
	monkeypatch.setattr(hpcp, '_XATTRS', (listing, lambda *args, **kwargs: b'', lambda *args, **kwargs: None,
		lambda *args, **kwargs: None))
	monkeypatch.setattr(hpcp, '_mac_sync_acl', lambda src, fd: False)
	assert hpcp._settle_identical(src, os.lstat(src), dest, os.lstat(dest))
	assert dest not in listed, listed  # never by the landing's path
	if kind == 'fifo':
		assert listed == [], listed
	else:
		assert [type(path) for path in listed] == [str, int], listed  # the source by its path, the landing by its descriptor


# -- The name probe before a copy, in a directory the runner's own mode shuts it out of ---------------------------------------------

def _can_lchmod_by_path():
	'''Whether this platform's chmod takes a path and says not to follow a link (macOS; glibc 2.32's fchmodat through /proc on
	Python 3.11 and later; not Python 3.6 on Linux).'''
	import tempfile
	with tempfile.TemporaryDirectory() as work:
		try:
			os.chmod(work, 0o700, follow_symlinks=False)
			return True
		except (OSError, NotImplementedError, ValueError):
			return False


@posix_only
@not_root
@pytest.mark.parametrize('route', ['proc', 'path', 'no-proc', 'read'])
@pytest.mark.parametrize('mode', [0o607, 0o407, 0o207, 0o007, 0o047])
def test_990_the_name_probe_gives_the_runners_own_directory_shut_to_its_owner_search_on_every_route(tmp_path, monkeypatch, route,
		mode):
	# Two names that fold alike (a and A) landing in a directory of the runner's own with a mode that shuts its owner out (0607,
	# which a non-root move gives one at its end and leaves after a skipped or stopped removal; on the read route also 0207, 0007
	# and 0047): the probe makes its names in the directory, and met EACCES ("nothing can be made in DIR to ask it (Permission
	# denied): make it writable and run again") on every rerun, where 9.89 had left 0707. The probe gives the directory owner
	# search as the door does. The routes: Linux with /proc (the probe holds the directory by O_PATH, and chmods through /proc);
	# macOS's (9.91: no O_PATH, a read descriptor for a mode with owner read, which the probe gives owner search by the path it
	# opened; by path, with lchmod by name, for one without it, as the door does); Linux without /proc (an O_PATH descriptor that
	# fchmod refuses: by path, as macOS's; no lchmod there, which a container's glibc still has through /proc); and the probe by
	# path throughout (Windows). The finish sets the exact mode again where it sets one: after the probe the directory has owner
	# rwx.
	if route == 'proc' and not (hpcp._PATH_FDS and hpcp._PROBE_BY_FD):
		pytest.skip('no /proc/self/fd here')
	if route == 'no-proc' and not hpcp._PROBE_BY_FD:
		pytest.skip('this platform has no O_PATH')
	if route != 'proc' and not mode & 0o400 and not _can_lchmod_by_path():
		pytest.skip('no chmod by name that does not follow a link here: the probe is refused (docs/limitations.md)')
	monkeypatch.setattr(hpcp, '_PATH_FDS', route == 'proc')
	monkeypatch.setattr(hpcp, '_ASK_BY_FD', route != 'path')
	if route == 'read':
		monkeypatch.setattr(hpcp, '_PARENT_FLAGS', os.O_RDONLY | os.O_DIRECTORY | getattr(os, 'O_CLOEXEC', 0))
	directory = tmp_path / 'd'
	directory.mkdir()
	os.chmod(str(directory), mode)
	try:
		answers, why = hpcp._ask_names(str(directory), ['a', 'A'])
	finally:
		now = stat.S_IMODE(os.stat(str(directory)).st_mode)
		os.chmod(str(directory), 0o755)
	assert not why and set(answers) == {'a', 'A'}, (answers, why)
	assert now == mode | 0o700, oct(now)
	assert os.listdir(str(directory)) == []  # the probe's own directory is gone


@posix_only
@not_root
def test_990_the_name_probe_of_a_directory_named_by_a_relative_path_gives_it_owner_read_by_name(tmp_path, monkeypatch):
	# The probe's directory can be a relative path (the destination as the operator typed it): its parent and name come from its
	# absolute path. macOS's route, a mode without owner read.
	if not _can_lchmod_by_path():
		pytest.skip('no chmod by name that does not follow a link here')
	monkeypatch.setattr(hpcp, '_PATH_FDS', False)
	monkeypatch.setattr(hpcp, '_ASK_BY_FD', True)
	monkeypatch.setattr(hpcp, '_PARENT_FLAGS', os.O_RDONLY | os.O_DIRECTORY | getattr(os, 'O_CLOEXEC', 0))
	monkeypatch.chdir(tmp_path)
	os.mkdir('d')
	os.chmod('d', 0o007)
	try:
		answers, why = hpcp._ask_names('d', ['a', 'A'])
	finally:
		now = stat.S_IMODE(os.stat('d').st_mode)
		os.chmod('d', 0o755)
	assert not why and set(answers) == {'a', 'A'}, (answers, why)
	assert now == 0o707, oct(now)


@posix_only
@not_root
def test_990_the_name_probe_still_refuses_a_directory_with_search_it_cannot_write(tmp_path):
	# Guards what 9.89 does: a 0555 directory has owner search and no write, so nothing is made in it and the refusal stands
	# (docs/limitations.md, "A directory that cannot be asked about a new pair"); the probe never makes a read-only directory
	# writable.
	directory = tmp_path / 'd'
	directory.mkdir()
	os.chmod(str(directory), 0o555)
	try:
		answers, why = hpcp._ask_names(str(directory), ['a', 'A'])
		now = stat.S_IMODE(os.stat(str(directory)).st_mode)
	finally:
		os.chmod(str(directory), 0o755)
	assert why and 'nothing can be made in' in sorted(why.values())[0], (answers, why)
	assert now == 0o555, oct(now)


# -- A directory whose mode shuts its owner out (0607) -------------------------------------------------------------------------

def _can_chmod_by_name_without_following():
	'''Whether this platform's chmod takes a name relative to a descriptor and says not to follow a link (macOS's lchmod, glibc 2.32's
	fchmodat on Python 3.11 and later; not Python 3.6 on Linux).'''
	import tempfile
	with tempfile.TemporaryDirectory() as work:
		os.mkdir(os.path.join(work, 'x'))
		fd = os.open(work, os.O_RDONLY | os.O_DIRECTORY)
		try:
			os.chmod('x', 0o700, dir_fd=fd, follow_symlinks=False)
			return True
		except (OSError, NotImplementedError, ValueError):
			return False
		finally:
			os.close(fd)


@posix_only
@not_root
@pytest.mark.parametrize('when', ['before the chmod', 'before the open'], ids=['before the chmod', 'before the open (a guard)'])
@pytest.mark.parametrize('mode', [0o007, 0o207, 0o047])
def test_990_the_owner_read_gives_a_hard_link_put_at_the_name_nothing_others_can_use(tmp_path, monkeypatch, mode, when):
	# N1: on the read route (macOS) out/d, the runner's own, has a mode without owner read (0007, 0207, 0047), and _owner_read
	# gave it its owner's rwx by a chmod by name. Another user who can write out/ swaps a hard link to one of the runner's files
	# in at the name (macOS has no protected_hardlinks; one renamex_np, or two renames, does it) after the lstat: the chmod did
	# not follow it, but a hard link is the file itself, so it got the directory's mode plus owner rwx, 0707 or 0747, and
	# others could write it. The name is given owner rwx alone first and the group and other bits go back through a descriptor
	# checked against the lstat, so a file swapped in has at most 0700. (Linux's glibc takes AT_SYMLINK_NOFOLLOW through /proc:
	# the route runs here with _PATH_FDS off.) "Before the open" is a guard, not a test of the fix: the open's O_DIRECTORY refuses a
	# file put at the name, so it passes on b7e892b's single chmod too; what reaches the (device, inode) check is a directory put
	# there, in the next test.
	if not _can_chmod_by_name_without_following():
		pytest.skip('no chmod by name that does not follow a link here: nothing changes')
	monkeypatch.setattr(hpcp, '_PATH_FDS', False)
	out = tmp_path / 'out'
	os.makedirs(str(out / 'd' / 'sub'))
	victim = write(tmp_path / 'victim', 'mine', mode=0o600)
	os.link(victim, str(out / 'lnk'))
	os.chmod(str(out / 'd'), mode)
	base = os.open(str(out), os.O_RDONLY | os.O_DIRECTORY)
	real_chmod, real_open, state = os.chmod, os.open, {'swapped': False, 'chmodded': False}
	def swap():
		state['swapped'] = True
		os.rename(str(out / 'd'), str(out / 'd.away'))  # the other user's renames
		os.rename(str(out / 'lnk'), str(out / 'd'))
	def chmod(path, new, *args, **kwargs):
		if kwargs.get('dir_fd') == base and not state['swapped'] and when == 'before the chmod':
			swap()
		result = real_chmod(path, new, *args, **kwargs)
		state['chmodded'] = state['chmodded'] or kwargs.get('dir_fd') == base
		return result
	def opening(path, flags, *args, **kwargs):
		if kwargs.get('dir_fd') == base and state['chmodded'] and not state['swapped'] and when == 'before the open':
			swap()
		return real_open(path, flags, *args, **kwargs)
	monkeypatch.setattr(os, 'chmod', chmod)
	monkeypatch.setattr(os, 'open', opening)
	try:
		with contextlib.suppress(OSError):
			hpcp._door_step(base, 'd', str(out / 'd'), False, None, 0)
	finally:
		monkeypatch.setattr(os, 'chmod', real_chmod)
		monkeypatch.setattr(os, 'open', real_open)
		os.close(base)
		os.chmod(str(out / 'd.away'), 0o700)
	assert state['swapped'], state
	assert stat.S_IMODE(os.stat(victim).st_mode) & 0o077 == 0, oct(os.stat(victim).st_mode)
	assert read(victim) == 'mine'


@posix_only
@not_root
def test_990_a_worker_that_meets_a_shut_directory_another_worker_has_just_opened_to_its_owner_opens_again(tmp_path, monkeypatch):
	# The pool's order: two workers walk below out/d, the runner's own and 0607. One's open of a name below it meets EACCES; by
	# the time it asks _owner_search the other has given out/d owner search, so _owner_search finds nothing to do (False) and the
	# walk raised EACCES for a directory that is open now (the rerun of a tree with 200 subdirectories exited 193 about one run in ten
	# on 3.11: the root-only test of that rerun below). It opens again, once.
	directory = tmp_path / 'd'
	os.makedirs(str(directory / 'sub'))
	os.chmod(str(directory), 0o607)
	flags = (hpcp._HOLD_PATH_FLAGS if hpcp._PATH_FDS else hpcp._HOLD_READ_FLAGS) | hpcp._O_NOFOLLOW
	base = os.open(str(directory), flags)
	real_search, state = hpcp._owner_search, {'calls': 0}
	def other_worker_first(fd):
		state['calls'] += 1
		os.chmod(str(directory), 0o707)  # the other worker's chmod, after this one's failed open
		return real_search(fd)
	monkeypatch.setattr(hpcp, '_owner_search', other_worker_first)
	try:
		sub = hpcp._door_step(base, 'sub', str(directory / 'sub'), False, None, 0)
		os.close(sub)
	finally:
		os.close(base)
		os.chmod(str(directory), 0o700)
	assert state['calls'] == 1, state


@posix_only
@not_root
@pytest.mark.parametrize('base_is', ['descriptor', 'path'])
@pytest.mark.parametrize('mode', [0o007, 0o207, 0o047])
def test_990_the_owner_read_leaves_a_directory_put_at_the_name_after_its_chmod_as_it_is(tmp_path, mode, base_is):
	# What reaches _owner_read's (device, inode) check: after the chmod of out/d to 0700, another directory is at its name by the
	# time of the open (the O_DIRECTORY open takes it). Without the check the full mode (the look's group and other bits and the
	# owner's rwx: 0707, 0747) is given to that directory, which is not the one looked at; with it, it keeps its own mode, and
	# out/d.away, the one looked at, stays 0700 (the known limitation "a directory of the runner's own with no owner read can be left 0700").
	if not _can_chmod_by_name_without_following():
		pytest.skip('no chmod by name that does not follow a link here: nothing changes')
	out = tmp_path / 'out'
	os.makedirs(str(out / 'd'))
	os.makedirs(str(out / 'other'))
	os.chmod(str(out / 'other'), 0o755)
	os.chmod(str(out / 'd'), mode)
	fd = os.open(str(out), os.O_RDONLY | os.O_DIRECTORY)
	base = fd if base_is == 'descriptor' else str(out)
	real_chmod, real_open, state = os.chmod, os.open, {'chmodded': False, 'swapped': False}
	def at_the_name(path, kwargs):
		return (path == 'd' and kwargs.get('dir_fd') == fd) or path == str(out / 'd')
	def chmod(path, new, *args, **kwargs):
		result = real_chmod(path, new, *args, **kwargs)
		state['chmodded'] = state['chmodded'] or at_the_name(path, kwargs)
		return result
	def opening(path, flags, *args, **kwargs):
		if state['chmodded'] and not state['swapped'] and at_the_name(path, kwargs):
			state['swapped'] = True
			os.rename(str(out / 'd'), str(out / 'd.away'))  # the other user's renames
			os.rename(str(out / 'other'), str(out / 'd'))
		return real_open(path, flags, *args, **kwargs)
	os.chmod, os.open = chmod, opening
	try:
		done = hpcp._owner_read(base, 'd')
	finally:
		os.chmod, os.open = real_chmod, real_open
		os.close(fd)
	try:
		assert state['swapped'], state
		assert not done
		assert stat.S_IMODE(os.stat(str(out / 'd')).st_mode) == 0o755, oct(os.stat(str(out / 'd')).st_mode)
		assert stat.S_IMODE(os.stat(str(out / 'd.away')).st_mode) == 0o700, oct(os.stat(str(out / 'd.away')).st_mode)
	finally:
		with contextlib.suppress(OSError):
			os.chmod(str(out / 'd.away'), 0o700)


@posix_only
@not_root
@pytest.mark.parametrize('route', ['proc', 'read'])
@pytest.mark.parametrize('mode', [0o607, 0o207, 0o007, 0o047])
def test_990_the_door_enters_the_runners_own_directory_shut_to_its_owner(tmp_path, fresh, monkeypatch, route, mode):
	# P1: a non-root copy or move gives a directory of mode 0607 that mode at its end; a rerun's folder sync of a subdirectory
	# that ran before the directory's own start met EACCES opening below it (exit 193). The door gives the runner's own such
	# directory owner search where it needs it (the finish sets the exact mode again). On the read route (macOS, Linux without
	# /proc) a mode that lacks owner read as well (0207, 0007, 0047: what a source's mode like 0047 gives the copy) is given owner
	# read, write and search by name below the held parent, where the first 9.90 build met EACCES; the /proc route holds the
	# directory by an O_PATH descriptor, which takes any mode.
	if route == 'proc' and not hpcp._PATH_FDS:
		pytest.skip('no /proc/self/fd here')
	if route == 'read' and not mode & 0o400 and not _can_chmod_by_name_without_following():
		pytest.skip('no chmod by name that does not follow a link here: the walk goes by path')
	monkeypatch.setattr(hpcp, '_PATH_FDS', route == 'proc')
	out = str(tmp_path / 'out')
	os.makedirs(os.path.join(out, 'd', 'sub'))
	os.chmod(os.path.join(out, 'd'), mode)
	_w1_run(monkeypatch, out)
	try:
		held = hpcp._dest_directory(os.path.join(out, 'd', 'sub'), make=False)
		assert isinstance(held, int) and os.path.samestat(os.fstat(held), os.stat(os.path.join(out, 'd', 'sub')))
	finally:
		hpcp._close_cursors()
		now = stat.S_IMODE(os.stat(os.path.join(out, 'd')).st_mode)
		os.chmod(os.path.join(out, 'd'), 0o755)
	assert now == mode | 0o700, oct(now)


_root_only = pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() != 0, reason='needs root to make another user\'s tree')


@posix_only
@_root_only
@pytest.mark.parametrize('remove', [False, True])
def test_990_a_rerun_over_a_0607_directory_with_many_jobs_below_it_completes(tmp_path, remove):
	# Guards P1 as an operator meets it, on the /proc route: src/d 0607, root's, with 200 subdirectories (more than one pool
	# job's 64 folders); a copy as nobody gives out/d 0607, and the rerun's folder sync of a subdirectory before out/d's own
	# start met EACCES (9.87 failed every rerun; 9.88 and 9.89 some, by the pool's order, and not in the run that measured it:
	# this passes on 9.89 on 3.14). The door test above ([proc] and [read]) is the one that fails on 9.89.
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


@posix_only
@_root_only
def test_990_the_rerun_of_a_skipped_move_may_bring_a_name_folding_like_one_there(tmp_path):
	# src/d is root's, 0607 (nobody, the runner, has no owner search on out/d's copy of it). A move as nobody skips its removal
	# for one unreadable file (212) and the late pass leaves out/d 0607. The rerun, with that file readable and a name G added
	# beside g, asks out/d about the pair before it copies: 9.90's first build was refused there, 213, on every rerun ("nothing
	# can be made in out/d to ask it (Permission denied): make it writable and run again", though it is writable and only lacks
	# search); 9.89 had left out/d 0707, and its rerun completed.
	import shutil
	from test_directories import _as_nobody, _shut_out_tree
	work = _shut_out_tree()
	try:
		unreadable = write(os.path.join(work, 'src', 'd', 'h'), 'secret', mode=0o600)  # root's: nobody cannot read it
		rc, output = _as_nobody(['src/', '-d', 'out/', '-rf'], work)
		assert rc != 0 and 'Remove skipped' in output, (rc, output[-3000:])
		out = os.path.join(work, 'out', 'd')
		assert stat.S_IMODE(os.lstat(out).st_mode) == 0o607, oct(os.lstat(out).st_mode)
		os.chmod(unreadable, 0o666)
		write(os.path.join(work, 'src', 'd', 'G'), 'upper', mode=0o666)
		rc, output = _as_nobody(['src/', '-d', 'out/', '-rf'], work)
		assert rc == 0, output[-3000:]
		assert (read(os.path.join(out, 'g')), read(os.path.join(out, 'G'))) == ('top', 'upper')
		assert stat.S_IMODE(os.lstat(out).st_mode) == 0o607 and not os.path.exists(os.path.join(work, 'src', 'd'))
	finally:
		shutil.rmtree(work, ignore_errors=True)


# -- macOS ------------------------------------------------------------------------------------------------------------------------

darwin_only = pytest.mark.skipif(sys.platform != 'darwin', reason='macOS')


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
def test_990_macos_skips_a_named_socket_as_its_cp_does():
	# The run itself on macOS, in the pool as by default: the socket is noted and not copied, exit 0 (9.89: 213), the file
	# beside it is copied, and a move leaves the socket where it is.
	with _a_short_tree() as work:
		rc, output = _cli(['src/s', 'src/f', '-d', 'out/'], work)
		assert rc == 0 and 'Special file skipped: src/s is a socket' in output, output[-3000:]
		assert os.listdir('out') == ['f']
		rc, output = _cli(['src/s', 'src/f', '-d', 'out/', '-rf'], work)
		assert rc == 0 and stat.S_ISSOCK(os.lstat('src/s').st_mode) and not os.path.lexists('src/f'), output[-3000:]
		assert os.listdir('out') == ['f']


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
		rc, output = _cli(['src/', '-d', 'out/'], tmp_path)
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
	rc, output = _cli(['src/', '-d', 'out/'], tmp_path)
	assert rc == 0, output[-3000:]
	landing, landing_dir = str(tmp_path / 'out' / 'd' / 'f'), str(tmp_path / 'out' / 'd')
	sources = (f, str(tmp_path / 'src' / 'd'))
	try:
		for path in sources:
			subprocess.run(['xattr', '-w', 'org.example.tag', 'one', path], check=True)
			subprocess.run(['chmod', '+a', 'everyone deny delete', path], check=True)
		rc, output = _cli(['src/', '-d', 'out/'], tmp_path)
		assert rc == 0, output[-3000:]
		for path in (landing, landing_dir):
			assert _mac_tag(path) == 'one', path
			assert any('deny delete' in line for line in _mac_acl(path)), (path, _mac_acl(path))
		for path in sources:
			subprocess.run(['xattr', '-d', 'org.example.tag', path], check=True)
			subprocess.run(['chmod', '-N', path], check=True)
		rc, output = _cli(['src/', '-d', 'out/'], tmp_path)
		assert rc == 0, output[-3000:]
		for path in (landing, landing_dir):
			assert _mac_tag(path) is None, path
			assert _mac_acl(path) == [], (path, _mac_acl(path))
	finally:
		# a deny-delete ACL left on a path keeps pytest from removing the tree
		for path in sources + (landing, landing_dir):
			subprocess.run(['chmod', '-N', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@darwin_only
def test_990_macos_a_rerun_keeps_the_marks_macos_puts_on_a_file_by_itself(tmp_path):
	# A rerun mirrors the source's xattrs: one only the landing has (a Finder tag, a program's own) is removed, as a Linux
	# rerun removes a user.* name. But com.apple.quarantine, com.apple.lastuseddate#PS, com.apple.provenance and com.apple.macl are
	# the system's own marks on a file: never removed (the first 9.90 build removed every name but com.apple.system.*).
	write(tmp_path / 'src' / 'f', 'x')
	rc, output = _cli(['src/', '-d', 'out/'], tmp_path)
	assert rc == 0, output[-3000:]
	landing = str(tmp_path / 'out' / 'f')
	quarantine = '0081;00000000;Safari;'
	subprocess.run(['xattr', '-w', 'org.example.mine', 'mine', landing], check=True)
	subprocess.run(['xattr', '-w', 'com.apple.quarantine', quarantine, landing], check=True)
	rc, output = _cli(['src/', '-d', 'out/'], tmp_path)
	assert rc == 0, output[-3000:]
	names = subprocess.run(['xattr', landing], stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout.split()
	assert 'org.example.mine' not in names and 'com.apple.quarantine' in names, names


@linux_only
def test_990_a_read_that_returned_bytes_counts_as_part_of_a_job():
	# The window the first 9.90 build missed (2 'EOFError' blocks in 2,400 early stops on Python 3.6 spawn): a stop is
	# handled after a read returns, before its bytes are counted, so a worker that has just read the 4-byte length (or some of
	# the payload) is mid-message although nothing else in its frames says so yet. Asked at every line of the read. Line events
	# only: the instant right after a read returns, before the call's result is stored, is not at a line boundary and is not seen
	# here (the entry in docs/limitations.md says "rarer", not gone: 184 of 200 aimed stops lost the length on 3.14 and 145 of
	# 200 on 3.6, against 200 of 200 on 9.89).
	import multiprocessing
	readable, writable = multiprocessing.Pipe(duplex=False)
	answers = []
	recv, recv_bytes = hpcp._RECV_CODE, hpcp._RECV_BYTES_CODE
	def tracer(frame, event, arg):
		if frame.f_code in (recv, recv_bytes):
			if event == 'line':
				answers.append((frame.f_code.co_name, frame.f_lineno, sorted(frame.f_locals), hpcp._in_message_read(frame)))
			return tracer
		return None
	previous = sys.gettrace()  # coverage's, where it runs
	try:
		writable.send_bytes(b'x' * 100)
		sys.settrace(tracer)
		try:
			data = readable.recv_bytes()
		finally:
			sys.settrace(previous)
	finally:
		readable.close()
		writable.close()
	assert data == b'x' * 100
	# Before the first read of the length nothing is consumed; from the first line after that read returned, everything is.
	first_read = [i for i, (name, _, names, _) in enumerate(answers) if name == '_recv' and 'chunk' in names][0]
	assert not any(answer for _, _, _, answer in answers[:first_read]), answers[:first_read]
	assert all(answer for _, _, _, answer in answers[first_read:]), [a for a in answers[first_read:] if not a[3]]


# -- A paced parallel remove's clock across spawned workers --------------------------------------------------------------------

@posix_only
@pytest.mark.skipif(sys.version_info < (3, 7), reason="Python 3.6's pool takes no initializer: spawned workers get no share")
def test_990_a_paced_parallel_remove_does_not_wait_out_the_parents_age(tmp_path, reset_hpcp_globals, monkeypatch):
	# The Mac suite's hang (test_parallel_plain_remove_honours_the_file_rate_limit, 16+ minutes): -pfl -F handed the workers
	# the parent's time.monotonic() start, and on macOS with Python 3.7 to 3.9 a spawned worker's monotonic counts from its
	# own start, so each unlink slept about the parent's age. Here the parent's monotonic runs 20 s ahead of the workers'.
	import multiprocessing
	import time
	src = tmp_path / 'src'
	for index in range(6):
		(src / ('d%d' % index)).mkdir(parents=True)
		for number in range(10):
			(src / ('d%d' % index) / ('f%d' % number)).write_text('x')
	monkeypatch.setattr(hpcp, 'FILES_RATE_LIMIT', 50)
	real = time.monotonic
	monkeypatch.setattr(time, 'monotonic', lambda: real() + 20)
	method = multiprocessing.get_start_method()
	multiprocessing.set_start_method('spawn', force=True)
	try:
		start = real()
		with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
			hpcp.process_remove([str(src)], single_thread=False, max_workers=2, parallel_file_listing=True, remove_force=True)
		took = real() - start
	finally:
		multiprocessing.set_start_method(method, force=True)
	assert not src.exists()
	assert 0.7 <= took < 10, took  # 60 files at 50 a second: 1.2 s and the workers' start; 20 s more with the parent's clock
