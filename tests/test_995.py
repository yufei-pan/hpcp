'''hpcp 9.95: -rm shows its progress as it removes, as 9.73 did (9.74 to 9.94 printed nothing between "Removing files from"
and the "Removed N files" line). Each test fails on 9.94 unless it says it guards what 9.94 already gets right. The pace
under -F is what makes a removal last: N files at -F 100 take at least N / 100 s whatever the machine, so a frame (every
0.5 s) is always due. The serial runs also check that the first frame came well before the end; a -pfl pool's start takes a time
that depends on the machine (a forkserver start under load ate that margin), so a -pfl run is checked for the order instead: its
first frame comes before its last job's report (the loop draws while the jobs unlink, not only the walk after them).'''
import os
import re
import select
import signal
import subprocess
import sys
import time

import pytest

import hpcp

posix_only = pytest.mark.skipif(os.name != 'posix', reason='POSIX')
HPCP = os.path.abspath(hpcp.__file__)

# The pace and the tree: 300 files at -F 100 take at least 3 s; the first frame is due at 0.5 s and files are still being
# unlinked then, so one is drawn by the serial walk (at an entry) or by the -pfl loop in the parent's wait (a job is another
# process and draws nothing). The pace counts from the run's start, and a -pfl pool takes its time to start (over a second on a
# busy machine, Python 3.9 as root: the first version of the cadence check failed there), so the run is long enough to show a
# frame after that.
FILES_PER_SECOND = 100
DIRECTORIES, PER_DIRECTORY = 6, 50
TOTAL = DIRECTORIES * PER_DIRECTORY
# The real program's tests only need a frame before the summary: 200 files, 2 s.
PROGRAM_DIRECTORIES, PROGRAM_PER_DIRECTORY = 4, 50
PROGRAM_TOTAL = PROGRAM_DIRECTORIES * PROGRAM_PER_DIRECTORY
# A frame that comes at the end (a loop that checks the clock only between directories, or after them) is no frame of the run:
# this is how far before the end the first one is, in the serial runs (-pfl's runs check the order: see the module docstring).
FIRST_FRAME_BEFORE_THE_END = 0.5
SHAPES = [pytest.param((DIRECTORIES, PER_DIRECTORY), id='tree'), pytest.param((1, TOTAL), id='one-wide-directory')]


def _tree(root, directories=DIRECTORIES, per_directory=PER_DIRECTORY):
	for index in range(directories):
		directory = root / ('d%d' % index)
		directory.mkdir(parents=True)
		for number in range(per_directory):
			(directory / ('f%d' % number)).write_text('x')


def _times_of_frames_and_job_reports(monkeypatch):
	'''Wrap what draws a frame (_RemoveProgress.show) and what takes a -pfl job's report (_task_result, called in the parent for each
	finished job): two lists of clock readings, the frames' and the reports'. A job is another process, so the order of the two is
	what shows that the loop draws while the jobs work, whatever the machine's speed.'''
	times, reported = [], []
	real_show = hpcp._RemoveProgress.show
	def show(self, *args, **kwargs):
		times.append(time.monotonic())
		return real_show(self, *args, **kwargs)
	monkeypatch.setattr(hpcp._RemoveProgress, 'show', show)
	real_task_result = hpcp._task_result
	def task_result(*args, **kwargs):
		reported.append(time.monotonic())
		return real_task_result(*args, **kwargs)
	monkeypatch.setattr(hpcp, '_task_result', task_result)
	return times, reported


def _frames(text):
	'''The frames of the progress line in text: it starts with a carriage return and carries "iteration:".'''
	return [frame for frame in re.split(r'[\r\n]', text) if 'iteration:' in frame]


def _before_the_summary(text, removed=TOTAL):
	'''What came before the "Removed N files and links" line, which must start a line of its own.'''
	summary = text.index('Removed %d files and links' % removed)
	assert summary == 0 or text[summary - 1] == '\n', 'the summary is on the last frame\'s line: %r' % text[max(0, summary - 80):summary + 40]
	return text[:summary]


@pytest.fixture
def paced(monkeypatch, reset_hpcp_globals):
	monkeypatch.setattr(hpcp, 'FILES_RATE_LIMIT', FILES_PER_SECOND)
	return hpcp


MODES = [
	pytest.param(dict(single_thread=True), True, id='serial'),
	pytest.param(dict(single_thread=True), False, id='serial-no-descriptors'),
	pytest.param(dict(single_thread=False, max_workers=2, parallel_file_listing=True), True, id='pfl'),
	pytest.param(dict(single_thread=False, max_workers=2, parallel_file_listing=True), False, id='pfl-no-descriptors'),
]


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('options, descriptors', MODES)
def test_995_plain_rm_shows_a_line_while_it_removes(tmp_path, capsys, paced, monkeypatch, options, descriptors, shape):
	'''9.74 to 9.94 removed in silence: no line between "Removing files from" and the summary. The first frame comes by the
	time it is due, not once the files are gone: in a wide directory only a check at an entry (the walk's unlinks) or the -pfl
	loop's wait in the parent can show it. Serial: the first frame is well before the end. -pfl: it is before the last job's
	report (the loop draws while the jobs unlink: without that draw the line comes from the walk after them).'''
	if not descriptors:
		monkeypatch.setattr(hpcp, '_DIR_FD_WALK', False)
	times, reported = _times_of_frames_and_job_reports(monkeypatch)
	_tree(tmp_path / 'src', *shape)
	hpcp.process_remove([str(tmp_path / 'src')], remove_force=True, **options)
	end = time.monotonic()
	assert not (tmp_path / 'src').exists()
	shown = _before_the_summary(capsys.readouterr().out)
	frames = _frames(shown)
	assert frames, 'no progress line in %r' % shown
	if options['single_thread']:
		assert end - times[0] >= FIRST_FRAME_BEFORE_THE_END, 'the first frame came %.2f s before the end' % (end - times[0])
	else:
		assert reported and times[0] < reported[-1], 'the first frame came %.3f s after the last job reported' % (times[0] - reported[-1])
	assert re.search(r'iteration:\d+ \| (Removed: [\d.]+ ?K? +)?(Folders: [\d.]+ ?K? +)?(\| )?Overall: [\d.]+ ?K? +Files', frames[-1]), frames[-1]
	# The Overall count is the pace's own (under -pfl the jobs' shared count): it only grows, and is never behind what the jobs reported.
	overall = [float(re.search(r'Overall: ([\d.]+) ', frame).group(1)) for frame in frames]
	removed = [float(re.search(r'Removed: ([\d.]+) ', frame).group(1)) if 'Removed:' in frame else 0 for frame in frames]
	# (The last frame can be stale on a loaded host, a run ending within 0.5 s of it: the summary has the total.)
	assert overall == sorted(overall), frames
	assert all(done >= count for done, count in zip(overall, removed)), frames


def test_995_the_line_is_the_listings_with_removed_and_folders(tmp_path, capsys, monkeypatch, reset_hpcp_globals):
	'''The directory's last 50 characters, then "iteration:N | Removed: N Folders: M" (the listing's frame has Files: Links:
	Folders: Size: where this has Removed: Folders:). With the interval 0 every entry shows a frame, and the counts grow to the last.'''
	monkeypatch.setattr(hpcp, '_LINE_INTERVAL', 0)
	root = tmp_path / ('long-directory-name-' * 4) / 'src'
	_tree(root, directories=3, per_directory=50)
	hpcp.process_remove([str(root)], single_thread=True, remove_force=True)
	frames = _frames(_before_the_summary(capsys.readouterr().out, 150))
	assert len(frames) >= 150
	assert len(str(root)) > 50
	directories = {(str(root) + tail)[-50:] for tail in ('', '/d0', '/d1', '/d2')}
	assert {frame.split(' | ')[0] for frame in frames} <= directories
	counts = [int(re.search(r'Removed: (\d+)\.00 ', frame).group(1)) for frame in frames if 'Removed:' in frame]
	assert counts == sorted(counts) and counts[0] > 0 and counts[-1] == 150
	assert 'Folders: ' in frames[-1] and 'Overall' not in frames[-1]
	assert re.search(r'iteration:\d+ \| Removed: ', frames[-1])


@pytest.mark.parametrize('descriptors', [True, False], ids=['descriptors', 'no-descriptors'])
def test_995_a_tree_of_empty_directories_shows_a_line_a_directory(tmp_path, capsys, monkeypatch, reset_hpcp_globals, descriptors):
	'''No file is unlinked, so only the walk's check at each directory (the rmdirs) can show a frame.'''
	if not descriptors:
		monkeypatch.setattr(hpcp, '_DIR_FD_WALK', False)
	monkeypatch.setattr(hpcp, '_LINE_INTERVAL', 0)
	for index in range(30):
		(tmp_path / 'src' / ('d%d' % index)).mkdir(parents=True)
	hpcp.process_remove([str(tmp_path / 'src')], single_thread=True, remove_force=True)
	frames = _frames(_before_the_summary(capsys.readouterr().out, 0))
	assert len(frames) >= 30 and re.search(r'Folders: 3[01]\.00 ', frames[-1]), frames
	assert not any('Removed:' in frame for frame in frames)


def test_995_under_the_file_rate_limit_the_line_has_the_overall_count(tmp_path, capsys, monkeypatch, reset_hpcp_globals):
	monkeypatch.setattr(hpcp, '_LINE_INTERVAL', 0)
	monkeypatch.setattr(hpcp, 'FILES_RATE_LIMIT', 100000)
	_tree(tmp_path / 'src', directories=2, per_directory=50)
	hpcp.process_remove([str(tmp_path / 'src')], single_thread=True, remove_force=True)
	frames = _frames(capsys.readouterr().out)
	assert frames and all('| Overall: ' in frame for frame in frames)
	assert re.search(r'Overall: 100\.00 +Files', frames[-1]), frames[-1]


def test_995_the_walk_adds_to_what_the_pfl_jobs_removed(tmp_path, capsys, monkeypatch, reset_hpcp_globals):
	'''-pfl: the jobs unlink the files, the walk after removes the directories; the line carries the count on, the Overall count
	(under -F, the jobs' shared one) as well.'''
	monkeypatch.setattr(hpcp, '_LINE_INTERVAL', 0)
	monkeypatch.setattr(hpcp, 'FILES_RATE_LIMIT', 100000)
	_tree(tmp_path / 'src', directories=4, per_directory=5)
	hpcp.process_remove([str(tmp_path / 'src')], single_thread=False, max_workers=2, parallel_file_listing=True, remove_force=True)
	frames = _frames(_before_the_summary(capsys.readouterr().out, 20))
	assert frames and all('Removed: 20.00 ' in frame for frame in frames[-2:]), frames[-2:]
	assert all(re.search(r'Overall: 20\.00 +Files', frame) for frame in frames[-2:]), frames[-2:]
	assert 'Folders: ' in frames[-1]
	# Every frame of the jobs' loop has the count of the files they had unlinked, never the parent's own (0): never behind what they reported.
	assert all(float(re.search(r'Overall: ([\d.]+) ', frame).group(1)) >= float(re.search(r'Removed: ([\d.]+) ', frame).group(1)) for frame in frames if 'Removed:' in frame), frames


def test_995_pfl_with_no_pace_draws_while_its_jobs_unlink(tmp_path, capsys, monkeypatch, reset_hpcp_globals):
	'''-pfl without -F (the default: no pace, no shared count): the loop shows the line while the jobs unlink too, not only the walk
	after them. A run that outlasts 0.5 s with no pace is the machine's doing, so the interval is 0 (the parent draws; the jobs
	are other processes and draw nothing): the first frame comes before the last job's report. Without the loop's draw the walk
	shows the only frames, after the jobs.'''
	monkeypatch.setattr(hpcp, 'FILES_RATE_LIMIT', 0)
	monkeypatch.setattr(hpcp, '_LINE_INTERVAL', 0)
	times, reported = _times_of_frames_and_job_reports(monkeypatch)
	_tree(tmp_path / 'src', directories=4, per_directory=5)
	hpcp.process_remove([str(tmp_path / 'src')], single_thread=False, max_workers=2, parallel_file_listing=True, remove_force=True)
	assert not (tmp_path / 'src').exists()
	frames = _frames(_before_the_summary(capsys.readouterr().out, 20))
	assert frames and not any('Overall' in frame for frame in frames), frames
	assert reported and times[0] < reported[-1], 'the first frame came %.3f s after the last job reported' % (times[0] - reported[-1])


def test_995_the_open_line_is_known_to_error_lines_and_ended_by_finish(capsys, monkeypatch, reset_hpcp_globals):
	'''show() sets the flag the parent's error lines consult (_BAR_DRAWN) and finish() clears it and ends the line. Where the error
	lines land is tested with the real program (below).'''
	progress = hpcp._RemoveProgress()
	assert not hpcp._BAR_DRAWN
	progress.show('/some/directory', 3, 1)
	assert hpcp._BAR_DRAWN
	shown = capsys.readouterr().out
	assert shown.startswith('\r') and not shown.endswith('\n') and re.search(r'Removed: 3\.00 +Folders: 1\.00', shown)
	progress.finish()
	assert not hpcp._BAR_DRAWN and capsys.readouterr().out == '\n'
	progress.finish()
	assert capsys.readouterr().out == ''


def test_995_finish_leaves_the_mark_of_a_line_it_did_not_draw(capsys, reset_hpcp_globals):
	'''-rwloff's job bar (Adaptive_Progress_Bar) sets _BAR_DRAWN and a stop skips its stop(): the removal's finish() has drawn no
	frame of its own, so it ends no line and leaves the mark for the stop's report to start off the bar's line.'''
	hpcp._BAR_DRAWN = True
	hpcp._RemoveProgress().finish()
	assert hpcp._BAR_DRAWN and capsys.readouterr().out == ''


def test_995_a_stop_inside_a_frames_draw_leaves_the_line_to_finish(capsys, monkeypatch, reset_hpcp_globals):
	'''A stop (KeyboardInterrupt) can come right after the frame is written, before show() returns: the line is open, and finish()
	ends it. (The program's stop test below hit that window once in 30 runs on Python 3.6 under a CPU quota: it signals as the
	first frame arrives.)'''
	def draw_and_stop(*args, **kwargs):
		print('\rframe', end='')
		raise KeyboardInterrupt
	monkeypatch.setattr(hpcp.multiCMD, 'print_progress_bar', draw_and_stop)
	progress = hpcp._RemoveProgress()
	with pytest.raises(KeyboardInterrupt):
		progress.show('/some/directory', 1)
	assert hpcp._BAR_DRAWN
	progress.finish()
	assert capsys.readouterr().out == '\rframe\n' and not hpcp._BAR_DRAWN


def test_995_a_run_that_shows_no_frame_prints_no_empty_line(tmp_path, capsys, reset_hpcp_globals):
	'''Guards what 9.94 gets right: a removal quicker than the first frame prints no blank line before the summary.'''
	_tree(tmp_path / 'src', directories=2, per_directory=3)
	hpcp.process_remove([str(tmp_path / 'src')], single_thread=True, remove_force=True)
	shown = capsys.readouterr().out
	assert '\n\n' not in shown and _frames(shown) == []
	assert not hpcp._BAR_DRAWN


def test_995_the_library_and_a_moves_walks_print_nothing(tmp_path, capsys, monkeypatch, reset_hpcp_globals):
	'''_rmtree (the library) and a walk given no progress object (a direct call) draw no line, whatever the interval. (A move's
	directory pass gives it one since 9.96: tests/test_996.py.) On 9.94 this fails only for the missing class.'''
	monkeypatch.setattr(hpcp, '_LINE_INTERVAL', 0)
	_tree(tmp_path / 'one')
	hpcp._rmtree(tmp_path / 'one')
	_tree(tmp_path / 'two')
	hpcp._remove_tree_fd(str(tmp_path / 'two'), hpcp._directory_identity(str(tmp_path / 'two')), remove_files=False)
	assert _frames(capsys.readouterr().out) == []
	assert not (tmp_path / 'one').exists()


def _run(args, cwd):
	'''hpcp as a program: stdout and stderr through one pipe, the bytes as written (a carriage return stays one). (exit status, output)'''
	done = subprocess.run([sys.executable, HPCP] + args, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
		stdin=subprocess.DEVNULL, timeout=120)
	return done.returncode, done.stdout.decode('utf-8', 'replace')


@pytest.mark.parametrize('extra', [[], ['-pfl']], ids=['serial', 'pfl'])
def test_995_hpcp_rm_through_a_pipe_prints_the_line(tmp_path, extra):
	'''The real program, its output a pipe: a line before the summary.'''
	_tree(tmp_path / 'src', PROGRAM_DIRECTORIES, PROGRAM_PER_DIRECTORY)
	rc, output = _run(['-rm', '-rf', '-F', str(FILES_PER_SECOND), '-j', '2'] + extra + ['src'], tmp_path)
	assert rc == 0 and not (tmp_path / 'src').exists(), output
	assert _frames(_before_the_summary(output, PROGRAM_TOTAL)), output


def _at_a_terminal(args, cwd, size):
	'''hpcp with a pty as its stdin, stdout and stderr (a terminal of size (rows, columns), or none: its size is 0 x 0).
	(exit status, what the terminal showed)'''
	import fcntl
	import pty
	import struct
	import termios
	pid, fd = pty.fork()
	if pid == 0:
		os.chdir(str(cwd))
		os.execv(sys.executable, [sys.executable, HPCP] + args)
	if size:
		fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack('HHHH', size[0], size[1], 0, 0))
	shown = b''
	deadline = time.monotonic() + 120
	try:
		while time.monotonic() < deadline:
			ready, _, _ = select.select([fd], [], [], 0.2)
			if not ready:
				continue
			try:
				chunk = os.read(fd, 4096)
			except OSError:
				break
			if not chunk:
				break
			shown += chunk
		else:
			os.kill(pid, 9)
			pytest.fail('no end in 120 s; the terminal showed %r' % shown)
	finally:
		os.close(fd)
	return os.WEXITSTATUS(os.waitpid(pid, 0)[1]), shown.decode('utf-8', 'replace').replace('\r\n', '\n')


@posix_only
@pytest.mark.parametrize('size', [(40, 120), None], ids=['sized-terminal', 'unsized-terminal'])
@pytest.mark.parametrize('extra', [[], ['-pfl']], ids=['serial', 'pfl'])
def test_995_hpcp_rm_at_a_terminal_prints_the_line_on_one_line(tmp_path, extra, size):
	'''At a terminal, sized or not (a pty from pty.fork has no size, as under `script`: 0 columns), each frame holds one "iteration:"
	and the Overall count (no two frames glued). The width is not measured: a frame wider than the terminal wraps (planned in multiCMD).'''
	_tree(tmp_path / 'src', PROGRAM_DIRECTORIES, PROGRAM_PER_DIRECTORY)
	rc, shown = _at_a_terminal(['-rm', '-rf', '-F', str(FILES_PER_SECOND), '-j', '2'] + extra + ['src'], tmp_path, size)
	assert rc == 0 and not (tmp_path / 'src').exists(), shown
	frames = _frames(_before_the_summary(shown, PROGRAM_TOTAL))
	assert frames, shown
	assert all(frame.count('iteration:') == 1 and '| Overall: ' in frame for frame in frames), frames


@pytest.fixture
def locked_tree(tmp_path):
	'''src/ with a/ of 300 files and z/ of 100, and src/a/sub/locked/ (mode 0555) holding two files the runner can not unlink: each is
	an error line, reached after the first frame. At -F 100 the 400 files take at least 4 s: serial, the walk is in a/ at the first
	frame; under -pfl, a's job is still unlinking, and the locked directory's job starts only once a's returns, after 3 s or more
	(the first frame is at 0.5 s, or when the pool has started: over a second under a CPU quota of 0.25, so a/ is long enough
	for the frame to come well before the error). Root can unlink them.'''
	if os.name != 'posix' or os.geteuid() == 0:
		pytest.skip('needs a directory its owner can not write into (not root)')
	root = tmp_path / 'src'
	locked = root / 'a' / 'sub' / 'locked'
	locked.mkdir(parents=True)
	for directory, count in (('a', 300), ('z', 100)):
		(root / directory).mkdir(exist_ok=True)
		for number in range(count):
			(root / directory / ('f%d' % number)).write_text('x')
	for name in ('k1', 'k2'):
		(locked / name).write_text('x')
	locked.chmod(0o555)
	yield root
	locked.chmod(0o755)


CLEAR_LINE = '\x1b[2K\r'


def _every_start_of_a_line(text, mark, starts=('\n',)):
	'''The places before the final "Error occurred:" report where mark does not come right after one of starts (a line's start: a
	newline, or at a terminal the clear-line and return, whole: a bare return would leave the bar's text on the line): each is a
	stretch of text ending in it. Fails when mark does not occur at all.'''
	text = text.partition('Error occurred:')[0]
	places = [match.start() for match in re.finditer(re.escape(mark), text)]
	assert places, 'no %r in %r' % (mark, text)
	return [text[max(0, place - 60):place + len(mark)] for place in places if place and not text[:place].endswith(tuple(starts))]


@posix_only
@pytest.mark.parametrize('extra', [[], ['-pfl']], ids=['serial', 'pfl'])
def test_995_an_error_line_while_the_line_is_open_starts_its_own_line_in_a_log(locked_tree, extra):
	'''9.95's first version: under -pfl a job's "Remove process exception:" came right after the open frame (a combined log: on the
	frame's line, after its padding), as a job of a plain -rm did not know the run drew one. 9.94 drew no frame. (The serial walk
	ends the frame's line before it reports its errors, in the parent: that case passes on the first build too, and guards the
	line's end. The test for an error printed while a frame is open, in the parent, is the named file's, below.)'''
	rc, output = _run(['-rm', '-rf', '-F', str(FILES_PER_SECOND), '-j', '2'] + extra + ['src'], locked_tree.parent)
	assert rc != 0 and not (locked_tree / 'a' / 'f0').exists(), output
	assert _frames(output), output
	assert _every_start_of_a_line(output, 'Remove process exception:') == [], output


@posix_only
@pytest.mark.parametrize('size', [(40, 120), None], ids=['sized-terminal', 'unsized-terminal'])
def test_995_an_error_line_while_the_line_is_open_clears_it_at_a_terminal(locked_tree, size):
	'''At a terminal the job's line starts with the clear-line and return a copy's error line does (the whole escape, not a bare
	return), in a pty with a size or none.'''
	rc, shown = _at_a_terminal(['-rm', '-rf', '-F', str(FILES_PER_SECOND), '-j', '2', '-pfl', 'src'], locked_tree.parent, size)
	assert rc != 0, shown
	assert _frames(shown), shown
	assert _every_start_of_a_line(shown, 'Remove process exception:', ('\n', CLEAR_LINE)) == [], shown


def _stopped(args, cwd, marker, sig, merged=False):
	'''hpcp as a program, sent sig as soon as marker (bytes) has come on its stdout: stdout and stderr two pipes, or one (merged). A
	SIGINT the parent ignores (a job its shell started with &) would be ignored by the program too, so it is given the default back.
	(exit status, what stdout held, what stderr held: b"" when merged)'''
	proc = subprocess.Popen([sys.executable, HPCP] + args, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT if merged else subprocess.PIPE, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
	seen = b''
	deadline = time.monotonic() + 60
	try:
		while marker not in seen:
			assert time.monotonic() < deadline, 'no %r in 60 s: %r' % (marker, seen)
			if select.select([proc.stdout], [], [], 0.2)[0]:
				chunk = os.read(proc.stdout.fileno(), 4096)
				assert chunk, 'the run ended before %r: %r' % (marker, seen)
				seen += chunk
		proc.send_signal(sig)
		out, err = proc.communicate(timeout=60)
	finally:
		proc.kill()
		proc.wait()
	return proc.returncode, (seen + out).decode('utf-8', 'replace'), (err or b'').decode('utf-8', 'replace')


STOPS = [pytest.param(signal.SIGTERM, -signal.SIGTERM, id='SIGTERM'), pytest.param(signal.SIGINT, 130, id='SIGINT')]


@posix_only
@pytest.mark.parametrize('sig, status', STOPS)
@pytest.mark.parametrize('extra', [[], ['-pfl']], ids=['serial', 'pfl'])
def test_995_a_stop_ends_the_open_line_before_the_report_on_stdout(tmp_path, extra, sig, status):
	'''A stop once the first frame has come, stdout one pipe and stderr another: "Error occurred: ..." (stdout) started right after
	the open frame, padding and all, as nothing ended the line when the removal was left by the stop. Ctrl-C (SIGINT) ends the
	program with status 130, a SIGTERM by the signal itself (-15, as in 9.94); the report says "return code: 130." for both, and the
	stop's own line is on stderr.'''
	_tree(tmp_path / 'src', PROGRAM_DIRECTORIES, PROGRAM_PER_DIRECTORY)
	returncode, output, err = _stopped(['-rm', '-rf', '-F', str(FILES_PER_SECOND), '-j', '2'] + extra + ['src'], tmp_path, b'iteration:', sig)
	assert 'Removed %d files' % PROGRAM_TOTAL not in output, 'the run was over before the stop: %r' % output
	assert returncode == status and 'return code: 130.' in output, (returncode, output, err)
	assert 'Keyboard Interrupt' in err
	report = output.index('Error occurred: ')
	assert output[report - 1] == '\n', output[max(0, report - 120):report + 60]


@posix_only
@pytest.mark.parametrize('sig, status', STOPS)
def test_995_a_stop_in_rwloffs_job_phase_leaves_its_bar_line_to_the_report(tmp_path, sig, status):
	'''-rwloff's jobs show the "Deleted" bar (its frames carry "F/s Limit" under -F), whose line a stop leaves open (its own end of line is skipped): the removal's
	finish() must not clear the mark that line set, or "Keyboard Interrupt: ..." is glued to the bar's text (stdout and stderr one
	pipe: 9.94 started it on its own line; 9.95's first fix round glued it).'''
	_tree(tmp_path / 'src', PROGRAM_DIRECTORIES, PROGRAM_PER_DIRECTORY)
	returncode, output, _ = _stopped(['-rm', '-rf', '-rwloff', '-F', str(FILES_PER_SECOND), '-j', '2', 'src'], tmp_path, b'F/s Limit', sig, merged=True)
	assert 'Removed %d files' % PROGRAM_TOTAL not in output, 'the run was over before the stop: %r' % output
	assert returncode == status, (returncode, output)
	report = output.index('Keyboard Interrupt: ')
	assert output[report - 1] == '\n', output[max(0, report - 120):report + 60]


@posix_only
def test_995_the_errors_after_the_last_frame_leave_no_blank_lines(locked_tree):
	'''The walk's errors are reported after its last frame, serially: the first ended the frame's line and each next one, and the
	summary after them, added an empty line (the open line was still known). Stdout and stderr one pipe.'''
	rc, output = _run(['-rm', '-rf', '-F', str(FILES_PER_SECOND), 'src'], locked_tree.parent)
	assert rc != 0 and _frames(output), output
	assert output.partition('Error occurred:')[0].count('Remove process exception:') == 2 and '\n\n' not in output, output


def test_995_files_named_as_arguments_show_a_line_while_they_are_removed(tmp_path, capsys, paced, monkeypatch):
	'''hpcp -rm dir/* names every file: no walk, so only the loop over the arguments can show the line (9.95's first version
	drew none: 6,000 files took seconds in silence). Each frame carries the count so far and the directory of the last file.'''
	times = []
	real_show = hpcp._RemoveProgress.show
	def show(self, *args, **kwargs):
		times.append(time.monotonic())
		return real_show(self, *args, **kwargs)
	monkeypatch.setattr(hpcp._RemoveProgress, 'show', show)
	(tmp_path / 'src').mkdir()
	names = [str(tmp_path / 'src' / ('f%d' % number)) for number in range(TOTAL)]
	for name in names:
		open(name, 'w').close()
	hpcp.process_remove(names, single_thread=True, remove_force=True)
	end = time.monotonic()
	assert os.listdir(str(tmp_path / 'src')) == []
	frames = _frames(_before_the_summary(capsys.readouterr().out))
	assert frames, 'no progress line'
	assert end - times[0] >= FIRST_FRAME_BEFORE_THE_END, 'the first frame came %.2f s before the end' % (end - times[0])
	assert all(frame.startswith(str(tmp_path / 'src')[-50:] + ' | iteration:') for frame in frames), frames
	counts = [int(re.search(r'Removed: (\d+)\.00 ', frame).group(1)) for frame in frames]
	# (A frame can come right after the last unlink, so the last count may be the total.)
	assert counts == sorted(counts) and 0 < counts[0] < TOTAL and counts[-1] <= TOTAL


def test_995_the_program_given_files_to_remove_prints_the_line(tmp_path):
	'''The real program as the shell runs it for -rm dir/*: every file an argument, its output a pipe.'''
	(tmp_path / 'src').mkdir()
	names = [os.path.join('src', 'f%d' % number) for number in range(PROGRAM_TOTAL)]
	for name in names:
		open(str(tmp_path / name), 'w').close()
	rc, output = _run(['-rm', '-rf', '-F', str(FILES_PER_SECOND)] + names, tmp_path)
	assert rc == 0 and os.listdir(str(tmp_path / 'src')) == [], output
	assert _frames(_before_the_summary(output, PROGRAM_TOTAL)), output


@pytest.mark.parametrize('descriptors', [True, False], ids=['descriptors', 'no-descriptors'])
def test_995_a_directory_gone_before_its_rmdir_is_counted_as_removed_in_every_walk(tmp_path, monkeypatch, reset_hpcp_globals, descriptors):
	'''A directory that something else removes between the walk's listing and its rmdir (the rmdir here finds it gone) counts under
	Folders: like any other, in both walks. (The fallback walk and the root already did: _rmdir_below answers None for a gone
	directory; the descriptor walk's inner directories did not.)'''
	if not descriptors:
		monkeypatch.setattr(hpcp, '_DIR_FD_WALK', False)
	_tree(tmp_path / 'src', directories=3, per_directory=2)
	real_rmdir = os.rmdir
	def rmdir(path, *args, **kwargs):
		real_rmdir(path, *args, **kwargs)
		if os.path.basename(path) == 'd1':
			raise FileNotFoundError(path)
	monkeypatch.setattr(os, 'rmdir', rmdir)
	progress = hpcp._RemoveProgress()
	count, _, errors = hpcp._remove_tree_fd(str(tmp_path / 'src'), hpcp._directory_identity(str(tmp_path / 'src')), remove_files=True, progress=progress)
	assert not errors and count == 6 and not (tmp_path / 'src').exists()
	assert progress.folders == 4, progress.folders


def test_995_after_rwloff_the_walks_overall_count_goes_on_from_the_jobs(tmp_path, capsys, monkeypatch, reset_hpcp_globals):
	'''-rwloff removes the listed files in pool jobs, then the walk takes the directories with a -F pace of its own: its frames
	counted from 0 ("Removed: 30" beside "Overall: 0").'''
	monkeypatch.setattr(hpcp, '_LINE_INTERVAL', 0)
	monkeypatch.setattr(hpcp, 'FILES_RATE_LIMIT', 100000)
	monkeypatch.setattr(hpcp, 'REMOVE_FILES_WHILE_LISTING', False)
	_tree(tmp_path / 'src', directories=3, per_directory=10)
	hpcp.process_remove([str(tmp_path / 'src')], single_thread=False, max_workers=2, remove_force=True)
	assert not (tmp_path / 'src').exists()
	# (The walk's frames: the listing before it draws its own, with the interval 0 too.)
	frames = [frame for frame in _frames(capsys.readouterr().out.partition('Time taken to get file list')[2]) if 'Overall:' in frame]
	assert frames and all('Removed: 30.00 ' in frame for frame in frames), frames
	assert all(re.search(r'Overall: 30\.00 +Files', frame) for frame in frames), frames


@posix_only
def test_995_an_error_line_of_a_named_file_while_the_line_is_open_starts_its_own_line(tmp_path):
	'''A file named as an argument that can not be unlinked is reported by the parent while the loop's frame is open: the line start
	the parent's error lines take from the open frame (_BAR_DRAWN in _bar_line_start). Stdout and stderr one pipe; not root.'''
	if os.geteuid() == 0:
		pytest.skip('root can unlink in a directory without write')
	(tmp_path / 'src').mkdir()
	names = [os.path.join('src', 'f%d' % number) for number in range(PROGRAM_TOTAL)]
	for name in names:
		open(str(tmp_path / name), 'w').close()
	locked = tmp_path / 'locked'
	locked.mkdir()
	(locked / 'k1').write_text('x')
	locked.chmod(0o555)
	try:
		rc, output = _run(['-rm', '-rf', '-F', str(FILES_PER_SECOND)] + names + ['locked/k1'], tmp_path)
	finally:
		locked.chmod(0o755)
	assert rc != 0 and _frames(output), output
	assert _every_start_of_a_line(output, 'Remove process exception:') == [], output
