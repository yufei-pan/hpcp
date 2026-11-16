'''hpcp 9.95: -rm shows its progress as it removes, as 9.73 did (9.74 to 9.94 printed nothing between "Removing files from"
and the "Removed N files" line). Each test fails on 9.94 unless it says it guards what 9.94 already gets right. The pace
under -F is what makes a removal last: N files at -F 100 take at least N / 100 s whatever the machine, so a frame (every
0.5 s) is always due, and no test waits on a machine's speed.'''
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
# unlinked then, so one is drawn by the walk, in a worker's directory or in the parent's wait. The pace counts from the run's
# start, and a -pfl pool takes its time to start (over a second on a busy machine, Python 3.9 as root: the first version of the
# cadence check failed there), so the run is long enough to show a frame after that.
FILES_PER_SECOND = 100
DIRECTORIES, PER_DIRECTORY = 6, 50
TOTAL = DIRECTORIES * PER_DIRECTORY
# The real program's tests only need a frame before the summary: 200 files, 2 s.
PROGRAM_DIRECTORIES, PROGRAM_PER_DIRECTORY = 4, 50
PROGRAM_TOTAL = PROGRAM_DIRECTORIES * PROGRAM_PER_DIRECTORY
# A frame that comes at the end (a loop that checks the clock only between directories, or after them) is no frame of the run:
# this is how far before the end the first one is.
FIRST_FRAME_BEFORE_THE_END = 0.5
SHAPES = [pytest.param((DIRECTORIES, PER_DIRECTORY), id='tree'), pytest.param((1, TOTAL), id='one-wide-directory')]


def _tree(root, directories=DIRECTORIES, per_directory=PER_DIRECTORY):
	for index in range(directories):
		directory = root / ('d%d' % index)
		directory.mkdir(parents=True)
		for number in range(per_directory):
			(directory / ('f%d' % number)).write_text('x')


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
	time it is due, not once the files are gone: in a wide directory only a check an entry (the walk's unlinks, a -pfl job's
	wait in the parent) can show it.'''
	if not descriptors:
		monkeypatch.setattr(hpcp, '_DIR_FD_WALK', False)
	times = []
	real_show = hpcp._RemoveProgress.show
	def show(self, *args, **kwargs):
		times.append(time.monotonic())
		return real_show(self, *args, **kwargs)
	monkeypatch.setattr(hpcp._RemoveProgress, 'show', show)
	_tree(tmp_path / 'src', *shape)
	hpcp.process_remove([str(tmp_path / 'src')], remove_force=True, **options)
	end = time.monotonic()
	assert not (tmp_path / 'src').exists()
	shown = _before_the_summary(capsys.readouterr().out)
	frames = _frames(shown)
	assert frames, 'no progress line in %r' % shown
	assert end - times[0] >= FIRST_FRAME_BEFORE_THE_END, 'the first frame came %.2f s before the end' % (end - times[0])
	assert re.search(r'iteration:\d+ \| (Removed: [\d.]+ ?K? +)?(Folders: [\d.]+ ?K? +)?(\| )?Overall: [\d.]+ ?K? +Files', frames[-1]), frames[-1]


def test_995_the_line_is_the_listings_with_removed_and_folders(tmp_path, capsys, monkeypatch, reset_hpcp_globals):
	'''The listing's style (get_file_list_serial): the directory's last 50 characters, then "iteration:N | Removed: N Folders:
	M". With the interval 0 every entry shows a frame, and the counts grow to the last.'''
	monkeypatch.setattr(hpcp._RemoveProgress, '_INTERVAL', 0)
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
	monkeypatch.setattr(hpcp._RemoveProgress, '_INTERVAL', 0)
	for index in range(30):
		(tmp_path / 'src' / ('d%d' % index)).mkdir(parents=True)
	hpcp.process_remove([str(tmp_path / 'src')], single_thread=True, remove_force=True)
	frames = _frames(_before_the_summary(capsys.readouterr().out, 0))
	assert len(frames) >= 30 and re.search(r'Folders: 3[01]\.00 ', frames[-1]), frames[-1]
	assert not any('Removed:' in frame for frame in frames)


def test_995_under_the_file_rate_limit_the_line_has_the_overall_count(tmp_path, capsys, monkeypatch, reset_hpcp_globals):
	monkeypatch.setattr(hpcp._RemoveProgress, '_INTERVAL', 0)
	monkeypatch.setattr(hpcp, 'FILES_RATE_LIMIT', 100000)
	_tree(tmp_path / 'src', directories=2, per_directory=50)
	hpcp.process_remove([str(tmp_path / 'src')], single_thread=True, remove_force=True)
	frames = _frames(capsys.readouterr().out)
	assert frames and all('| Overall: ' in frame for frame in frames)
	assert re.search(r'Overall: 100\.00 +Files', frames[-1]), frames[-1]


def test_995_the_walk_adds_to_what_the_pfl_jobs_removed(tmp_path, capsys, monkeypatch, reset_hpcp_globals):
	'''-pfl: the jobs unlink the files, the walk after removes the directories; the line carries the count on.'''
	monkeypatch.setattr(hpcp._RemoveProgress, '_INTERVAL', 0)
	_tree(tmp_path / 'src', directories=4, per_directory=5)
	hpcp.process_remove([str(tmp_path / 'src')], single_thread=False, max_workers=2, parallel_file_listing=True, remove_force=True)
	frames = _frames(_before_the_summary(capsys.readouterr().out, 20))
	assert frames and all('Removed: 20.00 ' in frame for frame in frames[-2:]), frames[-2:]
	assert 'Folders: ' in frames[-1]


def test_995_the_open_line_is_known_to_error_lines_and_ended_by_finish(capsys, monkeypatch, reset_hpcp_globals):
	'''An error line starts on its own line while a frame is open (_BAR_DRAWN), and the summary after finish() does.'''
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


def test_995_a_run_that_shows_no_frame_prints_no_empty_line(tmp_path, capsys, reset_hpcp_globals):
	'''Guards what 9.94 gets right: a removal quicker than the first frame prints no blank line before the summary.'''
	_tree(tmp_path / 'src', directories=2, per_directory=3)
	hpcp.process_remove([str(tmp_path / 'src')], single_thread=True, remove_force=True)
	shown = capsys.readouterr().out
	assert '\n\n' not in shown and _frames(shown) == []
	assert not hpcp._BAR_DRAWN


def test_995_the_library_and_a_moves_walks_print_nothing(tmp_path, capsys, monkeypatch, reset_hpcp_globals):
	'''_rmtree (the library) and the move's directory pass give the walk no progress object: no line, whatever the interval.'''
	monkeypatch.setattr(hpcp._RemoveProgress, '_INTERVAL', 0)
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
	'''At a terminal, sized or not (script's pty has no size: 0 columns), each frame is the one line.'''
	_tree(tmp_path / 'src', PROGRAM_DIRECTORIES, PROGRAM_PER_DIRECTORY)
	rc, shown = _at_a_terminal(['-rm', '-rf', '-F', str(FILES_PER_SECOND), '-j', '2'] + extra + ['src'], tmp_path, size)
	assert rc == 0 and not (tmp_path / 'src').exists(), shown
	frames = _frames(_before_the_summary(shown, PROGRAM_TOTAL))
	assert frames, shown
	assert all(frame.count('iteration:') == 1 and '| Overall: ' in frame for frame in frames), frames


@pytest.fixture
def locked_tree(tmp_path):
	'''src/ with a/ and z/ of 100 files each (at -F 100: the walk is in a/ at the first frame) and src/a/sub/locked/ (mode 0555)
	holding two files the runner can not unlink: each is an error line, reached after the first frame. Root can unlink them.'''
	if os.name != 'posix' or os.geteuid() == 0:
		pytest.skip('needs a directory its owner can not write into (not root)')
	root = tmp_path / 'src'
	locked = root / 'a' / 'sub' / 'locked'
	locked.mkdir(parents=True)
	for directory in ('a', 'z'):
		(root / directory).mkdir(exist_ok=True)
		for number in range(100):
			(root / directory / ('f%d' % number)).write_text('x')
	for name in ('k1', 'k2'):
		(locked / name).write_text('x')
	locked.chmod(0o555)
	yield root
	locked.chmod(0o755)


def _every_start_of_a_line(text, mark, starts='\n'):
	'''The places before the final "Error occurred:" report where mark is not right after a character of starts (a line's start, or
	the terminal's clear-and-return): each is a stretch of text ending in it. Fails when mark does not occur at all.'''
	text = text.partition('Error occurred:')[0]
	places = [match.start() for match in re.finditer(re.escape(mark), text)]
	assert places, 'no %r in %r' % (mark, text)
	return [text[max(0, place - 60):place + len(mark)] for place in places if place and text[place - 1] not in starts]


@posix_only
@pytest.mark.parametrize('extra', [[], ['-pfl']], ids=['serial', 'pfl'])
def test_995_an_error_line_while_the_line_is_open_starts_its_own_line_in_a_log(locked_tree, extra):
	'''9.95's first version: under -pfl a job's "Remove process exception:" came right after the open frame (a combined log: on the
	frame's line, after its padding), as a job of a plain -rm did not know the run drew one. 9.94 drew no frame. (The serial walk
	reports in the parent, which knows: that one guards.)'''
	rc, output = _run(['-rm', '-rf', '-F', str(FILES_PER_SECOND), '-j', '2'] + extra + ['src'], locked_tree.parent)
	assert rc != 0 and not (locked_tree / 'a' / 'f0').exists(), output
	assert _frames(output), output
	assert _every_start_of_a_line(output, 'Remove process exception:') == [], output


@posix_only
@pytest.mark.parametrize('size', [(40, 120), None], ids=['sized-terminal', 'unsized-terminal'])
def test_995_an_error_line_while_the_line_is_open_clears_it_at_a_terminal(locked_tree, size):
	'''At a terminal the job's line starts with the clear-line and return a copy's error line does, in a pty with a size or none.'''
	rc, shown = _at_a_terminal(['-rm', '-rf', '-F', str(FILES_PER_SECOND), '-j', '2', '-pfl', 'src'], locked_tree.parent, size)
	assert rc != 0, shown
	assert _frames(shown), shown
	assert _every_start_of_a_line(shown, 'Remove process exception:', '\n\r') == [], shown


@posix_only
@pytest.mark.parametrize('extra', [[], ['-pfl']], ids=['serial', 'pfl'])
def test_995_a_stop_ends_the_open_line_before_the_report_on_stdout(tmp_path, extra):
	'''A stop (SIGTERM, as Ctrl-C is) once the first frame has come, stdout one pipe and stderr another: "Error occurred: ..." (stdout)
	started right after the open frame, padding and all, as nothing ended the line when the removal was left by the stop. The
	status is what 9.94 gives (the run ends by the signal, "return code: 130." in the report) and the stop's own line is on stderr.'''
	_tree(tmp_path / 'src', PROGRAM_DIRECTORIES, PROGRAM_PER_DIRECTORY)
	proc = subprocess.Popen([sys.executable, HPCP, '-rm', '-rf', '-F', str(FILES_PER_SECOND), '-j', '2'] + extra + ['src'], cwd=str(tmp_path),
		stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	seen = b''
	deadline = time.monotonic() + 60
	try:
		while b'iteration:' not in seen:
			assert time.monotonic() < deadline, 'no frame in 60 s: %r' % seen
			if select.select([proc.stdout], [], [], 0.2)[0]:
				chunk = os.read(proc.stdout.fileno(), 4096)
				assert chunk, 'the run ended without a frame: %r' % seen
				seen += chunk
		proc.send_signal(signal.SIGTERM)
		out, err = proc.communicate(timeout=60)
	finally:
		proc.kill()
		proc.wait()
	output = (seen + out).decode('utf-8', 'replace')
	assert 'Removed %d files' % PROGRAM_TOTAL not in output, 'the run was over before the stop: %r' % output
	assert proc.returncode == -signal.SIGTERM and 'return code: 130.' in output, (proc.returncode, output, err)
	assert 'Keyboard Interrupt' in err.decode('utf-8', 'replace')
	report = output.index('Error occurred: ')
	assert output[report - 1] == '\n', output[max(0, report - 120):report + 60]


@posix_only
def test_995_the_errors_after_the_last_frame_leave_no_blank_lines(locked_tree):
	'''The walk's errors are reported after its last frame, serially: the first ended the frame's line and each next one, and the
	summary after them, added an empty line (the open line was still known). Stdout and stderr one pipe.'''
	rc, output = _run(['-rm', '-rf', '-F', str(FILES_PER_SECOND), 'src'], locked_tree.parent)
	assert rc != 0 and _frames(output), output
	assert output.partition('Error occurred:')[0].count('Remove process exception:') == 2 and '\n\n' not in output, output
