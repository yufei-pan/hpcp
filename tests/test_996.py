'''hpcp 9.96: the removal's progress line where 9.95 left none, and the lines around it. Each test fails on 9.95 unless it says
it guards what 9.95 already gets right. The pace under -F is what makes a run last: N files at -F 100 take at least N / 100 s
whatever the machine, so a frame (every 0.5 s) is due while the work goes on; the in-process tests set the interval to 0
(a frame at every check) instead of waiting on a clock.'''
import os
import re
import select
import subprocess
import sys
import time

import pytest

import hpcp

posix_only = pytest.mark.skipif(os.name != 'posix', reason='POSIX')
HPCP = os.path.abspath(hpcp.__file__)

# 200 files at -F 100: at least 2 s of removal, and a frame due every 0.5 s of it.
FILES_PER_SECOND = 100
DIRECTORIES, PER_DIRECTORY = 4, 50
TOTAL = DIRECTORIES * PER_DIRECTORY


def _tree(root, directories=DIRECTORIES, per_directory=PER_DIRECTORY):
	for index in range(directories):
		directory = os.path.join(str(root), 'd%d' % index)
		os.makedirs(directory)
		for number in range(per_directory):
			open(os.path.join(directory, 'f%d' % number), 'w').close()


def _frames(text, word='iteration:'):
	'''The progress frames in text: each starts with a carriage return and carries "iteration:".'''
	return [frame for frame in re.split(r'[\r\n]', text) if word in frame]


def _starts_a_line(text, mark):
	'''mark is in text, and at a line's start (after a newline, or at a terminal the clear-line and return).'''
	place = text.index(mark)
	assert place == 0 or text[:place].endswith(('\n', '\x1b[2K\r')), 'not at a line\'s start: %r' % text[max(0, place - 80):place + 40]


def _all_start_a_line(text, mark):
	'''mark is in text, and every one of it is at a line's start.'''
	assert mark in text, text
	for found in re.finditer(re.escape(mark), text):
		before = text[:found.start()]
		assert before == '' or before.endswith(('\n', '\x1b[2K\r')), 'not at a line\'s start: %r' % text[max(0, found.start() - 80):found.start() + 40]


def _run(args, cwd):
	'''hpcp as a program: stdout and stderr through one pipe, the bytes as written. (exit status, output)'''
	done = subprocess.run([sys.executable, HPCP] + args, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
		stdin=subprocess.DEVNULL, timeout=120)
	return done.returncode, done.stdout.decode('utf-8', 'replace')


def _at_a_terminal(args, cwd, size):
	'''hpcp with a pty as its stdin, stdout and stderr (a terminal of size (rows, columns), or none: 0 x 0).
	(exit status, what the terminal showed)'''
	import fcntl
	import pty
	import struct
	import termios
	pid, fd = pty.fork()
	if pid == 0:
		# (The child must never return into the test run: a failed chdir or exec ends it.)
		try:
			os.chdir(str(cwd))
			os.execv(sys.executable, [sys.executable, HPCP] + args)
		finally:
			os._exit(127)
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
			os.waitpid(pid, 0)
			pytest.fail('no end in 120 s; the terminal showed %r' % shown)
	finally:
		os.close(fd)
	return os.WEXITSTATUS(os.waitpid(pid, 0)[1]), shown.decode('utf-8', 'replace').replace('\r\n', '\n')


def _removal(output, start='Removing the', end='Time taken to remove files'):
	'''The removal's part of a run's output: from its first line to its summary, which must start a line of its own.'''
	_starts_a_line(output, end)
	return output[output.index(start):output.index(end)]


# ---- 1. A serial move's and -rme's removal, and a move's directory pass ----

@pytest.mark.parametrize('extra', [['-s'], ['-j', '1']], ids=['s', 'one-worker'])
def test_996_a_serial_moves_removal_prints_the_line_through_a_pipe(tmp_path, extra):
	'''9.73 to 9.95: "Removing the N files and links copied" then nothing until "Time taken to remove files" with -s (or one
	worker). Its frames are the removal's ("Removed: N"), not the copy's bar.'''
	_tree(tmp_path / 'src')
	rc, output = _run(['-rm', '-rf', '-F', str(FILES_PER_SECOND)] + extra + ['src', '-d', 'dst/'], tmp_path)
	assert rc == 0 and not (tmp_path / 'src').exists() and len(os.listdir(str(tmp_path / 'dst' / 'src' / 'd0'))) == PER_DIRECTORY, output
	frames = _frames(_removal(output))
	assert frames and all(re.search(r'\| iteration:\d+ \| Removed: [\d.]+ ', frame) for frame in frames), frames


@posix_only
@pytest.mark.parametrize('size', [(40, 120), None], ids=['sized-terminal', 'unsized-terminal'])
def test_996_a_serial_moves_removal_prints_the_line_at_a_terminal(tmp_path, size):
	_tree(tmp_path / 'src')
	rc, shown = _at_a_terminal(['-rm', '-rf', '-s', '-F', str(FILES_PER_SECOND), 'src', '-d', 'dst/'], tmp_path, size)
	assert rc == 0 and not (tmp_path / 'src').exists(), shown
	frames = _frames(_removal(shown))
	assert frames and all(frame.count('iteration:') == 1 and 'Removed: ' in frame for frame in frames), frames


def test_996_a_pooled_moves_directory_pass_draws_no_empty_frame(tmp_path, capsys, monkeypatch, reset_hpcp_globals):
	'''The pass over the source directories after a pool removed the files: its first frame is half a second into it. The build
	before 9.96's first fix round drew it at once (the line's first frame was due since the move began), an empty one ("src |
	iteration:2 |") after the bar's summary. The files' removal here takes 0.6 s, longer than the first frame's wait, and each
	directory the pass removes 0.3 s, so the pass draws frames, and each carries the folders removed so far.'''
	real_rmdir, real_remove_items = os.rmdir, hpcp._remove_items
	def slow_rmdir(path, *args, **kwargs):
		time.sleep(0.3)
		return real_rmdir(path, *args, **kwargs)
	def slow_remove_items(*args, **kwargs):
		result = real_remove_items(*args, **kwargs)
		time.sleep(0.6)
		return result
	monkeypatch.setattr(os, 'rmdir', slow_rmdir)
	monkeypatch.setattr(hpcp, '_remove_items', slow_remove_items)
	src = tmp_path / 'src'
	_tree(src, directories=4, per_directory=2)
	files = [os.path.join(str(src), 'd%d' % index, 'f%d' % number) for index in range(4) for number in range(2)]
	hpcp.remove_moved_sources([str(src)], files, {}, single_thread=False, max_workers=2, remove_force=True)
	assert not src.exists()
	frames = _frames(_removal(capsys.readouterr().out))
	assert frames and all(re.search(r'(Removed|Folders): [\d.]+ ', frame) for frame in frames), frames


@pytest.mark.parametrize('single_thread', [True, False], ids=['serial', 'pool'])
def test_996_a_moves_directory_pass_shows_the_line_in_every_mode(tmp_path, capsys, monkeypatch, reset_hpcp_globals, single_thread):
	'''The pass over the source directories after the files are gone (a pool's too: its jobs show the "Deleted" bar, the pass
	is the parent's) draws the line with the directories it removes: 9.95 drew none in any mode.'''
	monkeypatch.setattr(hpcp, '_LINE_INTERVAL', 0)
	src = tmp_path / 'src'
	_tree(src, directories=3, per_directory=2)
	files = [os.path.join(str(src), 'd%d' % index, 'f%d' % number) for index in range(3) for number in range(2)]
	hpcp.remove_moved_sources([str(src)], files, {}, single_thread=single_thread, max_workers=2, remove_force=True)
	assert not src.exists()
	shown = capsys.readouterr().out
	# (A frame comes before each directory: the last one counts all but the root.)
	frames = _frames(_removal(shown))
	assert any(re.search(r'Folders: 3\.00 ', frame) for frame in frames), frames


@pytest.mark.parametrize('descriptors', [True, False], ids=['descriptors', 'no-descriptors'])
def test_996_rme_removes_its_extra_files_and_directories_with_the_line(tmp_path, capsys, monkeypatch, reset_hpcp_globals, descriptors):
	'''-rme with -s: the extra files' removal and the extra directories' (every mode) draw the line, the counts growing to the
	total. 9.95 printed "Deleting extra directory:" lines and nothing while it removed.'''
	if not descriptors:
		monkeypatch.setattr(hpcp, '_DIR_FD_WALK', False)
	monkeypatch.setattr(hpcp, '_LINE_INTERVAL', 0)
	monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
	monkeypatch.setattr('builtins.input', lambda: 'y')
	(tmp_path / 'src').mkdir()
	(tmp_path / 'src' / 'kept').write_text('x')
	_tree(tmp_path / 'out' / 'src', directories=3, per_directory=4)
	assert hpcp.hpcp([str(tmp_path / 'src')], dest_paths=[str(tmp_path / 'out') + os.sep], remove_extra=True, single_thread=True) == 0
	assert sorted(os.listdir(str(tmp_path / 'out' / 'src'))) == ['kept']
	shown = capsys.readouterr().out
	# (A frame comes before each entry: the last one counts all but one.)
	files = _frames(_removal(shown, 'Do you want to delete them?', 'Time taken to remove extra files'))
	assert files and re.search(r'Removed: 11\.00 ', files[-1]) and not any('Folders:' in frame for frame in files), files
	directories = _frames(shown[shown.index('Deleting extra directory:'):])
	assert directories and re.search(r'Folders: 2\.00 ', directories[-1]), directories
	# The separator that follows the pass starts its own line (remove_extra_dirs ends the line the pass left open).
	_starts_a_line(shown[shown.index('Deleting extra directory:'):], '-' * 80)
	assert shown.endswith('Done.\n')
	_starts_a_line(shown, 'Done.')


def test_996_a_note_of_the_extra_directories_pass_starts_its_own_line(tmp_path, capsys, reset_hpcp_globals, monkeypatch):
	'''"Keeping non-empty directory:" (stdout) while the line is open: the frame's line ends first.'''
	monkeypatch.setattr(hpcp, '_LINE_INTERVAL', 0)
	(tmp_path / 'out' / 'a').mkdir(parents=True)
	(tmp_path / 'out' / 'b').mkdir()
	(tmp_path / 'out' / 'b' / 'stays').write_text('x')
	root = str(tmp_path / 'out')
	identity = hpcp._directory_identity(root)
	progress = hpcp._RemoveProgress()
	hpcp._remove_listed_files([(root, identity, 'a'), (root, identity, 'b')], rmdir=True, progress=progress)
	hpcp._end_line()
	shown = capsys.readouterr().out
	assert _frames(shown), shown
	_starts_a_line(shown, 'Keeping non-empty directory:')
	assert not (tmp_path / 'out' / 'a').exists() and progress.folders == 1


# ---- 2. A -pfl listing's line (and the listing's time on a line of its own) ----

# 40 folders of 5 files: the -pfl listing paces its jobs at -F 100 (200 files: 2 s), so the parent's loop has frames to draw.
LISTING_FOLDERS, LISTING_PER_FOLDER = 40, 5
# A listing's frame: the seconds, then the files listed so far (none before the first folder's), the folders (the root's own).
LISTING_FRAME = r'\| iteration:\d+ \| (Files: [\d.]+ +)?Folders: [\d.]+ '


def _listing(output):
	'''The listing's part of a copy's output, from "Getting file list for" to "Time taken to get file list", which must start a
	line of its own (9.73 to 9.95 printed it on the last frame's line).'''
	_starts_a_line(output, 'Time taken to get file list')
	return output[output.index('Getting file list for'):output.index('Time taken to get file list')]


@pytest.mark.parametrize('extra', [['-pfl'], []], ids=['pfl', 'serial'])
def test_996_a_listing_prints_its_line_through_a_pipe(tmp_path, extra):
	'''-pfl: 9.73 to 9.95 printed nothing between "Getting file list for" and "Time taken to get file list" in a tree of small
	folders (the jobs list, the parent waited in silence). Serial: the line was there, and the time landed on its last frame.'''
	_tree(tmp_path / 'src', LISTING_FOLDERS, LISTING_PER_FOLDER)
	rc, output = _run(['-F', str(FILES_PER_SECOND), '-j', '2'] + extra + ['src', '-d', 'dst/'], tmp_path)
	assert rc == 0 and len(os.listdir(str(tmp_path / 'dst' / 'src'))) == LISTING_FOLDERS, output
	# (A frame drawn before the first job's report has no file yet: the root holds folders only. A pool's start under a CPU
	# quota, with other tests beside it, took over 0.5 s.)
	frames = _frames(_listing(output))
	assert frames and all(re.search(LISTING_FRAME, frame) for frame in frames), frames


@posix_only
@pytest.mark.parametrize('size', [(40, 120), None], ids=['sized-terminal', 'unsized-terminal'])
def test_996_a_pfl_listing_prints_its_line_at_a_terminal(tmp_path, size):
	_tree(tmp_path / 'src', LISTING_FOLDERS, LISTING_PER_FOLDER)
	rc, shown = _at_a_terminal(['-F', str(FILES_PER_SECOND), '-j', '2', '-pfl', 'src', '-d', 'dst/'], tmp_path, size)
	assert rc == 0, shown
	frames = _frames(_listing(shown))
	assert frames and all(frame.count('iteration:') == 1 and re.search(LISTING_FRAME, frame) for frame in frames), frames


def test_996_a_pfl_jobs_listing_draws_no_frame(tmp_path, capsys, monkeypatch, reset_hpcp_globals):
	'''A -pfl job lists its folder in another process: its frames (a folder whose listing takes over 0.5 s drew them in 9.73 to
	9.95) would share the parent's line. The jobs are asked for none, and a listing asked for none draws none.'''
	monkeypatch.setattr(hpcp, '_LINE_INTERVAL', 0)
	_tree(tmp_path / 'src', 3, 4)
	submitted = []
	real = hpcp._submit_task
	def submit(executor, func, *args, **kwargs):
		if func is hpcp.get_file_list_serial:
			submitted.append(kwargs.get('show', True))
		return real(executor, func, *args, **kwargs)
	monkeypatch.setattr(hpcp, '_submit_task', submit)
	files, _, _, folders = hpcp.get_file_list_parallel(str(tmp_path / 'src'), max_workers=2)
	assert len(files) == 12 and len(folders) == 4
	assert submitted == [False] * 3
	# (The parent's own frames leave the line open: get_file_list ends it.)
	hpcp._end_line()
	capsys.readouterr()
	files, _, _, _ = hpcp.get_file_list_serial(str(tmp_path / 'src'), show=False)
	assert len(files) == 12 and capsys.readouterr().out == ''
	assert not hpcp._BAR_DRAWN


@posix_only
@pytest.mark.skipif(sys.platform == 'darwin', reason='forks the pool; macOS starts workers with spawn (fork there can crash a child)')
def test_996_a_pfl_listing_counts_a_wide_folder_while_its_job_lists_it(tmp_path, capsys, monkeypatch, reset_hpcp_globals):
	'''One -pfl job lists a folder of many entries: its frames name that folder, and the count rises as the job lists it (the job
	puts what it has listed in its worker's live cells; the parent shows them with what the finished jobs returned). 9.96's frames
	named the folder a job last finished, with the counts of the finished jobs only, until the wide folder's job returned (9.95's
	job drew its own frames). Forked workers, so the patches below reach them: the interval 0 (the job publishes at every entry
	and the parent draws a frame at every pass) and a listing that takes 1 ms an entry (300 entries last 0.3 s). The small folder
	holds files and two subfolders: no frame counts more than the tree has (what a finished job returned is not counted again
	from its worker's cells), and a wide folder's frame shows more than the small folder's whole count.'''
	import multiprocessing
	monkeypatch.setattr(hpcp, '_LINE_INTERVAL', 0)
	real = hpcp.get_file_repr
	def slow(*args):
		time.sleep(0.001)
		return real(*args)
	monkeypatch.setattr(hpcp, 'get_file_repr', slow)
	wide, small = tmp_path / 'src' / 'wide', tmp_path / 'src' / 'small'
	wide.mkdir(parents=True)
	(small / 's0').mkdir(parents=True)
	(small / 's1').mkdir()
	for number in range(300):
		(wide / ('f%d' % number)).write_text('x')
	for number in range(100):
		(small / ('f%d' % number)).write_text('x')
	method = multiprocessing.get_start_method()
	multiprocessing.set_start_method('fork', force=True)
	try:
		files, _, _, folders = hpcp.get_file_list_parallel(str(tmp_path / 'src'), max_workers=2)
	finally:
		multiprocessing.set_start_method(method, force=True)
	hpcp._end_line()
	assert len(files) == 400 and len(folders) == 5
	def count(frame, word):
		found = re.search(word + r': ([\d.]+) ', frame)
		return float(found.group(1)) if found else 0
	frames = [(frame.split(' | ')[0], count(frame, 'Files'), count(frame, 'Folders')) for frame in _frames(capsys.readouterr().out)]
	counts = [files for _, files, _ in frames]
	assert counts == sorted(counts), counts
	# Never more than is there: what the finished jobs returned is not counted again from their workers' cells.
	assert all(files <= 400 and folders <= 5 for _, files, folders in frames), frames
	assert any(name.endswith('wide') and 100 < files < 400 for name, files, _ in frames), frames


# ---- 3. One mark for stdout's open line: no blank lines between error lines, and a stop's report on a line of its own ----

not_root = pytest.mark.skipif(os.name != 'posix' or os.geteuid() == 0, reason='root reads and unlinks what these tests lock')


def _locked(directory, names):
	'''directory (made) holding names, mode 0555: its files can not be unlinked by their owner (not root).'''
	os.makedirs(str(directory))
	for name in names:
		open(os.path.join(str(directory), name), 'w').close()
	os.chmod(str(directory), 0o555)


def _unlock(root):
	for directory, _, _ in os.walk(str(root)):
		os.chmod(directory, 0o755)


@not_root
def test_996_error_lines_of_named_files_leave_no_blank_line_in_a_combined_log(tmp_path):
	'''hpcp -rm with 200 files named as arguments and three more that can not be unlinked, stdout and stderr one pipe: once the
	line had drawn, 9.95 put an empty line after each error line (the mark stayed set after an error line had ended the frame's
	line). Each error line starts a line, and none is glued to a frame.'''
	_tree(tmp_path / 'src', 1, TOTAL)
	_locked(tmp_path / 'locked', ['k0', 'k1', 'k2'])
	try:
		rc, output = _run(['-rm', '-rf', '-F', str(FILES_PER_SECOND), '-s'] + [os.path.join('src', 'd0', 'f%d' % number) for number in range(TOTAL)]
			+ ['locked/k0', 'locked/k1', 'locked/k2'], tmp_path)
	finally:
		_unlock(tmp_path / 'locked')
	assert rc == 192 and _frames(output), output
	assert output.count('Remove process exception:') == 3 + 3 and '\n\n' not in output, output
	_all_start_a_line(output.partition('Error occurred:')[0], 'Remove process exception:')


@not_root
def test_996_a_pfl_jobs_error_lines_leave_no_blank_line_in_a_combined_log(tmp_path):
	'''-rm -pfl: a job is another process and can not see the parent's line, so 9.95 started each of its error lines with a line
	end, an empty line where the line was already ended. The jobs' lines are left out now (the walk after them reports each file
	once), and the walk's start a line of their own.'''
	_tree(tmp_path / 'src', 1, 300)
	_locked(tmp_path / 'src' / 'd0' / 'sub' / 'locked', ['k0', 'k1', 'k2'])
	try:
		rc, output = _run(['-rm', '-rf', '-F', str(FILES_PER_SECOND), '-j', '2', '-pfl', 'src'], tmp_path)
	finally:
		_unlock(tmp_path / 'src')
	assert rc == 192 and _frames(output), output
	before = output.partition('Error occurred:')[0]
	assert '\n\n' not in before, output
	_all_start_a_line(before, 'Remove process exception:')


@not_root
def test_996_a_copys_job_error_lines_leave_no_blank_line_in_a_combined_log(tmp_path):
	'''A copy's job meets several files it can not read: 9.94 and 9.95 started each of its error lines with a line end, an empty
	line before each one after the first that no frame came between (16 of the 50 in one run). Each is still at a line's start.'''
	_tree(tmp_path / 'src', 2, 100)
	for number in range(0, 100, 2):
		os.chmod(str(tmp_path / 'src' / 'd1' / ('f%d' % number)), 0)
	rc, output = _run(['-F', str(FILES_PER_SECOND), '-j', '2', 'src', '-d', 'dst/'], tmp_path)
	assert rc == 190, output
	before = output.partition('Error occurred:')[0]
	assert len(re.findall(r'(?m)^Copy failed: ', before)) == 50 and '\n\nCopy failed' not in before, output


def _stopped(args, cwd, marker):
	'''hpcp as a program, stdout and stderr two pipes, sent SIGTERM as soon as marker (a regular expression over bytes) matches
	what has come on stdout. (exit status, stdout, stderr)'''
	import signal
	proc = subprocess.Popen([sys.executable, HPCP] + args, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
		stderr=subprocess.PIPE)
	seen = b''
	deadline = time.monotonic() + 60
	try:
		while not re.search(marker, seen):
			assert time.monotonic() < deadline, 'no %r in 60 s: %r' % (marker, seen)
			if select.select([proc.stdout], [], [], 0.2)[0]:
				chunk = os.read(proc.stdout.fileno(), 4096)
				assert chunk, 'the run ended before %r: %r' % (marker, seen)
				seen += chunk
		proc.send_signal(signal.SIGTERM)
		out, err = proc.communicate(timeout=60)
	finally:
		proc.kill()
		proc.wait()
	return proc.returncode, (seen + out).decode('utf-8', 'replace'), err.decode('utf-8', 'replace')


# The pace of the stop tests: 200 files at -F 20 last 10 s, so the stop at the marker always comes before the run's end.
STOP_PACE = '20'
# A bar's frame under -F carries "F/s Limit" (its prefix can be cut at a narrow width); the copy's comes after "Processing N files".
STOPPED_BARS = [
	pytest.param(['-F', STOP_PACE, '-j', '2', 'src', '-d', 'dst/'], rb'Processing \d+ files[^\r]*\r[^\n]*F/s Limit', id='copy-bar'),
	pytest.param(['-rm', '-rf', '-rwloff', '-F', STOP_PACE, '-j', '2', 'src'], rb'F/s Limit', id='rwloff-bar'),
	pytest.param(['-F', STOP_PACE, '-j', '2', 'src', '-d', 'dst/'], rb'iteration:', id='listing-line'),
]


@posix_only
@pytest.mark.parametrize('args, marker', STOPPED_BARS)
def test_996_a_stop_under_a_bar_ends_its_line_before_the_report_on_stdout(tmp_path, args, marker):
	'''stdout and stderr apart (> out 2> err), a SIGTERM while a copy's bar, -rwloff's "Deleted" bar or the listing's line is up:
	9.94 and 9.95 printed "Error occurred: ..." on that line in stdout (a stop skips the bar's own end of line).'''
	_tree(tmp_path / 'src')
	returncode, output, err = _stopped(args, tmp_path, marker)
	assert returncode == -15 and 'Keyboard Interrupt' in err, (returncode, output, err)
	assert 'return code: 130.' in output, output
	_starts_a_line(output, 'Error occurred: ')


def test_996_an_error_line_ends_the_open_line_only_where_it_is_stdouts_file(capsys, reset_hpcp_globals):
	'''One mark: an error line whose start ended stdout's line (stderr is stdout's file or terminal) clears it, so the next one
	starts no empty line; where stderr is elsewhere (a terminal, stdout a file) stdout's line is still open, for its next line to end.'''
	pid = os.getpid()
	try:
		hpcp._BAR_DRAWN = True
		hpcp._LINE_START[:] = [pid, '\n', True]
		hpcp.eprint('Remove process exception: one')
		hpcp.eprint('Remove process exception: two')
		assert capsys.readouterr().err == '\nRemove process exception: one\nRemove process exception: two\n' and not hpcp._BAR_DRAWN
		hpcp._BAR_DRAWN = True
		hpcp._LINE_START[:] = [pid, '\x1b[2K\r', False]
		hpcp.eprint('Remove process exception: three')
		assert capsys.readouterr().err == '\x1b[2K\rRemove process exception: three\n' and hpcp._BAR_DRAWN
		assert hpcp.get_rc_from_error() == 192
		assert capsys.readouterr().out.startswith('\nError occurred: ') and not hpcp._BAR_DRAWN
	finally:
		# (Not the fixture's to reset: a failure here must not leave this process's line start to the tests after it.)
		hpcp._LINE_START[:] = [None, '', False]


def test_996_a_tasks_error_lines_are_printed_by_the_parent_once(capsys, monkeypatch, reset_hpcp_globals):
	'''A pool task's eprint goes back with its result (_run_task), and the parent prints it as it takes the result (_settled); a
	caller whose next step reports the same again takes the result without them (report=False). (Where the line starts: the
	combined-log tests above.)'''
	import concurrent.futures
	monkeypatch.setattr(hpcp, '_worker_initialized', True)
	settings = {name: getattr(hpcp, name) for name in hpcp._TASK_SETTINGS}
	result = hpcp._run_task(settings, hpcp.eprint, 'Copy failed: a task\'s line')
	assert capsys.readouterr().err == '' and result[3] == ['Copy failed: a task\'s line\n']
	def future():
		done = concurrent.futures.Future()
		done.set_result(result)
		return done
	assert hpcp._task_result(future(), 'default', 'Copy process crashed') is None
	assert capsys.readouterr().err == 'Copy failed: a task\'s line\n' and hpcp.ERRORS == ['Copy failed: a task\'s line']
	assert hpcp._task_result(future(), 'default', 'Copy process crashed', report=False) is None
	assert capsys.readouterr().err == '' and hpcp.ERRORS == ['Copy failed: a task\'s line']


# ---- 4. A file -rm could not remove is reported once ----

@not_root
@pytest.mark.parametrize('extra', [['-pfl'], ['-rwloff']], ids=['pfl', 'rwloff'])
def test_996_an_unremovable_file_is_reported_once(tmp_path, extra):
	'''-rm with pool jobs: the job that met a file it could not unlink reported it, and the walk after the jobs met it again and
	reported it too (9.94 and 9.95: 6 lines and 6 entries in the list for 3 files). The walk reports it, once; the exit status is
	192 as before. Under -rwloff the "Removed N files and links" summary counts what was unlinked, not the files the jobs tried.'''
	_tree(tmp_path / 'src', 2, 10)
	_locked(tmp_path / 'src' / 'locked', ['k0', 'k1', 'k2'])
	try:
		rc, output = _run(['-rm', '-rf', '-j', '2'] + extra + ['src'], tmp_path)
	finally:
		_unlock(tmp_path / 'src')
	assert rc == 192 and sorted(os.listdir(str(tmp_path / 'src'))) == ['locked'], output
	lines, _, report = output.partition('Error occurred: ')
	assert lines.count('Remove process exception:') == 3 and report.count('Remove process exception:') == 3, output
	# The summary counts what was unlinked (20), not the 23 listed (9.95 and 9.96's -rwloff counted the failed unlinks too).
	assert 'Removed 20 files and links from' in lines, output
	for name in ('k0', 'k1', 'k2'):
		assert lines.count(os.path.join('locked', name) + ' generated an exception') == 1, output


@not_root
@pytest.mark.parametrize('extra', [[], ['-pfl']], ids=['serial', 'pfl'])
def test_996_an_unlistable_directory_is_reported_once(tmp_path, extra):
	'''A directory nobody can list (mode 0): a -pfl job's failure to list it is left out with the rest of its lines, and the walk
	after the jobs reports it, once (9.94 and 9.95 reported it twice under -pfl). The serial case is the control: with no jobs the
	walk alone reports the directory once (9.95 did too, so it passes on every build), and the -pfl case must say the same.'''
	_tree(tmp_path / 'src', 2, 10)
	closed = tmp_path / 'src' / 'd1' / 'closed'
	closed.mkdir()
	(closed / 'x').write_text('x')
	closed.chmod(0)
	try:
		rc, output = _run(['-rm', '-rf', '-j', '2'] + extra + ['src'], tmp_path)
	finally:
		closed.chmod(0o755)
	assert rc == 192, output
	lines, _, report = output.partition('Error occurred: ')
	assert lines.count('Remove process exception:') == 1 and lines.count(os.path.join('d1', 'closed')) == 1, output
	assert 'Removed 20 files and links from' in lines, output


@not_root
@pytest.mark.parametrize('extra', [['-s'], ['-j', '2']], ids=['serial', 'pool'])
def test_996_a_move_reports_an_unremovable_listed_file_once(tmp_path, extra):
	'''A move whose source has files that can not be unlinked: the removal of the listed files reports each, and the pass over the
	directories no longer names them again as "entries the copy did not list" (9.94 and 9.95 did, with the wrong reason).'''
	_tree(tmp_path / 'src', 2, 10)
	_locked(tmp_path / 'src' / 'locked', ['k0', 'k1', 'k2'])
	try:
		rc, output = _run(['-rm', '-rf'] + extra + ['src', '-d', 'dst/'], tmp_path)
	finally:
		_unlock(tmp_path / 'src')
		_unlock(tmp_path / 'dst')
	assert rc == 192, output
	lines, _, report = output.partition('Error occurred: ')
	assert lines.count('Remove process exception:') == 3 and report.count('Remove process exception:') == 3, output
	assert 'Move left entries' not in output, output
	assert sorted(os.listdir(str(tmp_path / 'src' / 'locked'))) == ['k0', 'k1', 'k2']


# ---- 5. A reader that closes the pipe early ----

def _reader_leaves_early(args, cwd, joined=False):
	'''hpcp ARGS | head -1 (joined: 2>&1 | head -1), head gone at the first line. (exit status, what hpcp wrote to stderr: empty
	where it is the pipe too)'''
	hp = subprocess.Popen([sys.executable, HPCP] + args, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT if joined else subprocess.PIPE)
	try:
		head = subprocess.Popen(['head', '-1'], stdin=hp.stdout, stdout=subprocess.PIPE)
		hp.stdout.close()
		first = head.communicate(timeout=120)[0]
		# The run is paced past 2 s and head left at its first line: what hpcp writes from here goes to a closed pipe.
		assert hp.poll() is None and first, first
		err = b'' if joined else hp.stderr.read()
		if not joined:
			hp.stderr.close()
		return hp.wait(timeout=120), err
	finally:
		if hp.poll() is None:
			hp.kill()
			hp.wait()


@posix_only
@pytest.mark.parametrize('joined', [False, True], ids=['stderr-apart', 'stderr-one-pipe'])
@pytest.mark.parametrize('args, done', [
	pytest.param(['-rm', '-rf', '-F', str(FILES_PER_SECOND), 'src'], 'removed', id='rm'),
	pytest.param(['-rm', '-rf', '-F', str(FILES_PER_SECOND), '-j', '2', '-pfl', 'src'], 'removed', id='rm-pfl'),
	pytest.param(['-F', str(FILES_PER_SECOND), '-j', '2', 'src', '-d', 'dst/'], 'copied', id='copy'),
])
def test_996_a_reader_that_leaves_early_ends_nothing(tmp_path, args, done, joined):
	'''hpcp ... | head -1: head leaves after the first line (the command line hpcp echoes at its start), and the run goes on writing
	to a closed pipe. 9.95: -rm removed the tree and exited 120 with a BrokenPipeError note (the interpreter's last flush), a copy
	ended at its pool's start with exit 70 and a traceback, nothing copied. Now the run's output is dropped from there and it goes
	on to its end: exit 0, the work done, nothing on stderr. (With stderr in the same pipe there is no stderr to read: that case
	shows it by the exit code and the work.)'''
	_tree(tmp_path / 'src')
	rc, err = _reader_leaves_early(args, tmp_path, joined)
	assert rc == 0 and err == b'', err
	if done == 'removed':
		assert not (tmp_path / 'src').exists()
	else:
		assert sum(len(files) for _, _, files in os.walk(str(tmp_path / 'dst'))) == TOTAL


@not_root
def test_996_a_closed_stderr_pipe_loses_error_lines_not_the_run(tmp_path):
	'''2>&1 | head -1 with files the copy can not read: the error lines come after head has gone (on a slow host a line may come
	first; the checks hold either way), into a closed pipe on both streams. The run goes on and ends with its own code (190),
	every readable file copied. (The lines are lost, which a closed pipe can not show: the test checks the code and the copy.)'''
	_tree(tmp_path / 'src')
	for number in range(3):
		os.chmod(str(tmp_path / 'src' / 'd1' / ('f%d' % number)), 0)
	rc, _ = _reader_leaves_early(['-F', str(FILES_PER_SECOND), '-j', '2', 'src', '-d', 'dst/'], tmp_path, joined=True)
	assert rc == 190
	assert sum(len(files) for _, _, files in os.walk(str(tmp_path / 'dst'))) == TOTAL - 3


# ---- 6. -rm -pfl counts a wide directory while one job empties it ----

@posix_only
def test_996_pfl_counts_a_wide_directory_as_its_job_empties_it(tmp_path):
	'''One directory of 100 files is one -pfl job, paced at -F 20 (5 s). 9.95's line had no count until the job returned (only the
	seconds, and "Overall" under -F). Each worker now puts what it has unlinked in a live count every 0.5 s, so a frame drawn while
	the job runs shows part of the directory removed.'''
	_tree(tmp_path / 'src', 1, 100)
	rc, output = _run(['-rm', '-rf', '-F', '20', '-j', '2', '-pfl', 'src'], tmp_path)
	assert rc == 0 and not (tmp_path / 'src').exists(), output
	counts = [float(re.search(r'Removed: ([\d.]+) ', frame).group(1)) for frame in _frames(output) if 'Removed:' in frame]
	assert any(0 < count < 100 for count in counts), counts
	assert counts == sorted(counts), counts
	# The frame names the directory the running job empties, not the last one a job finished (the root's, which listed it).
	assert any(0 < float(re.search(r'Removed: ([\d.]+) ', frame).group(1)) < 100 and frame.split(' | ')[0].endswith('/src/d0')
		for frame in _frames(output) if 'Removed:' in frame), _frames(output)


def test_996_pfl_removal_needs_no_shared_array(tmp_path, capsys, monkeypatch, reset_hpcp_globals):
	'''The jobs' live counts are shared memory made as the pool's slots are (_new_live): where none can be made the removal goes on
	(the line then shows what the jobs return; this test checks the removal and its summary). Python 3.6 backs a
	multiprocessing.Array with a file in the temporary directory, which a full disk (the case -rm is for) cannot take: 9.96 exited
	70 with a traceback there, nothing removed (9.95 made it under -F only). Both ways it is made are refused here: the anonymous
	mapping forked workers inherit (mmap) and the Array the other start methods get.'''
	import mmap
	import multiprocessing
	def refuse(*args, **kwargs):
		raise OSError('No usable temporary directory found')
	monkeypatch.setattr(multiprocessing, 'Array', refuse)
	monkeypatch.setattr(mmap, 'mmap', refuse)
	_tree(tmp_path / 'src', 3, 4)
	hpcp.process_remove([str(tmp_path / 'src')], single_thread=False, max_workers=2, parallel_file_listing=True, remove_force=True)
	assert not (tmp_path / 'src').exists()
	assert 'Removed 12 files and links' in capsys.readouterr().out


# ---- 7. A directory whose listing alone takes long shows the line while it is listed ----

@pytest.mark.skipif(not hpcp._DIR_FD_WALK, reason='no descriptor calls here (Windows): the walk lists through _walk, tested below')
@pytest.mark.parametrize('scandir', [True, False], ids=['scandir', 'listdir'])
def test_996_the_walk_shows_the_line_while_it_lists_a_directory(tmp_path, capsys, monkeypatch, reset_hpcp_globals, scandir):
	'''The walk lists a whole directory before it unlinks its first entry: 9.95 checked the time only at the unlinks, so a
	directory whose listing alone took over 0.5 s (millions of entries, a slow network filesystem) showed no line until it was
	listed. With the interval 0 every entry the listing reads is a frame's chance: frames come from inside the listing, with the
	counts so far.'''
	if not scandir:
		monkeypatch.setattr(hpcp, '_SCANDIR_FD', False)
		monkeypatch.setattr(hpcp, '_PROC_FD', False)
	monkeypatch.setattr(hpcp, '_LINE_INTERVAL', 0)
	_tree(tmp_path / 'src', 2, 20)
	shown, during = [], []
	real_show, real_entries = hpcp._RemoveProgress.show, hpcp._dir_entries
	def show(self, path, files=0, folders=0):
		shown.append(files + self.files)
		return real_show(self, path, files, folders)
	def entries(*args, **kwargs):
		before = len(shown)
		result = real_entries(*args, **kwargs)
		during.append(len(shown) - before)
		return result
	monkeypatch.setattr(hpcp._RemoveProgress, 'show', show)
	monkeypatch.setattr(hpcp, '_dir_entries', entries)
	hpcp.process_remove([str(tmp_path / 'src')], single_thread=True, remove_force=True)
	assert not (tmp_path / 'src').exists()
	# The root's listing (2 directories) and each directory's (20 files): a frame an entry.
	assert sorted(during) == [2, 20, 20], during
	assert shown == sorted(shown) and shown[-1] == 40, shown


def test_996_the_walk_without_descriptor_calls_shows_the_line_while_it_lists(tmp_path, capsys, monkeypatch, reset_hpcp_globals):
	'''The walk where there are no descriptor calls (Windows': os.walk's way, _walk) lists a directory before its files are
	unlinked too: its listing draws the line, with the interval 0 a frame at every entry its listing reads (the frames of the
	listing itself, not the unlinks' and the rmdirs' after it).'''
	monkeypatch.setattr(hpcp, '_DIR_FD_WALK', False)
	monkeypatch.setattr(hpcp, '_LINE_INTERVAL', 0)
	_tree(tmp_path / 'src', 2, 20)
	shown, ticks = [], []
	real_show = hpcp._RemoveProgress.show
	def show(self, *args, **kwargs):
		shown.append(args)
		return real_show(self, *args, **kwargs)
	monkeypatch.setattr(hpcp._RemoveProgress, 'show', show)
	real_walk = hpcp._walk
	def walk(*args, **kwargs):
		tick = kwargs.get('tick')
		def counted(dirpath):
			before = len(shown)
			tick(dirpath)
			ticks.append(len(shown) - before)
		if tick is not None:
			kwargs['tick'] = counted
		return real_walk(*args, **kwargs)
	monkeypatch.setattr(hpcp, '_walk', walk)
	hpcp.process_remove([str(tmp_path / 'src')], single_thread=True, remove_force=True)
	assert not (tmp_path / 'src').exists()
	# The walk's listing ticks at every entry it reads (2 + 20 + 20 and the roots'), and each tick drew a frame.
	assert len(ticks) == 42 and ticks == [1] * 42, ticks


# ---- 8. Found in the review of 9.96 ----

def _mark_in_a_worker():
	'''What a pool worker's stdout line mark is (a module-level function: a task is pickled by its name).'''
	return hpcp._BAR_DRAWN


@posix_only
def test_996_a_forked_worker_starts_with_no_line_open(monkeypatch, reset_hpcp_globals):
	'''A worker forked while the parent's progress line was open has the parent's mark: the line is the parent's, and the notes a
	worker prints itself ("Keeping non-empty directory:" ends the open line first) must not end it. Forked workers (the default
	before Python 3.14) are the ones that inherit it.'''
	import multiprocessing
	monkeypatch.setattr(hpcp, '_worker_initialized', False)
	method = multiprocessing.get_start_method()
	multiprocessing.set_start_method('fork', force=True)
	try:
		hpcp._BAR_DRAWN = True
		with hpcp._pool(1) as executor:
			result, errors, crash = hpcp._settled(hpcp._submit_task(executor, _mark_in_a_worker))
	finally:
		multiprocessing.set_start_method(method, force=True)
	assert crash is None and result is False, (result, crash)


@pytest.mark.parametrize('how', ['arguments', 'stored-list'])
def test_996_rme_with_rf_removes_the_extras_without_asking(tmp_path, how):
	'''-rf is "Remove all files without prompt" (its help): 9.95 asked all the same before -rme removed its extra files, and with no
	terminal to ask (stdin at EOF) stopped with exit 147, nothing removed. (-rf with a destination is a move: the sources go too.)'''
	(tmp_path / 'src' / 'd').mkdir(parents=True)
	(tmp_path / 'src' / 'd' / 'f').write_text('x')
	if how == 'arguments':
		(tmp_path / 'out' / 'src').mkdir(parents=True)
		(tmp_path / 'out' / 'src' / 'extra').write_text('x')
		rc, output = _run(['src', '-d', 'out/', '-rme', '-rf'], tmp_path)
		assert rc == 0 and sorted(os.listdir(str(tmp_path / 'out' / 'src'))) == ['d'] and not (tmp_path / 'src').exists(), output
	else:
		assert _run(['src', '-fl', 'list.tsv'], tmp_path)[0] == 0
		(tmp_path / 'src' / 'extra').write_text('x')
		rc, output = _run(['src', '-fl', 'list.tsv', '-rme', '-rf'], tmp_path)
		assert sorted(os.listdir(str(tmp_path / 'src'))) == ['d'], output
	assert 'Do you want to delete them?' not in output, output


def test_996_a_listing_after_named_files_starts_its_own_line(tmp_path):
	'''-rm -rwloff of named files and a tree: the named files' removal leaves its line open, and "Getting file list for" starts a
	line of its own.'''
	_tree(tmp_path / 'tree', 1, 5)
	_tree(tmp_path / 'one', 1, TOTAL)
	named = [os.path.join('one', 'd0', 'f%d' % number) for number in range(TOTAL)]
	rc, output = _run(['-rm', '-rf', '-rwloff', '-F', str(FILES_PER_SECOND), '-j', '2'] + named + ['tree'], tmp_path)
	assert rc == 0 and not (tmp_path / 'tree').exists(), output
	assert _frames(output[:output.index('Getting file list for')]), output
	_starts_a_line(output, 'Getting file list for')


@not_root
@pytest.mark.parametrize('extra', [['-pfl'], []], ids=['pfl', 'serial'])
def test_996_a_listing_error_while_the_listing_line_is_open_starts_its_own_line(tmp_path, extra):
	'''A folder the listing can not read, met while its progress line is up: its "Listing error" is at a line's start, with no
	empty line before it, in both listings (9.95: -pfl printed one after "Getting file list for", serial glued it to a frame).'''
	_tree(tmp_path / 'src', LISTING_FOLDERS, LISTING_PER_FOLDER)
	closed = tmp_path / 'src' / 'd7'
	closed.chmod(0)
	try:
		rc, output = _run(['-F', str(FILES_PER_SECOND), '-j', '2'] + extra + ['src', '-d', 'dst/'], tmp_path)
	finally:
		closed.chmod(0o755)
	assert rc == 209, output
	listing = _listing(output)
	assert _frames(listing) and '\n\n' not in listing, output
	_all_start_a_line(output.partition('Error occurred:')[0], 'Listing error:')
