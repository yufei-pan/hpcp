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


def _run(args, cwd):
	'''hpcp as a program: stdout and stderr through one pipe, the bytes as written. (exit status, output)'''
	done = subprocess.run([sys.executable, HPCP] + args, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
		stdin=subprocess.DEVNULL, timeout=120)
	return done.returncode, done.stdout.decode('utf-8', 'replace')


def _at_a_terminal(args, cwd, size, answers=()):
	'''hpcp with a pty as its stdin, stdout and stderr (a terminal of size (rows, columns), or none: 0 x 0). answers: (question,
	reply) pairs: reply is typed once question has come. (exit status, what the terminal showed)'''
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
	answers = list(answers)
	deadline = time.monotonic() + 120
	try:
		while time.monotonic() < deadline:
			if answers and answers[0][0] in shown:
				os.write(fd, answers.pop(0)[1])
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


def test_996_a_moves_directory_pass_shows_the_line_in_every_mode(tmp_path, capsys, monkeypatch, reset_hpcp_globals):
	'''The pass over the source directories after the files are gone (a pool's too: its jobs show the "Deleted" bar, the pass
	is the parent's) draws the line with the directories it removes: 9.95 drew none in any mode.'''
	monkeypatch.setattr(hpcp, '_LINE_INTERVAL', 0)
	src = tmp_path / 'src'
	_tree(src, directories=3, per_directory=2)
	files = [os.path.join(str(src), 'd%d' % index, 'f%d' % number) for index in range(3) for number in range(2)]
	hpcp.remove_moved_sources([str(src)], files, {}, single_thread=False, max_workers=2, remove_force=True)
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
	progress.finish()
	shown = capsys.readouterr().out
	assert _frames(shown), shown
	_starts_a_line(shown, 'Keeping non-empty directory:')
	assert not (tmp_path / 'out' / 'a').exists() and progress.folders == 1


# ---- 2. A -pfl listing's line (and the listing's time on a line of its own) ----

# 40 folders of 5 files: the -pfl listing paces its jobs at -F 100 (200 files: 2 s), so the parent's loop has frames to draw.
LISTING_FOLDERS, LISTING_PER_FOLDER = 40, 5


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
	frames = _frames(_listing(output))
	assert frames and all(re.search(r'\| iteration:\d+ \| Files: [\d.]+ ', frame) for frame in frames), frames


@posix_only
@pytest.mark.parametrize('size', [(40, 120), None], ids=['sized-terminal', 'unsized-terminal'])
def test_996_a_pfl_listing_prints_its_line_at_a_terminal(tmp_path, size):
	_tree(tmp_path / 'src', LISTING_FOLDERS, LISTING_PER_FOLDER)
	rc, shown = _at_a_terminal(['-F', str(FILES_PER_SECOND), '-j', '2', '-pfl', 'src', '-d', 'dst/'], tmp_path, size)
	assert rc == 0, shown
	frames = _frames(_listing(shown))
	assert frames and all(frame.count('iteration:') == 1 and 'Files: ' in frame for frame in frames), frames


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
