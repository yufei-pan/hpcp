'''hpcp 9.97: the deferred entries of 9.94's and 9.96's reviews. Each test fails on 9.96 unless it says it guards what 9.96
already gets right. The lines a pool task prints go back to the parent with its result, which prints them off its open
progress line; the in-process tests stand a task's state in (_IN_TASK, a done future) rather than wait on a pool.'''
import concurrent.futures
import os
import stat
import subprocess
import sys

import pytest

import hpcp

posix_only = pytest.mark.skipif(os.name != 'posix', reason='POSIX')
linux_only = pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux')
not_root = pytest.mark.skipif(os.name == 'posix' and os.geteuid() == 0, reason='root reads and writes past the modes')
HPCP = os.path.abspath(hpcp.__file__)


def _write(path, text='x'):
	path = str(path)
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'w') as handle:
		handle.write(text)


def _run(args, cwd):
	'''hpcp as a program: stdout and stderr through one pipe (a combined log), the bytes as written. (exit status, output)'''
	done = subprocess.run([sys.executable, HPCP] + args, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
		stdin=subprocess.DEVNULL, timeout=300)
	return done.returncode, done.stdout.decode('utf-8', 'replace')


def _done(result):
	'''A done future holding a task's result (_run_task's four-tuple), as a pool hands it to the parent.'''
	future = concurrent.futures.Future()
	future.set_result(result)
	return future


def _settings():
	return {name: getattr(hpcp, name) for name in hpcp._TASK_SETTINGS}


def _combined_log(monkeypatch):
	'''This process's stdout and stderr taken for one file (> log 2>&1): an error line ends an open frame's line.'''
	monkeypatch.setattr(hpcp, '_LINE_START', [os.getpid(), '\n', True])


# ---- 1. What a pool task prints goes back with its result, and the parent prints it off its open line ----

def test_997_a_task_hands_back_each_line_with_its_stream(capsys, monkeypatch, reset_hpcp_globals):
	# eprint's error line, a line the task writes to stdout itself, and a note: none is printed in the task (9.96 printed a
	# task's stdout lines and its own stderr writes where it was, onto the parent's open frame in a combined log).
	monkeypatch.setattr(hpcp, '_worker_initialized', True)
	def task():
		hpcp.eprint('Copy failed: one')
		hpcp._line('Error copying a to b, why\n', sys.stdout)
		hpcp._line('Note: B/sub is a link\n', sys.stdout, note=True)
		hpcp._line('Warning: cannot remove temporary file t: why\n')
		return 'done'
	result = hpcp._run_task(_settings(), task)
	assert capsys.readouterr() == ('', '')
	assert result == ('done', ['Copy failed: one'], None, [('err', 'Copy failed: one\n'), ('out', 'Error copying a to b, why\n'),
		('note', 'Note: B/sub is a link\n'), ('err', 'Warning: cannot remove temporary file t: why\n')])


def test_997_the_parent_prints_a_task_s_lines_off_its_open_line_and_a_note_once_a_run(capsys, monkeypatch, reset_hpcp_globals):
	# A frame is open (_BAR_DRAWN) in a combined log: the first error line ends it (one line end, and the mark clears), a stdout
	# line starts a line of its own, and a note two tasks hand back is printed once.
	_combined_log(monkeypatch)
	lines = [('note', 'Note: B/sub is a link\n'), ('err', 'Copy failed: one\n'), ('out', 'Error copying a to b, why\n')]
	hpcp._BAR_DRAWN = True
	hpcp._settled(_done((None, [], None, lines)))
	first = capsys.readouterr()
	assert first.out == '\nNote: B/sub is a link\nError copying a to b, why\n' and first.err == 'Copy failed: one\n'
	assert not hpcp._BAR_DRAWN
	hpcp._BAR_DRAWN = True
	hpcp._settled(_done((None, [], None, lines)))
	second = capsys.readouterr()
	assert second.out == 'Error copying a to b, why\n' and second.err == '\nCopy failed: one\n'


def test_997_a_task_keeps_a_bounded_number_of_lines_and_says_how_many_it_left_out(monkeypatch, reset_hpcp_globals):
	# A job that meets a great many failures kept every line until it returned (9.96); now _TASK_LINES_MAX of them and one
	# line saying how many more, while the error list keeps every error (the exit status and the run's last line are whole).
	monkeypatch.setattr(hpcp, '_worker_initialized', True)
	monkeypatch.setattr(hpcp, '_TASK_LINES_MAX', 3)
	def task():
		for index in range(5):
			hpcp.eprint('Copy failed: %d' % index)
	result, errors, crash, lines = hpcp._run_task(_settings(), task)
	assert errors == ['Copy failed: %d' % index for index in range(5)] and crash is None
	assert lines[:3] == [('err', 'Copy failed: %d\n' % index) for index in range(3)] and len(lines) == 4
	assert lines[3][0] == 'err' and lines[3][1].startswith('... and 2 more lines of this job, not shown'), lines
	# The next task starts its count afresh.
	assert hpcp._run_task(_settings(), hpcp.eprint, 'Copy failed: next')[3] == [('err', 'Copy failed: next\n')]


@posix_only
def test_997_a_worker_s_own_warnings_go_back_with_its_task(tmp_path, capsys, monkeypatch, reset_hpcp_globals):
	# A temporary file that cannot be removed, FAT attributes that cannot be set, the space warning: 9.96 wrote each to stderr
	# where the worker was, knowing nothing of the parent's frame.
	monkeypatch.setattr(hpcp, '_IN_TASK', True)
	hpcp._discard_temp(os.path.join(HPCP, 'not-a-directory', 'temp'))  # ENOTDIR: no FileNotFoundError
	_write(tmp_path / 'f')
	hpcp._set_fat_attributes(str(tmp_path / 'f'), 0x20, 'f')  # not vfat: the ioctl fails
	monkeypatch.setattr(hpcp, '_mount_type', lambda dev: 'tmpfs')
	monkeypatch.setattr(hpcp, '_claim_space', lambda dev: True)
	hpcp._space_warning(str(tmp_path / 'out'), str(tmp_path / 'f'), 1, 2, 0)
	kept = list(hpcp._TASK_LINES)
	del hpcp._TASK_LINES[:]
	assert capsys.readouterr() == ('', '')
	assert [stream for stream, _ in kept] == ['err', 'err', 'err'], kept
	assert kept[0][1].startswith('Warning: cannot remove temporary file ') and kept[1][1].startswith('Warning: cannot keep the FAT')
	assert kept[2][1].startswith('Space warning: %s has ' % tmp_path) and kept[2][1].endswith('again.\n'), kept


@posix_only
def test_997_a_copy_job_s_own_lines_go_back_with_its_task(tmp_path, capsys, monkeypatch, reset_hpcp_globals):
	# A copy that fails at its last destination says 'Error copying ...' and 'Trying to copy ... without sparse' on stdout, a
	# link below a destination replaced by a directory says so in a Note, and a crash that is no OSError prints its traceback:
	# 9.96 printed each where the worker was.
	_write(tmp_path / 'src' / 'f')
	_write(tmp_path / 'blob', 'b')
	os.makedirs(str(tmp_path / 'dst' / 'elsewhere'))
	os.symlink('elsewhere', str(tmp_path / 'dst' / 'sub'))
	monkeypatch.setattr(hpcp, '_IN_TASK', True)
	monkeypatch.setattr(hpcp, 'NAMED_DESTS', frozenset([str(tmp_path / 'dst')]))
	monkeypatch.setattr(hpcp, 'RUN_ID', '997-own-lines')
	monkeypatch.setattr(hpcp, 'NAMED_IDS', hpcp._named_ids(hpcp.NAMED_DESTS))
	try:
		hpcp.copy_file(str(tmp_path / 'src' / 'f'), [str(tmp_path / 'blob' / 'f')])  # a file on the way: cp fails
		hpcp.copy_file(str(tmp_path / 'src' / 'f'), [str(tmp_path / 'dst' / 'sub' / 'f')])  # the link becomes a directory
		def crash(*args, **kwargs):
			raise RuntimeError('stand-in')
		monkeypatch.setattr(hpcp, '_cp_flags', crash)
		hpcp.copy_file(str(tmp_path / 'src' / 'f'), [str(tmp_path / 'dst' / 'g')])
	finally:
		hpcp._close_cursors()
	kept = list(hpcp._TASK_LINES)
	del hpcp._TASK_LINES[:]
	assert capsys.readouterr() == ('', ''), kept
	out = [text for stream, text in kept if stream == 'out']
	assert any(text.startswith('Error copying %s' % (tmp_path / 'src' / 'f')) for text in out), kept
	assert any(text.startswith('Trying to copy from ') and text.endswith(' without sparse\n') for text in out), kept
	assert any(text.startswith('Note: %s was a link to elsewhere and is now a directory' % (tmp_path / 'dst' / 'sub')) for text in out), kept
	assert any(text.startswith('Traceback (most recent call last):') and 'RuntimeError: stand-in' in text for text in out), kept
	assert os.path.isfile(str(tmp_path / 'dst' / 'sub' / 'f')) and os.listdir(str(tmp_path / 'dst' / 'elsewhere')) == []


@posix_only
def test_997_the_space_warning_starts_no_empty_line_where_no_frame_is_open(tmp_path, capsys, monkeypatch, reset_hpcp_globals):
	# hpcp big -d /small/out/ > log 2>&1: 9.96 put an empty line before the warning (its line start whatever the mark said).
	_combined_log(monkeypatch)
	monkeypatch.setattr(hpcp, '_mount_type', lambda dev: 'tmpfs')
	monkeypatch.setattr(hpcp, '_claim_space', lambda dev: True)
	_write(tmp_path / 'f')
	hpcp._space_warning(str(tmp_path / 'out'), str(tmp_path / 'f'), 1, 2, 0)
	assert capsys.readouterr().err.startswith('Space warning: ')
	hpcp._BAR_DRAWN = True  # (with a frame open, it ends the frame's line: as 9.96)
	hpcp._space_warning(str(tmp_path / 'out'), str(tmp_path / 'f'), 1, 2, 0)
	assert capsys.readouterr().err.startswith('\nSpace warning: ') and not hpcp._BAR_DRAWN


@posix_only
def test_997_a_refusal_s_note_is_said_once_a_run(tmp_path):
	# hpcp src/ -d A/ -d B/ -ncd -nds -j 8, ten of B's directories links: each refusal's note once, where 9.94 to 9.96 said it
	# once for each worker that copied a file below it (94 lines for 20 such directories at -j 32, 2026-11-15).
	for directory in range(10):
		for index in range(20):
			_write(tmp_path / 'src' / ('d%d' % directory) / ('f%d' % index), str(index))
		os.makedirs(str(tmp_path / 'A' / ('d%d' % directory)))
		os.makedirs(str(tmp_path / 'elsewhere' / ('d%d' % directory)))
	os.makedirs(str(tmp_path / 'B'))
	for directory in range(10):
		os.symlink(str(tmp_path / 'elsewhere' / ('d%d' % directory)), str(tmp_path / 'B' / ('d%d' % directory)))
	rc, output = _run(['src/', '-d', 'A/', '-d', 'B/', '-ncd', '-nds', '-j', '8'], tmp_path)
	notes = [line for line in output.splitlines() if line.startswith('Note: ')]
	assert rc == 0 and sorted(os.listdir(str(tmp_path / 'A' / 'd9'))) == sorted('f%d' % index for index in range(20)), output[-3000:]
	assert len(notes) == 10 and len(set(notes)) == 10, output[-4000:]
	for note in notes:
		assert '-ncd makes no directory in its place; nothing is written below it (name it with -d ' in note, note


@posix_only
def test_997_a_link_bar_s_error_starts_a_line_of_its_own(tmp_path):
	# The link-tracking bar set no mark (9.96), so 'Failed to create symbolic link' landed on its frame in a combined log
	# ('...|10.0% | 971.23 F/sFailed to create symbolic link').
	_write(tmp_path / 'src' / 'target')
	for index in range(12):
		os.symlink('target', str(tmp_path / 'src' / ('a%02d' % index)))
	for index in (3, 6, 9, 11):
		os.makedirs(str(tmp_path / 'dst' / 'src' / ('a%02d' % index)))  # a directory where the link goes: an error
	rc, output = _run(['src', '-d', 'dst/', '-nds'], tmp_path)
	shown = output.split('Error occurred:')[0]  # (the error list at the end names them again)
	assert rc != 0 and shown.count('Failed to create symbolic link') == 4, output[-3000:]
	# The last link failed, so the bar never reached its total: its summary starts a line of its own too.
	_each_starts_a_line(shown, 'Failed to create symbolic link')
	_each_starts_a_line(shown, 'Time taken:')


@posix_only
@not_root
def test_997_a_link_that_cannot_be_made_says_so_off_the_bar(tmp_path):
	# Links whose directory takes no new name (0555): 'Could not create symbolic link ...' (9.96: written straight to stderr,
	# onto the bar's frame in a combined log) and 'Failed to create symbolic link: All destinations failed', each on a line of
	# its own. Ten links that can be made and five that cannot, in the order the copy hands them over: a frame comes before
	# at least one failure (all five first: 1 in 3,003).
	for directory, count in (('a', 10), ('b', 5)):
		_write(tmp_path / 'src' / directory / 'target')
		for index in range(count):
			os.symlink('target', str(tmp_path / 'src' / directory / ('l%d' % index)))
	_write(tmp_path / 'dst' / 'src' / 'b' / 'target')
	os.chmod(str(tmp_path / 'dst' / 'src' / 'b'), 0o555)
	try:
		rc, output = _run(['src', '-d', 'dst/', '-nds'], tmp_path)
	finally:
		os.chmod(str(tmp_path / 'dst' / 'src' / 'b'), 0o755)
	shown = output.split('Error occurred:')[0]
	assert rc != 0 and len(os.listdir(str(tmp_path / 'dst' / 'src' / 'a'))) == 11, output[-3000:]
	assert shown.count('Could not create symbolic link from target to ') == 5, output[-3000:]
	_each_starts_a_line(shown, 'Could not create symbolic link from target to ')
	_each_starts_a_line(shown, 'Failed to create symbolic link: All destinations failed')
	# A frame before one of them (each part but the last is what came before a failure, after the one before it).
	assert any('Symbolic Links Created' in part for part in shown.split('Could not create symbolic link')[:-1]), output[-3000:]


def _each_starts_a_line(text, mark):
	'''mark is in text, and each one starts a line (after a line end).'''
	assert mark in text, text[-3000:]
	start = text.find(mark)
	while start != -1:
		assert start == 0 or text[start - 1] == '\n', repr(text[max(0, start - 120):start + 40])
		start = text.find(mark, start + 1)


@posix_only
def test_997_the_mounted_dir_hint_is_left_out_where_a_file_is_in_the_way(tmp_path):
	# hpcp src -d blob/ with blob a file: the directory sync met it below blob and printed 'Destination path .../blob/src maybe
	# a mounted dir ... Continuing without creating dest folder...' (9.93 to 9.96); what comes after names the problem.
	_write(tmp_path / 'src' / 'f')
	_write(tmp_path / 'blob', 'b')
	rc, output = _run(['src', '-d', 'blob/'], tmp_path)
	assert rc != 0 and 'maybe a mounted dir' not in output and 'Continuing without creating' not in output, output[-3000:]
	assert 'Not a directory' in output, output[-3000:]
	with open(str(tmp_path / 'blob')) as handle:
		assert handle.read() == 'b'


# ---- 2. -pfl's live counts ----

@pytest.fixture
def a_worker_s_cells(monkeypatch):
	'''This process as a pool worker in slot 1 with live counts (_my_live: the cells 0 to 2), drawing a frame's worth at every
	check (_LINE_INTERVAL 0).'''
	live = [0.0] * (2 * hpcp._LIVE_CELLS)
	monkeypatch.setattr(hpcp, '_RATE_SHARE', (None, 0, 0, live))
	monkeypatch.setattr(hpcp, '_MY_SLOT', 1)
	monkeypatch.setattr(hpcp, '_LINE_INTERVAL', 0)
	return live


@posix_only
def test_997_a_listing_job_counts_its_files_links_and_folders_apart(tmp_path, a_worker_s_cells, reset_hpcp_globals):
	# 9.96 kept two cells a worker, the entries and the folders, so a running job's links showed under Files.
	for name in ('f1', 'f2', 'f3'):
		_write(tmp_path / 'd' / name)
	for name in ('l1', 'l2'):
		os.symlink('f1', str(tmp_path / 'd' / name))
	os.makedirs(str(tmp_path / 'd' / 'sub'))
	files, links, size, folders = hpcp.get_file_list_serial(str(tmp_path / 'd'), recurse=False, known_dir=True, show=False)
	assert (len(files), len(links), len(folders)) == (3, 2, 2)  # the folders: the root and sub
	assert a_worker_s_cells[:3] == [3, 2, 1] and a_worker_s_cells[3:] == [0, 0, 0]  # (the root is the parent's already)
	assert hpcp._live_sum(a_worker_s_cells, 1) == 2


@posix_only
def test_997_a_listing_job_that_raises_takes_its_counts_back(tmp_path, a_worker_s_cells, monkeypatch, reset_hpcp_globals):
	# A job that raises returns nothing of its folder: 9.96 left what it had published in its cells, and the line counted it
	# for the rest of the listing.
	for index in range(6):
		_write(tmp_path / 'd' / ('f%d' % index))
	real, seen = hpcp.get_file_repr, []
	def raises_late(path, *args, **kwargs):
		seen.append(path)
		if len(seen) > 3:
			raise MemoryError('stand-in')
		return real(path, *args, **kwargs)
	monkeypatch.setattr(hpcp, 'get_file_repr', raises_late)
	with pytest.raises(MemoryError):
		hpcp.get_file_list_serial(str(tmp_path / 'd'), recurse=False, known_dir=True, show=False)
	assert a_worker_s_cells == [0.0] * (2 * hpcp._LIVE_CELLS)


_LINKS_LISTED = '''
import multiprocessing, os, sys
if __name__ == '__main__':
	multiprocessing.set_start_method('fork', force=True)  # the workers take this process's _LINE_INTERVAL
	sys.path.insert(0, %(root)r)
	import hpcp
	hpcp._LINE_INTERVAL = 0
	frames = []
	hpcp._show_listing = lambda path, seconds, files, links, folders, size, rate_limiter=None: frames.append((files, links))
	files, links, size, folders = hpcp.get_file_list_parallel(%(tree)r, max_workers=2)
	print('result', len(files), len(links))
	print('frames', len(frames), 'files over 1', sum(1 for f in frames if f[0] > 1), 'links shown', sum(1 for f in frames if f[1]))
'''


@linux_only
def test_997_a_running_listing_job_s_links_show_under_links(tmp_path):
	# hpcp -pfl over a folder of many symlinks: while its job runs the line counted them under Files (9.96), and moved them to
	# Links when it returned. A frame at every check of the parent's loop (_LINE_INTERVAL 0), and the job publishes at every
	# entry: the frames while it runs show its links under Links, and Files never past the one file.
	tree = tmp_path / 'tree'
	_write(tree / 'file')
	os.makedirs(str(tree / 'wide'))
	for index in range(20000):
		os.symlink('../file', str(tree / 'wide' / ('l%d' % index)))
	code = _LINKS_LISTED % {'root': os.path.dirname(HPCP), 'tree': str(tree)}
	done = subprocess.run([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
	output = done.stdout.decode('utf-8', 'replace')
	assert 'result 1 20000' in output, output[-3000:]
	frames = output.split('frames ')[1].split()
	assert int(frames[0]) > 0 and frames[4] == '0' and int(frames[7]) > 0, output[-3000:]


# ---- 3. The pool's start ----

def test_997_python_3_6_makes_no_live_counts_its_spawned_workers_cannot_find(monkeypatch):
	# On 3.6 a worker that does not fork gets no slot (3.6's Array takes no 'q'), so it has no row of the live counts: 9.96 made
	# the Array all the same, a temporary-file arena for nothing.
	monkeypatch.setattr(hpcp.multiprocessing, 'get_start_method', lambda *args, **kwargs: 'spawn')
	monkeypatch.setattr(hpcp.sys, 'version_info', (3, 6, 15, 'final', 0))
	assert hpcp._new_live(4) is None


def test_997_without_a_temporary_directory_the_fork_server_gives_way_to_fork(monkeypatch):
	# Python 3.14 starts pool workers by a fork server, whose socket goes in the temporary directory: with none that takes a
	# file (/tmp read-only, a full disk) a pooled copy or -rm -pfl exited 70 with a traceback, nothing done (9.95, 9.96). The
	# real run (a read-only container) is in the 9.97 hand-back; here the choice, forced at the call.
	chosen = []
	monkeypatch.setattr(hpcp.multiprocessing, 'get_start_method', lambda *args, **kwargs: 'forkserver')
	monkeypatch.setattr(hpcp.multiprocessing, 'set_start_method', lambda method, force=False: chosen.append((method, force)))
	def none_found():
		raise FileNotFoundError(2, 'No usable temporary directory found')
	monkeypatch.setattr(hpcp.tempfile, 'gettempdir', none_found)
	hpcp._fork_without_tempdir()
	assert chosen == [('fork', True)]
	monkeypatch.setattr(hpcp.tempfile, 'gettempdir', lambda: '/tmp')
	hpcp._fork_without_tempdir()
	assert chosen == [('fork', True)]  # a temporary directory: the fork server stays


_PACED_36 = '''
import multiprocessing, runpy, sys, time
if __name__ == '__main__':
	multiprocessing.set_start_method('spawn')  # Windows's only start method
	sys.argv = [%(hpcp)r, '-rm', '-rf', '-pfl', '-j', '4', '-F', '100', %(tree)r]
	started = time.monotonic()
	try:
		runpy.run_path(%(hpcp)r, run_name='__main__')
	finally:
		print('seconds %%.2f' %% (time.monotonic() - started))
'''


@pytest.mark.skipif(sys.version_info >= (3, 7) or os.name != 'posix', reason='Python 3.6: its pool takes no initializer')
def test_997_python_3_6_spawned_workers_keep_the_pace(tmp_path):
	# -F reaches a -pfl job through the pool's initializer, which Python 3.6's pool does not take: with spawn (Windows) the
	# jobs ran unpaced (46,000 files at -F 2000: 1.7 s on 9.95, 2.7 s on 9.96 where about 23 s are expected). 300 files in 3
	# directories at -F 100 take at least about 3 s now.
	for directory in range(3):
		for index in range(100):
			_write(tmp_path / 'tree' / ('d%d' % directory) / ('f%d' % index))
	code = _PACED_36 % {'hpcp': HPCP, 'tree': str(tmp_path / 'tree')}
	done = subprocess.run([sys.executable, '-c', code], cwd=str(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
		timeout=300)
	output = done.stdout.decode('utf-8', 'replace')
	assert not os.path.exists(str(tmp_path / 'tree')), output[-3000:]
	assert float(output.split('seconds ')[-1].split()[0]) >= 2.5, output[-3000:]


# ---- 4. -co ----

def _user_xattrs(directory):
	'''Whether a file in directory takes a user.* xattr (Linux; not tmpfs before 6.6, not every overlay).'''
	probe = os.path.join(str(directory), 'xattr-probe')
	_write(probe, '')
	try:
		os.setxattr(probe, 'user.hpcp-probe', b'1')
		return True
	except (AttributeError, OSError):
		return False
	finally:
		os.unlink(probe)


@posix_only
@not_root
@pytest.mark.parametrize('route', ['held', 'proc', 'read'])
def test_997_a_co_rewrite_beside_of_a_read_only_source_keeps_a_user_xattr(tmp_path, monkeypatch, reset_hpcp_globals, route):
	# A non-root -co -wb rewrite of a 0444 source onto a file holding a user.* xattr: the new file had the source's mode when
	# the carry set the xattr, which needs write permission (through a descriptor too), so 9.93 to 9.96 kept the old copy (190,
	# 'an xattr cannot be given to the new copy (Permission denied)'). Each route the carry reaches the new file by: the
	# descriptor hpcp made it by (held), and by name (Linux with /proc: O_PATH; without, macOS: opened for reading).
	if not _user_xattrs(tmp_path):
		pytest.skip('no user.* xattrs here')
	if route != 'held':
		monkeypatch.setattr(hpcp, '_held_route', lambda: None)
		if route == 'proc' and not hpcp._PATH_FDS:
			pytest.skip('no O_PATH with /proc here')
		if route == 'read':
			monkeypatch.setattr(hpcp, '_PATH_FDS', False)
	elif not hpcp._held_route():
		pytest.skip('cp writes no held file here')
	_write(tmp_path / 'x', 'new')
	os.chmod(str(tmp_path / 'x'), 0o444)
	_write(tmp_path / 'out' / 'x', 'old')
	os.chmod(str(tmp_path / 'out' / 'x'), 0o644)
	os.setxattr(str(tmp_path / 'out' / 'x'), 'user.kept', b'yes')
	old = os.lstat(str(tmp_path / 'out' / 'x'))
	monkeypatch.setattr(hpcp, 'CONTENT_ONLY', True)
	monkeypatch.setattr(hpcp, 'WRITE_BESIDE', True)
	hpcp.copy_file(str(tmp_path / 'x'), [str(tmp_path / 'out' / 'x')])
	assert not hpcp.ERRORS, hpcp.ERRORS
	now = os.lstat(str(tmp_path / 'out' / 'x'))
	with open(str(tmp_path / 'out' / 'x')) as handle:
		assert handle.read() == 'new' and now.st_ino != old.st_ino
	assert stat.S_IMODE(now.st_mode) == 0o644 and os.getxattr(str(tmp_path / 'out' / 'x'), 'user.kept') == b'yes'
	assert sorted(os.listdir(str(tmp_path / 'out'))) == ['x']


# ---- 5. macOS: the lean binding of the read calls ----

darwin_only = pytest.mark.skipif(sys.platform != 'darwin', reason='macOS')


def test_997_without_ctypes_the_mac_read_calls_say_enotsup(monkeypatch):
	# A macOS Python without _ctypes: each read helper imported ctypes itself and raised ImportError (9.90 to 9.96), which no
	# caller skips (a directory's finishing ended with exit 70); now ENOTSUP, which every caller skips (_XATTR_SKIPPED).
	monkeypatch.setattr(hpcp, '_libc', lambda: None)
	monkeypatch.setattr(hpcp, '_LIBC_FUNCTIONS', {})
	monkeypatch.setitem(sys.modules, 'ctypes', None)  # import ctypes raises ImportError
	for call in (lambda: hpcp._mac_listxattr('/'), lambda: hpcp._mac_getxattr(0, 'com.example'), lambda: hpcp._mac_acl_text('/')):
		with pytest.raises(OSError) as raised:
			call()
		assert raised.value.errno == __import__('errno').ENOTSUP


@darwin_only
def test_997_macos_the_lean_reads_answer_as_the_xattr_tool_and_ls_do(tmp_path):
	# The three read calls through libc, by path and by descriptor: a value and a list of names larger than the one buffer
	# (asked their size and read again), a missing name (ENOATTR), an ACL and none. Guards what 9.96 gets right, through the
	# lean binding (one buffer a thread, each call bound once).
	import errno
	path = str(tmp_path / 'f')
	_write(path, 'x')
	big = 'v' * 9000
	subprocess.run(['xattr', '-w', 'org.example.big', big, path], check=True)
	names = ['org.example.name%03d-%s' % (index, 'n' * 40) for index in range(120)]  # over 6 KiB of names
	for name in names:
		subprocess.run(['xattr', '-w', name, 'small', path], check=True)
	fd = os.open(path, os.O_RDONLY)
	try:
		for target in (path, fd):
			assert hpcp._mac_getxattr(target, 'org.example.big') == big.encode()
			assert hpcp._mac_getxattr(target, names[7]) == b'small'
			assert set(hpcp._mac_listxattr(target)) >= set(names) | {'org.example.big'}
			with pytest.raises(OSError) as raised:
				hpcp._mac_getxattr(target, 'org.example.missing')
			assert raised.value.errno == errno.ENOATTR
			assert hpcp._mac_acl_text(target) == ''
		subprocess.run(['chmod', '+a', 'everyone deny delete', path], check=True)
		try:
			assert 'deny' in hpcp._mac_acl_text(path) and hpcp._mac_acl_text(fd) == hpcp._mac_acl_text(path)
		finally:
			subprocess.run(['chmod', '-N', path], check=True)
	finally:
		os.close(fd)


@darwin_only
@pytest.mark.parametrize('route', ['held', 'read'])
def test_997_macos_a_co_rewrite_beside_keeps_the_old_file_s_xattrs_and_acl(tmp_path, monkeypatch, reset_hpcp_globals, route):
	# A -co rewrite beside (-wb) gives the new file the old one's owner, group and mode, and on Linux its xattrs; on macOS it
	# gave it none of the old file's xattrs or ACL, and kept the source's xattr that cp copied (9.94 to 9.96: Python has no
	# os.listxattr there). Through the descriptor hpcp made the new file by (held), or one opened at its name (read).
	if route == 'read':
		monkeypatch.setattr(hpcp, '_held_route', lambda: None)
	_write(tmp_path / 'x', 'new')
	subprocess.run(['xattr', '-w', 'org.example.source', 'theirs', str(tmp_path / 'x')], check=True)
	landing = str(tmp_path / 'out' / 'x')
	_write(landing, 'old')
	subprocess.run(['xattr', '-w', 'org.example.tag', 'mine', landing], check=True)
	subprocess.run(['chmod', '+a', 'everyone allow read', landing], check=True)
	old = os.lstat(landing)
	monkeypatch.setattr(hpcp, 'CONTENT_ONLY', True)
	monkeypatch.setattr(hpcp, 'WRITE_BESIDE', True)
	try:
		hpcp.copy_file(str(tmp_path / 'x'), [landing])
		assert not hpcp.ERRORS, hpcp.ERRORS
		now = os.lstat(landing)
		with open(landing) as handle:
			assert handle.read() == 'new' and now.st_ino != old.st_ino
		names = subprocess.run(['xattr', landing], stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout.split()
		assert 'org.example.tag' in names and 'org.example.source' not in names, names
		acl = subprocess.run(['ls', '-led', landing], stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout
		assert 'everyone allow read' in acl, acl
	finally:
		subprocess.run(['chmod', '-N', landing], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---- 6. macOS: the copy check on a network mount ----

@pytest.mark.parametrize('kind, grouped', [('smbfs', True), ('nfs', True), ('apfs', False), ('hfs', False), (OSError, False),
	(ImportError, False)])
def test_997_macos_checks_a_destination_on_a_network_mount_by_directory(monkeypatch, kind, grouped):
	# 9.88 to 9.96 checked every macOS landing by its path, a network mount too: the 9.97 design check measured by directory
	# 2.3 to 3.0 times faster on an SMB mount (10,000 directories of one file) and on NFSv3 noac (50 of 1,000), as Linux takes
	# them. The type from statfs (_statfs_type: smbfs, nfs); a destination that cannot be asked, or a Python without ctypes,
	# stays by path. The platform and statfs stood in for.
	def statfs(path):
		if not isinstance(kind, str):
			raise kind('stand-in')
		return kind
	monkeypatch.setattr(hpcp.sys, 'platform', 'darwin')
	monkeypatch.setattr(hpcp, '_statfs_type', statfs)
	assert hpcp._on_network_fs(['/somewhere']) is grouped


@darwin_only
def test_997_macos_a_local_destination_stays_by_path(tmp_path):
	# On APFS the check by directory cost 2.3 times as much for 10,000 directories of one file each (9.88's first build took
	# every macOS destination by directory): a local destination is looked at by path. Guards what 9.96 gets right.
	assert hpcp._statfs_type(str(tmp_path)) == 'apfs' and hpcp._on_network_fs([str(tmp_path)]) is False


# ---- 7. Stops ----

_DD_COPY_STOPPED = '''
import os, signal, sys
from types import SimpleNamespace
sys.path.insert(0, %(root)r)
import hpcp
for name in ('SIGINT', 'SIGTERM', 'SIGHUP'):
	signal.signal(getattr(signal, name), signal.SIG_DFL)
hpcp._MAIN_PID = os.getpid()
hpcp._install_stops()
here = %(here)r
roots, copied, sent = [], [], []

def mount_dir():
	path = os.path.join(here, 'mount%%d' %% len(roots))
	os.mkdir(path)
	roots.append(path)
	if len(roots) %% 2 == 1:
		with open(os.path.join(path, 'data.txt'), 'w') as fh:  # each partition's source holds a file
			fh.write('partition data')
	return path

def run(command, **kwargs):
	return [] if command[0] == 'mount' else real_run(command, **kwargs)

real_run = hpcp.run_command_in_multicmd_with_path_check
real_process_copy = hpcp.process_copy

def copy_with_a_stop(*args, **kwargs):
	# The first partition's copy: Ctrl-C while it copies.
	if not sent:
		sent.append(1)
		os.kill(os.getpid(), signal.SIGINT)
	return real_process_copy(*args, **kwargs)

hpcp.validate_dd_source_path = lambda *a, **k: '/dev/fake-source'
hpcp.create_dd_dest_part_table = lambda *a, **k: ({'/dev/fake-source': {}, '1': {'size': 4096}, '2': {'size': 4096}}, [], None)
hpcp.get_partitions = lambda device: [device + '1', device + '2']
hpcp.is_device = lambda path: False
hpcp.os.geteuid = lambda: 0
hpcp.tempfile = SimpleNamespace(mkdtemp=mount_dir)
hpcp.os.path.ismount = lambda path: path in roots
hpcp.run_command_in_multicmd_with_path_check = run
hpcp.process_copy = copy_with_a_stop
hpcp.dd_partition = lambda *a, **k: copied.append(a[2])
rc = hpcp.hpcp(['/dev/fake-source'], dest_image=os.path.join(here, 'dest.img'), dd=True, single_thread=True, max_workers=1,
	no_directory_sync=True)
print('rc', rc, 'mounted', len(roots), 'dd copied partitions', copied)
'''


@linux_only
def test_997_a_stop_in_a_partition_s_copy_ends_the_dd_run_with_130(tmp_path):
	# Ctrl-C while -dd copies a partition's files: the nested run takes it and returns 130, and the outer run read that as a
	# failed partition ('Partition copy error: Copying partition 1 failed with return code 130.', exit 181; 9.93 to 9.96).
	# The disk setup and the mounts are stood in for (as test_994's -dd stop); the stop is a real SIGINT.
	code = _DD_COPY_STOPPED % {'root': os.path.dirname(HPCP), 'here': str(tmp_path)}
	done = subprocess.run([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120,
		start_new_session=True)
	output = done.stdout.decode('utf-8', 'replace')
	assert 'rc 130 mounted 2 dd copied partitions []' in output, output[-3000:]
	assert 'Partition copy error' not in output, output[-3000:]


_STOP_BEFORE_CLEANUP = '''
import os, signal, sys
sys.path.insert(0, %(root)r)
import hpcp
for name in ('SIGINT', 'SIGTERM', 'SIGHUP'):
	signal.signal(getattr(signal, name), signal.SIG_DFL)
hpcp._MAIN_PID = os.getpid()
hpcp._install_stops()
cleaned, sent, real_close = [], [], hpcp._close_cursors

def close_with_a_stop():
	# The first step of hpcp()'s finally, before its cleanup (the run's settings cleared, the file copied): Ctrl-C lands here.
	if not sent and hpcp.RUN_ID is None and os.path.exists(%(copied)r):
		sent.append(1)
		os.kill(os.getpid(), signal.SIGINT)
	return real_close()

hpcp._close_cursors = close_with_a_stop
hpcp.clean_up = lambda *args: cleaned.append(True)
try:
	rc = hpcp.hpcp([%(src)r], dest_paths=[%(dst)r], single_thread=True)
except KeyboardInterrupt:
	rc = 'raised'
print('rc', rc, 'sent', len(sent), 'cleaned', len(cleaned), 'stopped', hpcp._STOPPED == signal.SIGINT)
'''


@posix_only
def test_997_a_stop_between_the_run_and_its_cleanup_still_cleans_up(tmp_path):
	# Ctrl-C between hpcp()'s finally and clean_up() ended the run without the cleanup, where mounts or loop devices of an image
	# run could stay (9.93 to 9.96): the stop is now only recorded there, as in the cleanup, and acted on after it.
	_write(tmp_path / 'src' / 'f')
	code = _STOP_BEFORE_CLEANUP % {'root': os.path.dirname(HPCP), 'src': str(tmp_path / 'src'), 'dst': str(tmp_path / 'dst') + '/',
		'copied': str(tmp_path / 'dst' / 'src' / 'f')}
	done = subprocess.run([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120,
		start_new_session=True)
	output = done.stdout.decode('utf-8', 'replace')
	assert 'sent 1 cleaned 1 stopped True' in output and 'rc raised' not in output, output[-3000:]
