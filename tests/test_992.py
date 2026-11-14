'''hpcp 9.92: the cleanup after an image run runs every step, quiet, and shows what a command that did not end 0 printed on
stdout. Most tests force each failure in this process, at the step it meets it: a stdout whose reader quit raises
BrokenPipeError at a write that reaches it (a shown command's output, a flushed print), a full one OSError, a closed one
ValueError. The tests named "real" run real commands through the embedded multiCMD; none needs root or a loop device.'''
import errno
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

import hpcp

DELAYED = [['e2fsck', '-f', '-y', '/img'], ['tune2fs', '-U', 'random', '/img'], ['losetup', '--detach', '/dev/loop7']]
# what a write to stdout can raise: its reader quit (`hpcp -dd src dst | head`), a full disk under `> log`, a library caller
# that closed sys.stdout
STDOUT_ERRORS = [BrokenPipeError(errno.EPIPE, 'Broken pipe'), OSError(errno.ENOSPC, 'No space left on device'),
	ValueError('I/O operation on closed file.')]


def _raise(error):
	raise type(error)(*error.args)


class _GoneReader:
	'''A stdout that cannot be written: every write and flush raises the error it is given.'''
	encoding = 'utf-8'

	def __init__(self, error=STDOUT_ERRORS[0]):
		self.error = error

	def write(self, text):
		_raise(self.error)

	def flush(self):
		_raise(self.error)


def _record(monkeypatch, fail=(), mounted=(), error=STDOUT_ERRORS[0]):
	'''clean_up's three kinds of step recorded in order: a command call (its tool), a loop device's detach. A tool or a loop
	device in fail raises error when it is called, as a shown call or a print does once stdout cannot be written.
	mounted: what _is_mount_point answers, call by call (then False).'''
	steps = []
	mounted = list(mounted)

	def run(commands, **kwargs):
		for command in commands:
			steps.append(('run', command[0], kwargs.get('quiet'), kwargs.get('show_errors')))
			if command[0] in fail:
				_raise(error)
		return [[] for _ in commands]

	monkeypatch.setattr(hpcp, 'run_commands_in_multicmd_with_path_check', run)
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check', lambda command, **kwargs: run([command], **kwargs)[0])
	def detach(device):
		steps.append(('detach', device))
		if device in fail:
			_raise(error)
		return True

	monkeypatch.setattr(hpcp, 'detach_loop_device', detach)
	monkeypatch.setattr(hpcp, '_is_mount_point', lambda path: mounted.pop(0) if mounted else False)
	return steps


@pytest.mark.parametrize('error', STDOUT_ERRORS, ids=lambda error: type(error).__name__)
@pytest.mark.parametrize('raising, ran', [
	# the delayed pair is one call: tune2fs is skipped with the e2fsck that raised, the detaches are not
	('e2fsck', ['umount', 'umount', 'e2fsck']),
	('umount', ['umount', 'umount', 'e2fsck', 'tune2fs']),
	('/dev/loop5', ['umount', 'umount', 'e2fsck', 'tune2fs']),   # a detach that raises (its own print): the next devices are still detached
])
def test_a_step_that_raises_does_not_skip_the_steps_after_it(tmp_path, reset_hpcp_globals, monkeypatch, capsys, raising, ran, error):
	# 9.91: the call raised out of clean_up, so the later mounts, the delayed commands and every loop device were skipped
	steps = _record(monkeypatch, fail=(raising,), error=error)
	mounts = [str(tmp_path / 'first'), str(tmp_path / 'second')]
	for mount in mounts:
		os.mkdir(mount)
	loops = ['/dev/loop5', '/dev/loop6']
	hpcp.clean_up(list(mounts), loops, [list(command) for command in DELAYED])
	assert [step[1] for step in steps if step[0] == 'run'] == ran
	assert [step for step in steps if step[0] == 'detach'] == [('detach', '/dev/loop5'), ('detach', '/dev/loop6'), ('detach', '/dev/loop7')]
	assert os.listdir(str(tmp_path)) == []
	assert loops == []
	said = capsys.readouterr().err
	assert 'General Exception: cleanup step' in said and type(error).__name__ in said
	assert hpcp.get_rc_from_error() == 70   # the step that failed counts in the run's exit code


@pytest.mark.parametrize('error', STDOUT_ERRORS, ids=lambda error: type(error).__name__)
def test_a_print_into_a_stdout_that_cannot_take_it_does_not_skip_its_step(tmp_path, reset_hpcp_globals, monkeypatch, error):
	# 9.91: print raised first in each step, before its umount, rmdir and detach
	steps = _record(monkeypatch)
	monkeypatch.setattr(sys, 'stdout', _GoneReader(error))
	mount = str(tmp_path / 'mount')
	os.mkdir(mount)
	hpcp.clean_up([mount], ['/dev/loop5'], [list(command) for command in DELAYED])
	assert [step[1] for step in steps if step[0] == 'run'] == ['umount', 'e2fsck', 'tune2fs']
	assert [step for step in steps if step[0] == 'detach'] == [('detach', '/dev/loop5'), ('detach', '/dev/loop7')]
	assert not os.path.exists(mount)


def test_a_mount_still_busy_is_unmounted_once_more_quiet_too(tmp_path, reset_hpcp_globals, monkeypatch):
	steps = _record(monkeypatch, mounted=[True])
	monkeypatch.setattr(hpcp.time, 'sleep', lambda seconds: None)
	mount = str(tmp_path / 'mount')
	os.mkdir(mount)
	hpcp.clean_up([mount], [], [])
	assert steps == [('run', 'umount', True, True), ('run', 'umount', True, True)]


def test_every_command_of_the_cleanup_runs_quiet_and_its_failure_is_still_said(tmp_path, reset_hpcp_globals, monkeypatch, capsys):
	# 9.91 ran umount and the delayed pair shown (quiet False), and multiCMD stops a shown command once stdout's reader has quit
	calls = []

	def run_commands(commands, **kwargs):
		calls.append((os.path.basename(commands[0][0]), kwargs.get('quiet')))
		return [SimpleNamespace(command=list(command), returncode=4 if os.path.basename(command[0]) == 'e2fsck' else 0, stdout=[], stderr=['/img: bad superblock'])
			for command in commands]

	monkeypatch.setattr(hpcp.multiCMD, 'run_commands', run_commands)
	monkeypatch.setattr(hpcp, 'detach_loop_device', lambda device: True)
	monkeypatch.setattr(hpcp, '_is_mount_point', lambda path: False)
	monkeypatch.setattr(hpcp, 'check_path', lambda tool: True)
	mount = str(tmp_path / 'mount')
	os.mkdir(mount)
	hpcp.clean_up([mount], [], [['e2fsck', '-f', '-y', '/img'], ['tune2fs', '-U', 'random', '/img']])
	assert [tool for tool, _ in calls] == ['umount', 'e2fsck']   # the delayed pair is one call, led by e2fsck
	assert all(quiet is True for _, quiet in calls)
	said = capsys.readouterr()
	assert "Task return code error: Command" in said.err and 'e2fsck' in said.err and 'bad superblock' in said.err
	assert 'bad superblock' not in said.out
	assert any(error.startswith('Task return code error') for error in hpcp.ERRORS)


def test_a_quiet_command_that_fails_is_only_recorded_unless_its_errors_are_asked_for(reset_hpcp_globals, monkeypatch, capsys):
	# show_errors is the cleanup's: every other quiet caller (blkid, mount, partx) still says nothing on stderr
	monkeypatch.setattr(hpcp.multiCMD, 'run_commands', lambda commands, **kwargs: [
		SimpleNamespace(command='false', returncode=1, stdout=[], stderr=['no'])])
	monkeypatch.setattr(hpcp, 'check_path', lambda tool: True)
	hpcp.run_command_in_multicmd_with_path_check(['false'], quiet=True)
	assert capsys.readouterr().err == ''
	assert hpcp.ERRORS == ["Task return code error: Command 'false' failed with return code 1: no"]   # recorded: it counts in the exit code
	hpcp.run_command_in_multicmd_with_path_check(['false'], quiet=True, show_errors=True)
	assert 'failed with return code 1: no' in capsys.readouterr().err


@pytest.mark.parametrize('tool, returncode, shown, failed', [
	('e2fsck', 1, True, False),   # it repaired something: a success, and what it repaired is shown
	('e2fsck', 4, True, True),
	('tune2fs', 1, True, True),
	('e2fsck', 0, False, False),
])
def test_the_cleanup_shows_what_a_command_that_did_not_end_0_printed(reset_hpcp_globals, monkeypatch, capsys, tool, returncode, shown, failed):
	# 9.92's first build showed none of it: an e2fsck -f -y that cleared a file's entry on the clone ended the run 0 with no word
	lines = ["Entry 'f2' in /d (12) has deleted/unused inode 15.  Clear? yes", '/img: ***** FILE SYSTEM WAS MODIFIED *****']
	monkeypatch.setattr(hpcp.multiCMD, 'run_commands', lambda commands, **kwargs: [
		SimpleNamespace(command=list(commands[0]), returncode=returncode, stdout=list(lines), stderr=[])])
	monkeypatch.setattr(hpcp, 'check_path', lambda program: True)
	assert hpcp.run_commands_in_multicmd_with_path_check([[tool, '/img']], quiet=True, show_errors=True) == [lines]
	said = capsys.readouterr()
	assert (said.out.splitlines()[1:] == lines and said.out.splitlines()[0].endswith(tool + ' /img ended %d and printed:' % returncode)) if shown else said.out == ''
	assert bool(hpcp.ERRORS) == failed and ('Task return code error' in said.err) == failed
	# only the cleanup's: a quiet caller that reads the output itself (blkid, lsblk) prints none of it
	hpcp.run_commands_in_multicmd_with_path_check([[tool, '/img']], quiet=True)
	assert capsys.readouterr().out == ''


def test_what_a_command_printed_has_nowhere_to_go_once_the_reader_quit_and_the_cleanup_goes_on(reset_hpcp_globals, monkeypatch):
	monkeypatch.setattr(hpcp.multiCMD, 'run_commands', lambda commands, **kwargs: [
		SimpleNamespace(command=list(command), returncode=1, stdout=['fixed'], stderr=[]) for command in commands])
	monkeypatch.setattr(hpcp, 'check_path', lambda program: True)
	detached = []
	monkeypatch.setattr(hpcp, 'detach_loop_device', lambda device: detached.append(device) or True)
	monkeypatch.setattr(sys, 'stdout', _GoneReader())
	hpcp.clean_up([], ['/dev/loop5'], [['e2fsck', '-f', '-y', '/img']])
	assert detached == ['/dev/loop5'] and hpcp.ERRORS == []   # no 'cleanup step failed', and e2fsck's 1 is a success


def test_real_delayed_commands_are_named_and_only_the_one_that_did_not_end_0_is_shown(reset_hpcp_globals, capsys):
	# the real embedded multiCMD and real commands: sh stands for a tool that reports on stdout and ends 1
	hpcp.clean_up([], [], [['sh', '-c', 'echo all fine'], ['sh', '-c', 'echo repaired inode 15; echo oops >&2; exit 1']])
	said = capsys.readouterr()
	out = said.out.splitlines()
	assert out[0] == 'Running delayed commands: sh -c echo all fine; sh -c echo repaired inode 15; echo oops >&2; exit 1'
	assert 'repaired inode 15' in out and 'all fine' not in out   # a line of its own: the first line names the command
	assert [line for line in out if line.endswith(' ended 1 and printed:')] != []
	assert 'failed with return code 1: oops' in said.err and hpcp.get_rc_from_error() == 128


def test_real_cleanup_with_stdout_closed_runs_a_command_that_prints_for_seconds_to_its_end(tmp_path):
	# the reader of stdout has quit, and the delayed command prints for 2.5 s (multiCMD 1.64 stops a shown one after 1 s, and
	# 1.61 to 1.63 raise at its first line): quiet, it runs to its end, and clean_up returns. 9.91 fails this.
	code = ('import sys; sys.path.insert(0, %r); import hpcp\n'
		'hpcp.clean_up([], [], [["sh", "-c", "i=0; while [ $i -lt 25 ]; do echo line $i; sleep 0.1; i=$((i+1)); done; touch done-mark"]])\n'
		'sys.stderr.write("clean_up returned\\n")\n' % os.path.dirname(os.path.abspath(hpcp.__file__)))
	reader, writer = os.pipe()
	os.close(reader)   # quit before the child starts: its first write that reaches the pipe meets no reader
	try:
		task = subprocess.Popen([sys.executable, '-c', code], stdout=writer, stderr=subprocess.PIPE, cwd=str(tmp_path))
	finally:
		os.close(writer)
	try:
		said = task.communicate(timeout=120)[1].decode('utf-8', 'replace')   # a hang fails here, not the whole run
	finally:
		task.kill()
	assert 'clean_up returned' in said and os.path.exists(str(tmp_path / 'done-mark')), said[-2000:]


def test_a_library_call_with_an_ascii_stdout_draws_its_progress_bar(tmp_path):
	# multiCMD 1.61's print_progress_bar printed its non-ASCII bar again in its except branch: 9.91's library call on Python 3.6
	# under LC_ALL=C returned 70, its files copied (any Python, with an ASCII stdout forced); 1.62's falls back to ASCII
	src, dst = tmp_path / 'src', tmp_path / 'dst'
	src.mkdir()
	dst.mkdir()
	for i in range(30):
		(src / ('f%d' % i)).write_text('x' * 1000)
	code = ('import sys; sys.path.insert(0, %r); import hpcp; '
		'print("rc", hpcp.hpcp([%r], dest_paths=[%r], single_thread=True))' % (os.path.dirname(os.path.abspath(hpcp.__file__)), str(src) + '/', str(dst) + '/'))
	env = dict(os.environ, LC_ALL='C', PYTHONIOENCODING='ascii')
	task = subprocess.run([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=str(tmp_path), timeout=120)
	output = task.stdout.decode('ascii', 'replace')
	assert 'rc 0' in output.splitlines()[-1], output[-2000:]
	assert len(os.listdir(str(dst))) == 30
