'''hpcp 9.92: the cleanup after an image run runs every step, quiet. Each test fails on 9.91 (9348355). Every failure is forced at the
step it is timed by, in this process: a reader of stdout that quit makes a shown command's call raise BrokenPipeError (multiCMD
1.64) and a print raise it too; no sleep, no real tool, no loop device.'''
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

import hpcp

DELAYED = [['e2fsck', '-f', '-y', '/img'], ['tune2fs', '-U', 'random', '/img'], ['losetup', '--detach', '/dev/loop7']]


class _GoneReader:
	'''A stdout whose reader quit: every write and flush raises BrokenPipeError, as `hpcp -dd src dst | head` gives it.'''
	encoding = 'utf-8'

	def write(self, text):
		raise BrokenPipeError(32, 'Broken pipe')

	def flush(self):
		raise BrokenPipeError(32, 'Broken pipe')


def _record(monkeypatch, fail=()):
	'''clean_up's three kinds of step recorded in order: a command call (its tool), a loop device's detach. A tool in fail
	raises BrokenPipeError when it is called, as a shown call does once stdout's reader has quit.'''
	steps = []

	def run(commands, **kwargs):
		for command in commands:
			steps.append(('run', command[0], kwargs.get('quiet'), kwargs.get('show_errors')))
			if command[0] in fail:
				raise BrokenPipeError(32, 'Broken pipe')
		return [[] for _ in commands]

	monkeypatch.setattr(hpcp, 'run_commands_in_multicmd_with_path_check', run)
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check', lambda command, **kwargs: run([command], **kwargs)[0])
	monkeypatch.setattr(hpcp, 'detach_loop_device', lambda device: steps.append(('detach', device)) or True)
	monkeypatch.setattr(hpcp, '_is_mount_point', lambda path: False)
	return steps


@pytest.mark.parametrize('raising, ran', [
	# the delayed pair is one call: tune2fs is skipped with the e2fsck that raised, the detaches are not
	('e2fsck', ['umount', 'umount', 'e2fsck']),
	('umount', ['umount', 'umount', 'e2fsck', 'tune2fs']),
])
def test_a_step_that_raises_broken_pipe_does_not_skip_the_steps_after_it(tmp_path, reset_hpcp_globals, monkeypatch, capsys, raising, ran):
	# 9.91: the call raised out of clean_up, so the later mounts, the delayed commands and every loop device were skipped
	steps = _record(monkeypatch, fail=(raising,))
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
	assert 'General Exception: cleanup step' in said and 'BrokenPipeError' in said


def test_a_print_into_a_stdout_nobody_reads_does_not_skip_its_step(tmp_path, reset_hpcp_globals, monkeypatch):
	# 9.91: print raised first in each step, before its umount, rmdir and detach
	steps = _record(monkeypatch)
	monkeypatch.setattr(sys, 'stdout', _GoneReader())
	mount = str(tmp_path / 'mount')
	os.mkdir(mount)
	hpcp.clean_up([mount], ['/dev/loop5'], [list(command) for command in DELAYED])
	assert [step[1] for step in steps if step[0] == 'run'] == ['umount', 'e2fsck', 'tune2fs']
	assert [step for step in steps if step[0] == 'detach'] == [('detach', '/dev/loop5'), ('detach', '/dev/loop7')]
	assert not os.path.exists(mount)


def test_every_command_of_the_cleanup_runs_quiet_and_its_failure_is_still_said(tmp_path, reset_hpcp_globals, monkeypatch, capsys):
	# 9.91 ran umount and the delayed pair shown (quiet False): the output multiCMD 1.64 stops a command for
	calls = []

	def run_commands(commands, **kwargs):
		calls.append((os.path.basename(commands[0][0]), kwargs.get('quiet')))
		return [SimpleNamespace(command=' '.join(command), returncode=4 if command[0] == 'e2fsck' else 0, stdout=[], stderr=['/img: bad superblock'])
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
	hpcp.run_command_in_multicmd_with_path_check(['false'], quiet=True, show_errors=True)
	assert 'failed with return code 1: no' in capsys.readouterr().err


def test_a_library_call_with_an_ascii_stdout_draws_its_progress_bar(tmp_path):
	# multiCMD 1.61's print_progress_bar printed its non-ASCII bar again in its except branch: 9.91's library call on Python 3.6
	# under LC_ALL=C returned 70 after half the files (any Python, with an ASCII stdout forced); 1.62's falls back to ASCII
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
