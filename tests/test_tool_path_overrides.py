"""A <TOOL>_PATH override (DD_PATH, E2FSCK_PATH, ...) is the program hpcp starts, for every tool: through the command
runner as well as where hpcp starts a tool with multiCMD itself."""
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hpcp

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='runs shell scripts')


def _fake_tool(directory, name, body):
	path = os.path.join(str(directory), name)
	with open(path, 'w', encoding='utf-8') as f:
		f.write('#!/bin/sh\n' + body)
	os.chmod(path, 0o755)
	return path


@pytest.fixture(autouse=True)
def _fresh_tool_lookup(monkeypatch):
	"""Each test finds its tools anew, from the environment it sets, and leaves nothing of that behind."""
	monkeypatch.setattr(hpcp, '_binPaths', {})
	monkeypatch.setattr(hpcp, '_badOverrides', {})
	hpcp.check_path.cache_clear()
	yield
	hpcp.check_path.cache_clear()


def test_dumpe2fs_path_is_the_program_that_reads_the_ext_geometry(exec_dir, monkeypatch):
	# dumpe2fs was started by its bare name from PATH whatever DUMPE2FS_PATH said
	fake = _fake_tool(exec_dir, 'my-dumpe2fs', 'echo "Block size:               4096"\necho "Inode size:               256"\n')
	monkeypatch.setenv('DUMPE2FS_PATH', fake)
	params = hpcp._probe_ext('/dev/hpcp-no-such-device')
	assert params['block_size'] == 4096 and params['inode_size'] == 256


def test_blkid_path_is_the_program_that_reads_the_filesystem_type(exec_dir, monkeypatch):
	fake = _fake_tool(exec_dir, 'my-blkid', 'echo "$@" > "%s"\necho ext4\n' % (exec_dir / 'args'))
	monkeypatch.setenv('BLKID_PATH', fake)
	assert hpcp._probe_fs_type('/dev/hpcp-no-such-device') == 'ext4'
	assert (exec_dir / 'args').read_text().split() == ['-p', '-s', 'TYPE', '-o', 'value', '/dev/hpcp-no-such-device']


def test_cp_path_is_the_cp_whose_options_are_probed(exec_dir, monkeypatch):
	# the real cp here is GNU: only the override can make this False
	monkeypatch.setenv('CP_PATH', _fake_tool(exec_dir, 'my-cp', 'echo "my cp 1.0"\n'))
	hpcp.check_path('cp')  # as at import
	hpcp._SPARSE[:] = [None, False]
	try:
		assert hpcp.cp_supports_sparse() is False
	finally:
		hpcp._SPARSE[:] = [None, False]


def test_an_override_with_another_name_is_judged_as_the_tool(exec_dir, monkeypatch):
	# e2fsck exits 1 for "errors corrected", which is success; the override's own name must not change that
	monkeypatch.setenv('E2FSCK_PATH', _fake_tool(exec_dir, 'my-fsck', 'exit 1\n'))
	errors = len(hpcp.ERRORS)
	try:
		hpcp.run_command_in_multicmd_with_path_check(['e2fsck', '-f', '-y', '/dev/x'], quiet=True, strict=True)
		assert len(hpcp.ERRORS) == errors
		monkeypatch.setenv('DUMPE2FS_PATH', _fake_tool(exec_dir, 'my-dumpe2fs', 'exit 1\n'))
		hpcp.run_command_in_multicmd_with_path_check(['dumpe2fs', '-h', '/dev/x'], quiet=True)
		assert len(hpcp.ERRORS) == errors + 1 and hpcp.ERRORS[-1].startswith('Task return code error:')
	finally:
		del hpcp.ERRORS[errors:]


def test_every_bad_override_is_named_once_at_the_start_of_a_run_and_is_not_run(tmp_path, exec_dir, monkeypatch, capsys):
	# CP_PATH too, though cp runs in the workers; MOUNT_PATH and CP_PATH are variables other software sets to
	# directories and the like. Importing hpcp stays silent (test_importing_hpcp_prints_nothing_...).
	not_executable = tmp_path / 'dumpe2fs-data'
	not_executable.write_text('#!/bin/sh\necho "Block size:               9999"\n')
	not_executable.chmod(0o644)
	bad = {'DUMPE2FS_PATH': str(not_executable), 'CP_PATH': str(tmp_path), 'MOUNT_PATH': str(tmp_path / 'nope')}
	def run(*args):
		return subprocess.run([sys.executable, hpcp.__file__] + list(args), env=dict(os.environ, **bad), stdout=subprocess.PIPE,
			stderr=subprocess.PIPE, universal_newlines=True)
	(tmp_path / 'empty').mkdir()
	result = run(str(tmp_path / 'empty'), '-d', str(tmp_path / 'out'))
	assert result.returncode == 0, result.stderr
	for variable, value in bad.items():
		assert result.stderr.count('Warning: %s=%s is not an executable file' % (variable, value)) == 1
	# -V and -h print only what they are asked for
	for option in ('-V', '-h'):
		result = run(option)
		assert result.returncode == 0 and result.stdout and result.stderr == ''
	# the tool on PATH is the one that runs, and nothing more is printed when it does
	bin_dir = exec_dir / 'bin'
	bin_dir.mkdir()
	_fake_tool(bin_dir, 'dumpe2fs', 'echo "Block size:               2048"\n')
	monkeypatch.setenv('PATH', str(bin_dir) + os.pathsep + os.environ.get('PATH', ''))
	monkeypatch.setenv('DUMPE2FS_PATH', str(not_executable))
	assert hpcp._probe_ext('/dev/x')['block_size'] == 2048
	monkeypatch.setenv('CP_PATH', str(tmp_path))
	src = tmp_path / 'src'
	src.write_text('data')
	hpcp.run_command_in_multicmd_with_path_check(['cp', str(src), str(tmp_path / 'dst')], quiet=True, strict=True)
	assert (tmp_path / 'dst').read_text() == 'data'
	assert capsys.readouterr().err == ''


def test_a_stray_override_that_is_an_executable_file_now_runs_as_the_tool(exec_dir, monkeypatch):
	# the price of honouring every <TOOL>_PATH: MOUNT_PATH naming an executable is mount
	marker = exec_dir / 'ran'
	monkeypatch.setenv('MOUNT_PATH', _fake_tool(exec_dir, 'something-else', 'echo "$@" > "%s"\n' % marker))
	hpcp.run_command_in_multicmd_with_path_check(['mount', '-o', 'ro', '/dev/x', '/mnt/x'], quiet=True)
	assert marker.read_text().split() == ['-o', 'ro', '/dev/x', '/mnt/x']


def test_the_runner_takes_a_tuple_and_leaves_the_callers_list_alone(exec_dir, monkeypatch):
	# (a bogus device throughout: without the override the real dumpe2fs would be started, and fail on it)
	monkeypatch.setenv('DUMPE2FS_PATH', _fake_tool(exec_dir, 'my-dumpe2fs', 'echo "args:$*"\n'))
	assert hpcp.run_command_in_multicmd_with_path_check(('dumpe2fs', '-h', '/dev/hpcp-none'), quiet=True) == ['args:-h /dev/hpcp-none']
	# a path-like tool name, which the runner used to replace in the caller's list
	command = [pathlib.PurePath('dumpe2fs'), '-h', '/dev/hpcp-none']
	assert hpcp.run_command_in_multicmd_with_path_check(command, quiet=True) == ['args:-h /dev/hpcp-none']
	assert command == [pathlib.PurePath('dumpe2fs'), '-h', '/dev/hpcp-none']
	assert hpcp.run_commands_in_multicmd_with_path_check([['dumpe2fs', '/dev/hpcp-none']], quiet=True) == [['args:/dev/hpcp-none']]


def test_an_override_of_a_tool_not_looked_up_at_import_is_the_program_mkfs_runs(exec_dir, monkeypatch):
	# mkfs.hfs (built from the filesystem type) was not in _binCalled: _bin gave its bare name whatever its override said;
	# a tool not looked up yet (as here, the lookup cleared) is looked up by _bin
	fake = _fake_tool(exec_dir, 'my-mkfs-hfs', 'exit 0\n')
	monkeypatch.setenv('MKFS.HFS_PATH', fake)
	assert hpcp._bin('mkfs.hfs') == fake


def test_an_override_in_a_directory_with_an_equals_sign_runs(exec_dir, monkeypatch):
	# `env` in front of a command takes a leading NAME=value for an assignment; none is in front of it now
	directory = exec_dir / 'a=b'
	directory.mkdir()
	monkeypatch.setenv('DUMPE2FS_PATH', _fake_tool(directory, 'dumpe2fs', 'echo "Block size:               1024"\n'))
	assert hpcp._probe_ext('/dev/x')['block_size'] == 1024


def test_a_cp_version_that_gives_no_answer_is_asked_again(exec_dir, monkeypatch):
	# entry 5: 9.86 kept a timed-out `cp --version` (False) for the process's life: that worker's copies lost --sparse, and its
	# rewrites went to cp by name. A fake cp silent past the timeout once, then answering.
	marker = exec_dir / 'asked'
	fake = _fake_tool(exec_dir, 'slow-cp', 'if [ ! -e "%s" ]; then : > "%s"; sleep 3; fi\necho "cp (GNU coreutils) 9.4"\n' % (marker, marker))
	monkeypatch.setenv('CP_PATH', fake)
	monkeypatch.setattr(hpcp, '_CP_ASK_TIMEOUT', 0.2)
	monkeypatch.setattr(hpcp, '_SPARSE', [None, False])
	monkeypatch.setattr(hpcp, '_HELD', [None, None])
	# the held-file route, the first to ask: no answer, so it is asked again at the next rewrite
	assert hpcp._held_route() is None and hpcp._HELD[0] is None
	assert hpcp.cp_supports_sparse() is True and hpcp._SPARSE[1] is True
