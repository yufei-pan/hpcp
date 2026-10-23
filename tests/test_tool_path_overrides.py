"""A <TOOL>_PATH override (DD_PATH, E2FSCK_PATH, ...) is the program hpcp starts, for every tool: through the command
runner as well as where hpcp starts a tool with multiCMD itself."""
import os
import shutil
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
	monkeypatch.setattr(hpcp, '_warnedOverrides', set())
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
	hpcp.cp_supports_sparse.cache_clear()
	try:
		assert hpcp.cp_supports_sparse() is False
	finally:
		hpcp.cp_supports_sparse.cache_clear()


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


def test_a_bad_override_is_reported_when_the_tool_is_needed_and_is_not_run(tmp_path, exec_dir, monkeypatch, capsys):
	bin_dir = exec_dir / 'bin'
	bin_dir.mkdir()
	_fake_tool(bin_dir, 'dumpe2fs', 'echo "Block size:               2048"\n')
	monkeypatch.setenv('PATH', str(bin_dir) + os.pathsep + os.environ.get('PATH', ''))
	not_executable = tmp_path / 'dumpe2fs-data'
	not_executable.write_text('#!/bin/sh\necho "Block size:               9999"\n')
	not_executable.chmod(0o644)
	for bad in (str(not_executable), str(tmp_path / 'nope'), str(tmp_path)):
		hpcp._warnedOverrides.clear()
		monkeypatch.setenv('DUMPE2FS_PATH', bad)
		assert capsys.readouterr().err == ''
		assert hpcp._probe_ext('/dev/x')['block_size'] == 2048
		err = capsys.readouterr().err
		assert err.count('DUMPE2FS_PATH=' + bad) == 1 and 'is not an executable file' in err
		hpcp.check_path.cache_clear()
		hpcp._binPaths.clear()


def test_a_stray_mount_path_or_cp_path_is_silent_until_used(tmp_path, exec_dir, monkeypatch, capsys):
	# MOUNT_PATH and CP_PATH are variables other software sets to directories and the like
	monkeypatch.setenv('MOUNT_PATH', str(tmp_path))
	monkeypatch.setenv('CP_PATH', str(tmp_path / 'data-dir-of-something-else'))
	hpcp.check_path('mount')
	assert hpcp.check_path('cp') is True
	assert capsys.readouterr().err == ''
	# used: reported, and the cp on PATH is the one that runs
	src = tmp_path / 'src'
	src.write_text('data')
	hpcp.run_command_in_multicmd_with_path_check(['cp', str(src), str(tmp_path / 'dst')], quiet=True, strict=True)
	assert (tmp_path / 'dst').read_text() == 'data'
	assert 'CP_PATH=' in capsys.readouterr().err


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
	command = ['dumpe2fs', '-h', '/dev/hpcp-none']
	hpcp.run_command_in_multicmd_with_path_check(command, quiet=True)
	assert command == ['dumpe2fs', '-h', '/dev/hpcp-none']
	assert hpcp.run_commands_in_multicmd_with_path_check([['dumpe2fs', '/dev/hpcp-none']], quiet=True) == [['args:/dev/hpcp-none']]


def test_an_override_in_a_directory_with_an_equals_sign_runs(exec_dir, monkeypatch):
	# `env` in front of a command takes a leading NAME=value for an assignment; none is in front of it now
	directory = exec_dir / 'a=b'
	directory.mkdir()
	monkeypatch.setenv('DUMPE2FS_PATH', _fake_tool(directory, 'dumpe2fs', 'echo "Block size:               1024"\n'))
	assert hpcp._probe_ext('/dev/x')['block_size'] == 1024
