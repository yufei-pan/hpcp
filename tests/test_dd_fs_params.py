import os
import sys

import pytest

# Import sibling module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hpcp


def test_no_fs_param_mirror_flag_parses():
	old = sys.argv
	try:
		sys.argv = ['hpcp', '-nfp', '-dd', '/tmp/src.img', '/tmp/dest.img']
		args = hpcp.get_args()
		assert args.no_fs_param_mirror is True
	finally:
		sys.argv = old


def test_no_fs_param_mirror_default_off():
	old = sys.argv
	try:
		sys.argv = ['hpcp', '-dd', '/tmp/src.img', '/tmp/dest.img']
		args = hpcp.get_args()
		assert args.no_fs_param_mirror is False
	finally:
		sys.argv = old


def test_mirror_fs_params_global_defaults_true():
	assert hpcp.MIRROR_FS_PARAMS is True


def test_probe_fs_params_unknown_type_returns_empty():
	assert hpcp.probe_fs_params('/dev/null', 'no_such_fs') == {}


def test_probe_fs_params_never_raises_on_probe_error():
	def boom(device):
		raise RuntimeError('probe exploded')
	hpcp._FS_PARAM_PROBES['test_boom_fs'] = boom
	try:
		assert hpcp.probe_fs_params('/dev/null', 'test_boom_fs') == {}
	finally:
		del hpcp._FS_PARAM_PROBES['test_boom_fs']


def test_build_mkfs_params_unknown_type_returns_empty_list():
	assert hpcp.build_mkfs_params('no_such_fs', {'block_size': 4096}) == []


def test_build_mkfs_params_empty_params_returns_empty_list():
	assert hpcp.build_mkfs_params('ext4', {}) == []


def test_build_mkfs_params_never_raises_on_builder_error():
	def boom(params):
		raise RuntimeError('builder exploded')
	hpcp._FS_MKFS_BUILDERS['test_boom_fs'] = boom
	try:
		assert hpcp.build_mkfs_params('test_boom_fs', {'a': 1}) == []
	finally:
		del hpcp._FS_MKFS_BUILDERS['test_boom_fs']


def test_partition_details_dict_has_fs_params_key():
	# get_partition_details builds this key set; assert the contract without running sgdisk.
	import inspect
	src = inspect.getsource(hpcp.get_partition_details)
	assert "'fs_params'" in src


class _FakeTask:
	def __init__(self, returncode, stderr=None):
		self.returncode = returncode
		self.stderr = stderr or []


def test_mkfs_fallback_uses_mirrored_params_when_they_succeed(monkeypatch):
	calls = []

	def fake_run(commands, **kwargs):
		calls.append(list(commands[0]))
		return [_FakeTask(0)]

	monkeypatch.setattr(hpcp, '_binPaths', {})
	monkeypatch.setattr(hpcp.multiCMD, 'run_commands', fake_run)
	ok = hpcp._run_mkfs_with_fallback(['mkfs', '-t', 'ext4'], ['-b', '1024'], '/dev/fake1', 'ext4')
	assert ok is True
	assert len(calls) == 1
	assert calls[0] == ['mkfs', '-t', 'ext4', '-b', '1024', '/dev/fake1']


def test_mkfs_fallback_retries_without_params_on_failure(monkeypatch):
	calls = []

	def fake_run(commands, **kwargs):
		calls.append(list(commands[0]))
		# First attempt (with mirrored params) fails, second succeeds.
		return [_FakeTask(1, ['invalid block size'])] if len(calls) == 1 else [_FakeTask(0)]

	monkeypatch.setattr(hpcp, '_binPaths', {})
	monkeypatch.setattr(hpcp.multiCMD, 'run_commands', fake_run)
	ok = hpcp._run_mkfs_with_fallback(['mkfs', '-t', 'ext4'], ['-b', '1024'], '/dev/fake1', 'ext4')
	assert ok is True
	assert len(calls) == 2
	assert calls[0] == ['mkfs', '-t', 'ext4', '-b', '1024', '/dev/fake1']
	assert calls[1] == ['mkfs', '-t', 'ext4', '/dev/fake1']


def test_mkfs_fallback_reports_failure_when_both_attempts_fail(monkeypatch):
	def fake_run(commands, **kwargs):
		return [_FakeTask(1, ['no such device'])]

	monkeypatch.setattr(hpcp, '_binPaths', {})
	monkeypatch.setattr(hpcp.multiCMD, 'run_commands', fake_run)
	assert hpcp._run_mkfs_with_fallback(['mkfs', '-t', 'ext4'], ['-b', '1024'], '/dev/fake1', 'ext4') is False


def test_mkfs_fallback_single_attempt_when_no_params(monkeypatch):
	calls = []

	def fake_run(commands, **kwargs):
		calls.append(list(commands[0]))
		return [_FakeTask(0)]

	monkeypatch.setattr(hpcp, '_binPaths', {})
	monkeypatch.setattr(hpcp.multiCMD, 'run_commands', fake_run)
	assert hpcp._run_mkfs_with_fallback(['mkfs.xfs'], [], '/dev/fake1', 'xfs') is True
	assert calls == [['mkfs.xfs', '/dev/fake1']]


def test_mkfs_fallback_resolves_binary_path_from_binpaths(monkeypatch):
	calls = []

	def fake_run(commands, **kwargs):
		calls.append(list(commands[0]))
		return [_FakeTask(0)]

	monkeypatch.setattr(hpcp, '_binPaths', {'mkfs': '/usr/sbin/mkfs'})
	monkeypatch.setattr(hpcp.multiCMD, 'run_commands', fake_run)
	ok = hpcp._run_mkfs_with_fallback(['mkfs', '-t', 'ext4'], ['-b', '1024'], '/dev/fake1', 'ext4')
	assert ok is True
	assert len(calls) == 1
	# Verify the resolved path is used, remaining args are preserved
	assert calls[0][0] == '/usr/sbin/mkfs'
	assert calls[0][1:] == ['-t', 'ext4', '-b', '1024', '/dev/fake1']


def test_write_partition_info_applies_mirrored_params(monkeypatch):
	seen = {}

	def fake_target(image, partition_name):
		return '/dev/fake1', None

	def fake_run_cmd(command, **kwargs):
		return ['']

	def fake_mkfs(base_command, param_args, target_partition, fs_type):
		seen['base'] = list(base_command)
		seen['params'] = list(param_args)
		seen['target'] = target_partition
		return True

	monkeypatch.setattr(hpcp, 'get_target_partition', fake_target)
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check', fake_run_cmd)
	monkeypatch.setattr(hpcp, '_run_mkfs_with_fallback', fake_mkfs)
	monkeypatch.setattr(hpcp, 'MIRROR_FS_PARAMS', True)
	monkeypatch.setitem(hpcp._FS_MKFS_BUILDERS, 'ext4', lambda p: ['-b', '1024'])

	infos = {'2': {'partition_guid_code': '', 'unique_partition_guid': '', 'partition_name': '',
				   'partition_attrs': '', 'fs_type': 'ext4', 'fs_uuid': '', 'fs_label': 'BOOTFS',
				   'size': 0, 'fs_params': {'block_size': 1024}}}
	hpcp.write_partition_info('/dev/fakeimg', infos, '2')

	assert seen['params'] == ['-b', '1024']
	assert seen['target'] == '/dev/fake1'
	assert seen['base'][:3] == ['mkfs', '-t', 'ext4']
	assert '/dev/fake1' not in seen['base']


def test_write_partition_info_skips_mirroring_when_disabled(monkeypatch):
	seen = {}

	def fake_mkfs(base_command, param_args, target_partition, fs_type):
		seen['params'] = list(param_args)
		return True

	monkeypatch.setattr(hpcp, 'get_target_partition', lambda image, name: ('/dev/fake1', None))
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check', lambda command, **kwargs: [''])
	monkeypatch.setattr(hpcp, '_run_mkfs_with_fallback', fake_mkfs)
	monkeypatch.setattr(hpcp, 'MIRROR_FS_PARAMS', False)
	monkeypatch.setitem(hpcp._FS_MKFS_BUILDERS, 'ext4', lambda p: ['-b', '1024'])

	infos = {'2': {'partition_guid_code': '', 'unique_partition_guid': '', 'partition_name': '',
				   'partition_attrs': '', 'fs_type': 'ext4', 'fs_uuid': '', 'fs_label': '',
				   'size': 0, 'fs_params': {'block_size': 1024}}}
	hpcp.write_partition_info('/dev/fakeimg', infos, '2')

	assert seen['params'] == []
