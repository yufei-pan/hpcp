import os
import shutil
import struct
import subprocess
import sys
import tempfile

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


def test_probe_fs_params_truncates_errors_left_by_failed_tool():
	# Pins finding 2: run_command_in_multicmd_with_path_check appends
	# "Task return code error: ..." to the global ERRORS even with quiet=True
	# (hpcp.py ~465-467), so a probe tool that exits non-zero (e.g. fsck.fat -nv
	# on a FAT with the dirty bit set) must not leave that entry behind - a
	# clean copy must not come out of it with a poisoned exit code.
	def fake_probe_like_a_failed_quiet_tool(device):
		hpcp.ERRORS.append(f"Task return code error: Command '['dumpe2fs', '-h', '{device}']' failed with return code 19.")
		return {}
	hpcp._FS_PARAM_PROBES['test_quiet_fail_fs'] = fake_probe_like_a_failed_quiet_tool
	saved_errors = list(hpcp.ERRORS)
	hpcp.ERRORS.clear()
	try:
		result = hpcp.probe_fs_params('/dev/null', 'test_quiet_fail_fs')
		assert result == {}
		assert hpcp.ERRORS == []
	finally:
		del hpcp._FS_PARAM_PROBES['test_quiet_fail_fs']
		hpcp.ERRORS.clear()
		hpcp.ERRORS.extend(saved_errors)


def test_get_rc_from_error_returns_zero_for_fs_param_warning_only():
	# Pins finding 1: none of this branch's warning prefixes used to be
	# registered in ERROR_TO_RETURNCODE_TABLE, so when one was the only entry
	# in ERRORS, get_rc_from_error()'s max() over an empty filtered generator
	# raised ValueError at the very end of an otherwise successful -dd run.
	saved_errors = list(hpcp.ERRORS)
	hpcp.ERRORS.clear()
	hpcp.ERRORS.append('FS param warning: mkfs rejected mirrored btrfs parameters -O ^squota on /dev/loop0p1: err')
	try:
		assert hpcp.get_rc_from_error() == 0
	finally:
		hpcp.ERRORS.clear()
		hpcp.ERRORS.extend(saved_errors)


def test_get_rc_from_error_returns_nonzero_for_create_fs_error_only():
	# Sibling to the test above, pinning the follow-up correction: unlike the
	# three FS param *warnings* (a successful copy that degraded to mkfs
	# defaults), "Create fs error" means every fallback tier in
	# _run_mkfs_with_fallback failed and the destination partition has no
	# filesystem at all - a real failure that must keep a non-zero exit code,
	# not be flattened to 0 alongside the warnings.
	saved_errors = list(hpcp.ERRORS)
	hpcp.ERRORS.clear()
	hpcp.ERRORS.append('Create fs error: Failed to create ext4 on /dev/loop0p1: no such device')
	try:
		assert hpcp.get_rc_from_error() == 187
	finally:
		hpcp.ERRORS.clear()
		hpcp.ERRORS.extend(saved_errors)


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


def test_get_partition_details_skips_probe_when_mirroring_disabled(monkeypatch):
	# Pins finding 3: -nfp must stop hpcp from spawning probe tools at all
	# (dumpe2fs / xfs_info / fsck.fat -nv / btrfs dump-super / ...), not just
	# discard the probed result later in write_partition_info.
	sgdisk_output = [
		'Partition GUID code: C12A7328-F81F-11D2-BA4B-00A0C93EC93B (EFI System)',
		'Partition unique GUID: 11111111-1111-1111-1111-111111111111',
		"Partition name: 'ESP'",
		'Attribute flags: 0000000000000000',
		'Partition size: 2048 sectors (1.0 MiB)',
	]
	blkid_output = ['TYPE=vfat']

	def fake_run_cmd(command, **kwargs):
		if command[0] == 'sgdisk':
			return sgdisk_output
		if command[0] == 'blkid':
			return blkid_output
		return ['']

	probe_calls = []

	def spy_probe(target_partition, fs_type):
		probe_calls.append((target_partition, fs_type))
		return {}

	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check', fake_run_cmd)
	monkeypatch.setattr(hpcp, 'get_target_partition', lambda device, partition: (device, None))
	monkeypatch.setattr(hpcp, 'probe_fs_params', spy_probe)
	monkeypatch.setattr(hpcp, 'MIRROR_FS_PARAMS', False)

	hpcp.get_partition_details.cache_clear()
	try:
		result = hpcp.get_partition_details('/dev/fake_nfp_test_img', '1')
		assert probe_calls == []
		assert result['fs_type'] == 'vfat'
		assert result['fs_params'] == {}
	finally:
		hpcp.get_partition_details.cache_clear()


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


def test_mkfs_fallback_drops_only_dash_o_on_middle_retry(monkeypatch):
	# Pins finding 4: an unrecognised -O feature name must not drag down the
	# rest of the mirrored parameters (block size, inode size, -m, ...) with
	# it. When -O is present and the full set is rejected, the middle retry
	# must strip only the -O pair and keep everything else, before falling
	# all the way back to bare defaults.
	calls = []

	def fake_run(commands, **kwargs):
		calls.append(list(commands[0]))
		# First two attempts fail; the third (bare defaults) succeeds.
		return [_FakeTask(0)] if len(calls) >= 3 else [_FakeTask(1, [f'attempt {len(calls)} rejected'])]

	monkeypatch.setattr(hpcp, '_binPaths', {})
	monkeypatch.setattr(hpcp.multiCMD, 'run_commands', fake_run)
	param_args = ['-b', '1024', '-O', 'has_journal,^orphan_file', '-m', '0.00']
	ok = hpcp._run_mkfs_with_fallback(['mkfs', '-t', 'ext4'], param_args, '/dev/fake1', 'ext4')

	assert ok is True
	assert len(calls) == 3
	# Attempt 1: the full mirrored parameter set, -O included.
	assert calls[0] == ['mkfs', '-t', 'ext4', '-b', '1024', '-O', 'has_journal,^orphan_file', '-m', '0.00', '/dev/fake1']
	# Attempt 2: -O and its value gone, the other mirrored parameters kept.
	assert calls[1] == ['mkfs', '-t', 'ext4', '-b', '1024', '-m', '0.00', '/dev/fake1']
	# Attempt 3: bare defaults, no mirrored parameters at all.
	assert calls[2] == ['mkfs', '-t', 'ext4', '/dev/fake1']


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


def _partition_info_stub(fs_type, fs_uuid, partition='2'):
	return {partition: {'partition_guid_code': '', 'unique_partition_guid': '', 'partition_name': '',
						'partition_attrs': '', 'fs_type': fs_type, 'fs_uuid': fs_uuid, 'fs_label': '',
						'size': 0, 'fs_params': {}}}


@pytest.mark.parametrize('fs_type', ['ext2', 'ext3', 'ext4'])
def test_write_partition_info_fscks_ext_before_delayed_uuid(monkeypatch, fs_type):
	# tune2fs -U on a metadata_csum filesystem (default ext4) refuses unless
	# e2fsck -f has just run. UUID is delayed until after the copy so the live
	# source and new dest do not share a UUID while both are mounted, which
	# dirties the superblock - so the delayed pair must be e2fsck then tune2fs.
	target = '/dev/loop0p2'
	uuid = '9a7908e4-8af1-4be2-b7a0-83536e09ecc2'
	monkeypatch.setattr(hpcp, 'get_target_partition', lambda image, name: (target, None))
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check', lambda command, **kwargs: [''])
	monkeypatch.setattr(hpcp, '_run_mkfs_with_fallback', lambda *a, **k: True)

	delayed = hpcp.write_partition_info('/dev/fakeimg', _partition_info_stub(fs_type, uuid), '2')

	assert delayed[0] == ['e2fsck', '-f', '-y', target]
	assert delayed[1] == ['tune2fs', '-U', uuid, target]


def test_write_partition_info_does_not_fsck_ext_when_uuid_is_empty(monkeypatch):
	monkeypatch.setattr(hpcp, 'get_target_partition', lambda image, name: ('/dev/loop0p2', None))
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check', lambda command, **kwargs: [''])
	monkeypatch.setattr(hpcp, '_run_mkfs_with_fallback', lambda *a, **k: True)

	delayed = hpcp.write_partition_info('/dev/fakeimg', _partition_info_stub('ext4', ''), '2')

	assert delayed == []
	assert not any(cmd and cmd[0] == 'e2fsck' for cmd in delayed)


def test_e2fsck_exit_1_is_not_a_task_error(monkeypatch):
	# e2fsck uses a bitmask: 1 = errors corrected, 2 = reboot recommended.
	# A live-system clone almost always needs journal replay, so treating 1 as
	# "Task return code error" would still fail the clone after we fsck so
	# tune2fs -U can run.
	class Task:
		def __init__(self, command, returncode):
			self.command = command
			self.returncode = returncode
			self.stdout = []
			self.stderr = []

	saved_errors = list(hpcp.ERRORS)
	hpcp.ERRORS.clear()
	command = ['e2fsck', '-f', '-y', '/dev/loop0p2']
	monkeypatch.setattr(hpcp, '_binPaths', {'e2fsck': '/sbin/e2fsck'})
	monkeypatch.setattr(hpcp.multiCMD, 'run_commands', lambda commands, **kwargs: [Task(commands[0], 1)])
	try:
		hpcp.run_commands_in_multicmd_with_path_check([command], strict=False)
		assert hpcp.ERRORS == []
	finally:
		hpcp.ERRORS.clear()
		hpcp.ERRORS.extend(saved_errors)


def test_e2fsck_exit_4_is_still_a_task_error(monkeypatch):
	class Task:
		def __init__(self, command, returncode):
			self.command = command
			self.returncode = returncode
			self.stdout = []
			self.stderr = []

	saved_errors = list(hpcp.ERRORS)
	hpcp.ERRORS.clear()
	command = ['e2fsck', '-f', '-y', '/dev/loop0p2']
	monkeypatch.setattr(hpcp, '_binPaths', {'e2fsck': '/sbin/e2fsck'})
	monkeypatch.setattr(hpcp.multiCMD, 'run_commands', lambda commands, **kwargs: [Task(commands[0], 4)])
	try:
		hpcp.run_commands_in_multicmd_with_path_check([command], strict=False)
		assert any(err.startswith('Task return code error:') for err in hpcp.ERRORS)
	finally:
		hpcp.ERRORS.clear()
		hpcp.ERRORS.extend(saved_errors)


_DUMPE2FS_OUTPUT = """dumpe2fs 1.47.2 (1-Jan-2025)
Filesystem volume name:   BOOTFS
Last mounted on:          /tmp/tmp.BJpz2N4pR0
Filesystem UUID:          d0bac943-fd97-4462-bc04-0ba9a3097027
Filesystem magic number:  0xEF53
Filesystem revision #:    1 (dynamic)
Filesystem features:      has_journal ext_attr resize_inode orphan_file filetype extent flex_bg sparse_super large_file huge_file dir_nlink extra_isize
Filesystem flags:         signed_directory_hash
Default mount options:    user_xattr acl
Filesystem state:         clean
Inode count:              32768
Block count:              524288
Reserved block count:     0
Free blocks:              501204
Free inodes:              32754
First block:              1
Block size:               1024
Fragment size:            1024
Blocks per group:         8192
Inode size:               128
""".splitlines()


def test_probe_ext_parses_geometry_and_features(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: _DUMPE2FS_OUTPUT)
	params = hpcp._probe_ext('/dev/fake2')
	assert params['block_size'] == 1024
	assert params['inode_size'] == 128
	assert params['inode_count'] == 32768
	assert params['block_count'] == 524288
	assert params['reserved_block_count'] == 0
	assert 'has_journal' in params['features']
	assert '64bit' not in params['features']
	assert 'metadata_csum' not in params['features']


def test_build_ext_mirrors_geometry():
	args = hpcp._build_ext({'block_size': 1024, 'inode_size': 128})
	assert args[:4] == ['-b', '1024', '-I', '128']


def test_build_ext_negates_absent_curated_features():
	args = hpcp._build_ext({'features': ['has_journal', 'extent']})
	features = args[args.index('-O') + 1].split(',')
	# Present features are emitted plain...
	assert 'has_journal' in features
	assert 'extent' in features
	# ...and every curated feature the source lacks is explicitly negated,
	# because a plain -O list is merged with the mke2fs.conf defaults.
	assert '^64bit' in features
	assert '^metadata_csum' in features
	assert '^dir_index' in features


def test_build_ext_filters_runtime_state_features():
	args = hpcp._build_ext({'features': ['has_journal', 'needs_recovery', 'journal_dev']})
	features = args[args.index('-O') + 1].split(',')
	assert 'needs_recovery' not in features
	assert 'journal_dev' not in features
	assert '^needs_recovery' not in features


def test_build_ext_mirrors_inode_ratio_not_absolute_count():
	# 524288 blocks * 1024 bytes / 32768 inodes = 16384 bytes per inode
	args = hpcp._build_ext({'block_size': 1024, 'block_count': 524288, 'inode_count': 32768})
	assert args[args.index('-i') + 1] == '16384'
	assert '-N' not in args


def test_build_ext_mirrors_reserved_percentage():
	args = hpcp._build_ext({'block_count': 524288, 'reserved_block_count': 0})
	assert args[args.index('-m') + 1] == '0.00'
	args = hpcp._build_ext({'block_count': 200000, 'reserved_block_count': 10000})
	assert args[args.index('-m') + 1] == '5.00'


def test_ext_registered_for_all_three_types():
	for fs_type in ('ext2', 'ext3', 'ext4'):
		assert hpcp._FS_PARAM_PROBES[fs_type] is hpcp._probe_ext
		assert hpcp._FS_MKFS_BUILDERS[fs_type] is hpcp._build_ext


_XFS_INFO_OUTPUT = """meta-data=/dev/loop8p3           isize=1024   agcount=4, agsize=40127 blks
         =                       sectsz=512   attr=2, projid32bit=1
         =                       crc=1        finobt=1, sparse=1, rmapbt=1
         =                       reflink=0    bigtime=1 inobtcount=1 nrext64=1
         =                       exchange=0   metadir=0
data     =                       bsize=4096   blocks=160507, imaxpct=25
         =                       sunit=0      swidth=0 blks
naming   =version 2              bsize=8192   ascii-ci=0, ftype=1, parent=0
log      =internal log           bsize=4096   blocks=16384, version=2
         =                       sectsz=512   sunit=0 blks, lazy-count=1
realtime =none                   extsz=4096   blocks=0, rtextents=0
""".splitlines()


def test_probe_xfs_parses_sections(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: _XFS_INFO_OUTPUT)
	params = hpcp._probe_xfs('/dev/fake3')
	assert params['meta-data']['isize'] == '1024'
	assert params['meta-data']['sectsz'] == '512'
	assert params['meta-data']['reflink'] == '0'
	assert params['meta-data']['crc'] == '1'
	# bsize appears in three sections; each must land in its own bucket.
	assert params['data']['bsize'] == '4096'
	assert params['naming']['bsize'] == '8192'
	assert params['log']['bsize'] == '4096'
	assert params['data']['imaxpct'] == '25'


def test_build_xfs_mirrors_geometry_and_features(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: _XFS_INFO_OUTPUT)
	args = hpcp._build_xfs(hpcp._probe_xfs('/dev/fake3'))
	assert args[args.index('-b') + 1] == 'size=4096'
	assert args[args.index('-s') + 1] == 'size=512'
	assert args[args.index('-i') + 1] == 'size=1024,sparse=1,projid32bit=1,nrext64=1,maxpct=25'
	assert args[args.index('-n') + 1] == 'size=8192,ftype=1'
	assert args[args.index('-m') + 1] == 'crc=1,finobt=1,rmapbt=1,reflink=0,bigtime=1,inobtcount=1'


def test_build_xfs_omits_size_dependent_params(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: _XFS_INFO_OUTPUT)
	args = hpcp._build_xfs(hpcp._probe_xfs('/dev/fake3'))
	joined = ' '.join(args)
	# Log size and agcount scale with filesystem size and must never be mirrored.
	assert 'agcount' not in joined
	assert 'logdev' not in joined
	assert '-l' not in args


def test_build_xfs_skips_missing_keys():
	args = hpcp._build_xfs({'data': {'bsize': '4096'}})
	assert args == ['-b', 'size=4096']


def test_xfs_registered():
	assert hpcp._FS_PARAM_PROBES['xfs'] is hpcp._probe_xfs
	assert hpcp._FS_MKFS_BUILDERS['xfs'] is hpcp._build_xfs


_BTRFS_SUPER_OUTPUT = """superblock: bytenr=65536, device=/dev/fake4
---------------------------------------------------------
csum_type		0 (crc32c)
csum_size		4
bytenr			65536
flags			0x1
			( WRITTEN )
magic			_BHRfS_M [match]
label			ROOTFS
sectorsize		4096
nodesize		4096
leafsize (deprecated)	4096
stripesize		4096
num_devices		1
compat_flags		0x0
compat_ro_flags		0x3
			( FREE_SPACE_TREE |
			  FREE_SPACE_TREE_VALID )
incompat_flags		0x341
			( MIXED_BACKREF |
			  EXTENDED_IREF |
			  SKINNY_METADATA |
			  NO_HOLES )
""".splitlines()


def test_probe_btrfs_parses_super(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: _BTRFS_SUPER_OUTPUT)
	params = hpcp._probe_btrfs('/dev/fake4')
	assert params['nodesize'] == 4096
	assert params['sectorsize'] == 4096
	assert params['csum_type'] == 'crc32c'
	assert 'NO_HOLES' in params['incompat_flags']
	assert 'EXTENDED_IREF' in params['incompat_flags']
	assert 'FREE_SPACE_TREE' in params['compat_ro_flags']
	# The WRITTEN flag belongs to `flags`, not to the feature flag blocks.
	assert 'WRITTEN' not in params['incompat_flags']
	assert 'WRITTEN' not in params['compat_ro_flags']


def test_build_btrfs_mirrors_geometry_and_features(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: _BTRFS_SUPER_OUTPUT)
	args = hpcp._build_btrfs(hpcp._probe_btrfs('/dev/fake4'))
	assert args[args.index('-n') + 1] == '4096'
	assert args[args.index('-s') + 1] == '4096'
	assert args[args.index('--csum') + 1] == 'crc32c'
	features = args[args.index('-O') + 1].split(',')
	assert 'no-holes' in features
	assert 'extref' in features
	assert 'skinny-metadata' in features
	assert 'free-space-tree' in features
	# Absent curated features are negated so mkfs defaults cannot reintroduce them.
	assert '^raid56' in features
	assert '^block-group-tree' in features
	# Unmappable / non-creation flags are dropped entirely.
	assert 'MIXED_BACKREF' not in features
	assert 'FREE_SPACE_TREE_VALID' not in features


def test_build_btrfs_negates_features_when_source_has_none_mappable():
	args = hpcp._build_btrfs({'incompat_flags': ['MIXED_BACKREF'], 'compat_ro_flags': []})
	features = args[args.index('-O') + 1].split(',')
	assert all(f.startswith('^') for f in features)


def test_btrfs_registered():
	assert hpcp._FS_PARAM_PROBES['btrfs'] is hpcp._probe_btrfs
	assert hpcp._FS_MKFS_BUILDERS['btrfs'] is hpcp._build_btrfs


_BLKID_P_FAT32_OUTPUT = """DEVNAME=/dev/loop8p1
LABEL_FATBOOT=ESP
LABEL=ESP
UUID=A4D7-1D90
VERSION=FAT32
FSBLOCKSIZE=512
BLOCK_SIZE=512
TYPE=vfat
USAGE=filesystem
PART_ENTRY_TYPE=c12a7328-f81f-11d2-ba4b-00a0c93ec93b
""".splitlines()

_FSCK_FAT_OUTPUT = """fsck.fat 4.2 (2021-01-31)
Checking we can access the last sector of the filesystem
Boot sector contents:
System ID "mkfs.fat"
Media byte 0xf8 (hard disk)
       512 bytes per logical sector
       512 bytes per cluster
        32 reserved sectors
First FAT starts at byte 16384 (sector 32)
         2 FATs, 32 bit entries
   2097152 bytes per FAT (= 4096 sectors)
""".splitlines()


def _fake_fat_runner(command, **kwargs):
	return _BLKID_P_FAT32_OUTPUT if command[0] == 'blkid' else _FSCK_FAT_OUTPUT


def test_probe_vfat_reads_fat_width_and_geometry(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check', _fake_fat_runner)
	params = hpcp._probe_vfat('/dev/fake1')
	assert params['fat_bits'] == 32
	assert params['sector_size'] == 512
	assert params['cluster_size'] == 512
	assert params['reserved_sectors'] == 32
	assert params['fat_count'] == 2


def test_build_vfat_forces_source_fat_width(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check', _fake_fat_runner)
	args = hpcp._build_vfat(hpcp._probe_vfat('/dev/fake1'))
	# Without -F 32 mkfs.vfat picks FAT16 for a partition this size.
	assert args[args.index('-F') + 1] == '32'
	assert args[args.index('-S') + 1] == '512'
	assert args[args.index('-s') + 1] == '1'
	assert args[args.index('-f') + 1] == '2'
	assert args[args.index('-R') + 1] == '32'


def test_build_vfat_computes_sectors_per_cluster():
	args = hpcp._build_vfat({'sector_size': 512, 'cluster_size': 8192})
	assert args[args.index('-s') + 1] == '16'


def test_build_vfat_skips_bad_fat_width():
	assert '-F' not in hpcp._build_vfat({'fat_bits': 64, 'sector_size': 512})


def test_vfat_registered_for_all_fat_aliases():
	for fs_type in ('vfat', 'fat', 'fat12', 'fat16', 'fat32', 'msdos'):
		assert hpcp._FS_PARAM_PROBES[fs_type] is hpcp._probe_vfat
		assert hpcp._FS_MKFS_BUILDERS[fs_type] is hpcp._build_vfat


def test_dead_fat_width_branches_removed():
	# The old -F 16 / -F 12 branches keyed on blkid TYPE could never fire and
	# would now conflict with the probed width.
	import inspect
	src = inspect.getsource(hpcp.write_partition_info)
	assert "'-F', '16'" not in src
	assert "'-F', '12'" not in src


_NTFSINFO_OUTPUT = """Volume Information
	Name of device: /dev/fake5
	Device state: 11
	Volume Name: NT
	Volume State: 1
	Volume Version: 3.1
	Sector Size: 512
	Cluster Size: 8192
	Index Block Size: 4096
	Volume Size in Clusters: 25599
""".splitlines()

_DUMP_EXFAT_OUTPUT = """exfatprogs version : 1.2.9
-------------- Dump Boot sector region --------------
Volume Length(sectors):                  409600
FAT Offset(sector offset):               2048
FAT Length(sectors):                     13
Cluster Heap Offset (sector offset):     4096
Cluster Count:                           1584
Root Cluster (cluster offset):           4
Volume Serial:                           0xebe5bdee
Bytes per Sector:                        512
Sectors per Cluster:                     256

---------------- Show the statistics ----------------
Cluster size:                            131072
""".splitlines()

_DUMP_F2FS_OUTPUT = """Info: Debug level = 1
Info: superblock features = 4 : extra_attr
Info: superblock encrypt level = 0, salt = 00000000000000000000000000000000
magic                         		[0xf2f52010 : 4076150800]
major_ver                     		[0x       1 : 1]
volum_name                    		[F2]
log_sectorsize                		[0x       9 : 9]
log_sectors_per_block         		[0x       3 : 3]
log_blocksize                 		[0x       c : 12]
log_blocks_per_seg            		[0x       9 : 9]
segs_per_sec                  		[0x       2 : 2]
secs_per_zone                 		[0x       1 : 1]
block_count                   		[0x   12c00 : 76800]
""".splitlines()


def test_probe_ntfs_reads_cluster_and_sector_size(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: _NTFSINFO_OUTPUT)
	params = hpcp._probe_ntfs('/dev/fake5')
	assert params['cluster_size'] == 8192
	assert params['sector_size'] == 512


def test_build_ntfs():
	args = hpcp._build_ntfs({'cluster_size': 8192, 'sector_size': 512})
	assert args == ['-c', '8192', '-s', '512']


def test_probe_exfat_reads_geometry(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: _DUMP_EXFAT_OUTPUT)
	params = hpcp._probe_exfat('/dev/fake6')
	assert params['sector_size'] == 512
	assert params['cluster_size'] == 131072


def test_build_exfat():
	args = hpcp._build_exfat({'sector_size': 512, 'cluster_size': 131072})
	assert args == ['-s', '512', '-c', '131072']


def test_probe_f2fs_reads_geometry_and_features(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: _DUMP_F2FS_OUTPUT)
	params = hpcp._probe_f2fs('/dev/fake7')
	assert params['log_sectorsize'] == 9
	assert params['segs_per_sec'] == 2
	assert params['secs_per_zone'] == 1
	assert params['features'] == ['extra_attr']


def test_build_f2fs_converts_log_sector_size(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: _DUMP_F2FS_OUTPUT)
	args = hpcp._build_f2fs(hpcp._probe_f2fs('/dev/fake7'))
	assert args[args.index('-w') + 1] == '512'
	assert args[args.index('-s') + 1] == '2'
	assert args[args.index('-z') + 1] == '1'
	assert args[args.index('-O') + 1] == 'extra_attr'


def test_probe_f2fs_handles_no_features(monkeypatch):
	no_features = ['Info: superblock features = 0 : ', 'log_sectorsize                \t\t[0x       9 : 9]']
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: no_features)
	params = hpcp._probe_f2fs('/dev/fake7')
	assert params['features'] == []
	assert '-O' not in hpcp._build_f2fs(params)


def test_build_f2fs_translates_quota_ino_to_quota():
	args = hpcp._build_f2fs({'features': ['quota_ino']})
	assert args[args.index('-O') + 1] == 'quota'


def test_build_f2fs_drops_unrecognised_feature_but_keeps_geometry(capsys):
	params = {
		'log_sectorsize': 9,
		'segs_per_sec': 2,
		'secs_per_zone': 1,
		'features': ['extra_attr', 'some_future_feature'],
	}
	args = hpcp._build_f2fs(params)
	assert args[args.index('-w') + 1] == '512'
	assert args[args.index('-s') + 1] == '2'
	assert args[args.index('-z') + 1] == '1'
	assert args[args.index('-O') + 1] == 'extra_attr'
	assert 'some_future_feature' not in args
	assert 'some_future_feature' in capsys.readouterr().err


def test_build_f2fs_omits_dash_o_when_all_features_dropped():
	args = hpcp._build_f2fs({'features': ['some_future_feature']})
	assert '-O' not in args


def test_ntfs_exfat_f2fs_registered():
	for fs_type, probe, builder in (('ntfs', hpcp._probe_ntfs, hpcp._build_ntfs),
									('exfat', hpcp._probe_exfat, hpcp._build_exfat),
									('f2fs', hpcp._probe_f2fs, hpcp._build_f2fs)):
		assert hpcp._FS_PARAM_PROBES[fs_type] is probe
		assert hpcp._FS_MKFS_BUILDERS[fs_type] is builder


_UDFINFO_OUTPUT = """filename=/dev/fake8
label=UD
uuid=6aa09dfa158beac7
lvid=UD
vid=UD
blocksize=512
udfrev=2.01
integrity=closed
blocks=409600
usedblocks=103
""".splitlines()

_DEBUGREISERFS_OUTPUT = """debugreiserfs 3.6.27
Filesystem state: consistent
Reiserfs super block in block 16 on 0x0 of format 3.6 with standard journal
Count of blocks on the device: 76800
Blocksize: 4096
Hash function used to sort names: "r5"
sb_version: 2
""".splitlines()


def test_probe_udf_reads_blocksize_and_revision(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: _UDFINFO_OUTPUT)
	params = hpcp._probe_udf('/dev/fake8')
	assert params['block_size'] == 512
	assert params['udfrev'] == '2.01'


def test_build_udf():
	args = hpcp._build_udf({'block_size': 512, 'udfrev': '2.01'})
	assert args == ['--blocksize=512', '--udfrev=2.01']


def test_probe_reiserfs_reads_blocksize_format_and_hash(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: _DEBUGREISERFS_OUTPUT)
	params = hpcp._probe_reiserfs('/dev/fake9')
	assert params['block_size'] == 4096
	assert params['fs_format'] == '3.6'
	assert params['hash_function'] == 'r5'


def test_build_reiserfs():
	args = hpcp._build_reiserfs({'block_size': 4096, 'fs_format': '3.6', 'hash_function': 'r5'})
	assert args == ['-b', '4096', '--format', '3.6', '-h', 'r5']


def test_udf_reiserfs_registered():
	assert hpcp._FS_PARAM_PROBES['udf'] is hpcp._probe_udf
	assert hpcp._FS_MKFS_BUILDERS['udf'] is hpcp._build_udf
	assert hpcp._FS_PARAM_PROBES['reiserfs'] is hpcp._probe_reiserfs
	assert hpcp._FS_MKFS_BUILDERS['reiserfs'] is hpcp._build_reiserfs


def _write_superblock(path, offset, payload):
	with open(path, 'wb') as f:
		f.write(b'\0' * offset)
		f.write(payload)
		f.write(b'\0' * 512)


def test_probe_hfsplus_reads_block_size():
	# HFS+ volume header lives at byte 1024: 'H+' signature, blockSize at +40 big-endian.
	header = bytearray(64)
	struct.pack_into('>H', header, 0, 0x482B)
	struct.pack_into('>H', header, 2, 4)
	struct.pack_into('>I', header, 40, 8192)
	with tempfile.NamedTemporaryFile(suffix='.img', delete=False) as tmp:
		path = tmp.name
	try:
		_write_superblock(path, 1024, bytes(header))
		assert hpcp._probe_hfsplus(path) == {'block_size': 8192}
	finally:
		os.unlink(path)


def test_probe_hfsplus_rejects_foreign_signature():
	with tempfile.NamedTemporaryFile(suffix='.img', delete=False) as tmp:
		path = tmp.name
	try:
		_write_superblock(path, 1024, b'\0' * 64)
		assert hpcp._probe_hfsplus(path) == {}
	finally:
		os.unlink(path)


def test_build_hfsplus():
	assert hpcp._build_hfsplus({'block_size': 8192}) == ['-b', '8192']


def test_probe_minix_detects_version_and_name_length():
	# v1/v2 magic sits at 1024+16; v3 magic at 1024+24.
	cases = ((0x137F, 16, 1, 14), (0x138F, 16, 1, 30), (0x2468, 16, 2, 14),
			 (0x2478, 16, 2, 30), (0x4D5A, 24, 3, 60))
	for magic, offset, expected_version, expected_namelen in cases:
		sb = bytearray(64)
		struct.pack_into('<H', sb, offset, magic)
		with tempfile.NamedTemporaryFile(suffix='.img', delete=False) as tmp:
			path = tmp.name
		try:
			_write_superblock(path, 1024, bytes(sb))
			params = hpcp._probe_minix(path)
			assert params['fs_version'] == expected_version
			assert params['name_length'] == expected_namelen
		finally:
			os.unlink(path)


def test_build_minix_versions():
	assert hpcp._build_minix({'fs_version': 3, 'name_length': 60}) == ['-3']
	assert hpcp._build_minix({'fs_version': 2, 'name_length': 30}) == ['-2', '-n', '30']
	assert hpcp._build_minix({'fs_version': 1, 'name_length': 14}) == ['-1', '-n', '14']


def test_hfsplus_minix_registered():
	assert hpcp._FS_PARAM_PROBES['hfsplus'] is hpcp._probe_hfsplus
	assert hpcp._FS_PARAM_PROBES['hfs'] is hpcp._probe_hfsplus
	assert hpcp._FS_MKFS_BUILDERS['hfsplus'] is hpcp._build_hfsplus
	assert hpcp._FS_PARAM_PROBES['minix'] is hpcp._probe_minix
	assert hpcp._FS_MKFS_BUILDERS['minix'] is hpcp._build_minix


_ROUNDTRIP_TOOLS = ('losetup', 'sgdisk', 'mkfs.vfat', 'mkfs.ext4', 'mkfs.xfs',
					'blkid', 'dumpe2fs', 'xfs_info', 'fsck.fat', 'truncate',
					'mount', 'umount', 'udevadm')

requires_root_and_tools = pytest.mark.skipif(
	os.geteuid() != 0 or any(shutil.which(t) is None for t in _ROUNDTRIP_TOOLS),
	reason='dd round-trip test needs root and losetup/sgdisk/mkfs tools')


def _run(*command):
	return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def _partition_params(image, index, fs_type):
	loop = _run('losetup', '--partscan', '--find', '--show', '--read-only', image).strip()
	try:
		subprocess.run(['udevadm', 'settle'], check=False, capture_output=True)
		return hpcp.probe_fs_params(f'{loop}p{index}', fs_type)
	finally:
		subprocess.run(['losetup', '-d', loop], check=False, capture_output=True)


def _partition_uuid(image, index):
	loop = _run('losetup', '--partscan', '--find', '--show', '--read-only', image).strip()
	try:
		subprocess.run(['udevadm', 'settle'], check=False, capture_output=True)
		return _run('blkid', '-s', 'UUID', '-o', 'value', f'{loop}p{index}').strip()
	finally:
		subprocess.run(['losetup', '-d', loop], check=False, capture_output=True)


def _detach_loops_for_image(image):
	"""Detach every loop device still backed by `image`.

	hpcp.py used to leave its own read-only *source* loop attached after it
	returned; that leak is fixed now (see
	test_dd_does_not_leak_the_source_loop_device), but this stays as a safety
	net so a test that fails partway through still cannot accumulate attached
	loop devices on the host. Scoped strictly to this image's own backing file
	(via `losetup -j`), so unrelated loop devices - e.g. this host's
	snap-package loops - are never touched.
	"""
	try:
		output = _run('losetup', '-j', image)
	except subprocess.CalledProcessError:
		return
	for line in output.splitlines():
		device = line.partition(':')[0].strip()
		if device:
			subprocess.run(['losetup', '-d', device], check=False, capture_output=True)


def _run_hpcp_dd(extra_args, src, dest, timeout=900):
	"""Run `hpcp.py -dd [extra_args] src dest` and fail loudly if it crashes.

	A successful -dd run exits 0: hpcp's dd path deliberately ends by raising
	RuntimeError("Exiting after dd mode.") internally, but hpcp() catches
	that itself, eprints a message with no colon in it (so it is not counted
	as an error), and returns get_rc_from_error() == 0 - confirmed empirically
	by running a successful -dd here and inspecting its actual exit code
	before writing this assertion. A crash leaves a returncode that get_rc_from_error()
	maps away from 0, so asserting on it directly surfaces hpcp's own
	stdout/stderr in the pytest failure instead of a bare, opaque KeyError
	from a later probe of a destination that never got a filesystem.
	"""
	hpcp_py = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hpcp.py')
	open(dest, 'wb').close()
	result = subprocess.run([sys.executable, hpcp_py, '-dd', *extra_args, src, dest],
							 stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
	assert result.returncode == 0, (
		f"hpcp.py -dd {' '.join(extra_args)} {src} {dest} exited {result.returncode}\n"
		f"--- hpcp stdout ---\n{result.stdout}\n--- hpcp stderr ---\n{result.stderr}"
	)
	return result


def _build_three_partition_source(tmp_path):
	"""Build the shared 1400 MiB / three-partition source image.

	A 260 MiB FAT32 ESP (a size at which mkfs.vfat would otherwise pick
	FAT16), a 512 MiB ext4 built the way an older distro would, and an xfs
	with non-default inode and directory geometry. Used by both the positive
	round-trip test and the -nfp negative control so their source layouts
	can never drift apart. Single, whole-disk partition layouts are covered
	separately by test_dd_clones_a_single_partition_spanning_the_whole_disk.
	"""
	src = str(tmp_path / 'src.img')
	_run('truncate', '-s', '1400M', src)
	_run('sgdisk', '--clear',
		 '--new=1:0:+260M', '--typecode=1:ef00', '--change-name=1:EFI System',
		 '--new=2:0:+512M', '--typecode=2:8300', '--change-name=2:boot',
		 '--new=3:0:0', '--typecode=3:8300', '--change-name=3:root', src)

	loop = _run('losetup', '--partscan', '--find', '--show', src).strip()
	try:
		subprocess.run(['udevadm', 'settle'], check=False, capture_output=True)
		# Deliberately non-default: a FAT32 ESP small enough that mkfs.vfat would
		# otherwise pick FAT16, an ext4 built the way an older distro would, and
		# an xfs with non-default inode and directory geometry.
		_run('mkfs.vfat', '-F', '32', '-n', 'ESP', f'{loop}p1')
		_run('mkfs.ext4', '-q', '-F', '-b', '1024', '-I', '128',
			 '-O', '^metadata_csum,^64bit,^dir_index', '-m', '0', '-L', 'BOOTFS', f'{loop}p2')
		_run('mkfs.xfs', '-q', '-f', '-i', 'size=1024', '-n', 'size=8192',
			 '-m', 'reflink=0', '-L', 'ROOTFS', f'{loop}p3')
		for index in (1, 2, 3):
			mount_point = str(tmp_path / f'mnt{index}')
			os.makedirs(mount_point, exist_ok=True)
			_run('mount', f'{loop}p{index}', mount_point)
			try:
				os.makedirs(os.path.join(mount_point, 'dir'), exist_ok=True)
				with open(os.path.join(mount_point, 'dir', f'file{index}.txt'), 'w') as f:
					f.write(f'hello-{index}\n')
			finally:
				_run('umount', mount_point)
	finally:
		subprocess.run(['losetup', '-d', loop], check=False, capture_output=True)
	return src


@requires_root_and_tools
def test_dd_roundtrip_preserves_source_fs_params(tmp_path):
	src = _build_three_partition_source(tmp_path)
	dest = str(tmp_path / 'dest.img')

	try:
		src_fat = _partition_params(src, 1, 'vfat')
		src_ext = _partition_params(src, 2, 'ext4')
		src_xfs = _partition_params(src, 3, 'xfs')

		_run_hpcp_dd([], src, dest)

		dest_fat = _partition_params(dest, 1, 'vfat')
		dest_ext = _partition_params(dest, 2, 'ext4')
		dest_xfs = _partition_params(dest, 3, 'xfs')

		# The ESP must stay FAT32; plain mkfs.vfat picks FAT16 at this size.
		assert dest_fat['fat_bits'] == src_fat['fat_bits'] == 32
		assert dest_fat['cluster_size'] == src_fat['cluster_size']

		assert dest_ext['block_size'] == src_ext['block_size'] == 1024
		assert dest_ext['inode_size'] == src_ext['inode_size'] == 128
		assert dest_ext['reserved_block_count'] == 0
		assert sorted(dest_ext['features']) == sorted(src_ext['features'])
		for absent in ('64bit', 'metadata_csum', 'dir_index'):
			assert absent not in dest_ext['features']

		assert dest_xfs['meta-data']['isize'] == src_xfs['meta-data']['isize'] == '1024'
		assert dest_xfs['naming']['bsize'] == src_xfs['naming']['bsize'] == '8192'
		assert dest_xfs['meta-data']['reflink'] == src_xfs['meta-data']['reflink'] == '0'
	finally:
		# hpcp -dd leaves its own read-only source loop attached; see
		# _detach_loops_for_image's docstring.
		_detach_loops_for_image(src)


@requires_root_and_tools
def test_dd_roundtrip_uses_defaults_with_no_fs_param_mirror(tmp_path):
	# Reuses the same three-partition layout as the positive test, so the two
	# differ only in the -nfp flag and nothing else can explain a difference
	# in the result.
	src = _build_three_partition_source(tmp_path)
	dest = str(tmp_path / 'dest.img')

	try:
		src_ext = _partition_params(src, 2, 'ext4')
		assert src_ext['block_size'] == 1024

		_run_hpcp_dd(['-nfp'], src, dest)

		dest_ext = _partition_params(dest, 2, 'ext4')
		# -nfp restores today's behaviour: mkfs defaults, not the source's 1 KiB blocks.
		assert dest_ext['block_size'] == 4096
	finally:
		# hpcp -dd leaves its own read-only source loop attached; see
		# _detach_loops_for_image's docstring.
		_detach_loops_for_image(src)


def test_version_bumped():
	assert hpcp.version == '9.60'
	assert hpcp.__version__ == hpcp.version
	assert hpcp.COMMIT_DATE == '2026-09-10'


#%% -- Pre-existing -dd defects found while building the mirroring feature --
# These three bugs predate filesystem parameter mirroring: they reproduce on
# the commit this branch forked from, with none of the mirroring code present.


def test_validate_dd_source_path_registers_loop_in_empty_caller_list(monkeypatch, tmp_path):
	image = tmp_path / 'src.img'
	image.write_bytes(b'\0' * 1024)
	monkeypatch.setattr(hpcp, 'create_loop_device', lambda path, read_only=False: '/dev/fakeloop9')

	caller_loops = []
	dd_src = hpcp.validate_dd_source_path([str(image)], loop_devices=caller_loops)

	assert dd_src == '/dev/fakeloop9'
	# An empty list is falsy, so `if not loop_devices:` used to rebind the
	# parameter to a fresh local list. The caller's list never learned about the
	# source loop, so clean_up() could not detach it and every -dd run leaked one.
	assert caller_loops == ['/dev/fakeloop9']


def test_validate_dd_source_path_appends_to_populated_caller_list(monkeypatch, tmp_path):
	image = tmp_path / 'src.img'
	image.write_bytes(b'\0' * 1024)
	monkeypatch.setattr(hpcp, 'create_loop_device', lambda path, read_only=False: '/dev/fakeloop9')

	caller_loops = ['/dev/preexisting0']
	hpcp.validate_dd_source_path([str(image)], loop_devices=caller_loops)

	# A populated list was always truthy, which is why only the first loop leaked.
	assert caller_loops == ['/dev/preexisting0', '/dev/fakeloop9']


def test_write_partition_info_returns_list_when_partition_missing(monkeypatch):
	monkeypatch.setattr(hpcp, 'get_target_partition', lambda image, name: ('', None))
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check', lambda command, **kwargs: [''])

	infos = {'1': {'partition_guid_code': '', 'unique_partition_guid': '', 'partition_name': '',
				   'partition_attrs': '', 'fs_type': 'ext4', 'fs_uuid': '', 'fs_label': '',
				   'size': 0, 'fs_params': {}}}
	result = hpcp.write_partition_info('/dev/fakeimg', infos, '1')

	# Every caller does `delayed_commands.extend(write_partition_info(...))`, so
	# a bare `return` here turned a handled "partition not found" into
	# TypeError: 'NoneType' object is not iterable.
	assert result == []


def _build_single_partition_source(tmp_path):
	"""Build a 700 MiB image whose single partition spans the whole disk.

	This is the layout that used to be impossible to clone: with one partition
	the destination was sized at partition + 1 MiB, which the 1 MiB start
	alignment consumed entirely, leaving nothing for GPT's backup header.
	"""
	src = str(tmp_path / 'src.img')
	_run('truncate', '-s', '700M', src)
	_run('sgdisk', '--clear', '--new=1:0:0', '--typecode=1:8300', '--change-name=1:root', src)

	loop = _run('losetup', '--partscan', '--find', '--show', src).strip()
	try:
		subprocess.run(['udevadm', 'settle'], check=False, capture_output=True)
		# metadata_csum without metadata_csum_seed is the layout that makes
		# delayed tune2fs -U refuse unless e2fsck -f ran after the copy.
		_run('mkfs.ext4', '-q', '-F', '-b', '1024', '-I', '128',
			 '-O', '^metadata_csum_seed', '-L', 'ROOTFS', f'{loop}p1')
		mount_point = str(tmp_path / 'mnt1')
		os.makedirs(mount_point, exist_ok=True)
		_run('mount', f'{loop}p1', mount_point)
		try:
			os.makedirs(os.path.join(mount_point, 'dir'), exist_ok=True)
			with open(os.path.join(mount_point, 'dir', 'file1.txt'), 'w') as f:
				f.write('hello-single\n')
		finally:
			_run('umount', mount_point)
	finally:
		subprocess.run(['losetup', '-d', loop], check=False, capture_output=True)
	return src


@requires_root_and_tools
def test_dd_clones_a_single_partition_spanning_the_whole_disk(tmp_path):
	src = _build_single_partition_source(tmp_path)
	dest = str(tmp_path / 'dest.img')

	try:
		src_ext = _partition_params(src, 1, 'ext4')
		src_uuid = _partition_uuid(src, 1)
		assert src_ext['block_size'] == 1024
		assert src_uuid

		_run_hpcp_dd([], src, dest)

		dest_ext = _partition_params(dest, 1, 'ext4')
		assert dest_ext['block_size'] == src_ext['block_size'] == 1024
		assert dest_ext['inode_size'] == src_ext['inode_size'] == 128
		assert _partition_uuid(dest, 1) == src_uuid
	finally:
		_detach_loops_for_image(src)
		_detach_loops_for_image(dest)


@requires_root_and_tools
def test_dd_honors_dis_as_dest_image_size(tmp_path):
	# -dd used to print "Currently not supporting dest_image_size" and ignore
	# -dis. For an image-file destination, -dis should pad the clone to the
	# requested size so the result can be written onto a larger disk.
	src = _build_single_partition_source(tmp_path)
	dest = str(tmp_path / 'dest.img')
	requested = 900 * 1024 * 1024
	try:
		result = _run_hpcp_dd(['-dis', '900MiB'], src, dest)
		assert os.path.getsize(dest) == requested
		# The clone must still be a usable ext4 volume at the original UUID.
		assert _partition_uuid(dest, 1)
		assert 'dest_image_size in dd mode' not in (result.stderr or '')
	finally:
		_detach_loops_for_image(src)
		_detach_loops_for_image(dest)


@requires_root_and_tools
def test_dd_does_not_leak_the_source_loop_device(tmp_path):
	src = _build_single_partition_source(tmp_path)
	dest = str(tmp_path / 'dest.img')

	try:
		_run_hpcp_dd([], src, dest)
		# hpcp must detach its own read-only source loop before returning.
		assert subprocess.run(['losetup', '-j', src], capture_output=True, text=True).stdout.strip() == ''
		assert subprocess.run(['losetup', '-j', dest], capture_output=True, text=True).stdout.strip() == ''
	finally:
		_detach_loops_for_image(src)
		_detach_loops_for_image(dest)
