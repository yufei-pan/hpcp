"""The tools whose output hpcp parses run in the C locale (LC_ALL=C through multiCMD's extra_env, for that command
alone); every other command keeps the user's."""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hpcp


def _fake_tool(directory, name, body):
	path = os.path.join(str(directory), name)
	with open(path, 'w', encoding='utf-8') as f:
		f.write('#!/bin/sh\n' + body)
	os.chmod(path, 0o755)
	return path


def _install(monkeypatch, directory, name, body):
	"""A fake `name` that hpcp finds both ways: first on PATH and as its resolved path."""
	path = _fake_tool(directory, name, body)
	monkeypatch.setenv('PATH', str(directory) + os.pathsep + os.environ.get('PATH', ''))
	monkeypatch.setitem(hpcp._binPaths, name, path)
	return path


_XFS_ENGLISH = """meta-data=/dev/loop8p3           isize=1024   agcount=4, agsize=40127 blks
         =                       sectsz=512   attr=2, projid32bit=1
         =                       crc=1        finobt=1, sparse=1, rmapbt=1
data     =                       bsize=4096   blocks=160507, imaxpct=25
naming   =version 2              bsize=8192   ascii-ci=0, ftype=1, parent=0
log      =internal log           bsize=4096   blocks=16384, version=2
realtime =none                   extsz=4096   blocks=0, rtextents=0
"""

_DUMPE2FS_ENGLISH = """dumpe2fs 1.47.2 (1-Jan-2025)
Filesystem volume name:   BOOTFS
Filesystem features:      has_journal ext_attr resize_inode filetype extent flex_bg sparse_super
Inode count:              32768
Block count:              524288
Reserved block count:     0
Block size:               1024
Inode size:               128
"""


@pytest.mark.skipif(os.name != 'posix', reason='runs shell scripts')
def test_xfs_info_geometry_is_read_when_hpcp_runs_in_a_translated_locale(tmp_path, exec_dir, monkeypatch):
	# xfsprogs translates the section names (Debian's pl: metadane, dane, nazwy); in them no value would land
	# where _build_xfs looks, and isize, sectsz, bsize and ftype would silently not be mirrored.
	english = tmp_path / 'en.txt'
	english.write_text(_XFS_ENGLISH)
	polish = tmp_path / 'pl.txt'
	polish.write_text(_XFS_ENGLISH.replace('meta-data=', 'metadane= ').replace('data     =', 'dane     =').replace('naming   =', 'nazwy    ='))
	_install(monkeypatch, exec_dir, 'xfs_info', 'if [ "$LC_ALL" = C ]; then cat "%s"; else cat "%s"; fi\n' % (english, polish))
	monkeypatch.setenv('LC_ALL', 'pl_PL.UTF-8')
	monkeypatch.setenv('LANG', 'pl_PL.UTF-8')
	params = hpcp._probe_xfs('/dev/fake3')
	assert params['meta-data']['isize'] == '1024'
	assert params['data']['bsize'] == '4096'
	assert params['naming']['bsize'] == '8192'
	assert params['naming']['ftype'] == '1'
	# hpcp's own environment is not touched: only that command ran in the C locale
	assert os.environ['LC_ALL'] == 'pl_PL.UTF-8'


@pytest.mark.skipif(os.name != 'posix', reason='runs shell scripts')
def test_dumpe2fs_fields_are_read_when_hpcp_runs_in_a_translated_locale(tmp_path, exec_dir, monkeypatch):
	# e2fsprogs translates the field names ("Block size:" is "Blockgroesse:" in German)
	english = tmp_path / 'en.txt'
	english.write_text(_DUMPE2FS_ENGLISH)
	german = tmp_path / 'de.txt'
	german.write_text(_DUMPE2FS_ENGLISH.replace('Block size', 'Blockgroesse').replace('Inode size', 'Inode-Groesse').replace(
		'Filesystem features', 'Dateisystemfunktionen').replace('Inode count', 'Inode-Anzahl').replace('Block count', 'Blockanzahl'))
	_install(monkeypatch, exec_dir, 'dumpe2fs', 'if [ "$LC_ALL" = C ]; then cat "%s"; else cat "%s"; fi\n' % (english, german))
	monkeypatch.setenv('LC_ALL', 'de_DE.UTF-8')
	params = hpcp._probe_ext('/dev/fake2')
	assert params['block_size'] == 1024
	assert params['inode_size'] == 128
	assert params['inode_count'] == 32768
	assert 'has_journal' in params['features']


@pytest.mark.skipif(os.name != 'posix', reason='runs a shell script')
def test_a_command_without_extra_env_keeps_the_users_locale(exec_dir, monkeypatch):
	# what is shown to the user (mount, cp, fsck output) or not read stays in their language
	_install(monkeypatch, exec_dir, 'hpcptesttool', 'echo "LC_ALL=$LC_ALL"\n')
	monkeypatch.setenv('LC_ALL', 'pl_PL.UTF-8')
	assert hpcp.run_command_in_multicmd_with_path_check(['hpcptesttool'], quiet=True) == ['LC_ALL=pl_PL.UTF-8']
	assert hpcp.run_command_in_multicmd_with_path_check(['hpcptesttool'], quiet=True, extra_env=hpcp._C_LOCALE) == ['LC_ALL=C']
	assert os.environ['LC_ALL'] == 'pl_PL.UTF-8'
	assert hpcp._C_LOCALE == {'LC_ALL': 'C'}


def test_the_runner_hands_extra_env_to_multicmd_and_by_default_none(monkeypatch):
	seen = []

	def fake_run(commands, **kwargs):
		seen.append(kwargs.get('extra_env', 'not passed'))
		return [_Task(command) for command in commands]

	monkeypatch.setattr(hpcp, '_binPaths', {'true': '/bin/true'})
	monkeypatch.setattr(hpcp.multiCMD, 'run_commands', fake_run)
	hpcp.run_command_in_multicmd_with_path_check(['true'], quiet=True)
	hpcp.run_commands_in_multicmd_with_path_check([['true']], quiet=True, extra_env={'A': 'b'})
	assert seen == [None, {'A': 'b'}]


class _Task:
	def __init__(self, command, returncode=0):
		self.command = command
		self.returncode = returncode
		self.stdout = []
		self.stderr = []


def test_the_mke2fs_probe_sets_its_variables_through_extra_env_without_env(monkeypatch):
	# `env LC_ALL=C MKE2FS_CONFIG=... mkfs` made env the program that was found, started and judged
	seen = []

	def fake_run(commands, **kwargs):
		seen.append((commands, kwargs.get('extra_env')))
		return [_Task(command) for command in commands]

	monkeypatch.setitem(hpcp._binPaths, 'mkfs', '/usr/sbin/mkfs')
	monkeypatch.setattr(hpcp.multiCMD, 'run_commands', fake_run)
	assert hpcp._ext_mkfs_probe_run(['has_journal', 'extent']) == [True, True]
	commands, extra_env = seen[0]
	assert [command[0] for command in commands] == ['/usr/sbin/mkfs'] * 2
	assert all('env' not in command and not any('=' in word and word.startswith(('LC_ALL', 'MKE2FS')) for word in command) for command in commands)
	assert extra_env == {'LC_ALL': 'C', 'MKE2FS_CONFIG': os.devnull}


def test_cp_version_is_read_in_the_c_locale(monkeypatch):
	# 'GNU' is matched in `cp --version`
	seen = []

	def fake_run(commands, **kwargs):
		seen.append(kwargs.get('extra_env'))
		task = _Task(commands[0])
		task.stdout = ['cp (GNU coreutils) 9.4']
		return [task]

	hpcp.cp_supports_sparse.cache_clear()
	monkeypatch.setattr(hpcp.multiCMD, 'run_commands', fake_run)
	try:
		assert hpcp.cp_supports_sparse() is True
	finally:
		hpcp.cp_supports_sparse.cache_clear()
	assert seen == [{'LC_ALL': 'C'}]


def _record(monkeypatch):
	"""Replace the runner with one that records (tool, extra_env) of each call and prints nothing."""
	calls = []

	def recorder(command, **kwargs):
		calls.append((str(command[0]), kwargs.get('extra_env')))
		return []

	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check', recorder)
	monkeypatch.setattr(hpcp, 'run_commands_in_multicmd_with_path_check',
		lambda commands, **kwargs: [recorder(command, **kwargs) for command in commands])
	return calls


def _quietly(function, *args):
	try:
		function(*args)
	except Exception:
		pass  # the empty answers the recorder gives may not parse; only the calls matter


_PARSED = [
	# (what hpcp runs, the tools it must start in the C locale)
	(lambda: hpcp.get_fs_type_from_path('/'), {'df'}),
	(lambda: hpcp.get_fs_type('/dev/x'), {'lsblk'}),
	(lambda: hpcp.get_partitions('/dev/x'), {'lsblk'}),
	(lambda: hpcp.get_largest_partition('/dev/x'), {'lsblk'}),
	(lambda: hpcp.get_partition_details('/dev/x', '1', identity=False), {'sgdisk'}),
	(lambda: hpcp.get_partition_infos('/dev/x', identity=False), {'sgdisk'}),
	(lambda: hpcp._probe_ext('/dev/x'), {'dumpe2fs'}),
	(lambda: hpcp._probe_xfs('/dev/x'), {'xfs_info'}),
	(lambda: hpcp._probe_btrfs('/dev/x'), {'btrfs'}),
	(lambda: hpcp._probe_vfat('/dev/x'), {'blkid', 'fsck.fat'}),
	(lambda: hpcp._probe_ntfs('/dev/x'), {'ntfsinfo'}),
	(lambda: hpcp._probe_exfat('/dev/x'), {'dump.exfat'}),
	(lambda: hpcp._probe_f2fs('/dev/x'), {'dump.f2fs'}),
	(lambda: hpcp._probe_udf('/dev/x'), {'udfinfo'}),
	(lambda: hpcp._probe_reiserfs('/dev/x'), {'debugreiserfs'}),
	(lambda: hpcp._probe_fs_type('/dev/x'), {'blkid'}),
	(lambda: hpcp._probe_identity('/dev/x'), {'blkid'}),
	(lambda: hpcp._probe_raw_label('/dev/x'), {'blkid'}),
	(lambda: hpcp._partition_table_type('/dev/x'), {'blkid'}),
	(lambda: hpcp._refuse_if_image_in_use('/tmp/x.img'), {'losetup'}),
]


@pytest.mark.parametrize('run, tools', _PARSED, ids=[sorted(tools)[0] + str(i) for i, (_, tools) in enumerate(_PARSED)])
def test_every_parsed_tool_runs_in_the_c_locale(monkeypatch, run, tools):
	calls = _record(monkeypatch)
	_quietly(run)
	assert {tool for tool, _ in calls} >= tools
	assert all(extra_env == {'LC_ALL': 'C'} for tool, extra_env in calls if tool in tools), calls


def test_sgdisk_first_aligned_is_read_in_the_c_locale(monkeypatch):
	calls = []

	def recorder(command, **kwargs):
		calls.append((command[1], kwargs.get('extra_env')))
		return ['2048'] if command[1] == '--first-aligned-in-largest' else []

	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check', recorder)
	monkeypatch.setattr(hpcp, 'create_loop_device', lambda image: '/dev/loop99')
	monkeypatch.setattr(hpcp, 'write_partition_info', lambda *args: [])
	infos = {'/dev/src': {'disk_identifier': 'x', 'sector_size': 512}, 'p1': {'size': 1048576}}
	hpcp.create_partition_table('/tmp/hpcp-test.img', infos, ['p1', '/dev/src'])
	assert ('--first-aligned-in-largest', {'LC_ALL': 'C'}) in calls


def test_what_is_shown_to_the_user_keeps_their_locale(monkeypatch):
	# losetup --show prints a device path and is shown; e2fsck's, umount's and mount's output is shown, and read
	# only for the exit status
	calls = _record(monkeypatch)
	_quietly(hpcp.create_loop_device, '/tmp/x.img')
	hpcp.fix_fs('/dev/x', 'ext4')
	_quietly(hpcp.clean_up, [], [])
	hpcp.run_command_in_multicmd_with_path_check(['umount', '/mnt/hpcp-test'])
	assert {tool for tool, _ in calls} >= {'losetup', 'e2fsck', 'umount'}
	assert all(extra_env is None for _, extra_env in calls)
