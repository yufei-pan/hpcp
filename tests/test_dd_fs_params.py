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
