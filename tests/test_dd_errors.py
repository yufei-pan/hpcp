"""Exercise recursive DD error handling without attaching real block devices."""
import os
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('max_workers', [1, 2])
@pytest.mark.parametrize('fail_copy', [True, False])
def test_dd_propagates_partition_copy_status(
	tmp_path, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch,
	max_workers, fail_copy,
):
	# The copy itself and cp failures are real. Only disk setup/mounting is simulated.
	roots = []
	cleaned = []
	real_run = hpcp_mod.run_command_in_multicmd_with_path_check

	def mount_dir():
		path = tmp_path / ('mount' + str(len(roots)))
		path.mkdir()
		roots.append(path)
		if len(roots) % 2 == 1:
			(path / 'sub').mkdir()
			(path / 'sub' / 'data.txt').write_text('partition data')
		return str(path)

	def run(command, **kwargs):
		if command[0] == 'mount':
			return []
		return real_run(command, **kwargs)

	monkeypatch.setattr(hpcp_mod, 'validate_dd_source_path', lambda *a, **k: '/dev/fake-source')
	monkeypatch.setattr(hpcp_mod, 'create_dd_dest_part_table', lambda *a, **k: (
		{'/dev/fake-source': {}, '1': {'size': 4096}, '2': {'size': 4096}}, [], None,
	))
	monkeypatch.setattr(hpcp_mod, 'get_partitions', lambda device: [device + '1', device + '2'])
	monkeypatch.setattr(hpcp_mod.os, 'geteuid', lambda: 0)
	monkeypatch.setattr(hpcp_mod, 'tempfile', SimpleNamespace(mkdtemp=mount_dir))
	monkeypatch.setattr(hpcp_mod.os.path, 'ismount', lambda path: any(str(p) == str(path).rstrip(os.sep) for p in roots))
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', run)
	monkeypatch.setattr(hpcp_mod, 'clean_up', lambda mounts, *a: cleaned.append(list(mounts)))
	rc = hpcp_mod.hpcp(
		['/dev/fake-source'], dest_image=str(tmp_path / 'dest.img'), dd=True,
		single_thread=max_workers == 1, max_workers=max_workers,
		no_directory_sync=True, no_create_dir=fail_copy,
	)
	assert cleaned[-1] == [str(p) for p in roots]
	if fail_copy:
		assert rc != 0
		assert len(roots) == 2  # Stop before copying the second partition.
		assert not (roots[1] / 'sub' / 'data.txt').exists()
	else:
		assert rc == 0
		assert len(roots) == 4
		assert (roots[1] / 'sub' / 'data.txt').read_text() == 'partition data'
		assert (roots[3] / 'sub' / 'data.txt').read_text() == 'partition data'


def test_dd_partition_copy_status_ignores_errors_recorded_before_it(
	tmp_path, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch,
):
	"""Review C6: an error the outer -dd setup recorded (an NTFS UUID that cannot be set) must not make the first
	partition's copy look failed; it stays in the run's result."""
	roots = []
	real_run = hpcp_mod.run_command_in_multicmd_with_path_check

	def mount_dir():
		path = tmp_path / ('mount' + str(len(roots)))
		path.mkdir()
		roots.append(path)
		if len(roots) % 2 == 1:
			(path / 'data.txt').write_text('partition data')
		return str(path)

	def run(command, **kwargs):
		if command[0] == 'mount':
			return []
		return real_run(command, **kwargs)

	def part_table(*args, **kwargs):
		hpcp_mod.eprint('Cannot set fs uuid for ntfs: Skipping.')
		return {'/dev/fake-source': {}, '1': {'size': 4096}, '2': {'size': 4096}}, [], None

	monkeypatch.setattr(hpcp_mod, 'validate_dd_source_path', lambda *a, **k: '/dev/fake-source')
	monkeypatch.setattr(hpcp_mod, 'create_dd_dest_part_table', part_table)
	monkeypatch.setattr(hpcp_mod, 'get_partitions', lambda device: [device + '1', device + '2'])
	monkeypatch.setattr(hpcp_mod.os, 'geteuid', lambda: 0)
	monkeypatch.setattr(hpcp_mod, 'tempfile', SimpleNamespace(mkdtemp=mount_dir))
	monkeypatch.setattr(hpcp_mod.os.path, 'ismount', lambda path: any(str(p) == str(path).rstrip(os.sep) for p in roots))
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', run)
	monkeypatch.setattr(hpcp_mod, 'clean_up', lambda *a: None)
	rc = hpcp_mod.hpcp(['/dev/fake-source'], dest_image=str(tmp_path / 'dest.img'), dd=True,
		single_thread=True, max_workers=1, no_directory_sync=True)
	assert len(roots) == 4
	assert (roots[3] / 'data.txt').read_text() == 'partition data'
	assert rc == 152


def test_dd_mounts_xfs_partitions_with_nouuid(tmp_path, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch):
	"""A -dd source that is an image of a disk whose xfs is mounted (the live node it was taken from) has that UUID.
	The source is only read, so it is mounted read-only (ultrareview: a read-write nouuid mount wrote to a clone)."""
	roots = []
	mounts = []
	real_run = hpcp_mod.run_command_in_multicmd_with_path_check

	def mount_dir():
		path = tmp_path / ('mount' + str(len(roots)))
		path.mkdir()
		roots.append(path)
		if len(roots) % 2 == 1:
			(path / 'data.txt').write_text('partition data')
		return str(path)

	def run(command, **kwargs):
		if command[0] == 'mount':
			mounts.append(list(command))
			return []
		return real_run(command, **kwargs)

	monkeypatch.setattr(hpcp_mod, 'validate_dd_source_path', lambda *a, **k: '/dev/fake-source')
	monkeypatch.setattr(hpcp_mod, 'create_dd_dest_part_table', lambda *a, **k: ({'/dev/fake-source': {}, '1': {'size': 4096}}, [], None))
	monkeypatch.setattr(hpcp_mod, 'get_partitions', lambda device: [device + '1'])
	monkeypatch.setattr(hpcp_mod, '_probe_fs_type', lambda device: 'xfs')
	monkeypatch.setattr(hpcp_mod.os, 'geteuid', lambda: 0)
	monkeypatch.setattr(hpcp_mod, 'tempfile', SimpleNamespace(mkdtemp=mount_dir))
	monkeypatch.setattr(hpcp_mod.os.path, 'ismount', lambda path: any(str(p) == str(path).rstrip(os.sep) for p in roots))
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', run)
	monkeypatch.setattr(hpcp_mod, 'clean_up', lambda *a: None)
	assert hpcp_mod.hpcp(['/dev/fake-source'], dest_image=str(tmp_path / 'dest.img'), dd=True,
		single_thread=True, max_workers=1, no_directory_sync=True) == 0
	assert [command[:3] for command in mounts] == [['mount', '-o', 'ro,nouuid'], ['mount', '-o', 'nouuid']]


@pytest.mark.parametrize('fs_type, options', [('ext4', 'ro'), ('btrfs', 'ro'), ('xfs', 'ro,nouuid')])
def test_dd_source_mount_command_is_read_only(hpcp_mod, monkeypatch, fs_type, options):
	monkeypatch.setattr(hpcp_mod, '_probe_fs_type', lambda device: fs_type)
	assert hpcp_mod._ro_source_mount_command('/dev/sdb1', '/mnt/x') == ['mount', '-o', options, '/dev/sdb1', '/mnt/x']
