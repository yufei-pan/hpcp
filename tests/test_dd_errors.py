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
