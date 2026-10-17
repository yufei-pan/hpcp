"""9.75: -rme leaves a filesystem mounted below the destination alone, its files as well as its directories."""
import contextlib
import os
import subprocess
import sys

import pytest


@contextlib.contextmanager
def _mounted(target, bind_source=None):
	"""A tmpfs (another device) or a bind mount of bind_source (the same device) at target, always unmounted after."""
	if os.geteuid() != 0:
		pytest.skip('needs root to mount')
	target.mkdir(parents=True, exist_ok=True)
	command = ['mount', '--bind', str(bind_source), str(target)] if bind_source else ['mount', '-t', 'tmpfs', 'hpcp-test', str(target)]
	result = subprocess.run(command, capture_output=True, text=True)
	if result.returncode:
		pytest.skip(f'mount failed: {result.stderr.strip()}')
	try:
		yield
	finally:
		subprocess.run(['umount', '-l', str(target)], capture_output=True)


def _answer_yes(monkeypatch):
	monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
	monkeypatch.setattr('builtins.input', lambda: 'y')


def _failures(hpcp_mod):
	return [error for error in hpcp_mod.ERRORS if hpcp_mod._is_failure(error)]


def _fill(directory):
	(directory / 'sub' / 'empty').mkdir(parents=True)
	(directory / 'x.txt').write_text('other filesystem')
	(directory / 'sub' / 'y.txt').write_text('other filesystem')


@pytest.mark.parametrize('kind', ['tmpfs', 'bind'])
def test_remove_extra_leaves_a_filesystem_mounted_below_the_destination(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, kind):
	"""9.74 left the mounted filesystem's directories alone but deleted its files (they are not in the source)."""
	src, dst = tmp_path / 'src', tmp_path / 'dst'
	src.mkdir()
	dst.mkdir()
	(dst / 'stale.txt').write_text('stale')
	(dst / 'stale-dir').mkdir()
	other = tmp_path / 'other'
	other.mkdir()
	with _mounted(dst / 'mnt', bind_source=other if kind == 'bind' else None):
		_fill(dst / 'mnt')
		_answer_yes(monkeypatch)
		hpcp_mod.remove_extra_files([], [str(dst) + os.sep], 2, False, 1, single_thread=True, src_paths=[str(src) + os.sep])
		hpcp_mod.remove_extra_dirs([str(src) + os.sep], [str(dst) + os.sep])
		assert (dst / 'mnt' / 'x.txt').read_text() == 'other filesystem'
		assert (dst / 'mnt' / 'sub' / 'y.txt').read_text() == 'other filesystem'
		assert (dst / 'mnt' / 'sub' / 'empty').is_dir()
	assert not (dst / 'stale.txt').exists() and not (dst / 'stale-dir').exists()
	assert _failures(hpcp_mod) == []


@pytest.mark.parametrize('single_thread', [True, False])
def test_hpcp_rme_leaves_a_mounted_filesystem_below_the_destination(tmp_path, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch, single_thread):
	src, dst = tmp_path / 'src', tmp_path / 'dst'
	src.mkdir()
	(src / 'a.txt').write_text('a')
	dst.mkdir()
	(dst / 'stale.txt').write_text('stale')
	with _mounted(dst / 'mnt'):
		_fill(dst / 'mnt')
		_answer_yes(monkeypatch)
		rc = hpcp_mod.hpcp([str(src) + os.sep], dest_paths=[str(dst) + os.sep], remove_extra=True, single_thread=single_thread,
			max_workers=1 if single_thread else 2, batch=True)
		assert (dst / 'mnt' / 'x.txt').exists() and (dst / 'mnt' / 'sub' / 'y.txt').exists()
	assert rc == 0
	assert (dst / 'a.txt').read_text() == 'a' and not (dst / 'stale.txt').exists()


def test_remove_extra_cleans_a_destination_that_is_itself_a_mount_point(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Only filesystems mounted below the destination are left alone, not the destination's own (a -di image, a disk)."""
	src, dst = tmp_path / 'src', tmp_path / 'dst'
	src.mkdir()
	with _mounted(dst):
		(dst / 'stale.txt').write_text('stale')
		(dst / 'stale-dir').mkdir()
		_answer_yes(monkeypatch)
		hpcp_mod.remove_extra_files([], [str(dst) + os.sep], 2, False, 1, single_thread=True, src_paths=[str(src) + os.sep])
		hpcp_mod.remove_extra_dirs([str(src) + os.sep], [str(dst) + os.sep])
		assert not (dst / 'stale.txt').exists() and not (dst / 'stale-dir').exists()
	assert _failures(hpcp_mod) == []


@pytest.mark.parametrize('single_thread', [True, False])
def test_removal_cursor_stops_at_another_device_it_was_not_told_about(tmp_path, hpcp_mod, reset_hpcp_globals, single_thread):
	"""The backstop: a mount (or a btrfs subvolume) the extras list did not know about, met while removing, is left."""
	dst = tmp_path / 'dst'
	dst.mkdir()
	(dst / 'stale.txt').write_text('stale')
	with _mounted(dst / 'mnt'):
		_fill(dst / 'mnt')
		root_id = hpcp_mod._path_identity(str(dst))
		items = [(str(dst), root_id, rel) for rel in ('mnt/sub/y.txt', 'mnt/x.txt', 'stale.txt')]
		hpcp_mod._remove_items(sorted(items, key=lambda item: item[2]), single_thread=single_thread, max_workers=2, one_filesystem=True)
		assert (dst / 'mnt' / 'x.txt').exists() and (dst / 'mnt' / 'sub' / 'y.txt').exists()
	assert not (dst / 'stale.txt').exists()
	assert _failures(hpcp_mod) == []
	assert any(error.startswith('Other filesystem left') for error in hpcp_mod.ERRORS)
