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


# ---- 9.75: review of main..HEAD ----

def test_remove_extra_does_not_list_into_a_mounted_filesystem(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, capsys):
	"""Review: the destination listing went into the mount and filtered after it: a hung NFS mount stalled -rme, an
	unreadable FUSE mount failed the run. As rsync -x, the listing stops at the mount."""
	src, dst = tmp_path / 'src', tmp_path / 'dst'
	src.mkdir()
	dst.mkdir()
	(dst / 'stale.txt').write_text('stale')
	with _mounted(dst / 'share'):
		_fill(dst / 'share')
		real = os.scandir
		def scandir(path='.'):
			if isinstance(path, (str, bytes, os.PathLike)) and os.fspath(path).startswith(str(dst / 'share')):
				raise PermissionError(13, 'Permission denied (an unreadable FUSE mount)', os.fspath(path))
			return real(path)
		monkeypatch.setattr(hpcp_mod.os, 'scandir', scandir)
		_answer_yes(monkeypatch)
		hpcp_mod.remove_extra_files([], [str(dst) + os.sep], 2, False, 1, single_thread=True, src_paths=[str(src) + os.sep])
	assert not (dst / 'stale.txt').exists()
	assert _failures(hpcp_mod) == []


def test_remove_extra_prompt_lists_nothing_on_another_device_mountinfo_does_not_show(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, capsys):
	"""Review: a btrfs subvolume (another device, not a mount) was offered for deletion, then refused. Here a tmpfs that
	mountinfo is made not to show stands in for it: the listing stops at the device, so the prompt does not offer it."""
	src, dst = tmp_path / 'src', tmp_path / 'dst'
	src.mkdir()
	dst.mkdir()
	(dst / 'stale.txt').write_text('stale')
	with _mounted(dst / 'subvol'):
		_fill(dst / 'subvol')
		monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda *args, **kwargs: [])
		_answer_yes(monkeypatch)
		hpcp_mod.remove_extra_files([], [str(dst) + os.sep], 2, False, 1, single_thread=True, src_paths=[str(src) + os.sep])
		hpcp_mod.remove_extra_dirs([str(src) + os.sep], [str(dst) + os.sep])
		assert (dst / 'subvol' / 'x.txt').exists() and (dst / 'subvol' / 'sub' / 'empty').is_dir()
	output = capsys.readouterr().out
	assert 'x.txt' not in output and 'stale.txt' in output
	assert not (dst / 'stale.txt').exists()
	assert _failures(hpcp_mod) == []


def test_remove_extra_leaves_a_bind_mounted_file(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Review: a file that is a mount point (/etc/hosts in a container) was offered and failed to unlink (EBUSY)."""
	src, dst = tmp_path / 'src', tmp_path / 'dst'
	src.mkdir()
	dst.mkdir()
	(tmp_path / 'hosts-source').write_text('127.0.0.1 localhost')
	(dst / 'hosts').write_text('')
	if os.geteuid() != 0:
		pytest.skip('needs root to mount')
	result = subprocess.run(['mount', '--bind', str(tmp_path / 'hosts-source'), str(dst / 'hosts')], capture_output=True, text=True)
	if result.returncode:
		pytest.skip(f'mount failed: {result.stderr.strip()}')
	try:
		_answer_yes(monkeypatch)
		hpcp_mod.remove_extra_files([], [str(dst) + os.sep], 2, False, 1, single_thread=True, src_paths=[str(src) + os.sep])
		assert (dst / 'hosts').read_text() == '127.0.0.1 localhost'
	finally:
		subprocess.run(['umount', '-l', str(dst / 'hosts')], capture_output=True)
	assert _failures(hpcp_mod) == []


def test_a_mount_hidden_under_a_later_mount_does_not_shield_the_visible_directory(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Review: dst/share was a mount, then a filesystem was mounted over dst: the share mountinfo still lists is hidden,
	and dst/share is a plain directory of the visible filesystem, cleaned like any other."""
	src, dst = tmp_path / 'src', tmp_path / 'dst'
	src.mkdir()
	with _mounted(dst / 'share'):
		(dst / 'share' / 'old.txt').write_text('on the hidden mount')
		with _mounted(dst):
			(dst / 'share').mkdir()
			(dst / 'share' / 'stale.txt').write_text('stale')
			_answer_yes(monkeypatch)
			hpcp_mod.remove_extra_files([], [str(dst) + os.sep], 2, False, 1, single_thread=True, src_paths=[str(src) + os.sep])
			hpcp_mod.remove_extra_dirs([str(src) + os.sep], [str(dst) + os.sep])
			assert not (dst / 'share').exists()
		assert (dst / 'share' / 'old.txt').read_text() == 'on the hidden mount'
	assert _failures(hpcp_mod) == []


def test_a_directory_mounted_on_after_the_walk_is_kept_without_an_error(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Review: an extra directory mounted on between the walk and its rmdir failed with EBUSY as an error."""
	src, dst = tmp_path / 'src', tmp_path / 'dst'
	src.mkdir()
	(dst / 'extra').mkdir(parents=True)
	if os.geteuid() != 0:
		pytest.skip('needs root to mount')
	real_walk = hpcp_mod._walk
	mounted = []
	def walk(*args, **kwargs):
		yield from real_walk(*args, **kwargs)
		if not mounted and subprocess.run(['mount', '-t', 'tmpfs', 'hpcp-test', str(dst / 'extra')], capture_output=True).returncode == 0:
			mounted.append(True)
	monkeypatch.setattr(hpcp_mod, '_walk', walk)
	try:
		hpcp_mod.remove_extra_dirs([str(src) + os.sep], [str(dst) + os.sep])
		assert mounted and (dst / 'extra').is_dir()
	finally:
		subprocess.run(['umount', '-l', str(dst / 'extra')], capture_output=True)
	assert _failures(hpcp_mod) == []


def test_deep_destination_with_a_mount_below_it(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Review: the mount check recursed once per path component (a deep chain hit Python's recursion limit); the
	listing now stops at mounts itself. 600 levels: the listing's own recursion (pre-existing, also in the copy) stops
	near 1000."""
	src, dst = tmp_path / 'src', tmp_path / 'dst'
	src.mkdir()
	deep = dst
	dst.mkdir()
	for _ in range(600):
		deep = deep / 'a'
		os.mkdir(str(deep))
	(deep / 'stale.txt').write_text('stale')
	with _mounted(dst / 'mnt'):
		_answer_yes(monkeypatch)
		hpcp_mod.remove_extra_files([], [str(dst) + os.sep], 2, False, 1, single_thread=True, src_paths=[str(src) + os.sep])
	assert not (deep / 'stale.txt').exists()


def test_fallback_removal_stops_at_another_device(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Review: without descriptor support the one_filesystem backstop did not exist."""
	dst = tmp_path / 'dst'
	dst.mkdir()
	(dst / 'stale.txt').write_text('stale')
	with _mounted(dst / 'mnt'):
		_fill(dst / 'mnt')
		monkeypatch.setattr(hpcp_mod, '_DIR_FD_WALK', False)
		root_id = hpcp_mod._path_identity(str(dst))
		items = [(str(dst), root_id, rel) for rel in ('mnt/sub/y.txt', 'mnt/x.txt', 'stale.txt')]
		hpcp_mod._remove_items(items, single_thread=True, one_filesystem=True)
		assert (dst / 'mnt' / 'x.txt').exists() and (dst / 'mnt' / 'sub' / 'y.txt').exists()
	assert not (dst / 'stale.txt').exists()
	assert _failures(hpcp_mod) == []


# ---- 9.75: re-review of 127f141..HEAD ----

@pytest.mark.parametrize('dest', ['dst/', './dst/', 'x/../dst/'])
def test_remove_extra_with_a_relative_destination_leaves_a_bind_mount_below_it(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, dest):
	"""Re-review: the listing names entries as the destination was typed ('dst/share'), the mount points were absolute:
	a bind mount of the same filesystem below a relative destination was listed and its files removed."""
	(tmp_path / 'src').mkdir()
	(tmp_path / 'dst').mkdir()
	(tmp_path / 'x').mkdir()
	(tmp_path / 'dst' / 'stale.txt').write_text('stale')
	other = tmp_path / 'other'
	other.mkdir()
	with _mounted(tmp_path / 'dst' / 'share', bind_source=other):
		_fill(tmp_path / 'dst' / 'share')
		monkeypatch.chdir(tmp_path)
		_answer_yes(monkeypatch)
		hpcp_mod.remove_extra_files([], [dest], 2, False, 1, single_thread=True, src_paths=['src/'])
		hpcp_mod.remove_extra_dirs(['src/'], [dest])
		assert (other / 'x.txt').read_text() == 'other filesystem' and (other / 'sub' / 'empty').is_dir()
	assert not (tmp_path / 'dst' / 'stale.txt').exists()
	assert _failures(hpcp_mod) == []


def test_remove_extra_dirs_does_not_stat_into_a_mount_point(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Re-review: an lstat (islink) of a mount point goes to the mounted filesystem: a dead NFS server hangs it. The
	mountinfo check, a string lookup, comes first."""
	src, dst = tmp_path / 'src', tmp_path / 'dst'
	src.mkdir()
	dst.mkdir()
	with _mounted(dst / 'nfs'):
		looked_at = []
		real_islink, real_lstat = os.path.islink, os.lstat
		monkeypatch.setattr(hpcp_mod.os.path, 'islink', lambda path: looked_at.append(os.path.abspath(path)) or real_islink(path))
		monkeypatch.setattr(hpcp_mod.os, 'lstat', lambda path, *a, **k: looked_at.append(os.path.abspath(path)) or real_lstat(path, *a, **k))
		hpcp_mod.remove_extra_dirs([str(src) + os.sep], [str(dst) + os.sep])
		assert str(dst / 'nfs') not in looked_at


def test_no_device_boundary_on_windows(tmp_path, hpcp_mod, monkeypatch):
	"""Re-review: DirEntry.stat reports st_dev 0 on Windows (os.stat the volume serial): every directory looked foreign."""
	monkeypatch.setattr(hpcp_mod.os, 'name', 'nt')
	boundary = hpcp_mod._filesystem_boundary(str(tmp_path), str(tmp_path), visible=frozenset())
	assert boundary[-1] is None


def test_cursor_closes_a_directory_whose_fstat_failed(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	root = tmp_path / 'dst'
	(root / 'sub').mkdir(parents=True)
	(root / 'sub' / 'f').write_text('x')
	real_fstat = os.fstat
	calls = []
	def fstat(fd):
		calls.append(fd)
		if len(calls) == 3:
			raise OSError(116, 'Stale file handle')
		return real_fstat(fd)
	before = len(os.listdir('/proc/self/fd'))
	monkeypatch.setattr(hpcp_mod.os, 'fstat', fstat)
	hpcp_mod._remove_listed_files([(str(root), hpcp_mod._path_identity(str(root)), 'sub/f')], one_filesystem=True)
	monkeypatch.setattr(hpcp_mod.os, 'fstat', real_fstat)
	assert len(calls) >= 3
	assert len(os.listdir('/proc/self/fd')) == before


def test_fallback_backstop_checks_every_directory_above_the_entry(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Re-review: root (dev A)/mnt is dev B, root/mnt/back a bind of a dev-A directory: the entry's parent is dev A, but
	it lies below another filesystem. The descriptor cursor stops at mnt; the fallback must too."""
	dst, other = tmp_path / 'dst', tmp_path / 'other'
	dst.mkdir()
	other.mkdir()
	(other / 'f.txt').write_text('keep')
	with _mounted(dst / 'mnt'):
		with _mounted(dst / 'mnt' / 'back', bind_source=other):
			monkeypatch.setattr(hpcp_mod, '_DIR_FD_WALK', False)
			hpcp_mod._remove_items([(str(dst), hpcp_mod._path_identity(str(dst)), 'mnt/back/f.txt')], single_thread=True, one_filesystem=True)
			assert (other / 'f.txt').read_text() == 'keep'


def test_rmdir_ebusy_that_is_no_mount_point_is_an_error(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Re-review: only a directory that is a mount point now is kept as one; another EBUSY (a FUSE server's) is reported."""
	root = tmp_path / 'dst'
	(root / 'extra').mkdir(parents=True)
	real_rmdir = os.rmdir
	def rmdir(path, *args, **kwargs):
		if os.fspath(path) == 'extra':
			raise OSError(16, 'Device or resource busy', 'extra')
		return real_rmdir(path, *args, **kwargs)
	monkeypatch.setattr(hpcp_mod.os, 'rmdir', rmdir)
	hpcp_mod._remove_listed_files([(str(root), hpcp_mod._path_identity(str(root)), 'extra')], rmdir=True, one_filesystem=True)
	assert [error.partition(':')[0] for error in _failures(hpcp_mod)] == ['Remove directory failed']


# ---- 9.77: a filesystem that appears only when its directory is entered (an NFSv4 submount, an automount) ----

class _StatAsParent:
	"""A scandir entry whose stat says the parent's device for one name: what an untriggered automount point (an
	NFSv4 crossmnt submount) looks like before it is entered."""
	def __init__(self, entry, name, device):
		self._entry, self._name, self._device = entry, name, device
	def __getattr__(self, attribute):
		return getattr(self._entry, attribute)
	def stat(self, follow_symlinks=True):
		st = self._entry.stat(follow_symlinks=follow_symlinks)
		if self._entry.name != self._name:
			return st
		fields = list(st)[:10]
		fields[2] = self._device
		return os.stat_result(fields)


class _Scandir:
	def __init__(self, real, target, name, device):
		self._iterator, self._name, self._device = real(target), name, device
	def __enter__(self):
		return self
	def __exit__(self, *exc):
		self._iterator.close()
	def __iter__(self):
		return (_StatAsParent(entry, self._name, self._device) for entry in self._iterator)
	def close(self):
		self._iterator.close()


def _looks_unmounted(hpcp_mod, monkeypatch, name, device):
	"""Hide mountinfo and make the scandir entry 'name' report the parent's device (before it is entered)."""
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda *args, **kwargs: [])
	real = os.scandir
	monkeypatch.setattr(hpcp_mod.os, 'scandir', lambda target='.': _Scandir(real, target, name, device))


def test_remove_extra_prompt_lists_nothing_from_a_filesystem_mounted_on_entering(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, scan_mode, capsys):
	"""9.75 follow-up: an NFSv4 submount not mounted yet has its parent's device and no mountinfo entry, so it passed
	the boundary until the listing entered it; its files reached the y/n prompt (the backstop then kept them). The
	listing checks the device of each directory once open."""
	src, dst = tmp_path / 'src', tmp_path / 'dst'
	src.mkdir()
	dst.mkdir()
	(dst / 'stale.txt').write_text('stale')
	with _mounted(dst / 'sub'):
		_fill(dst / 'sub')
		_looks_unmounted(hpcp_mod, monkeypatch, 'sub', os.stat(str(dst)).st_dev)
		_answer_yes(monkeypatch)
		hpcp_mod.remove_extra_files([], [str(dst) + os.sep], 2, False, 1, single_thread=True, src_paths=[str(src) + os.sep])
		assert (dst / 'sub' / 'x.txt').exists() and (dst / 'sub' / 'sub' / 'y.txt').exists()
	output = capsys.readouterr().out
	assert 'x.txt' not in output and 'y.txt' not in output and 'stale.txt' in output
	assert not (dst / 'stale.txt').exists()
	assert not any(error.startswith('Other filesystem left') for error in hpcp_mod.ERRORS)
	assert _failures(hpcp_mod) == []


@pytest.mark.parametrize('parallel', [False, True])
def test_destination_listing_stops_at_a_filesystem_mounted_on_entering(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, scan_mode, parallel):
	dst = tmp_path / 'dst'
	dst.mkdir()
	(dst / 'own.txt').write_text('own')
	with _mounted(dst / 'sub'):
		_fill(dst / 'sub')
		_looks_unmounted(hpcp_mod, monkeypatch, 'sub', os.stat(str(dst)).st_dev)
		boundary = hpcp_mod._filesystem_boundary(str(dst) + os.sep, str(dst), visible=frozenset())
		files, links, size, folders = hpcp_mod.get_file_list(str(dst) + os.sep, max_workers=2, parallel_file_listing=parallel, boundary=boundary)
	assert sorted(os.path.basename(path) for path in files) == ['own.txt']


def test_remove_extra_dirs_does_not_walk_a_filesystem_mounted_on_entering(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, scan_mode, capsys):
	src, dst = tmp_path / 'src', tmp_path / 'dst'
	src.mkdir()
	dst.mkdir()
	with _mounted(dst / 'sub'):
		(dst / 'sub' / 'empty').mkdir()
		monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda *args, **kwargs: [])
		_stub_until_opened(hpcp_mod, monkeypatch, dst / 'sub', os.stat(str(dst)).st_dev)
		hpcp_mod.remove_extra_dirs([str(src) + os.sep], [str(dst) + os.sep])
		assert (dst / 'sub' / 'empty').is_dir()
	output = capsys.readouterr().out
	assert 'empty' not in output
	assert not any(error.startswith('Other filesystem left') for error in hpcp_mod.ERRORS)
	assert _failures(hpcp_mod) == []


# ---- 9.77: review of main..HEAD ----

def test_remove_extra_dirs_does_not_offer_an_automount_point_itself(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, scan_mode, capsys):
	"""Review: the parent's loop decided before entering: the untriggered mount point was 'Deleting extra directory'
	and rmdir'd (EBUSY saved it; an expired automount would leave the stub to rmdir)."""
	src, dst = tmp_path / 'src', tmp_path / 'dst'
	src.mkdir()
	dst.mkdir()
	(dst / 'stale').mkdir()
	with _mounted(dst / 'sub'):
		monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda *args, **kwargs: [])
		_stub_until_opened(hpcp_mod, monkeypatch, dst / 'sub', os.stat(str(dst)).st_dev)
		hpcp_mod.remove_extra_dirs([str(src) + os.sep], [str(dst) + os.sep])
		assert (dst / 'sub').is_dir()
	output = capsys.readouterr().out
	assert 'Deleting extra directory: ' + str(dst / 'sub') not in output and 'Keeping mount point' not in output
	assert not (dst / 'stale').exists()
	assert _failures(hpcp_mod) == []


def _stub_until_opened(hpcp_mod, monkeypatch, path, device=None):
	"""stat and lstat of path (by name) report another identity than the directory opened there (device: that device,
	its parent's): an NFSv4 submount (an automount) not mounted yet, which a stat does not mount and an open does."""
	def stubbed(real):
		def call(target, *args, **kwargs):
			st = real(target, *args, **kwargs)
			if isinstance(target, str) and kwargs.get('dir_fd') is None and os.path.abspath(target) == str(path):
				# Every field kept (the times in nanoseconds too), the inode and device changed.
				result_type, (fields, extra) = st.__reduce__()
				fields = list(fields)
				fields[1] += 777
				fields[2] = device if device is not None else fields[2] + 12345
				return result_type(fields, extra)
			return st
		return call
	monkeypatch.setattr(hpcp_mod.os, 'stat', stubbed(os.stat))
	monkeypatch.setattr(hpcp_mod.os, 'lstat', stubbed(os.lstat))


def test_a_destination_that_is_itself_mounted_on_entering_is_cleaned(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Review: the destination's pin and the boundary's device came from lstat / stat, which do not mount an automount:
	opened, it had another identity (the removal refused it) and another device (every directory below looked foreign:
	nothing cleaned, no note)."""
	src, dst = tmp_path / 'src', tmp_path / 'dst'
	src.mkdir()
	(src / 'kept.txt').write_text('kept')
	(dst / 'a').mkdir(parents=True)
	(dst / 'a' / 'stale.txt').write_text('stale')
	_stub_until_opened(hpcp_mod, monkeypatch, dst)
	_answer_yes(monkeypatch)
	rc = hpcp_mod.hpcp([str(src) + os.sep], dest_paths=[str(dst) + os.sep], remove_extra=True, single_thread=True, batch=True)
	assert not (dst / 'a').exists()
	assert (dst / 'kept.txt').read_text() == 'kept'
	assert rc == 0 and _failures(hpcp_mod) == []


def test_a_source_that_is_itself_mounted_on_entering_is_moved(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Review: the moved source was pinned by lstat (the unmounted stub); opened by the removal it was another
	directory, so every source file was refused as replaced since the copy."""
	src, dst = tmp_path / 'src', tmp_path / 'dst'
	(src / 'd').mkdir(parents=True)
	(src / 'd' / 'f.txt').write_text('f')
	_stub_until_opened(hpcp_mod, monkeypatch, src)
	rc = hpcp_mod.hpcp([str(src) + os.sep], dest_paths=[str(dst) + os.sep], remove_force=True, single_thread=True, batch=True)
	assert (dst / 'd' / 'f.txt').read_text() == 'f'
	assert not (src / 'd').exists()
	assert rc == 0 and _failures(hpcp_mod) == []


def test_rmtree_removes_a_directory_mounted_on_entering(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	tree = tmp_path / 'tree'
	(tree / 'd').mkdir(parents=True)
	(tree / 'd' / 'f.txt').write_text('f')
	_stub_until_opened(hpcp_mod, monkeypatch, tree)
	hpcp_mod._rmtree(str(tree))
	assert not tree.exists()


def test_parallel_listing_leaves_a_foreign_folder_out_of_its_folders(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, scan_mode):
	dst = tmp_path / 'dst'
	(dst / 'own').mkdir(parents=True)
	with _mounted(dst / 'sub'):
		_fill(dst / 'sub')
		_looks_unmounted(hpcp_mod, monkeypatch, 'sub', os.stat(str(dst)).st_dev)
		boundary = hpcp_mod._filesystem_boundary(str(dst) + os.sep, str(dst), visible=frozenset())
		_, _, _, folders = hpcp_mod.get_file_list(str(dst) + os.sep, max_workers=2, parallel_file_listing=True, boundary=boundary)
	assert os.path.join(str(dst), 'sub') not in folders and os.path.join(str(dst), 'own') in folders


def test_boundary_listing_does_not_lstat_each_directory(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, scan_mode):
	"""Review: an lstat, then an open and an fstat, per directory: the open says ELOOP / ENOTDIR for a swapped entry."""
	dst = tmp_path / 'dst'
	for index in range(20):
		(dst / ('d%d' % index)).mkdir(parents=True)
	boundary = hpcp_mod._filesystem_boundary(str(dst), str(dst), visible=frozenset())
	calls = []
	real_lstat = os.lstat
	monkeypatch.setattr(hpcp_mod.os, 'lstat', lambda *a, **k: calls.append(a[0]) or real_lstat(*a, **k))
	_, _, _, folders = hpcp_mod.get_file_list_serial(str(dst), boundary=boundary)
	assert len(folders) == 21
	assert len(calls) <= 2


def test_a_directory_swapped_for_a_symlink_before_it_is_entered_is_listed_as_a_link(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, scan_mode):
	"""Review: the open's ELOOP became a 'Listing error'; 9.76 listed the entry as the link it had become."""
	dst = tmp_path / 'dst'
	(dst / 'sub').mkdir(parents=True)
	boundary = hpcp_mod._filesystem_boundary(str(dst), str(dst), visible=frozenset())
	real = hpcp_mod._enter_directory
	def enter(path, device):
		if os.path.basename(path) == 'sub' and not os.path.islink(path):
			os.rmdir(path)
			os.symlink('/nowhere', path)
		return real(path, device)
	monkeypatch.setattr(hpcp_mod, '_enter_directory', enter)
	files, links, _, folders = hpcp_mod.get_file_list_serial(str(dst), boundary=boundary)
	assert os.path.join(str(dst), 'sub') in links
	assert _failures(hpcp_mod) == []
