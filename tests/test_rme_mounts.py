"""9.75: -rme leaves a filesystem mounted below the destination alone, its files as well as its directories."""
import contextlib
import errno
import os
import shutil
import subprocess
import sys

import pytest

# The test's own names need a UTF-8 filesystem encoding to be made (Python 3.6 with LANG unset has ASCII); hpcp's side of
# such a locale is test_landings' test_a_non_ascii_name_copies_with_lang_unset.
utf8_names = pytest.mark.skipif(sys.getfilesystemencoding().lower().replace('-', '') != 'utf8',
	reason='needs a UTF-8 filesystem encoding for its own non-ASCII names')

HPCP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hpcp.py')


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


def test_a_move_onto_a_bind_mount_of_itself_is_refused(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""One directory by two paths: every file is identical to itself, so the removal would delete the only copy."""
	src = tmp_path / 'src'
	src.mkdir()
	(src / 'f').write_text('keep')
	(tmp_path / 'bind').mkdir()
	with _mounted(tmp_path / 'bind', bind_source=src):
		assert hpcp_mod.hpcp([str(src) + '/'], dest_paths=[str(tmp_path / 'bind') + '/'], remove_force=True, batch=True) == 241
		assert (src / 'f').read_text() == 'keep'


def test_a_landing_through_a_bind_mount_is_a_landing(tmp_path):
	"""m2 is m1 by another path: d1/x and m2/d1/x both land on m1/d1/x, and the second is the first's source."""
	run = lambda: subprocess.run([sys.executable, HPCP, 'd1', 'm2/d1/', '-d', 'm1/', '-rf', '-s'], cwd=str(tmp_path), stdin=subprocess.DEVNULL,
		stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	(tmp_path / 'd1').mkdir()
	(tmp_path / 'd1/x').write_text('A')
	(tmp_path / 'm1/d1').mkdir(parents=True)
	(tmp_path / 'm1/d1/x').write_text('B')
	(tmp_path / 'm2').mkdir()
	with _mounted(tmp_path / 'm2', bind_source=tmp_path / 'm1'):
		task = run()
		assert task.returncode == 213, task.stdout[-2000:]
		assert (tmp_path / 'd1/x').read_text() == 'A' and (tmp_path / 'm1/d1/x').read_text() == 'B'


def test_a_landing_by_another_case_is_a_landing(tmp_path):
	"""On a casefolded directory D2 and d2 are one: the second source is what the first would land on."""
	mkfs, image, mount = shutil.which('mkfs.ext4'), tmp_path / 'fold.img', tmp_path / 'fold'
	if os.geteuid() != 0 or not mkfs or not shutil.which('chattr'):
		pytest.skip('needs root, mkfs.ext4 and chattr')
	image.write_bytes(b'')
	os.truncate(str(image), 64 << 20)
	if subprocess.run([mkfs, '-q', '-O', 'casefold', '-E', 'encoding=utf8', str(image)], capture_output=True).returncode:
		pytest.skip('mkfs.ext4 without casefold')
	mount.mkdir()
	if subprocess.run(['mount', '-o', 'loop', str(image), str(mount)], capture_output=True).returncode:
		pytest.skip('this kernel does not mount a casefold filesystem')
	try:
		fold = mount / 'w'
		fold.mkdir()
		if subprocess.run(['chattr', '+F', str(fold)], capture_output=True).returncode:
			pytest.skip('casefold flag refused')
		(fold / 'd1').mkdir()
		(fold / 'd1/x').write_text('A')
		(fold / 'd2/d1').mkdir(parents=True)
		(fold / 'd2/d1/x').write_text('B')
		task = subprocess.run([sys.executable, HPCP, 'd1', 'D2/d1/', '-d', 'd2/', '-rf', '-s'], cwd=str(fold), stdin=subprocess.DEVNULL,
			stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
		assert task.returncode == 213, task.stdout[-2000:]
		assert (fold / 'd1/x').read_text() == 'A' and (fold / 'd2/d1/x').read_text() == 'B'
	finally:
		subprocess.run(['umount', '-l', str(mount)], capture_output=True)


_FOLDING = {
	'ext4': ('mkfs.ext4', ['-q', '-O', 'casefold', '-E', 'encoding=utf8'], []),
	'f2fs': ('mkfs.f2fs', ['-q', '-O', 'casefold', '-C', 'utf8'], []),
	'vfat': ('mkfs.vfat', [], []),
	'vfat-utf8': ('mkfs.vfat', [], ['-o', 'iocharset=utf8']),
	'exfat': ('mkfs.exfat', [], []),
	'ntfs': ('mkfs.ntfs', ['-F', '-f', '-q'], ['-t', 'lowntfs-3g', '-o', 'ignore_case']),
	'vfat-iso8859-9': ('mkfs.vfat', [], ['-o', 'iocharset=iso8859-9,utf8=0,codepage=857']),
	'vfat-codepage-850': ('mkfs.vfat', [], ['-o', 'iocharset=utf8,codepage=850']),
}


@contextlib.contextmanager
def _folding_directory(tmp_path, kind, table=_FOLDING):
	"""A directory on a filesystem of its own: ext4 or f2fs with the casefold flag, or the root of the others."""
	tool, mkfs_options, mount_options = table[kind]
	mkfs = shutil.which(tool) or shutil.which('/usr/sbin/' + tool) or shutil.which('/sbin/' + tool)
	image, mount = tmp_path / 'fold.img', tmp_path / 'fold'
	if os.geteuid() != 0 or not mkfs or not shutil.which('chattr'):
		pytest.skip('needs root, mkfs and chattr')
	image.write_bytes(b'')
	os.truncate(str(image), 128 << 20)
	if subprocess.run([mkfs] + mkfs_options + [str(image)], capture_output=True).returncode:
		pytest.skip(f'mkfs for {kind} refused')
	mount.mkdir()
	if subprocess.run(['mount', '-o', 'loop', str(image), str(mount)] + mount_options, capture_output=True).returncode:
		pytest.skip(f'this kernel does not mount {kind}, or not with casefold')
	try:
		fold = mount
		if kind in ('ext4', 'f2fs'):
			fold = mount / 'w'
			fold.mkdir()
			if subprocess.run(['chattr', '+F', str(fold)], capture_output=True).returncode:
				pytest.skip('casefold flag refused')
		yield fold
	finally:
		subprocess.run(['umount', '-l', str(mount)], capture_output=True)


def _hpcp(args, cwd):
	return subprocess.run([sys.executable, HPCP] + args, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _filled(path, content):
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(content)


_PAIRS = {'case': ('X', 'x'), 'sharp-s': ('Stra\u00dfe', 'STRASSE'), 'normal-form': ('caf\u00e9', 'cafe\u0301'), 'joiner': ('a\u200db', 'ab')}


@utf8_names
@pytest.mark.parametrize('kind', sorted(_FOLDING))
@pytest.mark.parametrize('pair', sorted(_PAIRS))
def test_a_run_agrees_with_the_filesystem_about_a_pair(tmp_path, hpcp_mod, kind, pair):
	"""The filesystem is the judge, whatever its rules: one name is a clash, two names both land. Each filesystem's own answer
	is read here by making both files."""
	first, second = _PAIRS[pair]
	_filled(tmp_path / 'a' / first, 'first')
	_filled(tmp_path / 'b' / second, 'second')
	with _folding_directory(tmp_path, kind) as fold:
		try:
			(fold / first).write_text('')
		except OSError:
			pytest.skip(f'{kind} takes no such name')
		try:
			os.close(os.open(str(fold / second), os.O_WRONLY | os.O_CREAT | os.O_EXCL))
			one = False
			(fold / second).unlink()
		except FileExistsError:
			one = True
		(fold / first).unlink()
		task = _hpcp(['a/', 'b/', '-d', str(fold / 'out') + os.sep], tmp_path)
		assert task.returncode == (213 if one else 0), task.stdout[-2000:]
		assert len(os.listdir(fold / 'out')) == (1 if one else 2), task.stdout[-2000:]
		assert not [name for name in os.listdir(fold) if name.startswith('.hpcp')]


@pytest.mark.parametrize('kind', ['ext4', 'f2fs', 'vfat', 'exfat'])
def test_a_move_of_two_names_that_are_one_removes_nothing(tmp_path, kind):
	"""It was exit 0 with one file's content gone, from the source too."""
	_filled(tmp_path / 'a/X', 'upper')
	_filled(tmp_path / 'b/x', 'lower')
	with _folding_directory(tmp_path, kind) as fold:
		task = _hpcp(['a/', 'b/', '-d', str(fold / 'out') + os.sep, '-rf'], tmp_path)
		assert task.returncode == 213 and b'where the destination folds case' in task.stdout, task.stdout[-2000:]
		assert (fold / 'out/x').read_text() == 'upper'
	assert (tmp_path / 'a/X').read_text() == 'upper' and (tmp_path / 'b/x').read_text() == 'lower'


@pytest.mark.parametrize('kind, first, second', [
	# The kernel's iso8859-9 table makes ASCII 'i' the other case of DD, dotted 'I'.
	('vfat-iso8859-9', 'i\u015f.txt'.encode('iso8859-9'), '\u0130\u015f.txt'.encode('iso8859-9')),
	# Codepage 850 gives 'k\u0131z.txt' the short name 'KIZ.TXT'.
	('vfat-codepage-850', b'KIZ.TXT', 'k\u0131z.txt'.encode('utf-8')),
], ids=['iso8859-9', 'codepage-850'])
@pytest.mark.parametrize('order', ['ab', 'ba'])
def test_a_move_of_turkish_names_that_the_mount_takes_for_one_removes_nothing(tmp_path, kind, first, second, order):
	"""Both were exit 0 with one content gone: iso8859-9 always, codepage 850 in about half the runs."""
	base = os.fsencode(str(tmp_path))
	for source, name, content in ((b'a', first, 'A'), (b'b', second, 'B')):
		os.makedirs(os.path.join(base, source))
		with open(os.path.join(base, source, name), 'w') as f:
			f.write(content)
	with _folding_directory(tmp_path, kind) as fold:
		sources = ['a/', 'b/'] if order == 'ab' else ['b/', 'a/']
		task = _hpcp(sources + ['-d', str(fold / 'out') + os.sep, '-rf'], tmp_path)
		assert task.returncode == 213 and b'where the destination folds case' in task.stdout, task.stdout[-2000:]
		assert len(os.listdir(fold / 'out')) == 1
	with open(os.path.join(base, b'a', first)) as f:
		assert f.read() == 'A'
	with open(os.path.join(base, b'b', second)) as f:
		assert f.read() == 'B'


def test_names_below_a_directory_that_is_not_utf8_are_still_folded_as_utf8(tmp_path):
	"""Only the name that is not UTF-8 is read byte by byte: below it, 'Stra\u00dfe' and 'STRASSE', and 'caf\u00e9' composed and
	decomposed, are one name each to ext4 casefold, and to the net."""
	pairs = [('Stra\u00dfe', 'STRASSE'), ('caf\u00e9', 'cafe\u0301')]
	odd = b'\xff'
	base = os.fsencode(str(tmp_path))
	for index, (first, second) in enumerate(pairs):
		for source, name, content in ((b'a', first, 'A%d' % index), (b'b', second, 'B%d' % index)):
			os.makedirs(os.path.join(base, source, odd), exist_ok=True)
			with open(os.path.join(base, source, odd, name.encode('utf-8')), 'w') as f:
				f.write(content)
	with _folding_directory(tmp_path, 'ext4') as fold:
		try:
			os.mkdir(os.path.join(os.fsencode(str(fold)), odd))
			os.rmdir(os.path.join(os.fsencode(str(fold)), odd))
		except OSError:
			pytest.skip('this ext4 refuses a name that is not UTF-8 in a casefold directory')
		task = _hpcp(['a/', 'b/', '-d', str(fold / 'out') + os.sep, '-rf'], tmp_path)
		assert task.returncode == 213 and b'where the destination folds case' in task.stdout, task.stdout[-2000:]
		assert len(os.listdir(os.path.join(os.fsencode(str(fold)), b'out', odd))) == len(pairs)
	for index, (first, second) in enumerate(pairs):
		for source, name, content in ((b'a', first, 'A%d' % index), (b'b', second, 'B%d' % index)):
			with open(os.path.join(base, source, odd, name.encode('utf-8'))) as f:
				assert f.read() == content


# Mounts that take two names for one where no rule on names says so, and the pairs: what only the move's backstop catches.
_BACKSTOP_MOUNTS = {
	'vfat': ('mkfs.vfat', [], []),
	'vfat-cp1251': ('mkfs.vfat', [], ['-o', 'iocharset=cp1251,utf8=0,codepage=866']),
	'vfat-koi8-r': ('mkfs.vfat', [], ['-o', 'iocharset=koi8-r,utf8=0,codepage=866']),
	'vfat-iso8859-9': ('mkfs.vfat', [], ['-o', 'iocharset=iso8859-9,utf8=0,codepage=857']),
	'exfat-euc-jp': ('mkfs.exfat', [], ['-t', 'exfat', '-o', 'iocharset=euc-jp']),
}
_BACKSTOP_PAIRS = {
	# vfat gives the long name the short name LONGFI~1.TXT, which the second is.
	'8.3-alias': ('vfat', b'longfilename.txt', b'LONGFI~1.TXT'),
	# Valid UTF-8 by chance against its other case, which is not.
	'cp1251-utf8-by-chance': ('vfat-cp1251', b'\xc3\xa9t', b'\xe3\xa9t'),
	# Both valid UTF-8, not as each other's case: KOI8-R '\u0431\u0451' and '\u0431\u0401'.
	'koi8-r-both-utf8': ('vfat-koi8-r', b'\xc2\xa3', b'\xc2\xb3'),
	# ASCII against ISO 8859-9 '\u0130', which that table makes the other case of 'i'.
	'iso8859-9-ascii': ('vfat-iso8859-9', b'istanbul', b'\xddstanbul'),
	# Two cases of different lengths: EUC-JP '\u03c2' is 8F A6 F8, '\u03a3' A6 B2.
	'euc-jp-lengths': ('exfat-euc-jp', b'\x8f\xa6\xf8', b'\xa6\xb2'),
}


@pytest.mark.parametrize('form', ['move-ab', 'move-ba', 'rename'])
@pytest.mark.parametrize('pair', sorted(_BACKSTOP_PAIRS))
def test_a_move_keeps_a_source_its_landing_does_not_hold_where_no_rule_on_names_says_why(tmp_path, pair, form):
	"""9.83 exited 0 with one content gone (a rename: the file gone). Whatever exit, both contents are still there, and an exit
	of 0 is two landed files. -j 1: in source order, so that an 8.3 alias is the first name's when the second comes."""
	kind, first, second = _BACKSTOP_PAIRS[pair]
	base = os.fsencode(str(tmp_path))
	with _folding_directory(tmp_path, kind, _BACKSTOP_MOUNTS) as fold:
		out = os.path.join(os.fsencode(str(fold)), b'out')
		os.makedirs(out)
		try:
			for name in (first, second):
				os.close(os.open(os.path.join(out, name), os.O_WRONLY | os.O_CREAT | os.O_EXCL))
				os.unlink(os.path.join(out, name))
		except OSError:
			pytest.skip(f'{kind} takes no such names')
		if form == 'rename':
			with open(os.path.join(out, first), 'w') as f:
				f.write('AAAA')
			task = subprocess.run([sys.executable, HPCP, first, '-d', second, '-rf'], cwd=out, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
			assert task.returncode == 213 and b'is the source itself' in task.stdout, task.stdout[-2000:]
			assert [open(os.path.join(out, name)).read() for name in os.listdir(out)] == ['AAAA']
			return
		for source, name, content in ((b'a', first, 'AAAA'), (b'b', second, 'BBBB')):
			os.makedirs(os.path.join(base, source))
			with open(os.path.join(base, source, name), 'w') as f:
				f.write(content)
		sources = ['a/', 'b/'] if form == 'move-ab' else ['b/', 'a/']
		task = _hpcp(sources + ['-d', os.fsdecode(out) + os.sep, '-rf', '-j', '1'], tmp_path)
		landed = [open(os.path.join(out, name)).read() for name in os.listdir(out)]
		left = [open(os.path.join(path, name)).read() for path, _, names in os.walk(os.path.join(base, b'a')) for name in names]
		left += [open(os.path.join(path, name)).read() for path, _, names in os.walk(os.path.join(base, b'b')) for name in names]
		assert {'AAAA', 'BBBB'} <= set(landed + left), task.stdout[-2000:]
		if task.returncode == 0:
			assert sorted(landed) == ['AAAA', 'BBBB'], task.stdout[-2000:]
		else:
			assert task.returncode == 213 and b'Destination clash: not removing' in task.stdout, task.stdout[-2000:]


def test_a_folding_directory_below_a_plain_destination_is_asked(tmp_path):
	"""The top says nothing of what lands below it."""
	_filled(tmp_path / 'a/sub/X', 'upper')
	_filled(tmp_path / 'a/sub/x', 'lower')
	with _folding_directory(tmp_path, 'ext4') as fold:
		plain = fold.parent / 'plain'
		(plain / 'sub').mkdir(parents=True)
		if subprocess.run(['chattr', '+F', str(plain / 'sub')], capture_output=True).returncode:
			pytest.skip('casefold flag refused')
		task = _hpcp(['a/', '-d', str(plain) + os.sep], tmp_path)
		assert task.returncode == 213 and os.listdir(plain / 'sub') == ['X'] and (plain / 'sub/X').read_text() == 'upper', task.stdout[-2000:]


def test_a_folding_mount_below_a_plain_destination_is_asked(tmp_path):
	_filled(tmp_path / 'a/usb/X', 'upper')
	_filled(tmp_path / 'a/usb/x', 'lower')
	(tmp_path / 'top/usb').mkdir(parents=True)
	with _folding_directory(tmp_path, 'vfat') as fold, _mounted(tmp_path / 'top/usb', bind_source=fold):
		task = _hpcp(['a/', '-d', str(tmp_path / 'top') + os.sep], tmp_path)
		assert task.returncode == 213 and os.listdir(tmp_path / 'top/usb') == ['X'], task.stdout[-2000:]


def test_a_link_in_another_case_that_leads_onto_a_source_is_refused(tmp_path):
	"""out/X -> ../src/y with src/x: the copy of x writes through X, onto src/y. The same in one case was always 213."""
	with _folding_directory(tmp_path, 'ext4') as fold:
		_filled(fold / 'src/x', 'x')
		_filled(fold / 'src/y', 'y')
		(fold / 'out').mkdir()
		os.symlink('../src/y', fold / 'out/X')
		task = _hpcp(['src/', '-d', 'out/'], fold)
		assert task.returncode == 213 and (fold / 'src/y').read_text() == 'y', task.stdout[-2000:]


def _mirror(hpcp_mod, monkeypatch, sources, out, **kwargs):
	_answer_yes(monkeypatch)
	return hpcp_mod.hpcp(sources, [str(out) + os.sep], single_thread=True, max_workers=1, batch=True, remove_extra=True, **kwargs)


@pytest.mark.parametrize('kind', ['ext4', 'vfat'])
def test_rme_leaves_a_file_written_through_another_case(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, kind):
	"""a/x onto out/X: 'X' looked extra, and with -rf the file was gone from both sides, exit 0."""
	_filled(tmp_path / 'a/x', 'new')
	monkeypatch.chdir(tmp_path)
	with _folding_directory(tmp_path, kind) as fold:
		_filled(fold / 'out/X', 'old')
		assert _mirror(hpcp_mod, monkeypatch, ['a/'], fold / 'out', remove_force=True) == 0
		assert (fold / 'out/X').read_text() == 'new'


def test_rme_leaves_files_of_directories_that_merged(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""a/D/f and b/d/g onto an empty destination: g lands in out/D, where out/d/g was expected."""
	_filled(tmp_path / 'a/D/f', 'f')
	_filled(tmp_path / 'b/d/g', 'g')
	monkeypatch.chdir(tmp_path)
	with _folding_directory(tmp_path, 'ext4') as fold:
		assert _mirror(hpcp_mod, monkeypatch, ['a/', 'b/'], fold / 'out', remove_force=True) == 0
		assert sorted(os.listdir(fold / 'out/D')) == ['f', 'g']


@pytest.mark.parametrize('move', [False, True], ids=['mirror', 'move'])
def test_rme_after_a_rename_in_case_only_keeps_the_directory(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, move):
	_filled(tmp_path / 'a/Dir/f', 'f')
	_filled(tmp_path / 'a/keep', 'k')
	monkeypatch.chdir(tmp_path)
	with _folding_directory(tmp_path, 'ext4') as fold:
		assert _mirror(hpcp_mod, monkeypatch, ['a/'], fold / 'out') == 0
		os.rename(tmp_path / 'a/Dir', tmp_path / 'a/dir')
		assert _mirror(hpcp_mod, monkeypatch, ['a/'], fold / 'out', remove_force=move) == 0
		assert (fold / 'out/dir/f').read_text() == 'f'


def test_rme_after_a_file_is_renamed_in_case_only_keeps_it_and_still_cleans(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	_filled(tmp_path / 'a/Photo.JPG', 'p')
	monkeypatch.chdir(tmp_path)
	with _folding_directory(tmp_path, 'ext4') as fold:
		assert _mirror(hpcp_mod, monkeypatch, ['a/'], fold / 'out') == 0
		os.rename(tmp_path / 'a/Photo.JPG', tmp_path / 'a/photo.jpg')
		_filled(fold / 'out/stale', 's')
		assert _mirror(hpcp_mod, monkeypatch, ['a/'], fold / 'out') == 0
		assert (fold / 'out/photo.jpg').read_text() == 'p' and not (fold / 'out/stale').exists()


def test_an_image_that_folds_is_asked(tmp_path):
	"""hpcp a/ -di x.img on a vfat image with a/X and a/x lost one."""
	mkfs = shutil.which('mkfs.vfat') or shutil.which('/usr/sbin/mkfs.vfat') or shutil.which('/sbin/mkfs.vfat')
	if os.geteuid() != 0 or not mkfs:
		pytest.skip('needs root and mkfs.vfat')
	_filled(tmp_path / 'a/X', 'upper')
	_filled(tmp_path / 'a/x', 'lower')
	image = tmp_path / 'x.img'
	image.write_bytes(b'')
	os.truncate(str(image), 32 << 20)
	if subprocess.run([mkfs, str(image)], capture_output=True).returncode:
		pytest.skip('mkfs.vfat refused')
	task = _hpcp(['a/', '-di', 'x.img'], tmp_path)
	assert task.returncode == 213 and b'where the destination folds case' in task.stdout, task.stdout[-2000:]


def test_a_full_disk_says_so(tmp_path):
	"""'Copy failed' said only 'return code 1'."""
	(tmp_path / 'big').write_bytes(os.urandom(1 << 20))
	out = tmp_path / 'out'
	out.mkdir()
	if os.geteuid() != 0:
		pytest.skip('needs root to mount')
	if subprocess.run(['mount', '-t', 'tmpfs', '-o', 'size=64k', 'hpcp-test', str(out)], capture_output=True).returncode:
		pytest.skip('mount failed')
	try:
		task = subprocess.run([sys.executable, HPCP, 'big', '-d', str(out) + os.sep], cwd=str(tmp_path), stdin=subprocess.DEVNULL,
			stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
		assert task.returncode == 190 and b'No space left on device' in task.stdout, task.stdout[-2000:]
	finally:
		subprocess.run(['umount', '-l', str(out)], capture_output=True)


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


# ---- 9.77: re-review of 1504353..499c981 ----

def test_remove_extra_dirs_keeps_a_directory_on_another_device_it_cannot_open(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, capsys):
	"""Re-review: the parent's lstat device check went with 9.77's check on entering, so a directory on another device
	that the walk could not open (EACCES, EIO, a mount that failed) stayed extra: an empty btrfs subvolume (not a mount
	point) would have been rmdir'd. One not opened is checked by its lstat, as in 9.76."""
	src, dst = tmp_path / 'src', tmp_path / 'dst'
	src.mkdir()
	dst.mkdir()
	with _mounted(dst / 'sub'):
		monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda *args, **kwargs: [])
		real = hpcp_mod._enter_directory
		def enter(path, device):
			if os.path.basename(path) == 'sub':
				raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), path)
			return real(path, device)
		monkeypatch.setattr(hpcp_mod, '_enter_directory', enter)
		removed = []
		real_rmdir = os.rmdir
		monkeypatch.setattr(hpcp_mod.os, 'rmdir', lambda path, *args, **kwargs: removed.append(path) or real_rmdir(path, *args, **kwargs))
		hpcp_mod.remove_extra_dirs([str(src) + os.sep], [str(dst) + os.sep])
		assert (dst / 'sub').is_dir()
	assert 'Deleting extra directory: ' + str(dst / 'sub') not in capsys.readouterr().out.splitlines()
	assert 'sub' not in removed and str(dst / 'sub') not in removed


@pytest.mark.parametrize('how', ['by name', 'on entering'])
def test_remove_extra_dirs_does_not_offer_the_directories_holding_a_mount_point(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, capsys, how):
	"""Re-review: the directories above a mount point were still offered ('Deleting extra directory'), then kept as
	non-empty: they hold another filesystem."""
	src, dst = tmp_path / 'src', tmp_path / 'dst'
	src.mkdir()
	(dst / 'x').mkdir(parents=True)
	(dst / 'stale').mkdir()
	with _mounted(dst / 'x' / 'mnt'):
		if how == 'on entering':
			monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda *args, **kwargs: [])
			_stub_until_opened(hpcp_mod, monkeypatch, dst / 'x' / 'mnt', os.stat(str(dst)).st_dev)
		hpcp_mod.remove_extra_dirs([str(src) + os.sep], [str(dst) + os.sep])
		assert (dst / 'x' / 'mnt').is_dir()
	output = capsys.readouterr().out
	assert 'Deleting extra directory: ' + str(dst / 'x') not in output.splitlines()
	assert 'Keeping non-empty directory' not in output
	assert not (dst / 'stale').exists()


def test_parallel_listing_keeps_a_folder_whose_listing_crashed(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Re-review: a folder whose listing task crashed (its default result) was dropped from folders too; kept, as in
	9.76 (its 'Listing error' is recorded)."""
	src = tmp_path / 'src'
	(src / 'boom').mkdir(parents=True)
	(src / 'fine').mkdir()
	boom = os.path.join(str(src), 'boom')
	real = hpcp_mod._task_result
	def task_result(future, default, crash_prefix):
		result = real(future, default, crash_prefix)
		if result is not default and boom in result[3]:
			hpcp_mod.eprint(f'{crash_prefix}: simulated crash')
			return default
		return result
	monkeypatch.setattr(hpcp_mod, '_task_result', task_result)
	_, _, _, folders = hpcp_mod.get_file_list(str(src), max_workers=2, parallel_file_listing=True)
	assert boom in folders and os.path.join(str(src), 'fine') in folders


def test_remove_extra_dirs_does_not_lstat_each_directory(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, scan_mode):
	"""Re-review: the parent's loop still lstat'ed each directory for a link (a round trip each on Lustre); the walk's
	d_type says it."""
	src, dst = tmp_path / 'src', tmp_path / 'dst'
	for index in range(20):
		(src / ('d%d' % index)).mkdir(parents=True)
		(dst / ('d%d' % index)).mkdir(parents=True)
	os.symlink(str(tmp_path), str(dst / 'link'))
	calls = []
	real_lstat = os.lstat
	monkeypatch.setattr(hpcp_mod.os, 'lstat', lambda path, *args, **kwargs: calls.append(path) or real_lstat(path, *args, **kwargs))
	hpcp_mod.remove_extra_dirs([str(src) + os.sep], [str(dst) + os.sep])
	assert [path for path in calls if isinstance(path, str) and path.startswith(str(dst) + os.sep)] == []
	assert (dst / 'link').is_symlink() and len(list(dst.iterdir())) == 21


@pytest.mark.parametrize('scan_mode', ['path'], indirect=True)
def test_a_directory_swapped_after_it_is_opened_is_not_read_by_its_path(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, scan_mode):
	"""Re-review: without scandir on a descriptor or /proc (Python 3.6 off Linux) the held directory was read by its
	path: one swapped for a symlink after the open had the link's target listed as if below the destination."""
	dst, outside = tmp_path / 'dst', tmp_path / 'outside'
	(dst / 'sub').mkdir(parents=True)
	outside.mkdir()
	(outside / 'secret.txt').write_text('x')
	boundary = hpcp_mod._filesystem_boundary(str(dst), str(dst), visible=frozenset())
	real = hpcp_mod._enter_directory
	def enter(path, device):
		fd = real(path, device)
		if os.path.basename(path) == 'sub':
			os.rename(path, path + '.away')
			os.symlink(str(outside), path)
		return fd
	monkeypatch.setattr(hpcp_mod, '_enter_directory', enter)
	files, _, _, _ = hpcp_mod.get_file_list_serial(str(dst), boundary=boundary)
	assert not any('secret.txt' in path for path in files)
	assert any(error.startswith('Listing error: ' + str(dst / 'sub') + ':') for error in hpcp_mod.ERRORS)


# ---- 9.83, review round 2 ----

@pytest.mark.parametrize('kind', ['ext4', 'f2fs', 'vfat', 'vfat-utf8', 'exfat', 'ntfs'])
@pytest.mark.parametrize('layout', [('X', 'x'), ('d/X', 'd/x'), ('d/X', 'D/x')], ids=['file', 'in-a-directory', 'directory-in-another-case'])
def test_a_move_that_renames_a_file_in_case_only_removes_nothing(tmp_path, kind, layout):
	"""It was exit 0 and the file was gone: the copy skipped it as identical, and the move removed the only copy."""
	source, dest = layout
	with _folding_directory(tmp_path, kind) as fold:
		_filled(fold / source, 'data')
		task = _hpcp([source, '-d', dest, '-rf'], fold)
		assert task.returncode == 241 and (fold / source).read_text() == 'data', task.stdout[-2000:]
		assert [name for name in os.listdir(fold) if name.startswith('.hpcp')] == []


def test_a_rename_in_case_on_a_directory_that_keeps_both_names_still_moves(tmp_path):
	with _folding_directory(tmp_path, 'ext4') as fold:
		plain = fold.parent / 'plain'
		plain.mkdir()
		_filled(plain / 'X', 'data')
		task = _hpcp(['X', '-d', 'x', '-rf'], plain)
		assert task.returncode == 0 and os.listdir(plain) == ['x'], task.stdout[-2000:]


@utf8_names
@pytest.mark.parametrize('kind', ['ext4', 'f2fs', 'vfat', 'vfat-utf8', 'exfat', 'ntfs'])
@pytest.mark.parametrize('environment', [{}, {'LC_ALL': 'C', 'PYTHONUTF8': '0', 'PYTHONCOERCECLOCALE': '0'}], ids=['locale', 'ascii'])
def test_a_pair_that_only_the_filesystem_folds_is_caught_whatever_the_locale(tmp_path, kind, environment):
	"""\u00c9 and \u00e9 under the C locale of Python 3.6 (or forced on a newer one) are escapes that nothing folds: exit 0, both
	sources removed, one content lost."""
	_filled(tmp_path / 'a' / '\u00c9', 'upper')
	_filled(tmp_path / 'b' / '\u00e9', 'lower')
	with _folding_directory(tmp_path, kind) as fold:
		try:
			(fold / '\u00c9').write_text('')
		except OSError:
			pytest.skip(f'{kind} takes no such name')
		try:
			os.close(os.open(str(fold / '\u00e9'), os.O_WRONLY | os.O_CREAT | os.O_EXCL))
			(fold / '\u00e9').unlink()
			one = False
		except FileExistsError:
			one = True
		(fold / '\u00c9').unlink()
		if not one:
			pytest.skip(f'{kind} keeps these two names apart')
		task = subprocess.run([sys.executable, HPCP, 'a/', 'b/', '-d', str(fold / 'out') + os.sep, '-rf'], cwd=str(tmp_path), stdin=subprocess.DEVNULL,
			stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=dict(os.environ, **environment))
		assert task.returncode == 213 and len(os.listdir(fold / 'out')) == 1, task.stdout[-2000:]
	assert (tmp_path / 'a' / '\u00c9').exists() and (tmp_path / 'b' / '\u00e9').exists()


@pytest.mark.parametrize('kind', ['vfat', 'vfat-utf8', 'exfat'])
@pytest.mark.parametrize('pair', [('a/x.', 'b/x'), ('a/d./f', 'b/d/f')], ids=['file', 'directory'])
def test_names_a_trailing_dot_apart_are_one_name_on_fat_and_exfat(tmp_path, kind, pair):
	"""a/x. and b/x: exit 0, both sources removed, one content gone."""
	_filled(tmp_path / pair[0], 'one')
	_filled(tmp_path / pair[1], 'two')
	with _folding_directory(tmp_path, kind) as fold:
		task = _hpcp(['a/', 'b/', '-d', str(fold / 'out') + os.sep, '-rf'], tmp_path)
		assert task.returncode == 213 and b'are one name where the destination folds case' in task.stdout, task.stdout[-2000:]
	assert (tmp_path / pair[0]).exists() and (tmp_path / pair[1]).exists()


def test_names_a_trailing_dot_apart_are_two_names_where_the_directory_keeps_them_apart(tmp_path):
	_filled(tmp_path / 'a/x.', 'one')
	_filled(tmp_path / 'b/x', 'two')
	with _folding_directory(tmp_path, 'ext4') as fold:
		task = _hpcp(['a/', 'b/', '-d', str(fold / 'out') + os.sep, '-rf'], tmp_path)
		assert task.returncode == 0 and sorted(os.listdir(fold / 'out')) == ['x', 'x.'], task.stdout[-2000:]


@pytest.mark.parametrize('kind', ['ext4', 'f2fs', 'vfat', 'vfat-utf8', 'exfat', 'ntfs'])
def test_two_directories_in_another_case_clash_on_every_run_and_keep_the_first(tmp_path, kind):
	"""The first run was 213 with D/f from a/; the second, with nothing changed, exit 0 with D/f from b/."""
	_filled(tmp_path / 'a/D/f', 'one')
	_filled(tmp_path / 'b/d/f', 'two')
	with _folding_directory(tmp_path, kind) as fold:
		for run_count in range(3):
			task = _hpcp(['a/', 'b/', '-d', str(fold / 'out') + os.sep], tmp_path)
			assert task.returncode == 213, (run_count, task.stdout[-2000:])
			assert (fold / 'out/D/f').read_text() == 'one'


@pytest.mark.parametrize('kind', ['ext4', 'vfat', 'ntfs'])
def test_one_tree_with_two_spellings_onto_an_existing_directory_clashes(tmp_path, kind):
	"""a/D/f and a/d/f onto an existing out/D: exit 0, and one file never landed."""
	_filled(tmp_path / 'a/D/f', 'one')
	_filled(tmp_path / 'a/d/f', 'two')
	with _folding_directory(tmp_path, kind) as fold:
		(fold / 'out/D').mkdir(parents=True)
		task = _hpcp(['a/', '-d', str(fold / 'out') + os.sep, '-rf'], tmp_path)
		assert task.returncode == 213 and (fold / 'out/D/f').read_text() == 'one', task.stdout[-2000:]
	assert (tmp_path / 'a/D/f').exists() and (tmp_path / 'a/d/f').exists()


def test_a_rerun_with_both_names_there_is_not_a_clash_on_a_read_only_mount(tmp_path):
	"""Two names on a case-sensitive directory below a read-only mount: nothing can be made there to ask, and nothing needs to be."""
	if os.geteuid() != 0:
		pytest.skip('needs root to mount')
	_filled(tmp_path / 'a/s/X', '1')
	_filled(tmp_path / 'a/s/x', '2')
	assert _hpcp(['a/', '-d', 'out/'], tmp_path).returncode == 0
	if subprocess.run(['mount', '--bind', str(tmp_path / 'out'), str(tmp_path / 'out')], capture_output=True).returncode:
		pytest.skip('bind mount failed')
	try:
		if subprocess.run(['mount', '-o', 'remount,bind,ro', str(tmp_path / 'out')], capture_output=True).returncode:
			pytest.skip('read-only remount failed')
		task = _hpcp(['a/', '-d', 'out/'], tmp_path)
		# The copy itself cannot write there (193, as 9.82 ends); it is no clash.
		assert task.returncode == 193 and b'clash' not in task.stdout and b'Read-only file system' in task.stdout, task.stdout[-2000:]
	finally:
		subprocess.run(['umount', '-l', str(tmp_path / 'out')], capture_output=True)


@pytest.mark.parametrize('move', [False, True], ids=['mirror', 'move'])
def test_rme_after_a_rename_in_case_only_keeps_an_empty_directory_and_does_not_announce_it(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, capsys, move):
	(tmp_path / 'a/Empty').mkdir(parents=True)
	(tmp_path / 'a/P/Q').mkdir(parents=True)
	_filled(tmp_path / 'a/k/f', 'f')
	monkeypatch.chdir(tmp_path)
	with _folding_directory(tmp_path, 'ext4') as fold:
		assert _mirror(hpcp_mod, monkeypatch, ['a/'], fold / 'out') == 0
		os.rename(tmp_path / 'a/Empty', tmp_path / 'a/empty')
		os.rename(tmp_path / 'a/P', tmp_path / 'a/p')
		capsys.readouterr()
		assert _mirror(hpcp_mod, monkeypatch, ['a/'], fold / 'out', remove_force=move) == 0
		assert 'Deleting extra directory' not in capsys.readouterr().out
		assert sorted(os.listdir(fold / 'out')) == ['Empty', 'P', 'k'] and os.listdir(fold / 'out/P') == ['Q']
