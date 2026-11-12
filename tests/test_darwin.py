'''9.87 on macOS (the temporary macOS job, and any Mac later): cp writes the file hpcp holds through /dev/fd/N (H: rewrites,
L1: first copies, L2: the beside route's new file), a move asks each destination filesystem once for F_FULLFSYNC (M4),
multiCMD's pass_fds through hpcp's embedded copy (E), and a CLI smoke run on the runner's case-insensitive APFS. Skipped off
macOS; Linux's held route is in test_held.py.'''
import contextlib
import io
import itertools
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time

import pytest

import hpcp

pytestmark = pytest.mark.skipif(sys.platform != 'darwin', reason='macOS: /dev/fd/N, F_FULLFSYNC, APFS')

HPCP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hpcp.py')
_HELD_PREFIX = '/dev/fd/'
_RUNS = itertools.count()


def write(path, text, mode=None):
	path = str(path)
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'w') as handle:
		handle.write(text)
	if mode is not None:
		os.chmod(path, mode)
	return path


def read(path):
	with open(str(path)) as handle:
		return handle.read()


@pytest.fixture
def root(tmp_path):
	'''The test's directory by its real path: macOS's temporary directory is below /var, a link to /private/var.'''
	return os.path.realpath(str(tmp_path))


@pytest.fixture
def fresh(monkeypatch, reset_hpcp_globals):
	'''Each test asks cp anew (the held route) and leaves nothing behind.'''
	monkeypatch.setattr(hpcp, '_binPaths', {})
	monkeypatch.setattr(hpcp, '_badOverrides', {})
	monkeypatch.setattr(hpcp, '_SPARSE', [None, False])
	monkeypatch.setattr(hpcp, '_HELD', [None, None])
	hpcp.check_path.cache_clear()
	yield hpcp
	hpcp.check_path.cache_clear()


def _copy_file(monkeypatch, src, dest):
	'''copy_file of one source to one landing, its folder named by the operator.'''
	monkeypatch.setattr(hpcp, 'NAMED_DESTS', frozenset([os.path.dirname(os.path.abspath(str(dest)))]))
	monkeypatch.setattr(hpcp, 'RUN_ID', 'darwin-%d' % next(_RUNS))
	with contextlib.redirect_stdout(io.StringIO()):
		return hpcp.copy_file(str(src), [str(dest)])


def _named(target):
	'''What cp's destination argument leads to: the held file's path for /dev/fd/N (F_GETPATH, asked while cp runs).'''
	if not target.startswith(_HELD_PREFIX):
		return target
	import fcntl
	return os.fsdecode(fcntl.fcntl(int(target[len(_HELD_PREFIX):]), fcntl.F_GETPATH, bytes(1024)).rstrip(b'\0'))


def _record_cp(monkeypatch, before=None, after=None):
	'''Each cp the copy runs, as (its destination argument, what that leads to, pass_fds); before(named) and after(named)
	run around it.'''
	seen = []
	real = hpcp.run_command_in_multicmd_with_path_check
	def cp(command, *args, **kwargs):
		named = _named(command[-1])
		seen.append((command[-1], named, tuple(kwargs.get('pass_fds', ()))))
		if before is not None:
			before(named)
		result = real(command, *args, **kwargs)
		if after is not None:
			after(named)
		return result
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check', cp)
	return seen


def _held_or_fail():
	route = hpcp._held_route()
	assert route in ('c', 'cb'), route


def _xattrs(path):
	'''xattr -l's listing of path (names and values), '' with none.'''
	return subprocess.run(['xattr', '-l', path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
		universal_newlines=True).stdout


def _acl(path):
	'''path's ACL entries, as ls -le lists them under its line.'''
	lines = subprocess.run(['ls', '-led', path], stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout.splitlines()
	return lines[1:]


# ---- H: the probe

def test_the_probe_answers_and_leaves_nothing(tmp_path, fresh, monkeypatch):
	# Apple's cp opens /dev/fd/N (a dup, which ignores O_TRUNC) and copyfile truncates after its copy: 'c'; a cp whose write
	# leaves an old tail gets 'cb', and hpcp empties the held file first.
	probes = str(tmp_path / 'probe-tmp')
	os.makedirs(probes)
	monkeypatch.setattr(tempfile, 'tempdir', probes)
	route = hpcp._held_route()
	print('held route:', route, 'cp:', shutil.which('cp'))
	assert route in ('c', 'cb') and os.listdir(probes) == []


@pytest.mark.parametrize('case', ['first', 'rewrite', 'beside'])
def test_cp_writes_what_hpcp_holds_through_dev_fd(root, fresh, monkeypatch, case):
	_held_or_fail()
	src = write(os.path.join(root, 'a/x'), 'new content')
	landing = os.path.join(root, 'out/x')
	if case != 'first':
		write(landing, 'old')
	monkeypatch.setattr(hpcp, 'WRITE_BESIDE', case == 'beside')
	seen = _record_cp(monkeypatch)
	_copy_file(monkeypatch, src, landing)
	assert read(landing) == 'new content' and not hpcp.ERRORS and len(seen) == 1, (seen, hpcp.ERRORS)
	target, named, fds = seen[0]
	assert target == _HELD_PREFIX + str(fds[0]), seen
	if case == 'beside':
		assert os.path.dirname(named) == os.path.join(root, 'out') and os.path.basename(named).startswith('.hpcp_tmp_'), seen
	else:
		assert named == landing, seen


# ---- H: a rewrite in place ends as cp by name ends it

def _decorated(path):
	'''path with what cp -a carries on macOS: a mode, times, a flag (hidden), an xattr and an ACL entry.'''
	os.chmod(path, 0o640)
	subprocess.run(['xattr', '-w', 'org.example.hpcp', 'kept', path], check=True)
	subprocess.run(['chmod', '+a', 'everyone deny delete', path], check=True)
	os.chflags(path, stat.UF_HIDDEN)
	os.utime(path, ns=(1600000000 * 10 ** 9, 1600000000 * 10 ** 9 + 123456789))


def _state(path):
	st = os.lstat(path)
	return (oct(stat.S_IMODE(st.st_mode)), st.st_size, st.st_mtime_ns, st.st_flags, _xattrs(path), _acl(path), read(path))


@pytest.mark.parametrize('old, new', [('o' * 100, 'new'), ('o', 'n' * 100)], ids=['old-longer', 'new-longer'])
def test_a_rewrite_in_place_ends_as_cp_by_name_ends_it(root, fresh, monkeypatch, old, new):
	# Content, size and inode (in place), and mode, times, flags, xattrs and ACL as the same rewrite with the held route off.
	_held_or_fail()
	src = write(os.path.join(root, 'a/x'), new)
	_decorated(src)
	seen = _record_cp(monkeypatch)
	states = {}
	for route in ('held', 'named'):
		landing = write(os.path.join(root, route, 'x'), old, 0o600)
		inode = os.lstat(landing).st_ino
		if route == 'named':
			monkeypatch.setattr(hpcp, '_held_route', lambda: None)
		_copy_file(monkeypatch, src, landing)
		assert os.lstat(landing).st_ino == inode and not hpcp.ERRORS, (route, hpcp.ERRORS)
		states[route] = _state(landing)
	assert seen[0][0].startswith(_HELD_PREFIX) and not seen[1][0].startswith(_HELD_PREFIX), seen
	assert states['held'] == states['named'], states
	assert states['held'][6] == new and 'org.example.hpcp' in states['held'][4], states


# ---- L1: a first copy ends as cp by name makes it

_MODES = (0o644, 0o600, 0o755, 0o444, 0o000)


def _outcome(path):
	try:
		st = os.lstat(path)
	except FileNotFoundError:
		return None
	return oct(stat.S_IMODE(st.st_mode)), st.st_uid, st.st_gid, st.st_flags, _acl(path), read(path)


@pytest.mark.parametrize('umask', [0o022, 0o077], ids=['022', '077'])
@pytest.mark.parametrize('content_only', [False, True], ids=['plain', 'content-only'])
def test_a_first_copy_ends_as_cp_by_name_makes_it(root, fresh, monkeypatch, content_only, umask):
	# The modes a runner (not root) can make; a 0000 source cannot be read, and fails both ways leaving nothing.
	_held_or_fail()
	monkeypatch.setattr(hpcp, 'CONTENT_ONLY', content_only)
	sources = [write(os.path.join(root, 'src', 'm%04o' % mode), 'mode %o' % mode, mode) for mode in _MODES]
	seen = _record_cp(monkeypatch)
	old = os.umask(umask)
	try:
		for path in sources:
			_copy_file(monkeypatch, path, os.path.join(root, 'held', os.path.basename(path)))
		monkeypatch.setattr(hpcp, '_held_route', lambda: None)
		for path in sources:
			_copy_file(monkeypatch, path, os.path.join(root, 'named', os.path.basename(path)))
	finally:
		os.umask(old)
		os.chmod(sources[-1], 0o600)
	assert any(target.startswith(_HELD_PREFIX) for target, _, _ in seen), seen
	cells = {os.path.basename(path): (_outcome(os.path.join(root, 'held', os.path.basename(path))),
		_outcome(os.path.join(root, 'named', os.path.basename(path)))) for path in sources}
	assert all(held == named for held, named in cells.values()), {name: cell for name, cell in cells.items() if cell[0] != cell[1]}
	assert cells['m0644'][0] is not None and cells['m0000'] == (None, None), cells


# ---- names swapped mid-copy: the victim is never written

@pytest.mark.parametrize('case', ['link-at-a-first-copy', 'link-at-a-rewrite', 'hard-link-at-the-beside-temp'])
def test_a_name_swapped_mid_copy_never_leads_the_copy_elsewhere(root, fresh, monkeypatch, case):
	# A link put at the landing's name after hpcp opened it (a first copy's new file, a rewrite's landing): cp writes the
	# file hpcp holds, now under the other name. The beside route's new file swapped for a second name of the victim once
	# cp wrote it: the rename finds the name leads elsewhere, 190, the old copy kept.
	_held_or_fail()
	victim = write(os.path.join(root, 'victim/secret'), 'victim', 0o600)
	subprocess.run(['xattr', '-w', 'org.example.keep', '1', victim], check=True)
	before = os.lstat(victim)
	src = write(os.path.join(root, 'src/f'), 'NEW-CONTENT', 0o600)
	landing = os.path.join(root, 'out/f')
	os.makedirs(os.path.dirname(landing))
	if case != 'link-at-a-first-copy':
		write(landing, 'old')
	monkeypatch.setattr(hpcp, 'WRITE_BESIDE', case == 'hard-link-at-the-beside-temp')
	def swap(named):
		if case != 'hard-link-at-the-beside-temp':
			os.rename(landing, landing + '.away')
			os.symlink(victim, landing)
	def link(named):
		if case == 'hard-link-at-the-beside-temp' and '.hpcp_tmp_' in named:
			os.rename(named, named + '.away')
			os.link(victim, named)
	seen = _record_cp(monkeypatch, before=swap, after=link)
	_copy_file(monkeypatch, src, landing)
	after = os.lstat(victim)
	assert read(victim) == 'victim' and (after.st_mode, after.st_uid, after.st_gid) == (before.st_mode, before.st_uid, before.st_gid)
	assert 'org.example.keep' in _xattrs(victim) and seen[0][0].startswith(_HELD_PREFIX), seen
	if case == 'hard-link-at-the-beside-temp':
		assert read(landing) == 'old', read(landing)
		assert any('something else was put at its name' in error for error in hpcp.ERRORS), hpcp.ERRORS
	else:
		assert read(landing + '.away') == 'NEW-CONTENT', hpcp.ERRORS


# ---- M4: one F_FULLFSYNC per destination filesystem in a move

@contextlib.contextmanager
def _ram_volume():
	'''A second filesystem, an APFS volume on a 64 MiB RAM disk mounted under /Volumes, or None where the runner lets the
	job make none.'''
	attach = subprocess.run(['hdiutil', 'attach', '-nomount', 'ram://131072'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
		universal_newlines=True)
	device = attach.stdout.split()[0] if attach.returncode == 0 and attach.stdout.split() else None
	if device is None:
		print('no RAM disk:', attach.stdout.strip())
		yield None
		return
	try:
		name = 'hpcp987ram%d' % os.getpid()
		erase = subprocess.run(['diskutil', 'erasevolume', 'APFS', name, device], stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT, universal_newlines=True)
		mount = os.path.join('/Volumes', name)
		if erase.returncode != 0 or not os.path.ismount(mount):
			print('no APFS volume on the RAM disk:', erase.stdout.strip())
			yield None
		else:
			yield os.path.realpath(mount)
	finally:
		subprocess.run(['hdiutil', 'detach', device, '-force'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def test_a_move_asks_each_destination_filesystem_once_for_a_full_flush(root, fresh, monkeypatch):
	import fcntl
	with _ram_volume() as volume:
		dests = [os.path.join(root, 'out')] + ([os.path.join(volume, 'out')] if volume else [])
		print('destinations:', dests)
		src = os.path.join(root, 'a')
		# Several -d spill: each file goes to one destination. Picked at random (-rds), 40 files land on both filesystems
		# (all on one: 2 in 2^40).
		files = {os.path.join('d%d' % (i % 4), 'f%d' % i): 'content %d' % i for i in range(40)}
		for name, text in files.items():
			write(os.path.join(src, name), text)
		events = []
		real_fsync, real_fcntl, real_remove = os.fsync, fcntl.fcntl, hpcp.remove_moved_sources
		def fsync(fd):
			events.append(('fsync',))
			return real_fsync(fd)
		def full(fd, command, *args):
			if command != fcntl.F_FULLFSYNC:
				return real_fcntl(fd, command, *args)
			start = time.monotonic()
			result = real_fcntl(fd, command, *args)
			events.append(('full', os.fstat(fd).st_dev, result, time.monotonic() - start))
			return result
		def remove(*args, **kwargs):
			events.append(('remove',))
			return real_remove(*args, **kwargs)
		monkeypatch.setattr(hpcp.os, 'fsync', fsync)
		monkeypatch.setattr(fcntl, 'fcntl', full)
		monkeypatch.setattr(hpcp, 'remove_moved_sources', remove)
		with contextlib.redirect_stdout(io.StringIO()):
			rc = hpcp.hpcp([src + os.sep], dest_paths=[dest + os.sep for dest in dests], single_thread=True, remove=True,
				remove_force=True, random_destination_selection=True)
		monkeypatch.setattr(fcntl, 'fcntl', real_fcntl)
		monkeypatch.setattr(hpcp.os, 'fsync', real_fsync)
		fulls = [event for event in events if event[0] == 'full']
		for _, dev, result, seconds in fulls:
			print('F_FULLFSYNC on st_dev %d: returned %r in %.6f s' % (dev, result, seconds))
		kinds = [event[0] for event in events]
		assert rc == 0 and not os.path.exists(src), (rc, hpcp.ERRORS)
		landed = {name: [dest for dest in dests if os.path.exists(os.path.join(dest, name))] for name in files}
		assert all(len(where) == 1 and read(os.path.join(where[0], name)) == files[name] for name, where in landed.items()), landed
		assert len({where[0] for where in landed.values()}) == len(dests), landed
		assert sorted(dev for _, dev, _, _ in fulls) == sorted({os.stat(dest).st_dev for dest in dests}), events
		assert all(result == 0 for _, _, result, _ in fulls), fulls
		last_fsync = max(i for i, kind in enumerate(kinds) if kind == 'fsync')
		assert last_fsync < kinds.index('full') and max(i for i, kind in enumerate(kinds) if kind == 'full') < kinds.index('remove'), kinds


# ---- E: pass_fds through hpcp's embedded multiCMD

def test_a_command_reads_a_passed_descriptor_through_hpcp_s_multicmd():
	r, w = os.pipe()
	try:
		os.write(w, b'through\n')
		os.close(w)
		task = hpcp.multiCMD.run_commands([['cat', '/dev/fd/%d' % r]], quiet=True, return_object=True, pass_fds=(r,))[0]
	finally:
		os.close(r)
	print('multiCMD', hpcp.multiCMD.version)
	assert (task.returncode, task.stdout) == (0, ['through']), (task.returncode, task.stdout, task.stderr)


# ---- a CLI smoke run on the runner's disk (APFS, case-insensitive)

def test_the_cli_copies_reruns_and_moves(tmp_path):
	write(tmp_path / 'src/a.txt', 'a')
	write(tmp_path / 'src/Sub/b.txt', 'b')
	def run(*args):
		task = subprocess.run([sys.executable, HPCP] + list(args), cwd=str(tmp_path), stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT, universal_newlines=True, timeout=300)
		assert task.returncode == 0, task.stdout[-3000:]
	run('src/', '-d', 'dst/')
	run('src/', '-d', 'dst/')
	assert read(tmp_path / 'dst/a.txt') == 'a' and read(tmp_path / 'dst/Sub/b.txt') == 'b'
	run('src/', '-d', 'moved/', '-rf')
	assert read(tmp_path / 'moved/Sub/b.txt') == 'b' and not os.path.exists(str(tmp_path / 'src/a.txt'))


# ---- 9.88 (macos988: added in 9.88 and not yet run on macOS; select them with -m macos988)

@pytest.mark.macos988
@pytest.mark.parametrize('handler', ['hpcp', 'python'])
def test_a_stop_anywhere_in_the_probe_leaves_nothing_on_macos(tmp_path, fresh, monkeypatch, stop_alarms, handler):
	# S6: macOS has no O_TMPFILE, so the probe's held file is made with a name and unlinked at once. Each of the probe's files
	# is held by C as it is made (_create), and each name goes whatever a stop does (9.87 used tempfile, whose own files a stop
	# could leave). cp is left out: the runner answers at once.
	probes = os.path.realpath(str(tmp_path / 'probe-tmp'))
	os.makedirs(probes)
	monkeypatch.setattr(tempfile, 'tempdir', probes)
	class Done:
		returncode = 0
	monkeypatch.setattr(hpcp.multiCMD, 'run_commands', lambda *args, **kwargs: [Done()])
	def call(index, got):
		hpcp._HELD[:] = [None, None]
		got.extend(map(lambda _: hpcp._held_route() or True, (None,)))
	stopped = stop_alarms(handler, call, 2000)
	assert stopped and os.listdir(probes) == [], (len(stopped), os.listdir(probes)[:10])


@pytest.mark.macos988
def test_a_link_swapped_at_an_identical_fifo_gets_no_owner_or_mode(root, fresh, monkeypatch):
	# N1: _settle_identical sets an identical FIFO's metadata by name (no descriptor for a node off Linux's /proc): a link put
	# at the name after its look took the source's mode (9.87's chmod followed it). lchmod and lchown now change the link.
	source, landing, victim = (os.path.join(root, name) for name in ('p', 'q', 'victim'))
	os.mkfifo(source, 0o644)
	os.mkfifo(landing, 0o600)
	write(victim, 'victim', mode=0o600)
	look = os.lstat(landing)
	real_lstat = os.lstat
	def swapped(path, *args, **kwargs):
		result = real_lstat(path, *args, **kwargs)
		if path == landing and stat.S_ISFIFO(result.st_mode):
			os.unlink(landing)
			os.symlink(victim, landing)  # after the look _settle_identical takes
		return result
	monkeypatch.setattr(os, 'lstat', swapped)
	hpcp._settle_identical(source, real_lstat(source), landing, look)
	monkeypatch.setattr(os, 'lstat', real_lstat)
	assert stat.S_IMODE(os.stat(victim).st_mode) == 0o600 and read(victim) == 'victim'


@pytest.mark.macos988
@pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() == 0, reason='root sets system flags')
def test_a_list_of_paths_below_a_directory_with_a_system_flag_copies_and_moves_as_non_root(root):
	# P2: a list-mode copy spells each source in full, so the copy syncs every directory on its way (/Users, /private) into the
	# destination, and their finish gave the copies the source's flags: sunlnk (SF_NOUNLINK) is root's to set, chflags was
	# EPERM for the whole call, "Directory sync error: /Users to out/Users: [Errno 1] Operation not permitted", exit 193, and
	# a move (-rf) never removed its sources. Now the user flags are set, the system ones left (_flags_to_set).
	parts = root.split(os.sep)
	flagged = [path for path in (os.sep.join(parts[:end]) or os.sep for end in range(1, len(parts)))
		if getattr(os.lstat(path), 'st_flags', 0) & ~0xffff]
	if not flagged:
		pytest.skip('no directory above the test directory has a system flag here')
	source = write(os.path.join(root, 'src/a/x.txt'), 'x')
	for args in (['-d', 'out'], ['-d', 'out'], ['-d', 'moved', '-rf']):
		task = subprocess.run([sys.executable, HPCP] + args, cwd=root, input=source + '\n', stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT, universal_newlines=True, timeout=300)
		assert task.returncode == 0 and 'Operation not permitted' not in task.stdout, (args, task.stdout[-3000:])
	assert read(os.path.join(root, 'out' + source)) == 'x' and read(os.path.join(root, 'moved' + source)) == 'x'
	assert not os.path.exists(source)


# ---- W1 (9.88) on macOS: the door holds read descriptors (no O_PATH, and /dev/fd/D/NAME names nothing below D)

def _victim(path):
	'''A directory a run must never write: (path, its state now). The runner's own here (no root on the Mac).'''
	os.makedirs(path)
	for name in ('x', 'y'):
		write(os.path.join(path, name), 'victim ' + name, 0o600)
	return path, _tree_state(path)


def _tree_state(path):
	found = {}
	for name in sorted(os.listdir(path)):
		st = os.lstat(os.path.join(path, name))
		found[name] = (st.st_mode, st.st_mtime_ns, st.st_ino, read(os.path.join(path, name)))
	st = os.lstat(path)
	return found, (st.st_mode, st.st_mtime_ns)


def _w1_run(monkeypatch, out):
	'''One run's settings: out named (-d), a run of its own (RUN_ID).'''
	monkeypatch.setattr(hpcp, 'NAMED_DESTS', frozenset([os.path.abspath(out)]))
	monkeypatch.setattr(hpcp, 'RUN_ID', 'w1-mac-%d' % next(_RUNS))


def _swap_for_link(directory, target):
	os.rename(directory, directory + '.away')
	os.symlink(target, directory)


@pytest.mark.macos988
def test_the_door_holds_read_descriptors_on_macos(root, fresh, monkeypatch):
	# macOS has no O_PATH, and /dev/fd/D/NAME is no path below the directory D holds (ENOENT or ENOTDIR, macOS 26): the door
	# holds read descriptors, and every call goes relative to them (dir_fd), or by name where Python has no such form.
	out = os.path.join(root, 'out')
	os.makedirs(os.path.join(out, 'a'))
	write(os.path.join(out, 'a/x'), 'x')
	assert hpcp._PATH_FDS is False
	_w1_run(monkeypatch, out)
	base = hpcp._dest_directory(os.path.join(out, 'a'), make=False)
	try:
		assert isinstance(base, int)
		import fcntl
		assert fcntl.fcntl(base, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
		assert os.lstat('x', dir_fd=base).st_size == 1
		with pytest.raises(OSError):
			os.stat('/dev/fd/%d/x' % base)
	finally:
		hpcp._close_base(base)


@pytest.mark.macos988
@pytest.mark.parametrize('case', ['first', 'rewrite', 'beside'])
def test_w1_a_directory_swapped_after_the_door_never_leads_the_copy_elsewhere_on_macos(root, fresh, monkeypatch, case):
	# W1 case 1 on the read route: out/a is swapped for a link to victim/ after the door looked at it and before the file is
	# made, opened or renamed. 9.87 made, opened and renamed the landing by path, through the link.
	victim, before = _victim(os.path.join(root, 'victim'))
	out = os.path.join(root, 'out')
	os.makedirs(os.path.join(out, 'a'))
	src = write(os.path.join(root, 'src/a/x'), 'NEW-CONTENT')
	if case != 'first':
		write(os.path.join(out, 'a/x'), 'old')
	monkeypatch.setattr(hpcp, 'WRITE_BESIDE', case == 'beside')
	_w1_run(monkeypatch, out)
	real, done = hpcp._claim_device, []
	def swapping(*args, **kwargs):
		if not done:
			done.append(True)
			_swap_for_link(os.path.join(out, 'a'), victim)
		return real(*args, **kwargs)
	monkeypatch.setattr(hpcp, '_claim_device', swapping)
	with contextlib.redirect_stdout(io.StringIO()):
		hpcp.copy_file(src, [os.path.join(out, 'a/x')])
	assert done and os.path.islink(os.path.join(out, 'a'))
	assert _tree_state(victim) == before, (before, _tree_state(victim))
	assert read(os.path.join(out, 'a.away/x')) == 'NEW-CONTENT' and not hpcp.ERRORS, hpcp.ERRORS


@pytest.mark.macos988
def test_w1_a_directory_swapped_between_two_files_never_leads_the_second_elsewhere_on_macos(root, fresh, monkeypatch):
	# W1 case 2 on the read route: out/a/x is copied, out/a is swapped for a link to victim/, then out/a/y is copied. 9.87
	# looked at out/a once a process and wrote y by path, through the link.
	victim, before = _victim(os.path.join(root, 'victim'))
	out = os.path.join(root, 'out')
	os.makedirs(os.path.join(out, 'a'))
	for name in 'xy':
		write(os.path.join(root, 'src/a', name), 'NEW ' + name)
	_w1_run(monkeypatch, out)
	with contextlib.redirect_stdout(io.StringIO()):
		hpcp.copy_file(os.path.join(root, 'src/a/x'), [os.path.join(out, 'a/x')])
		_swap_for_link(os.path.join(out, 'a'), victim)
		hpcp.copy_file(os.path.join(root, 'src/a/y'), [os.path.join(out, 'a/y')])
	assert _tree_state(victim) == before, (before, _tree_state(victim))
	assert read(os.path.join(out, 'a.away/x')) == 'NEW x' and read(os.path.join(out, 'a.away/y')) == 'NEW y'
	assert not hpcp.ERRORS, hpcp.ERRORS
