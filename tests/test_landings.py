"""What is at a landing. 9.86: a regular file with one name that opens for writing is written in place by cp, through the
file hpcp holds open (on Linux cp gets /proc/PID/fd/N), so its inode stays and a link put at its name is never written
through; anything else there (a file with other names, one that does not open for writing, a FIFO, device node or link) is
replaced by a new file renamed over it once whole, never opened or written through, as every rewrite is under -wb. Only
what the operator named (-d) is followed. A stop runs the cleanup. A copy short of room is warned about, then tried."""
import contextlib
import errno
import os
import random
import re
import shutil
import signal
import stat
import subprocess
import sys
import time

import pytest

HPCP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hpcp.py')

posix_only = pytest.mark.skipif(os.name != 'posix', reason='POSIX file types, signals and permissions')
not_root = pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() == 0, reason='root writes every directory')
root_only = pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() != 0, reason='needs root')
# A small filesystem of its own, for the space cases: HPCP_TEST_SMALL_FS names a writable mount of 5 MiB (a Docker run
# gives one with --tmpfs /small:size=5m,mode=1777). Each test uses a directory of its own in it, and empties it.
SMALL_FS = os.environ.get('HPCP_TEST_SMALL_FS')


@pytest.fixture
def _small_fs_alone():
	"""One test at a time on the small filesystem, across pytest-xdist's workers: each fills it to leave a set room, which a
	test running beside it would take (two of them failed in each of two -n 8 runs)."""
	import fcntl
	with open(os.path.join(SMALL_FS, '.lock'), 'a') as lock:
		fcntl.flock(lock, fcntl.LOCK_EX)
		yield


def small_fs(test):
	return pytest.mark.skipif(not SMALL_FS, reason='HPCP_TEST_SMALL_FS: a 5 MiB filesystem of its own')(
		pytest.mark.usefixtures('_small_fs_alone')(test))


def run(args, cwd, stdin=b'', env=None, timeout=120, unset=()):
	"""The CLI in a session of its own: (return code, stdout and stderr as bytes); ('timeout', output) when it hung. A cp
	that hangs (on a FIFO, in 9.84) holds the pipe in a process group of its own: the run's group is killed, and the output is
	read only as far as is there."""
	task = subprocess.Popen([sys.executable, HPCP] + args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
		cwd=str(cwd), env={k: v for k, v in dict(os.environ, **(env or {})).items() if k not in unset}, start_new_session=True)
	try:
		output = task.communicate(stdin, timeout=timeout)[0]
	except subprocess.TimeoutExpired:
		os.killpg(task.pid, signal.SIGKILL)
		task.wait()
		task.stdout.close()
		return 'timeout', b''
	return task.returncode, output



def write(path, content='x'):
	path = str(path)
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'wb') as f:
		f.write(content if isinstance(content, bytes) else content.encode())


def _user_xattrs(path):
	# The user.* names only: a host with SELinux labels every file (security.selinux), the copy's own label not a carry.
	return sorted(name for name in os.listxattr(str(path)) if name.startswith('user.'))


def read(path):
	with open(str(path), 'rb') as f:
		return f.read().decode()


def read_bytes(path):
	with open(str(path), 'rb') as f:
		return f.read()


def temporaries(root):
	"""Every .hpcp_tmp_ name below root."""
	return [os.path.join(path, name) for path, dirs, names in os.walk(str(root)) for name in names + dirs if name.startswith('.hpcp_tmp_')]


# ---- entry 1: a source that fails part way; what goes beside, and what in place ----

def _cp_writing_a_part(hpcp_mod, monkeypatch):
	"""Every cp writes a part to the name it is given and fails, as a source failing on read in the middle does. A rewrite in
	place gives cp /proc/PID/fd/N, which opening writes the held file."""
	calls = []
	def cp(command, *args, **kwargs):
		calls.append(command[-1])
		with open(command[-1], 'w') as target:
			target.write('par')
		raise RuntimeError('cp: error reading: Input/output error')
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', cp)
	return calls


def _source_fails_on_read(hpcp_mod, monkeypatch):
	"""-rbw's read of the source fails (EIO), as a bad sector does."""
	read = []
	def fails(path):
		read.append(path)
		return OSError(errno.EIO, os.strerror(errno.EIO), path)
	monkeypatch.setattr(hpcp_mod, '_read_through', fails)
	return read


@posix_only
@pytest.mark.parametrize('mode', ['default', 'rbw', 'wb'])
@pytest.mark.parametrize('content_only', [False, True], ids=['plain', 'content-only'])
def test_a_source_that_fails_on_read_during_a_rewrite(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, content_only, mode):
	# Default: written in place as in 9.84, so the old copy, truncated first, goes with the torn file (the approved known
	# limitation), 190, the source kept. -rbw reads the source first and keeps the old copy and its inode; -wb writes beside
	# it and keeps it, with no temporary file left.
	write(tmp_path / 'a/x', 'the new and longer content')
	write(tmp_path / 'out/x', 'the last good copy')
	inode = os.stat(str(tmp_path / 'out/x')).st_ino
	monkeypatch.setattr(hpcp_mod, 'CONTENT_ONLY', content_only)
	monkeypatch.setattr(hpcp_mod, 'READ_FIRST', mode == 'rbw')
	monkeypatch.setattr(hpcp_mod, 'WRITE_BESIDE', mode == 'wb')
	monkeypatch.setattr(hpcp_mod, 'NAMED_DESTS', frozenset([str(tmp_path / 'out')]))
	calls = _cp_writing_a_part(hpcp_mod, monkeypatch)
	reads = _source_fails_on_read(hpcp_mod, monkeypatch) if mode == 'rbw' else None
	assert hpcp_mod.copy_file(str(tmp_path / 'a/x'), [str(tmp_path / 'out/x')])[0] == 0
	assert read_bytes(tmp_path / 'a/x') == b'the new and longer content'
	failed = [error for error in hpcp_mod.ERRORS if error.startswith('Copy failed')]
	assert len(failed) == 1 and hpcp_mod.get_rc_from_error() == 190
	if mode == 'default':
		assert not os.path.lexists(str(tmp_path / 'out/x')) and temporaries(tmp_path / 'out') == []
		assert 'Input/output error' in failed[0] and '/proc/' not in failed[0], failed
	else:
		assert read(tmp_path / 'out/x') == 'the last good copy' and os.stat(str(tmp_path / 'out/x')).st_ino == inode
		assert temporaries(tmp_path / 'out') == [] and 'leaving ' in failed[0], failed
	if mode == 'rbw':
		# Read before cp ran, once per attempt: cp never ran, so nothing was truncated.
		assert calls == [] and reads and set(reads) == {str(tmp_path / 'a/x')} and 'failed on read (Input/output error)' in failed[0]


@posix_only
@not_root
@pytest.mark.skipif(not os.path.exists('/proc/self/mem'), reason='/proc/self/mem: a file that opens and fails on read')
@pytest.mark.parametrize('mode', [[], ['-rbw'], ['-wb']], ids=['default', 'rbw', 'wb'])
@pytest.mark.parametrize('flags', [[], ['-co']], ids=['plain', 'content-only'])
def test_a_source_that_fails_on_read_in_a_directory_that_takes_no_new_name(tmp_path, flags, mode):
	# The review's A3: out 0555 with a writable out/mem. Default: cp truncated it and failed, and the directory keeps the
	# name, so out/mem is left empty (9.84's loss, 190). -rbw reads first and keeps it; -wb cannot make a new name there and
	# keeps it, saying that without -wb it would be written in place.
	write(tmp_path / 'out/mem', 'the last good copy')
	os.chmod(str(tmp_path / 'out/mem'), 0o666)
	os.chmod(str(tmp_path / 'out'), 0o555)
	try:
		rc, output = run(['/proc/self/mem', '-d', 'out/'] + flags + mode, tmp_path)
	finally:
		os.chmod(str(tmp_path / 'out'), 0o755)
	assert rc == 190, output[-2000:]
	if not mode:
		assert read(tmp_path / 'out/mem') == '' and b'/proc/self/mem' in output, output[-2000:]
		return
	assert read(tmp_path / 'out/mem') == 'the last good copy' and b'leaving ' in output, output[-2000:]
	if mode == ['-rbw']:
		assert b'failed on read' in output
	else:
		assert b'-wb writes beside it; ' in output and b'; without -wb it would be written in place' in output, output[-2000:]


@posix_only
@not_root
@pytest.mark.parametrize('flags', [[], ['-co']], ids=['plain', 'content-only'])
def test_a_growing_rewrite_in_a_directory_that_takes_no_new_name(tmp_path, flags):
	# Written in place, it needs no new name: exit 0. -wb needs one: 190, the old copy kept (the approved "beside route never
	# falls back").
	write(tmp_path / 'a/x', 'a longer new content')
	write(tmp_path / 'out/x', 'old')
	os.chmod(str(tmp_path / 'out/x'), 0o666)
	inode = os.stat(str(tmp_path / 'out/x')).st_ino
	os.chmod(str(tmp_path / 'out'), 0o555)
	try:
		rc, output = run(['a/x', '-d', 'out/'] + flags, tmp_path)
		assert rc == 0 and read(tmp_path / 'out/x') == 'a longer new content', output[-2000:]
		assert os.stat(str(tmp_path / 'out/x')).st_ino == inode
		write(tmp_path / 'a/x', 'an even longer new content')
		rc, output = run(['a/x', '-d', 'out/', '-wb'] + flags, tmp_path)
	finally:
		os.chmod(str(tmp_path / 'out'), 0o755)
	assert rc == 190 and read(tmp_path / 'out/x') == 'a longer new content', output[-2000:]
	assert b'; without -wb it would be written in place' in output


@posix_only
@pytest.mark.skipif(not hasattr(os, 'link'), reason='hard links')
@pytest.mark.parametrize('content_only', [False, True], ids=['plain', 'content-only'])
def test_a_landing_with_other_names_is_never_written_in_place(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, content_only):
	# A write in place would change every name of it: it goes beside, and the other names keep the old content. Beside
	# failed: both names keep the old content, 190, and the message says why it went beside.
	write(tmp_path / 'a/x', 'new')
	write(tmp_path / 'out/x', 'old')
	os.link(str(tmp_path / 'out/x'), str(tmp_path / 'out/other'))
	monkeypatch.setattr(hpcp_mod, 'CONTENT_ONLY', content_only)
	real = hpcp_mod.run_command_in_multicmd_with_path_check
	refuse = []
	def cp(command, *args, **kwargs):
		# cp writes the new file hpcp made beside the landing (/proc/self/fd/N, which names it), never the landing
		target = os.readlink(command[-1]) if command[-1].startswith('/proc/self/fd/') else command[-1]
		assert os.path.basename(target).startswith('.hpcp_tmp_'), command
		if refuse:
			raise RuntimeError('cp: cannot create regular file: Permission denied')
		return real(command, *args, **kwargs)
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', cp)
	refuse.append(True)
	hpcp_mod.copy_file(str(tmp_path / 'a/x'), [str(tmp_path / 'out/x')])
	assert read(tmp_path / 'out/x') == 'old' and read(tmp_path / 'out/other') == 'old' and temporaries(tmp_path) == []
	failed = [error for error in hpcp_mod.ERRORS if error.startswith('Copy failed')]
	assert failed and ('leaving %s as it is: it has other names, which a write in place would change; the new copy beside it '
		'failed: ' % (tmp_path / 'out/x')) in failed[-1], failed
	hpcp_mod.ERRORS.clear()
	refuse.clear()
	hpcp_mod.copy_file(str(tmp_path / 'a/x'), [str(tmp_path / 'out/x')])
	assert read(tmp_path / 'out/x') == 'new' and read(tmp_path / 'out/other') == 'old' and not hpcp_mod.ERRORS
	assert os.stat(str(tmp_path / 'out/x')).st_nlink == 1 and temporaries(tmp_path) == []


@posix_only
def test_a_refused_rename_on_the_beside_route_keeps_the_old_copy(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# -wb: the temporary file was whole and the rename was refused (a sticky directory, a bind-mounted file): no write in
	# place follows (9.85 wrote it in place at once), 190, the old copy kept, nothing left beside it.
	write(tmp_path / 'a/x', 'a longer new content')
	write(tmp_path / 'out/x', 'old')
	monkeypatch.setattr(hpcp_mod, 'WRITE_BESIDE', True)
	cps = []
	real = hpcp_mod.run_command_in_multicmd_with_path_check
	def cp(command, *args, **kwargs):
		# the beside route's new file, held by hpcp, is named by cp's /proc/self/fd/N
		cps.append(os.readlink(command[-1]) if command[-1].startswith('/proc/self/fd/') else command[-1])
		return real(command, *args, **kwargs)
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', cp)
	real_rename = os.rename
	def rename(source, target, **kwargs):
		if os.path.basename(str(source)).startswith('.hpcp_tmp_'):
			raise PermissionError(errno.EPERM, 'Operation not permitted', target)
		return real_rename(source, target, **kwargs)
	monkeypatch.setattr(hpcp_mod.os, 'rename', rename)
	assert hpcp_mod.copy_file(str(tmp_path / 'a/x'), [str(tmp_path / 'out/x')])[0] == 0
	assert read(tmp_path / 'out/x') == 'old' and temporaries(tmp_path / 'out') == []
	landings = [path for path in cps if not os.path.basename(path).startswith('.hpcp_tmp_')]
	assert landings == [] and cps and all(os.path.dirname(path) == str(tmp_path / 'out') for path in cps), cps
	failed = [error for error in hpcp_mod.ERRORS if error.startswith('Copy failed')]
	assert ('-wb writes beside it; it could not be replaced by the new copy (Operation not permitted); without -wb it would be '
		'written in place') in failed[-1], failed


@posix_only
@root_only
def test_a_sticky_directory_keeps_9_84_s_outcome_for_another_user_s_file(tmp_path, tmp_path_factory):
	# The re-review's stickyprobe: a root-owned sticky directory, another user's 0666 file, a growing source, run as nobody.
	# It opens for writing, so it is written in place, as 9.84 did: the content is new and the owner kept. Under -wb the rename
	# over it is refused (EPERM): 190, the file as it was.
	base, above = str(tmp_path_factory.getbasetemp()), str(tmp_path)
	while above.startswith(base):
		os.chmod(above, 0o755)  # pytest makes its own temporary directories 0700
		above = os.path.dirname(above)
	for path in (os.path.realpath(sys.executable), os.path.realpath(HPCP), str(tmp_path)):
		above = os.path.dirname(path)
		while above != os.path.dirname(above):
			if not os.stat(above).st_mode & stat.S_IXOTH:
				pytest.skip(f'nobody cannot reach {path}: {above} is not searchable by others')
			above = os.path.dirname(above)
	os.chmod(str(tmp_path), 0o755)
	os.makedirs(str(tmp_path / 'out'))
	os.chmod(str(tmp_path / 'out'), 0o1777)
	write(tmp_path / 'a/x', 'a longer new content')
	os.chmod(str(tmp_path / 'a'), 0o755)
	os.chmod(str(tmp_path / 'a/x'), 0o644)
	def as_nobody(flags):
		return subprocess.run([os.path.realpath(sys.executable), HPCP, 'a/x', '-d', 'out/'] + flags, cwd=str(tmp_path), stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT, timeout=120, preexec_fn=lambda: (os.setgid(65534), os.setuid(65534)))
	for flags in ([], ['-co']):
		write(tmp_path / 'out/x', 'old')
		os.chown(str(tmp_path / 'out/x'), 1000, 1000)
		os.chmod(str(tmp_path / 'out/x'), 0o666)
		inode = os.stat(str(tmp_path / 'out/x')).st_ino
		task = as_nobody(flags)
		assert read(tmp_path / 'out/x') == 'a longer new content' and os.stat(str(tmp_path / 'out/x')).st_uid == 1000, task.stdout[-2000:]
		assert os.stat(str(tmp_path / 'out/x')).st_ino == inode
		write(tmp_path / 'out/x', 'old')
		task = as_nobody(flags + ['-wb'])
		assert task.returncode == 190 and read(tmp_path / 'out/x') == 'old' and temporaries(tmp_path / 'out') == [], task.stdout[-2000:]


# ---- 9.86: a rewrite in place through the file the worker holds open ----

@posix_only
@pytest.mark.parametrize('size', [1, 3 << 20], ids=['1-byte', '3-MiB'])
@pytest.mark.parametrize('flags', [['-s'], ['-j', '4'], ['-s', '-co'], ['-j', '4', '-co']], ids=['serial', 'parallel', 'serial-co', 'parallel-co'])
def test_a_rewrite_keeps_the_inode(tmp_path, size, flags):
	# The inode, and what lives on it, stays (9.85 made a new one each rewrite). Plain: the source's mode and times, as cp -a
	# gives them; -co: the old mode.
	for name in 'abcd':
		write(tmp_path / 'a' / name, os.urandom(size))
		os.chmod(str(tmp_path / 'a' / name), 0o640)
		write(tmp_path / 'out' / name, b'o' * size)
		os.chmod(str(tmp_path / 'out' / name), 0o606)
	inodes = {name: os.stat(str(tmp_path / 'out' / name)).st_ino for name in 'abcd'}
	rc, output = run(['a/', '-d', 'out/'] + flags, tmp_path)
	assert rc == 0 and temporaries(tmp_path / 'out') == [], output[-2000:]
	for name in 'abcd':
		source, landing = os.stat(str(tmp_path / 'a' / name)), os.stat(str(tmp_path / 'out' / name))
		assert read_bytes(tmp_path / 'out' / name) == read_bytes(tmp_path / 'a' / name) and landing.st_ino == inodes[name]
		if '-co' in flags:
			assert stat.S_IMODE(landing.st_mode) == 0o606
		else:
			assert stat.S_IMODE(landing.st_mode) == 0o640 and landing.st_mtime_ns == source.st_mtime_ns


@posix_only
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='cp gets the held file through /proc only on Linux')
@pytest.mark.parametrize('content_only', [False, True], ids=['plain', 'content-only'])
def test_a_link_swapped_in_after_the_open_is_not_written_through(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, content_only):
	# The race the landing had: between hpcp's look and cp's open, the name is renamed away and a link to another file put
	# there. cp writes the file hpcp holds open, not the name: the link's target is untouched.
	write(tmp_path / 'a/x', 'new content')
	write(tmp_path / 'out/x', 'old')
	write(tmp_path / 't/f', 'not yours')
	inode = os.stat(str(tmp_path / 'out/x')).st_ino
	monkeypatch.setattr(hpcp_mod, 'CONTENT_ONLY', content_only)
	monkeypatch.setattr(hpcp_mod, 'NAMED_DESTS', frozenset([str(tmp_path / 'out')]))
	real = hpcp_mod.run_command_in_multicmd_with_path_check
	seen = []
	def swap_then_cp(command, *args, **kwargs):
		seen.append(command[-1])
		os.rename(str(tmp_path / 'out/x'), str(tmp_path / 'out/x.held'))
		os.symlink('../t/f', str(tmp_path / 'out/x'))
		return real(command, *args, **kwargs)
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', swap_then_cp)
	assert hpcp_mod.copy_file(str(tmp_path / 'a/x'), [str(tmp_path / 'out/x')])[0] > 0 and not hpcp_mod.ERRORS
	assert len(seen) == 1 and seen[0].startswith('/proc/self/fd/'), seen
	assert read(tmp_path / 't/f') == 'not yours' and os.path.islink(str(tmp_path / 'out/x'))
	assert read(tmp_path / 'out/x.held') == 'new content' and os.stat(str(tmp_path / 'out/x.held')).st_ino == inode


@posix_only
def test_write_beside_gives_a_new_inode_and_leaves_no_temporary_file(tmp_path):
	write(tmp_path / 'a/x', 'new content')
	write(tmp_path / 'out/x', 'old')
	inode = os.stat(str(tmp_path / 'out/x')).st_ino
	rc, output = run(['a/', '-d', 'out/', '-wb'], tmp_path)
	assert rc == 0 and read(tmp_path / 'out/x') == 'new content' and temporaries(tmp_path) == [], output[-2000:]
	assert os.stat(str(tmp_path / 'out/x')).st_ino != inode


@posix_only
@pytest.mark.skipif(not hasattr(os, 'link'), reason='hard links')
@pytest.mark.parametrize('case', ['in-place', 'first-copy', 'other-names', 'write-beside'])
def test_read_before_write_reads_before_cp_only_on_the_in_place_route(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, case):
	# -rbw guards only a write into an old copy: a first copy has none, and the beside route writes a new file.
	write(tmp_path / 'a/x', 'new content')
	if case != 'first-copy':
		write(tmp_path / 'out/x', 'old')
	if case == 'other-names':
		os.link(str(tmp_path / 'out/x'), str(tmp_path / 'other'))
	monkeypatch.setattr(hpcp_mod, 'READ_FIRST', True)
	monkeypatch.setattr(hpcp_mod, 'WRITE_BESIDE', case == 'write-beside')
	events = []
	real_read, real_cp = hpcp_mod._read_through, hpcp_mod.run_command_in_multicmd_with_path_check
	monkeypatch.setattr(hpcp_mod, '_read_through', lambda path: events.append('read') or real_read(path))
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', lambda command, *a, **k: events.append('cp') or real_cp(command, *a, **k))
	hpcp_mod.copy_file(str(tmp_path / 'a/x'), [str(tmp_path / 'out/x')])
	assert read(tmp_path / 'out/x') == 'new content' and not hpcp_mod.ERRORS
	assert events == (['read', 'cp'] if case == 'in-place' else ['cp']), events


@posix_only
@pytest.mark.parametrize('content_only', [False, True], ids=['plain', 'content-only'])
def test_off_linux_the_landing_is_written_by_path(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, content_only):
	# Where cp cannot be given the held file (no /proc/self/fd: macOS, the BSDs), cp gets the name, as in 9.84 (the approved
	# known limitation: a link swapped in mid-run is written through). The open still sorts what is there: a link at the
	# landing goes beside, its target untouched.
	write(tmp_path / 'a/x', 'new content')
	write(tmp_path / 'a/y', 'new y')
	write(tmp_path / 'out/x', 'old')
	write(tmp_path / 't/f', 'not yours')
	os.symlink('../t/f', str(tmp_path / 'out/y'))
	inode = os.stat(str(tmp_path / 'out/x')).st_ino
	monkeypatch.setattr(hpcp_mod, 'CONTENT_ONLY', content_only)
	monkeypatch.setattr(hpcp_mod, 'NAMED_DESTS', frozenset([str(tmp_path / 'out')]))
	monkeypatch.setattr(hpcp_mod, '_held_route', lambda: None)
	real = hpcp_mod.run_command_in_multicmd_with_path_check
	seen = []
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', lambda command, *a, **k: seen.append(command[-1]) or real(command, *a, **k))
	for name in 'xy':
		hpcp_mod.copy_file(str(tmp_path / 'a' / name), [str(tmp_path / 'out' / name)])
	assert not hpcp_mod.ERRORS and seen[0] == str(tmp_path / 'out/x') and os.path.basename(seen[1]).startswith('.hpcp_tmp_'), seen
	assert read(tmp_path / 'out/x') == 'new content' and os.stat(str(tmp_path / 'out/x')).st_ino == inode
	assert read(tmp_path / 'out/y') == 'new y' and not os.path.islink(str(tmp_path / 'out/y')) and read(tmp_path / 't/f') == 'not yours'


_NO_PROC_CP = '''#!/bin/sh
# cp that may not open the held file's /proc/self/fd/N (an LSM rule): it fails, writing nothing, as GNU cp fails its stat of
# the destination ("cannot stat ...: Permission denied"). PROBE=pass lets the held-file probe (cp of a temporary file named
# hpcp-held-probe-*) through, as if the refusal came only later.
case "$1" in --version) exec cp "$@";; esac
for last; do :; done
case "$last" in
/proc/*)
	case "$2" in */hpcp-held-probe-*) if [ "$PROBE" = pass ]; then exec cp "$@"; fi;; esac
	echo "cp: cannot stat '$last': Permission denied" >&2; exit 1;;
esac
exec cp "$@"
'''


@posix_only
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='the held-file route is Linux only')
@pytest.mark.parametrize('flags', [[], ['-co']], ids=['plain', 'content-only'])
def test_a_cp_that_cannot_open_the_held_file_is_given_the_name(tmp_path, flags):
	# The probe at a worker's first rewrite in place finds that cp cannot reach /proc/PID/fd/N: cp gets the name, as in 9.84,
	# and the rewrite works, the inode kept.
	cp = tmp_path / 'noproccp'
	cp.write_text(_NO_PROC_CP)
	os.chmod(str(cp), 0o755)
	for name in 'abcd':
		write(tmp_path / 'a' / name, 'new ' + name)
		write(tmp_path / 'out' / name, 'old')
	inodes = {name: os.stat(str(tmp_path / 'out' / name)).st_ino for name in 'abcd'}
	for mode, content in ((['-s'], 'new '), (['-j', '4'], 'newer ')):
		for name in 'abcd':
			write(tmp_path / 'a' / name, content + name)
		rc, output = run(['a/', '-d', 'out/'] + mode + flags, tmp_path, env={'CP_PATH': str(cp)})
		assert rc == 0, output[-2000:]
		assert all(read(tmp_path / 'out' / name) == content + name and os.stat(str(tmp_path / 'out' / name)).st_ino == inodes[name] for name in 'abcd')


@posix_only
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='the held-file route is Linux only')
def test_a_cp_refused_the_held_file_after_the_probe_keeps_the_old_copy(tmp_path):
	# cp passed the probe but cannot open the held file now (an LSM rule that only some files meet): cp fails before it opens
	# anything, so nothing was truncated: the old copy and its inode stay, 190, and the message names the landing, not
	# /proc. Never a silent success, never a torn file.
	cp = tmp_path / 'noproccp'
	cp.write_text(_NO_PROC_CP)
	os.chmod(str(cp), 0o755)
	write(tmp_path / 'a/x', 'new content')
	write(tmp_path / 'out/x', 'the last good copy')
	inode = os.stat(str(tmp_path / 'out/x')).st_ino
	rc, output = run(['a/x', '-d', 'out/', '-s'], tmp_path, env={'CP_PATH': str(cp), 'PROBE': 'pass'})
	assert rc == 190 and read(tmp_path / 'out/x') == 'the last good copy' and os.stat(str(tmp_path / 'out/x')).st_ino == inode, output[-2000:]
	assert ("cp: cannot stat '%s': Permission denied" % (tmp_path / 'out/x')).encode() in output and b'/proc/' not in output, output[-2000:]


@posix_only
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='the held-file route is Linux only')
def test_the_held_file_probe_runs_once_per_process(hpcp_mod, monkeypatch):
	# One probe however many rewrites in place; a forked worker (another pid) asks for itself. cp gets the descriptor it
	# inherits under its own process's name (/proc/self/fd/N): 9.86 named this process's (/proc/PID/fd/N), which hidepid, a
	# worker /proc refuses and another PID namespace's /proc did not show to cp.
	monkeypatch.setattr(hpcp_mod, '_HELD', [None, None])
	probes = []
	real_run = hpcp_mod.multiCMD.run_commands
	def run_commands(commands, *args, **kwargs):
		if os.path.basename(str(commands[0][-2])).startswith('hpcp-held-probe-'):
			probes.append((commands[0], kwargs.get('pass_fds')))
		return real_run(commands, *args, **kwargs)
	monkeypatch.setattr(hpcp_mod.multiCMD, 'run_commands', run_commands)
	answers = [hpcp_mod._held_route() for _ in range(3)]
	assert answers == [answers[0]] * 3 and len(probes) == 1, probes
	command, pass_fds = probes[0]
	assert command[-1] == '/proc/self/fd/%d' % pass_fds[0], probes
	assert not os.path.exists(command[-2])  # the probe's source is gone
	monkeypatch.setattr(hpcp_mod, '_HELD', [os.getpid() + 1, None])
	assert hpcp_mod._held_route() == answers[0] and len(probes) == 2


@posix_only
@pytest.mark.skipif(not hasattr(os, 'link'), reason='hard links')
@pytest.mark.parametrize('flags', [[], ['-co']], ids=['plain', 'content-only'])
def test_a_hard_linked_landing_goes_beside_and_its_other_names_keep_the_old_content(tmp_path, flags):
	write(tmp_path / 'a/x', 'new content')
	write(tmp_path / 'out/x', 'old')
	os.chmod(str(tmp_path / 'out/x'), 0o604)
	os.link(str(tmp_path / 'out/x'), str(tmp_path / 'elsewhere'))
	xattrs = True
	try:
		os.setxattr(str(tmp_path / 'out/x'), 'user.mine', b'kept')
	except OSError:
		xattrs = False
	inode = os.stat(str(tmp_path / 'out/x')).st_ino
	rc, output = run(['a/', '-d', 'out/'] + flags, tmp_path)
	assert rc == 0 and read(tmp_path / 'out/x') == 'new content' and read(tmp_path / 'elsewhere') == 'old', output[-2000:]
	st = os.stat(str(tmp_path / 'out/x'))
	assert st.st_ino != inode and st.st_nlink == 1 and os.stat(str(tmp_path / 'elsewhere')).st_ino == inode
	if flags:
		assert stat.S_IMODE(st.st_mode) == 0o604 and (not xattrs or _user_xattrs(tmp_path / 'out/x') == ['user.mine'])
	else:
		assert stat.S_IMODE(st.st_mode) == stat.S_IMODE(os.stat(str(tmp_path / 'a/x')).st_mode)


@posix_only
@not_root
@pytest.mark.parametrize('flags', [[], ['-co']], ids=['plain', 'content-only'])
def test_an_own_read_only_file_goes_beside(tmp_path, flags):
	# 0444: it does not open for writing (root's would), so it is replaced by a new file renamed over it. Plain gets the
	# source's mode, as cp -a gives it; -co keeps 0444 (_carry_over).
	write(tmp_path / 'a/x', 'new content')
	os.chmod(str(tmp_path / 'a/x'), 0o640)
	write(tmp_path / 'out/x', 'old')
	os.chmod(str(tmp_path / 'out/x'), 0o444)
	inode = os.stat(str(tmp_path / 'out/x')).st_ino
	rc, output = run(['a/', '-d', 'out/'] + flags, tmp_path)
	assert rc == 0 and read(tmp_path / 'out/x') == 'new content' and temporaries(tmp_path) == [], output[-2000:]
	st = os.stat(str(tmp_path / 'out/x'))
	assert st.st_ino != inode and stat.S_IMODE(st.st_mode) == (0o444 if flags else 0o640)


@posix_only
@pytest.mark.skipif(not os.path.exists('/bin/sleep'), reason='/bin/sleep')
def test_a_running_executable_at_the_landing_goes_beside(tmp_path):
	# A running program does not open for writing (ETXTBSY): its new copy is renamed over it, and the running process keeps its
	# old inode, untouched.
	import shutil
	os.makedirs(str(tmp_path / 'a'))
	os.makedirs(str(tmp_path / 'out'))
	shutil.copy2('/bin/sleep', str(tmp_path / 'out/exe'))
	write(tmp_path / 'a/exe', '#!/bin/sh\necho new\n')
	os.chmod(str(tmp_path / 'a/exe'), 0o755)
	inode = os.stat(str(tmp_path / 'out/exe')).st_ino
	process = subprocess.Popen([str(tmp_path / 'out/exe'), '60'])
	try:
		time.sleep(0.2)
		assert process.poll() is None
		rc, output = run(['a/', '-d', 'out/'], tmp_path)
		assert rc == 0 and read(tmp_path / 'out/exe') == '#!/bin/sh\necho new\n' and temporaries(tmp_path) == [], output[-2000:]
		assert os.stat(str(tmp_path / 'out/exe')).st_ino != inode and process.poll() is None
	finally:
		process.kill()
		process.wait()


@posix_only
@not_root
@pytest.mark.skipif(not hasattr(os, 'link'), reason='hard links')
def test_a_file_with_other_names_in_a_directory_that_takes_no_new_name_keeps_the_old_content(tmp_path):
	write(tmp_path / 'a/x', 'new content')
	write(tmp_path / 'out/x', 'old')
	os.chmod(str(tmp_path / 'out/x'), 0o666)
	os.link(str(tmp_path / 'out/x'), str(tmp_path / 'elsewhere'))
	os.chmod(str(tmp_path / 'out'), 0o555)
	try:
		# A file source: a directory source would sync out's mode (0755) before its files.
		rc, output = run(['a/x', '-d', 'out/'], tmp_path)
	finally:
		os.chmod(str(tmp_path / 'out'), 0o755)
	assert rc == 190 and read(tmp_path / 'out/x') == 'old' and read(tmp_path / 'elsewhere') == 'old', output[-2000:]
	assert ('leaving %s as it is: it has other names, which a write in place would change; the new copy beside it failed: '
		% (tmp_path / 'out/x')).encode() in output, output[-2000:]


# ---- what is at a landing: replaced, never opened or followed ----

@posix_only
@pytest.mark.parametrize('route', [[], ['-wb']], ids=['default', 'write-beside'])
@pytest.mark.parametrize('flags', [[], ['-co']], ids=['plain', 'content-only'])
def test_a_lone_fifo_at_a_landing_is_replaced_and_never_opened(tmp_path, flags, route):
	# 9.84 opened it for writing and waited for a reader: the run never ended.
	write(tmp_path / 'a/x', 'A')
	os.makedirs(str(tmp_path / 'out'))
	os.mkfifo(str(tmp_path / 'out/x'))
	rc, output = run(['a/', '-d', 'out/'] + flags + route, tmp_path, timeout=60)
	assert rc == 0 and stat.S_ISREG(os.lstat(str(tmp_path / 'out/x')).st_mode) and read(tmp_path / 'out/x') == 'A', output[-2000:]


@posix_only
def test_a_fifo_named_by_d_is_written_through(tmp_path):
	# A pipe feed: the reader gets the file, as cp gives it.
	write(tmp_path / 'a/x', 'through the pipe')
	os.mkfifo(str(tmp_path / 'feed'))
	reader = subprocess.Popen(['cat', str(tmp_path / 'feed')], stdout=subprocess.PIPE)
	rc, output = run(['a/x', '-d', 'feed'], tmp_path, timeout=60)
	got = reader.communicate(timeout=60)[0]
	assert rc == 0 and got == b'through the pipe' and stat.S_ISFIFO(os.lstat(str(tmp_path / 'feed')).st_mode), output[-2000:]


@posix_only
@not_root
@pytest.mark.parametrize('route', [[], ['-wb']], ids=['default', 'write-beside'])
@pytest.mark.parametrize('kind', ['fifo', 'link'])
def test_what_cannot_be_replaced_is_refused_and_never_opened(tmp_path, kind, route):
	# The review's A1 (a lone FIFO in a 0555 directory hung) and A2 (a link there was written through): a 213 naming the
	# directory and the fix, at once, and the FIFO is unopened (no writer ever shows at its other end).
	write(tmp_path / 'a/x', 'A')
	write(tmp_path / 't/f', 'kept')
	os.makedirs(str(tmp_path / 'out'))
	if kind == 'fifo':
		os.mkfifo(str(tmp_path / 'out/x'))
	else:
		os.symlink('../t/f', str(tmp_path / 'out/x'))
	os.chmod(str(tmp_path / 'out'), 0o555)
	started = time.monotonic()
	try:
		rc, output = run(['a/x', '-d', 'out/'] + route, tmp_path, timeout=60)
	finally:
		os.chmod(str(tmp_path / 'out'), 0o755)
	assert rc == 213 and time.monotonic() - started < 30, output[-2000:]
	what = b'the FIFO' if kind == 'fifo' else b'the link'
	assert b'Destination clash: cannot replace ' + what + b' ' + str(tmp_path / 'out/x').encode() in output
	assert b'make ' + str(tmp_path / 'out').encode() + b' writable' in output
	assert read(tmp_path / 't/f') == 'kept'
	if kind == 'fifo':
		# Nobody opened it to write: a reader opening it now without blocking sees no writer.
		fd = os.open(str(tmp_path / 'out/x'), os.O_RDONLY | os.O_NONBLOCK)
		try:
			assert os.read(fd, 10) == b''
		finally:
			os.close(fd)


@posix_only
@pytest.mark.skipif(not os.path.exists('/dev/null'), reason='/dev/null')
@pytest.mark.parametrize('route', [[], ['-wb']], ids=['default', 'write-beside'])
@pytest.mark.parametrize('flags', [[], ['-co']], ids=['plain', 'content-only'])
def test_a_link_to_a_device_at_a_landing_is_replaced(tmp_path, flags, route):
	# 9.84 wrote into /dev/null through it and failed on its times, 190, on every run.
	write(tmp_path / 'a/x', 'A')
	os.makedirs(str(tmp_path / 'out'))
	os.symlink('/dev/null', str(tmp_path / 'out/x'))
	for _ in range(2):
		rc, output = run(['a/', '-d', 'out/'] + flags + route, tmp_path)
		assert rc == 0 and not os.path.islink(str(tmp_path / 'out/x')) and read(tmp_path / 'out/x') == 'A', output[-2000:]


@posix_only
@root_only
@pytest.mark.parametrize('route', [[], ['-wb']], ids=['default', 'write-beside'])
@pytest.mark.parametrize('flags', [[], ['-co']], ids=['plain', 'content-only'])
def test_a_device_node_at_a_landing_is_replaced_never_written(tmp_path, flags, route):
	# A character node of /dev/null's numbers in the tree: 9.84 wrote the file into the device.
	write(tmp_path / 'a/x', 'A')
	os.makedirs(str(tmp_path / 'out'))
	os.mknod(str(tmp_path / 'out/x'), 0o666 | stat.S_IFCHR, os.stat('/dev/null').st_rdev)
	rc, output = run(['a/', '-d', 'out/'] + flags + route, tmp_path)
	assert rc == 0 and stat.S_ISREG(os.lstat(str(tmp_path / 'out/x')).st_mode) and read(tmp_path / 'out/x') == 'A', output[-2000:]


@posix_only
@pytest.mark.parametrize('route', [[], ['-wb']], ids=['default', 'write-beside'])
def test_a_link_at_a_file_s_landing_is_replaced_with_its_target_left_alone(tmp_path, route):
	write(tmp_path / 'a/x', 'A')
	write(tmp_path / 't/f', 'kept')
	os.chmod(str(tmp_path / 't/f'), 0o640)
	os.makedirs(str(tmp_path / 'out'))
	os.symlink('../t/f', str(tmp_path / 'out/x'))
	os.chmod(str(tmp_path / 'a/x'), 0o600)
	rc, output = run(['a/', '-d', 'out/'] + route, tmp_path)
	assert rc == 0 and read(tmp_path / 'out/x') == 'A' and not os.path.islink(str(tmp_path / 'out/x')), output[-2000:]
	assert read(tmp_path / 't/f') == 'kept' and stat.S_IMODE(os.stat(str(tmp_path / 't/f')).st_mode) == 0o640


@posix_only
@pytest.mark.parametrize('route', [[], ['-wb']], ids=['default', 'write-beside'])
def test_a_named_link_is_followed_and_its_target_rewritten(tmp_path, route):
	# What -d names is followed, as cp follows its destination: its target is written in place (its inode kept), or under -wb
	# beside itself and renamed over itself; the link stays.
	write(tmp_path / 'a/x', 'A')
	write(tmp_path / 't/f', 'old')
	inode = os.stat(str(tmp_path / 't/f')).st_ino
	os.makedirs(str(tmp_path / 'out'))
	os.symlink('../t/f', str(tmp_path / 'out/x'))
	rc, output = run(['a/x', '-d', 'out/x'] + route, tmp_path)
	assert rc == 0 and os.path.islink(str(tmp_path / 'out/x')) and read(tmp_path / 't/f') == 'A', output[-2000:]
	assert temporaries(tmp_path) == [] and (os.stat(str(tmp_path / 't/f')).st_ino == inode) == (route == [])


@posix_only
@pytest.mark.parametrize('order', ['parents-first', 'children-first'])
def test_a_link_at_a_directory_s_landing_is_replaced_with_a_note(tmp_path, order):
	# out/d -> ../elsewhere: 9.84 put src/d's files into elsewhere and gave it src/d's mode. Two levels, either order.
	write(tmp_path / 'src/d/e/f', 'F')
	os.chmod(str(tmp_path / 'src/d'), 0o700)
	os.makedirs(str(tmp_path / 'elsewhere'))
	os.chmod(str(tmp_path / 'elsewhere'), 0o755)
	os.makedirs(str(tmp_path / 'out'))
	os.symlink('../elsewhere', str(tmp_path / 'out/d'))
	args = ['src/', '-d', 'out/'] + (['-s'] if order == 'parents-first' else ['-j', '4'])
	rc, output = run(args, tmp_path)
	assert rc == 0 and not os.path.islink(str(tmp_path / 'out/d')) and read(tmp_path / 'out/d/e/f') == 'F', output[-2000:]
	assert os.listdir(str(tmp_path / 'elsewhere')) == [] and stat.S_IMODE(os.stat(str(tmp_path / 'elsewhere')).st_mode) == 0o755
	assert stat.S_IMODE(os.stat(str(tmp_path / 'out/d')).st_mode) == 0o700
	note = (f'Note: {tmp_path / "out/d"} was a link to ../elsewhere and is now a directory; nothing was written to ../elsewhere. '
		f'To write through a link, name it as the destination: hpcp {tmp_path / "src/d"}/ -d {tmp_path / "out/d"}/').encode()
	assert note in output


@posix_only
@pytest.mark.parametrize('listed', [False, True], ids=['tree', 'list'])
def test_a_link_to_a_sibling_directory_below_the_destination_is_not_taken_for_it(tmp_path, listed):
	# out/l -> d, with l and d both directories in the source: l/s/x lands at out/l/s/x, never at out/d/s/x through the link,
	# so the two are two landings, no clash, and the link becomes a directory of its own.
	write(tmp_path / 'd/s/x', 'D')
	write(tmp_path / 'l/s/x', 'L')
	os.makedirs(str(tmp_path / 'out/d/s'))
	os.symlink('d', str(tmp_path / 'out/l'))
	for _ in range(2):
		if listed:
			rc, output = run(['-sfl', '-', '-d', 'out/', '-j', '4'], tmp_path, b'd/s/x\nl/s/x\n')
		else:
			rc, output = run(['d', 'l', '-d', 'out/', '-j', '4'], tmp_path)
		assert rc == 0 and not os.path.islink(str(tmp_path / 'out/l')), output[-2000:]
		assert read(tmp_path / 'out/d/s/x') == 'D' and read(tmp_path / 'out/l/s/x') == 'L'


@posix_only
def test_a_link_two_levels_up_is_replaced_when_the_children_sync_first(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# The folder sync hands folders out in any order: the door replaces the link at out/d whoever reaches it first.
	write(tmp_path / 'src/d/e/f', 'F')
	os.makedirs(str(tmp_path / 'elsewhere/e'))
	os.makedirs(str(tmp_path / 'out'))
	os.symlink('../elsewhere', str(tmp_path / 'out/d'))
	monkeypatch.setattr(hpcp_mod, 'NAMED_DESTS', frozenset([str(tmp_path / 'out')]))
	hpcp_mod.sync_directory_metadata(str(tmp_path / 'src/d/e'), [str(tmp_path / 'out/d/e')])
	hpcp_mod.sync_directory_metadata(str(tmp_path / 'src/d'), [str(tmp_path / 'out/d')])
	assert not os.path.islink(str(tmp_path / 'out/d')) and os.path.isdir(str(tmp_path / 'out/d/e'))
	assert os.listdir(str(tmp_path / 'elsewhere')) == ['e']


@posix_only
@not_root
def test_a_link_that_cannot_be_replaced_by_a_directory_is_refused_per_file(tmp_path):
	write(tmp_path / 'src/d/f', 'F')
	write(tmp_path / 'src/d/g', 'G')
	os.makedirs(str(tmp_path / 'elsewhere'))
	os.makedirs(str(tmp_path / 'out/p'))
	os.symlink('../../elsewhere', str(tmp_path / 'out/p/d'))
	os.chmod(str(tmp_path / 'out/p'), 0o555)
	try:
		rc, output = run(['src/d', '-d', 'out/p/'], tmp_path)
	finally:
		os.chmod(str(tmp_path / 'out/p'), 0o755)
	assert rc == 213 and os.listdir(str(tmp_path / 'elsewhere')) == [] and os.path.islink(str(tmp_path / 'out/p/d')), output[-2000:]
	for name in ('f', 'g'):
		assert (b'Destination clash: cannot replace the link ' + str(tmp_path / 'out/p/d').encode() + b' (to ../../elsewhere) by a directory'
			in output and b'not copying ' + str(tmp_path / 'src/d' / name).encode() + b' below it' in output)


@posix_only
def test_a_link_under_ncd_is_refused_naming_the_d_form(tmp_path):
	# -ncd makes no directory in place of anything: the link stays, a 213 for each file below it (9.84 wrote through it).
	write(tmp_path / 'src/d/f', 'F')
	os.makedirs(str(tmp_path / 'elsewhere'))
	os.makedirs(str(tmp_path / 'out'))
	os.symlink('../elsewhere', str(tmp_path / 'out/d'))
	rc, output = run(['src/', '-d', 'out/', '-ncd'], tmp_path)
	assert rc == 213 and os.listdir(str(tmp_path / 'elsewhere')) == [] and os.path.islink(str(tmp_path / 'out/d')), output[-2000:]
	assert (b'is a link to ../elsewhere and -ncd makes no directory in its place: not copying ' + str(tmp_path / 'src/d/f').encode()
		+ b' below it; name it with -d ' + str(tmp_path / 'out/d').encode() + b'/ to write through it') in output


@posix_only
@pytest.mark.parametrize('form', ['nds', 'ncd', 'list-nds'])
def test_the_skip_check_never_syncs_a_file_behind_a_link_below_the_destination(tmp_path, form):
	# out/d -> ../elsewhere holds the same file (content, mtime), mode 600: 9.84 and 9.85 before this gave it src's 644
	# through the link under -nds and -ncd, the link kept. Now the door meets the link first.
	write(tmp_path / 'src/d/x', 'same')
	write(tmp_path / 'elsewhere/x', 'same')
	os.chmod(str(tmp_path / 'src/d/x'), 0o644)
	os.chmod(str(tmp_path / 'elsewhere/x'), 0o600)
	os.utime(str(tmp_path / 'src/d/x'), (1000000000, 1000000000))
	os.utime(str(tmp_path / 'elsewhere/x'), (1000000000, 1000000000))
	xattrs = True
	try:
		os.setxattr(str(tmp_path / 'elsewhere/x'), 'user.mine', b'1')
	except OSError:
		xattrs = False
	os.makedirs(str(tmp_path / 'out'))
	os.symlink('../elsewhere', str(tmp_path / 'out/d'))
	if form == 'list-nds':
		rc, output = run(['-sfl', '-', '-d', str(tmp_path / 'out') + '/', '-nds'], tmp_path / 'src', b'd/x\n')
	else:
		rc, output = run(['src/', '-d', 'out/', '-' + form], tmp_path)
	ex = os.stat(str(tmp_path / 'elsewhere/x'))
	assert stat.S_IMODE(ex.st_mode) == 0o600 and read(tmp_path / 'elsewhere/x') == 'same', (rc, output[-2000:])
	if xattrs:
		assert _user_xattrs(tmp_path / 'elsewhere/x') == ['user.mine']
	if form == 'ncd':
		assert rc == 213 and os.path.islink(str(tmp_path / 'out/d')), output[-2000:]
	else:
		assert rc == 0 and not os.path.islink(str(tmp_path / 'out/d')) and read(tmp_path / 'out/d/x') == 'same', output[-2000:]


@posix_only
@root_only
def test_the_skip_check_never_chowns_a_file_behind_a_link_below_the_destination(tmp_path):
	write(tmp_path / 'src/d/x', 'same')
	write(tmp_path / 'elsewhere/x', 'same')
	os.chown(str(tmp_path / 'elsewhere/x'), 1000, 1000)
	for path in ('src/d/x', 'elsewhere/x'):
		os.utime(str(tmp_path / path), (1000000000, 1000000000))
	os.makedirs(str(tmp_path / 'out'))
	os.symlink('../elsewhere', str(tmp_path / 'out/d'))
	for flag in ('-nds', '-ncd'):
		run(['src/', '-d', 'out/', flag], tmp_path)
		st = os.stat(str(tmp_path / 'elsewhere/x'))
		assert (st.st_uid, st.st_gid) == (1000, 1000)


@posix_only
def test_a_named_link_that_dangles_is_not_written_through(tmp_path):
	# As cp and 9.84: exit 190, nothing made where the link points.
	write(tmp_path / 'a/x', 'A')
	os.makedirs(str(tmp_path / 'out'))
	os.symlink('../nowhere/f', str(tmp_path / 'out/y'))
	rc, output = run(['a/x', '-d', 'out/y'], tmp_path)
	assert rc == 190 and not os.path.lexists(str(tmp_path / 'nowhere')) and os.path.islink(str(tmp_path / 'out/y')), output[-2000:]
	assert (str(tmp_path / 'out/y') + ' is a link to ../nowhere/f, which does not exist').encode() in output


def test_the_door_looks_again_when_another_worker_makes_the_directory(tmp_path, hpcp_mod, monkeypatch):
	# The race the review reasoned: this worker sees the link, another unlinks it, this unlink meets ENOENT, the next lstat
	# finds nothing, the other's mkdir lands, and this mkdir meets EEXIST: a directory, not vfat's case.
	out, d = str(tmp_path / 'out'), str(tmp_path / 'out/d')
	os.makedirs(out)
	os.symlink('elsewhere', d)
	real_lstat, real_unlink, real_mkdir = os.lstat, os.unlink, os.mkdir
	state = {'unlinked': False}
	def unlink(path, *args, **kwargs):
		if str(path) == d and not state['unlinked']:
			state['unlinked'] = True
			real_unlink(d)  # the other worker's
			raise FileNotFoundError(errno.ENOENT, 'No such file or directory', path)
		return real_unlink(path, *args, **kwargs)
	def mkdir(path, *args, **kwargs):
		if str(path) == d and not os.path.lexists(d):
			real_mkdir(d)  # the other worker's lands first
			raise FileExistsError(errno.EEXIST, 'File exists', path)
		return real_mkdir(path, *args, **kwargs)
	monkeypatch.setattr(hpcp_mod.os, 'unlink', unlink)
	monkeypatch.setattr(hpcp_mod.os, 'mkdir', mkdir)
	monkeypatch.setattr(hpcp_mod, 'NAMED_DESTS', frozenset([out]))
	monkeypatch.setattr(hpcp_mod, 'RUN_ID', object())
	hpcp_mod._dest_directory(d)  # its answer is the cursor's own descriptor, closed with the cursors the door keeps for the run
	try:
		assert os.path.isdir(d) and not os.path.islink(d)
	finally:
		hpcp_mod._close_cursors()


@posix_only
def test_a_link_at_an_implied_parent_of_a_list_entry_is_replaced(tmp_path):
	write(tmp_path / 'src/d/x', 'x')
	os.makedirs(str(tmp_path / 'elsewhere'))
	os.makedirs(str(tmp_path / 'out/src'))
	os.symlink('../../elsewhere', str(tmp_path / 'out/src/d'))
	rc, output = run(['-d', 'out'], tmp_path, b'src/d/x\n')
	assert rc == 0 and read(tmp_path / 'out/src/d/x') == 'x' and os.listdir(str(tmp_path / 'elsewhere')) == [], output[-2000:]


@posix_only
def test_rme_mirrors_a_directory_the_destination_held_as_a_link_into_the_source(tmp_path):
	# out/d -> ../src/d: 9.84 stopped at -rme's prompt, or removed the link as an extra and mirrored nothing of d.
	write(tmp_path / 'src/d/f', 'F')
	os.makedirs(str(tmp_path / 'out'))
	os.symlink('../src/d', str(tmp_path / 'out/d'))
	rc, output = run(['src/', '-d', 'out/', '-rme'], tmp_path)
	assert rc == 0 and not os.path.islink(str(tmp_path / 'out/d')) and read(tmp_path / 'out/d/f') == 'F', output[-2000:]
	assert read(tmp_path / 'src/d/f') == 'F'


@posix_only
@pytest.mark.parametrize('route', [[], ['-wb']], ids=['default', 'write-beside'])
def test_a_stale_user_xattr_goes_after_a_rewrite(tmp_path, route):
	# cp -a sets the source's xattrs and never removes one: written in place, the old copy's stale user.* xattr is removed
	# through the held file's descriptor; beside, the new file never had it.
	write(tmp_path / 'a/x', 'new')
	write(tmp_path / 'out/x', 'old')
	try:
		os.setxattr(str(tmp_path / 'out/x'), 'user.stale', b'1')
		os.setxattr(str(tmp_path / 'a/x'), 'user.kept', b'2')
	except OSError:
		pytest.skip('no user xattrs here')
	inode = os.stat(str(tmp_path / 'out/x')).st_ino
	rc, output = run(['a/', '-d', 'out/'] + route, tmp_path)
	assert rc == 0, output[-2000:]
	assert _user_xattrs(tmp_path / 'out/x') == ['user.kept'] and (os.stat(str(tmp_path / 'out/x')).st_ino == inode) == (route == [])


@posix_only
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='the held file is reached through /proc only on Linux')
def test_a_rewrite_in_place_gives_the_landing_the_source_s_selinux_label(tmp_path):
	# cp -a sets an existing destination's label by name, which through /proc/PID/fd/N reaches the /proc link (coreutils
	# 8.22 on): hpcp sets it through the descriptor, as 9.84's cp by name and 9.85's new file had it.
	write(tmp_path / 'a/x', 'new')
	write(tmp_path / 'out/x', 'old')
	try:
		label = os.getxattr(str(tmp_path / 'a/x'), 'security.selinux')
		user, role, kind, level = label.rstrip(b'\0').split(b':', 3)
		if kind == b'usr_t':
			pytest.skip('the source is usr_t already')
		os.setxattr(str(tmp_path / 'a/x'), 'security.selinux', b':'.join((user, role, b'usr_t', level)))
	except (OSError, ValueError):
		pytest.skip('no SELinux label that can be set here')
	inode = os.stat(str(tmp_path / 'out/x')).st_ino
	rc, output = run(['a/', '-d', 'out/'], tmp_path)
	assert rc == 0 and read(tmp_path / 'out/x') == 'new' and os.stat(str(tmp_path / 'out/x')).st_ino == inode, output[-2000:]
	assert os.getxattr(str(tmp_path / 'out/x'), 'security.selinux') == os.getxattr(str(tmp_path / 'a/x'), 'security.selinux')


@posix_only
@pytest.mark.skipif(not hasattr(os, 'getxattr'), reason='needs xattrs')
@pytest.mark.parametrize('case', ['differs', 'same', 'refused', 'no-label'])
def test_the_label_of_a_held_file_is_set_only_where_it_differs_and_never_fails_the_copy(tmp_path, hpcp_mod, monkeypatch, case):
	write(tmp_path / 'src', 'new')
	write(tmp_path / 'held', 'old')
	labels = {str(tmp_path / 'src'): b'u:r:new_t:s0', 'held': b'u:r:new_t:s0' if case == 'same' else b'u:r:old_t:s0'}
	calls = []

	def getxattr(path, name, **how):
		assert name == 'security.selinux'
		if case == 'no-label' and not isinstance(path, int):
			raise OSError(errno.ENODATA, 'No data available')
		return labels['held' if isinstance(path, int) else path]

	def setxattr(path, name, value, **how):
		calls.append((isinstance(path, int), name, value))
		if case == 'refused':
			raise PermissionError(errno.EPERM, 'Operation not permitted')
	monkeypatch.setattr(hpcp_mod.os, 'getxattr', getxattr)
	monkeypatch.setattr(hpcp_mod.os, 'setxattr', setxattr)
	fd = os.open(str(tmp_path / 'held'), os.O_WRONLY)
	try:
		hpcp_mod._label_like(str(tmp_path / 'src'), fd)
	finally:
		os.close(fd)
	assert calls == ([(True, 'security.selinux', b'u:r:new_t:s0')] if case in ('differs', 'refused') else [])


@posix_only
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='the held file is reached through /proc only on Linux')
@pytest.mark.parametrize('content_only', [False, True], ids=['plain', 'content-only'])
def test_a_rewrite_through_the_held_file_asks_for_the_source_s_label_only_on_a_plain_run(tmp_path, hpcp_mod,
		reset_hpcp_globals, monkeypatch, content_only):
	# Runs without SELinux too: a plain rewrite through the held file gives it the source's label (_label_like), and -co,
	# which keeps the old label in every version, does not.
	if not hpcp_mod._held_route():
		pytest.skip('cp cannot reach the held file here')
	write(tmp_path / 'a/x', 'new content')
	write(tmp_path / 'out/x', 'old')
	inode = os.stat(str(tmp_path / 'out/x')).st_ino
	monkeypatch.setattr(hpcp_mod, 'CONTENT_ONLY', content_only)
	monkeypatch.setattr(hpcp_mod, 'NAMED_DESTS', frozenset([str(tmp_path / 'out')]))
	asked = []
	monkeypatch.setattr(hpcp_mod, '_label_like', lambda src, fd: asked.append((src, os.fstat(fd).st_ino)))
	hpcp_mod.copy_file(str(tmp_path / 'a/x'), [str(tmp_path / 'out/x')])
	assert not hpcp_mod.ERRORS and read(tmp_path / 'out/x') == 'new content' and os.stat(str(tmp_path / 'out/x')).st_ino == inode
	assert asked == ([] if content_only else [(str(tmp_path / 'a/x'), inode)])


# ---- -co, option A': decide first, then carry exactly what was there ----

@posix_only
@pytest.mark.parametrize('route', ['default', 'write-beside', 'other-names'])
def test_content_only_keeps_the_old_file_s_mode_and_xattrs(tmp_path, route):
	# By default the inode is written in place and keeps all of it; beside (-wb, or a file with other names by default) the
	# new inode is given the old one's mode and carried xattrs (_carry_over), and the other names keep the old content.
	write(tmp_path / 'a/x', 'new and longer')
	write(tmp_path / 'out/x', 'old')
	os.chmod(str(tmp_path / 'out/x'), 0o604)
	try:
		os.setxattr(str(tmp_path / 'out/x'), 'user.mine', b'kept')
		os.setxattr(str(tmp_path / 'a/x'), 'user.source', b'not carried')
	except OSError:
		pytest.skip('no user xattrs here')
	if route == 'other-names':
		os.link(str(tmp_path / 'out/x'), str(tmp_path / 'other'))
	inode = os.stat(str(tmp_path / 'out/x')).st_ino
	rc, output = run(['a/', '-d', 'out/', '-co'] + (['-wb'] if route == 'write-beside' else []), tmp_path)
	assert rc == 0 and read(tmp_path / 'out/x') == 'new and longer', output[-2000:]
	st = os.stat(str(tmp_path / 'out/x'))
	assert (st.st_ino == inode) == (route == 'default') and stat.S_IMODE(st.st_mode) == 0o604 and st.st_nlink == 1
	assert _user_xattrs(tmp_path / 'out/x') == ['user.mine'] and os.getxattr(str(tmp_path / 'out/x'), 'user.mine') == b'kept'
	if route == 'other-names':
		assert read(tmp_path / 'other') == 'old' and os.stat(str(tmp_path / 'other')).st_ino == inode


def _acl(*entries):
	"""A POSIX ACL as its xattr holds it: version 2, then (tag, perm, id) each, little-endian (u16, u16, u32)."""
	import struct
	return struct.pack('<I', 2) + b''.join(struct.pack('<HHI', tag, perm, ident) for tag, perm, ident in entries)

# Tags: user_obj, user, group_obj, mask, other. 0xffffffff is "no id".
_NO_ID = 0xffffffff


@posix_only
@pytest.mark.parametrize('route', [[], ['-wb']], ids=['default', 'write-beside'])
def test_content_only_carries_the_old_acl_and_drops_one_the_directory_gave(tmp_path, route):
	# out has a default ACL, which a new file in it takes as its access ACL: removed, as the old file had none. A file
	# that had an access ACL keeps exactly it.
	write(tmp_path / 'a/x', 'new')
	write(tmp_path / 'a/y', 'new')
	write(tmp_path / 'out/x', 'old')
	write(tmp_path / 'out/y', 'old')
	access = _acl((1, 6, _NO_ID), (2, 4, 12345), (4, 4, _NO_ID), (0x10, 4, _NO_ID), (0x20, 4, _NO_ID))
	default = _acl((1, 7, _NO_ID), (2, 6, 54321), (4, 5, _NO_ID), (0x10, 7, _NO_ID), (0x20, 5, _NO_ID))
	try:
		os.setxattr(str(tmp_path / 'out/y'), 'system.posix_acl_access', access)
		os.setxattr(str(tmp_path / 'out'), 'system.posix_acl_default', default)
	except OSError:
		pytest.skip('no ACLs here')
	before = os.getxattr(str(tmp_path / 'out/y'), 'system.posix_acl_access')
	rc, output = run(['a/', '-d', 'out/', '-co'] + route, tmp_path)
	assert rc == 0 and read(tmp_path / 'out/x') == 'new' and read(tmp_path / 'out/y') == 'new', output[-2000:]
	assert 'system.posix_acl_access' not in os.listxattr(str(tmp_path / 'out/x'))
	assert os.getxattr(str(tmp_path / 'out/y'), 'system.posix_acl_access') == before



@pytest.mark.skipif(not hasattr(os, 'listxattr'), reason='xattrs')
def test_content_only_carries_an_acl_that_xfs_also_lists_by_its_own_name(tmp_path, hpcp_mod, monkeypatch):
	# Root on EL7's XFS (kernel 3.10) lists an ACL twice: system.posix_acl_access and XFS's own trusted.SGI_ACL_FILE (a default
	# ACL: SGI_ACL_DEFAULT). The system.* name carries it; taken for an xattr hpcp cannot carry, -co -wb exited 190 for
	# every such file. Without its system.* twin a trusted.* name still keeps the file from being carried.
	write(tmp_path / 'f', 'x')
	names = {'security.selinux': b'l', 'system.posix_acl_access': b'a', 'trusted.SGI_ACL_FILE': b'A',
		'system.posix_acl_default': b'd', 'trusted.SGI_ACL_DEFAULT': b'D'}
	monkeypatch.setattr(hpcp_mod.os, 'listxattr', lambda path, **kwargs: list(names))
	monkeypatch.setattr(hpcp_mod.os, 'getxattr', lambda path, name, **kwargs: names[name])
	st = os.lstat(str(tmp_path / 'f'))
	assert hpcp_mod._co_carry(str(tmp_path / 'f'), st) == {'security.selinux': b'l', 'system.posix_acl_access': b'a',
		'system.posix_acl_default': b'd'}
	del names['system.posix_acl_access']
	assert hpcp_mod._co_carry(str(tmp_path / 'f'), st) is None

def shutil_which(name):
	import shutil
	return shutil.which(name)


@posix_only
@root_only
@pytest.mark.parametrize('route', [[], ['-wb']], ids=['default', 'write-beside'])
def test_content_only_drops_a_file_capability_as_a_write_in_place_does(tmp_path, route):
	# VM 902: a write in place strips security.capability (setuid kept); carrying it would give new content a capability.
	if not shutil_which('setcap'):
		pytest.skip('setcap')
	write(tmp_path / 'a/x', 'new')
	write(tmp_path / 'out/x', 'old')
	os.chmod(str(tmp_path / 'out/x'), 0o4755)
	if subprocess.run(['setcap', 'cap_net_raw+ep', str(tmp_path / 'out/x')]).returncode:
		pytest.skip('no file capabilities here')
	rc, output = run(['a/', '-d', 'out/', '-co'] + route, tmp_path)
	assert rc == 0 and read(tmp_path / 'out/x') == 'new', output[-2000:]
	assert 'security.capability' not in os.listxattr(str(tmp_path / 'out/x'))
	assert stat.S_IMODE(os.stat(str(tmp_path / 'out/x')).st_mode) == 0o4755


@posix_only
@not_root
@pytest.mark.parametrize('route', [[], ['-wb']], ids=['default', 'write-beside'])
def test_content_only_drops_setuid_for_a_non_root_owner_as_the_kernel_does(tmp_path, route):
	# suidprobe: the owner writing in place lost 4700 -> 700 and 6755 -> 755, kept 2745.
	for name, mode, kept in (('u', 0o4700, 0o700), ('ug', 0o6755, 0o755), ('g', 0o2745, 0o2745)):
		write(tmp_path / 'a' / name, 'new')
		write(tmp_path / 'out' / name, 'old')
		os.chmod(str(tmp_path / 'out' / name), mode)
	rc, output = run(['a/', '-d', 'out/', '-co'] + route, tmp_path)
	assert rc == 0, output[-2000:]
	for name, mode, kept in (('u', 0o4700, 0o700), ('ug', 0o6755, 0o755), ('g', 0o2745, 0o2745)):
		assert stat.S_IMODE(os.stat(str(tmp_path / 'out' / name)).st_mode) == kept and read(tmp_path / 'out' / name) == 'new'


@posix_only
@pytest.mark.parametrize('route', ['default', 'write-beside'])
def test_content_only_where_an_xattr_cannot_be_carried(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, route):
	# trusted.*, security.ima, a filesystem's own: no new file can carry it. By default the file opens for writing and is
	# written in place, keeping it; under -wb the old copy stays, 190, naming the cause.
	write(tmp_path / 'a/x', 'new')
	write(tmp_path / 'out/x', 'old')
	inode = os.stat(str(tmp_path / 'out/x')).st_ino
	monkeypatch.setattr(hpcp_mod, 'CONTENT_ONLY', True)
	monkeypatch.setattr(hpcp_mod, 'WRITE_BESIDE', route == 'write-beside')
	real = os.listxattr
	monkeypatch.setattr(hpcp_mod.os, 'listxattr', lambda path, **kwargs: ['trusted.overlay'] if str(path).endswith('out/x') else real(path, **kwargs))
	assert hpcp_mod._co_carry(str(tmp_path / 'out/x'), os.lstat(str(tmp_path / 'out/x'))) is None
	hpcp_mod.copy_file(str(tmp_path / 'a/x'), [str(tmp_path / 'out/x')])
	assert os.stat(str(tmp_path / 'out/x')).st_ino == inode and temporaries(tmp_path) == []
	if route == 'default':
		assert read(tmp_path / 'out/x') == 'new' and not hpcp_mod.ERRORS
	else:
		assert read(tmp_path / 'out/x') == 'old'
		assert any(error.startswith('Copy failed') and error.endswith('-wb writes beside it; its owner, group or an xattr cannot be '
			'given to a new file; without -wb it would be written in place') for error in hpcp_mod.ERRORS), hpcp_mod.ERRORS


@posix_only
@pytest.mark.parametrize('xattr', ['security.selinux', 'security.capability'])
def test_content_only_carries_a_label_and_never_a_capability(tmp_path, hpcp_mod, monkeypatch, xattr):
	write(tmp_path / 'out/x', 'old')
	real_list, real_get = os.listxattr, os.getxattr
	monkeypatch.setattr(hpcp_mod.os, 'listxattr', lambda path, **kwargs: [xattr])
	monkeypatch.setattr(hpcp_mod.os, 'getxattr', lambda path, name, **kwargs: b'value')
	carried = hpcp_mod._co_carry(str(tmp_path / 'out/x'), os.lstat(str(tmp_path / 'out/x')))
	assert carried == ({xattr: b'value'} if xattr == 'security.selinux' else {})


@posix_only
def test_content_only_skips_a_label_the_filesystem_gives_itself(tmp_path, hpcp_mod, monkeypatch):
	# vfat with SELinux enforcing lists security.selinux and refuses to set it (EOPNOTSUPP): v1's prototype exited 190.
	write(tmp_path / 'temp', 'new')
	def refuse(path, name, value, **kwargs):
		raise OSError(errno.EOPNOTSUPP, 'Operation not supported', path)
	monkeypatch.setattr(hpcp_mod.os, 'setxattr', refuse)
	monkeypatch.setattr(hpcp_mod.os, 'listxattr', lambda path, **kwargs: [])
	st = os.lstat(str(tmp_path / 'temp'))
	hpcp_mod._carry_over(st, str(tmp_path / 'temp'), {'security.selinux': b'system_u:object_r:dosfs_t:s0'})
	with pytest.raises(OSError):
		hpcp_mod._carry_over(st, str(tmp_path / 'temp'), {'user.other': b'x'})


@posix_only
@not_root
def test_content_only_cannot_carry_another_user_s_file(tmp_path, hpcp_mod, monkeypatch):
	write(tmp_path / 'out/x', 'old')
	st = os.lstat(str(tmp_path / 'out/x'))
	other = os.stat_result((st.st_mode, st.st_ino, st.st_dev, st.st_nlink, st.st_uid + 1, st.st_gid) + tuple(st)[6:])
	assert hpcp_mod._co_carry(str(tmp_path / 'out/x'), other) is None
	foreign_group = os.stat_result((st.st_mode, st.st_ino, st.st_dev, st.st_nlink, st.st_uid, 4242) + tuple(st)[6:])
	if 4242 not in os.getgroups() and os.getegid() != 4242:
		assert hpcp_mod._co_carry(str(tmp_path / 'out/x'), foreign_group) is None


@posix_only
@not_root
@pytest.mark.parametrize('mode', [0o666, 0o444], ids=['opens', 'read-only'])
def test_content_only_rewrites_another_user_s_file_only_in_place(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, mode):
	# Another user's file (_co_carry: None, as no new file can have its owner): written in place when it opens for writing,
	# so it keeps its owner and mode; one that does not open for writing would need a new file, which -co cannot give its
	# metadata: 190, the old copy kept (9.84's cp -f unlinked it and made a file the runner owns).
	write(tmp_path / 'a/x', 'new')
	write(tmp_path / 'out/x', 'old')
	os.chmod(str(tmp_path / 'out/x'), mode)
	inode = os.stat(str(tmp_path / 'out/x')).st_ino
	monkeypatch.setattr(hpcp_mod, 'CONTENT_ONLY', True)
	asked = []
	monkeypatch.setattr(hpcp_mod, '_co_carry', lambda target, st: asked.append(target))
	try:
		hpcp_mod.copy_file(str(tmp_path / 'a/x'), [str(tmp_path / 'out/x')])
		st = os.stat(str(tmp_path / 'out/x'))
		assert st.st_ino == inode and stat.S_IMODE(st.st_mode) == mode and temporaries(tmp_path) == []
		if mode == 0o666:
			assert read(tmp_path / 'out/x') == 'new' and asked == [] and not hpcp_mod.ERRORS
		else:
			assert read(tmp_path / 'out/x') == 'old' and asked
			assert any(error.endswith('leaving %s as it is: it cannot be opened for writing (Permission denied); its owner, group '
				'or an xattr cannot be given to a new file' % (tmp_path / 'out/x')) for error in hpcp_mod.ERRORS), hpcp_mod.ERRORS
	finally:
		os.chmod(str(tmp_path / 'out/x'), 0o644)


# ---- rule 8: a block device named by -d ----

@posix_only
def test_a_block_device_in_use_is_refused(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# Linux fails O_EXCL on a mounted or held block device with EBUSY (checked on EL7 for a mounted loop device).
	write(tmp_path / 'a/x', 'A')
	write(tmp_path / 'disk', 'DISK')
	real_lstat, real_open = os.lstat, os.open
	block = os.stat_result((stat.S_IFBLK | 0o660,) + tuple(real_lstat(str(tmp_path / 'disk')))[1:])
	monkeypatch.setattr(hpcp_mod.os, 'lstat', lambda path, **kwargs: block if str(path) == str(tmp_path / 'disk') else real_lstat(path, **kwargs))
	def opened(path, flags, *args, **kwargs):
		if str(path) == str(tmp_path / 'disk') and flags & os.O_EXCL:
			raise OSError(errno.EBUSY, 'Device or resource busy', path)
		return real_open(path, flags, *args, **kwargs)
	monkeypatch.setattr(hpcp_mod.os, 'open', opened)
	monkeypatch.setattr(hpcp_mod, 'NAMED_DESTS', frozenset([str(tmp_path / 'disk')]))
	hpcp_mod.copy_file(str(tmp_path / 'a/x'), [str(tmp_path / 'disk')])
	assert read(tmp_path / 'disk') == 'DISK'
	assert any(error == f'Destination clash: {tmp_path / "disk"} is in use (mounted, swap, or held by device-mapper, md or LVM): '
		f'not writing {tmp_path / "a/x"} over it' for error in hpcp_mod.ERRORS)


# ---- rule 11: a path through another source's link ----

@posix_only
@pytest.mark.parametrize('flags', [[], ['-rf']], ids=['copy', 'move'])
def test_a_path_through_a_link_another_source_puts_there_is_a_clash(tmp_path, flags):
	# lb -> b named and lb/x listed: out/lb is the link, and lb/x would land through it. Every run ends the same: the link is
	# copied, lb/x is a 213, and nothing is moved (a move that reported an error removes nothing).
	for attempt in range(5):
		root = tmp_path / str(attempt)
		write(root / 'b/x', 'X')
		os.symlink('b', str(root / 'lb'))
		rc, output = run(['-sfl', '-', 'lb', '-d', 'out/', '-j', '8'] + flags, root, b'lb/x\n')
		assert rc == 213 and os.readlink(str(root / 'out/lb')) == 'b' and not os.path.exists(str(root / 'out/b')), output[-2000:]
		assert (b'Destination clash: ' + str(root / 'out/lb/x').encode() + b' lies below ' + str(root / 'out/lb').encode() +
			b', where this run puts the link ' + str(root / 'lb').encode() + b': not copying ' + str(root / 'lb/x').encode()) in output
		assert read(root / 'b/x') == 'X' and os.path.islink(str(root / 'lb'))


# ---- a small filesystem of its own ----

def _fill(out, free):
	"""Fill out's filesystem so that free bytes are left."""
	fill = os.path.join(out, 'fill')
	if os.path.exists(fill):
		os.unlink(fill)
	vfs = os.statvfs(out)
	write(fill, b'f' * (vfs.f_bavail * vfs.f_frsize - free))


# ---- 9.86: the full-disk warning: warn, then try and fail as GNU cp does ----

_MIB = 1 << 20
_HEADER = b'Space warnings in this run (repeated from above):'


@contextlib.contextmanager
def _on_small(name):
	"""A directory of its own on the small filesystem, emptied after."""
	out = os.path.join(SMALL_FS, '%s-%d' % (name, os.getpid()))
	subprocess.run(['rm', '-rf', out])
	os.makedirs(out)
	try:
		yield out
	finally:
		subprocess.run(['rm', '-rf', out])


def _warnings(output, where):
	"""(the live warning lines naming where, each starting a line of its own; the lines after the end's header)."""
	head, _, tail = output.partition(_HEADER)
	live = [line for line in head.split(b'\n') if line.startswith(b'Space warning: ') and where.encode() in line]
	repeated = [line for line in tail.split(b'\n') if line.startswith(b'Space warning: ')]
	return live, repeated


_LOGGING_CP = '''#!/bin/sh
# cp that notes each copy it starts in $LOG, then (after $PAUSE seconds, when set) runs the real cp. --version and the
# held-file probe (cp of a temporary file named hpcp-held-probe-*) are not copies.
case "$1 $2" in --version*|*/hpcp-held-probe-*) exec cp "$@";; esac
echo CP-STARTED >> "$LOG"
if [ -n "$PAUSE" ]; then sleep "$PAUSE"; fi
exec cp "$@"
'''


def _logged_run(tmp_path, args, to_file=True, pause=None):
	"""The CLI with cp noting each copy it starts in a log: (return code, the run's stdout and stderr). to_file: the run writes
	them to the same log, appended, so the log shows the order of the two; else they go to a pipe."""
	cp = tmp_path / 'logcp'
	cp.write_text(_LOGGING_CP)
	os.chmod(str(cp), 0o755)
	log = tmp_path / 'log'
	env = dict(os.environ, CP_PATH=str(cp), LOG=str(log), **({'PAUSE': str(pause)} if pause else {}))
	if not to_file:
		return run(args, tmp_path, env=env)
	with open(str(log), 'ab') as out:
		task = subprocess.run([sys.executable, HPCP] + args, cwd=str(tmp_path), stdin=subprocess.DEVNULL, stdout=out,
			stderr=subprocess.STDOUT, env=env, timeout=120)
	return task.returncode, log.read_bytes()


@posix_only
@small_fs
def test_the_space_warning_costs_nothing_where_a_file_fits(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# 1 MiB over 1 MiB with 3 MiB free: no line, and nothing of the warning runs (no stat, no mountinfo, no lock).
	with _on_small('fits') as out:
		write(os.path.join(out, 'x'), b'o' * _MIB)
		write(tmp_path / 'a/x', os.urandom(_MIB))
		_fill(out, 3 * _MIB)
		rc, output = run([str(tmp_path / 'a/x'), '-d', out + '/'], tmp_path)
		assert rc == 0 and b'Space warning' not in output and read_bytes(os.path.join(out, 'x')) == read_bytes(tmp_path / 'a/x'), output[-2000:]
		def never(*args):
			raise AssertionError('the warning ran on the normal path')
		monkeypatch.setattr(hpcp_mod, '_mount_type', never)
		monkeypatch.setattr(hpcp_mod, '_space_warning', never)
		write(tmp_path / 'a/x', os.urandom(_MIB))
		assert hpcp_mod.copy_file(str(tmp_path / 'a/x'), [os.path.join(out, 'x')])[0] > 0 and not hpcp_mod.ERRORS


@posix_only
@small_fs
def test_the_space_warning_comes_before_the_copy_it_is_about(tmp_path):
	# 0.5 MiB free, 1 MiB over a 1 MiB old copy: it fits only by reusing the old copy's room. The line is in the output before
	# cp starts, on a line of its own, and the copy then succeeds (exit 0, the inode kept).
	with _on_small('before') as out:
		write(os.path.join(out, 'x'), b'o' * _MIB)
		inode = os.stat(os.path.join(out, 'x')).st_ino
		write(tmp_path / 'a/x', os.urandom(_MIB))
		_fill(out, _MIB // 2)
		rc, output = _logged_run(tmp_path, [str(tmp_path / 'a/x'), '-d', out + '/'])
		assert rc == 0 and read_bytes(os.path.join(out, 'x')) == read_bytes(tmp_path / 'a/x'), output[-3000:]
		assert os.stat(os.path.join(out, 'x')).st_ino == inode
		live, repeated = _warnings(output, out)
		assert len(live) == 1 and repeated == live, output[-3000:]
		assert output.index(live[0]) < output.index(b'CP-STARTED'), output[-3000:]
		assert (b"; it fits only by reusing its old copy's room, which cp frees first, so if another writer takes that room the "
			b'copy fails and the old copy is lost (the source is kept). hpcp copies anyway, as cp does;') in live[0]
		assert live[0].startswith(('Space warning: %s has ' % out).encode()) and output.rstrip().endswith(b'Done.'), output[-3000:]


@posix_only
@small_fs
@pytest.mark.parametrize('form', ['pipe', 'verbose', 'file'])
def test_the_space_warning_is_printed_once_per_filesystem_and_repeated_at_the_end(tmp_path, form):
	# Four 1 MiB rewrites at once (-j 4) with 0.5 MiB free: each worker that runs one in the forced pass could warn; the run's
	# shared slots let one. cp waits 2 s before it frees an old copy, so every worker that checks the room in that time finds
	# 0.5 MiB and goes on to the warning (the first one always does). The end repeats it once, and the run ends Done. (a
	# warning is no error). Each line starts a line of its own, in a pipe and in a file.
	with _on_small('once') as out:
		for name in 'abcd':
			write(os.path.join(out, name), b'o' * _MIB)
			write(tmp_path / 'src' / name, os.urandom(_MIB))
		_fill(out, _MIB // 2)
		args = [str(tmp_path / 'src') + '/', '-d', out + '/', '-j', '4'] + (['-v'] if form == 'verbose' else [])
		rc, output = _logged_run(tmp_path, args, to_file=form == 'file', pause=2)
		assert rc == 0 and all(read_bytes(os.path.join(out, name)) == read_bytes(tmp_path / 'src' / name) for name in 'abcd'), output[-3000:]
		live, repeated = _warnings(output, out)
		assert len(live) == 1 and repeated == live, output[-3000:]
		assert b'Done.' in output and b'Error occurred' not in output, output[-3000:]


@posix_only
@small_fs
def test_a_doomed_rewrite_is_warned_about_and_still_tried(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# 3 MiB over a 1 MiB old copy with 1 MiB free: it may not fit. cp still runs (GNU cp's way), fails on the full disk, and the
	# old copy it truncated goes with the torn file (the approved known limitation): 190, the source kept. The warning
	# changes nothing in that: the same outcome without it.
	with _on_small('doomed') as out:
		landing = os.path.join(out, 'x')
		write(tmp_path / 'a/x', os.urandom(3 * _MIB))
		write(landing, b'o' * _MIB)
		_fill(out, _MIB)
		rc, output = _logged_run(tmp_path, [str(tmp_path / 'a/x'), '-d', out + '/'])
		assert rc == 190 and b'Copy failed' in output and b'CP-STARTED' in output, output[-3000:]
		live, _ = _warnings(output, out)
		assert len(live) == 1 and b', so it may not fit; if the disk runs out the copy fails, and a rewrite\'s old copy is lost' in live[0]
		assert output.index(live[0]) < output.index(b'CP-STARTED')
		assert not os.path.lexists(landing) and len(read_bytes(tmp_path / 'a/x')) == 3 * _MIB
		write(landing, b'o' * _MIB)
		_fill(out, _MIB)
		monkeypatch.setattr(hpcp_mod, '_space_warning', lambda *args: None)
		hpcp_mod.copy_file(str(tmp_path / 'a/x'), [landing])
		assert not os.path.lexists(landing) and any(error.startswith('Copy failed') for error in hpcp_mod.ERRORS)


@posix_only
@small_fs
def test_a_doomed_new_file_is_warned_about_and_still_tried(tmp_path):
	with _on_small('new') as out:
		write(tmp_path / 'a/x', os.urandom(6 * _MIB))
		rc, output = _logged_run(tmp_path, [str(tmp_path / 'a/x'), '-d', out + '/'])
		assert rc == 190 and b'CP-STARTED' in output and os.listdir(out) == [], output[-3000:]
		live, repeated = _warnings(output, out)
		assert len(live) == 1 and repeated == live and b'so it may not fit' in live[0], output[-3000:]


@posix_only
@small_fs
def test_no_space_warning_where_the_estimate_does_not_hold(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# btrfs (compression, shared blocks) and a source on an XFS's own filesystem (cp may clone it) get no line; the copy is
	# tried as ever.
	with _on_small('scope') as out:
		write(os.path.join(out, 'x'), b'o' * _MIB)
		write(tmp_path / 'a/x', os.urandom(_MIB))
		_fill(out, _MIB // 2)
		monkeypatch.setattr(hpcp_mod, '_mount_type', lambda dev: 'btrfs')
		assert hpcp_mod.copy_file(str(tmp_path / 'a/x'), [os.path.join(out, 'x')])[0] > 0
		assert not hpcp_mod.ERRORS and read_bytes(os.path.join(out, 'x')) == read_bytes(tmp_path / 'a/x')
		monkeypatch.undo()
		os.unlink(os.path.join(out, 'fill'))
		write(os.path.join(out, 'src/y'), os.urandom(_MIB))
		write(os.path.join(out, 'dst/y'), b'o' * _MIB)
		_fill(out, _MIB // 2)
		monkeypatch.setattr(hpcp_mod, '_mount_type', lambda dev: 'xfs')
		assert hpcp_mod.copy_file(os.path.join(out, 'src/y'), [os.path.join(out, 'dst/y')])[0] > 0
		assert not hpcp_mod.ERRORS and read_bytes(os.path.join(out, 'dst/y')) == read_bytes(os.path.join(out, 'src/y'))


@posix_only
@small_fs
def test_the_space_warning_is_printed_for_a_source_on_the_same_tmpfs(tmp_path):
	# ext2/3/4 and tmpfs never clone, so the source on the landing's own filesystem gets the warning as any other (9.86 as
	# built skipped every same-device copy): 1 MiB over a 1 MiB old copy with 0.5 MiB free fits only by reusing its room.
	with _on_small('same') as out:
		write(os.path.join(out, 'src/y'), os.urandom(_MIB))
		write(os.path.join(out, 'dst/y'), b'o' * _MIB)
		_fill(out, _MIB // 2)
		rc, output = run([os.path.join(out, 'src/y'), '-d', os.path.join(out, 'dst') + '/'], tmp_path)
		assert rc == 0 and read_bytes(os.path.join(out, 'dst/y')) == read_bytes(os.path.join(out, 'src/y')), output[-3000:]
		live, repeated = _warnings(output, os.path.join(out, 'dst'))
		assert len(live) == 1 and repeated == live, output[-3000:]


@posix_only
@small_fs
def test_the_space_warning_under_write_beside_says_the_old_copy_stays(tmp_path):
	# -wb needs room for a second copy: with 0.5 MiB free nothing of the old copy's room is reused (old 0 on the beside route),
	# so it may not fit; it does not, and the old copy stays, 190, no temporary file.
	with _on_small('beside') as out:
		write(os.path.join(out, 'x'), b'o' * _MIB)
		write(tmp_path / 'a/x', os.urandom(_MIB))
		_fill(out, _MIB // 2)
		rc, output = run([str(tmp_path / 'a/x'), '-d', out + '/', '-wb'], tmp_path)
		assert rc == 190 and read_bytes(os.path.join(out, 'x')) == b'o' * _MIB and temporaries(out) == [], output[-3000:]
		live, _ = _warnings(output, out)
		assert len(live) == 1 and b"so it may not fit; if the disk runs out the copy fails, and a rewrite's old copy stays (-wb)" in live[0], output[-3000:]


@posix_only
@small_fs
def test_several_destinations_spill_over_without_a_warning(tmp_path):
	# Unchanged: the first pass skips a destination without room, and the file lands on the next; the attempt that ran had
	# room, so nothing is warned about.
	with _on_small('spill') as out:
		os.makedirs(os.path.join(out, 'a'))
		write(tmp_path / 'src/x', os.urandom(3 * _MIB))
		_fill(out, _MIB)
		rc, output = run([str(tmp_path / 'src/x'), '-d', os.path.join(out, 'a') + '/', '-d', str(tmp_path / 'big') + '/'], tmp_path)
		assert rc == 0 and read_bytes(tmp_path / 'big/x') == read_bytes(tmp_path / 'src/x') and b'Space warning' not in output, output[-3000:]
		assert not os.path.exists(os.path.join(out, 'a/x'))


def test_the_space_slots_are_shared_by_the_run_s_workers(hpcp_mod, monkeypatch):
	# The run's own slots, under this Python's start method (fork: an inherited mapping; forkserver, 3.14's default: an
	# Array through the pool's initializer): a filesystem the parent claimed is claimed for the workers too, and one a
	# worker claimed for the parent.
	share = hpcp_mod._space_share()
	assert share is not None
	monkeypatch.setattr(hpcp_mod, '_SPACE_WARNED', share)
	assert hpcp_mod._claim_space(111) is True and hpcp_mod._claim_space(111) is False
	with hpcp_mod._copy_pool(2) as pool:
		assert [pool.submit(hpcp_mod._claim_space, 111).result() for _ in range(4)] == [False] * 4
		assert pool.submit(hpcp_mod._claim_space, 222).result() is True
		assert pool.submit(hpcp_mod._claim_space, 222).result() is False
	assert hpcp_mod._claim_space(222) is False and hpcp_mod._claim_space(333) is True


@pytest.mark.skipif(sys.version_info < (3, 7), reason="Python 3.6's pool takes no initializer (its workers fork)")
def test_the_space_slots_reach_spawned_workers(hpcp_mod, monkeypatch):
	import multiprocessing
	context = multiprocessing.get_context('spawn')
	array = context.Array('Q', hpcp_mod._SPACE_SLOTS)
	monkeypatch.setattr(hpcp_mod, '_SPACE_WARNED', (array.get_lock(), array.get_obj(), array))
	assert hpcp_mod._claim_space(7) is True
	from concurrent.futures import ProcessPoolExecutor
	with ProcessPoolExecutor(2, mp_context=context, initializer=hpcp_mod._init_space_share, initargs=(array,)) as pool:
		assert [pool.submit(hpcp_mod._claim_space, 7).result() for _ in range(3)] == [False] * 3
		assert pool.submit(hpcp_mod._claim_space, 8).result() is True
	assert hpcp_mod._claim_space(8) is False


def test_a_space_slot_lock_that_cannot_be_taken_prints_anyway(hpcp_mod, monkeypatch):
	# A worker that died holding the lock: after 1 s the warning prints anyway, and nothing hangs.
	# Held from another thread: the Array's lock (forkserver, spawn) is an RLock, which its own holder takes again at once.
	import threading
	share = hpcp_mod._space_share()
	assert share is not None
	monkeypatch.setattr(hpcp_mod, '_SPACE_WARNED', share)
	held, done = threading.Event(), threading.Event()
	def hold():
		with share[0]:
			held.set()
			done.wait(60)
	holder = threading.Thread(target=hold)
	holder.start()
	try:
		assert held.wait(30)
		started = time.monotonic()
		assert hpcp_mod._claim_space(5) is True and hpcp_mod._claim_space(5) is True
		assert 1.5 < time.monotonic() - started < 30
	finally:
		done.set()
		holder.join(60)
	assert hpcp_mod._claim_space(5) is True and hpcp_mod._claim_space(5) is False


def test_every_slot_taken_prints_each_time(hpcp_mod, monkeypatch):
	share = hpcp_mod._space_share()
	monkeypatch.setattr(hpcp_mod, '_SPACE_WARNED', share)
	assert all(hpcp_mod._claim_space(dev) for dev in range(hpcp_mod._SPACE_SLOTS))
	assert hpcp_mod._claim_space(10 ** 6) is True and hpcp_mod._claim_space(10 ** 6) is True and hpcp_mod._claim_space(3) is False


@pytest.mark.skipif(not hasattr(os, 'makedev'), reason='os.makedev')
def test_a_mount_type_is_read_from_mountinfo_s_own_fields(tmp_path, hpcp_mod, monkeypatch):
	sample = tmp_path / 'mountinfo'
	sample.write_bytes(b'22 1 259:2 / / rw,relatime shared:1 - xfs /dev/nvme0n1p2 rw,attr2\n'
		b'61 22 0:52 / /mnt/with\\040space rw,nosuid shared:30 master:2 - tmpfs tmpfs rw,size=5120k\n'
		b'62 22 0:53 / /mnt/- rw - ext4 /dev/loop0 rw\n'
		b'63 22 0:54 / /odd rw\n'
		b'garbage\n')
	real_entries = hpcp_mod._mountinfo_entries
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda: real_entries(str(sample)))
	monkeypatch.setattr(hpcp_mod, '_MOUNT_TYPES', {})
	assert [hpcp_mod._mount_type(os.makedev(*key)) for key in ((259, 2), (0, 52), (0, 53), (0, 54))] == ['xfs', 'tmpfs', 'ext4', None]


@pytest.mark.skipif(not hasattr(os, 'makedev'), reason='os.makedev')
def test_a_mount_type_miss_reads_mountinfo_once_more_then_is_kept(hpcp_mod, monkeypatch):
	reads = []
	tables = [{(8, 1): 'ext4'}, {(8, 1): 'ext4', (8, 2): 'xfs'}, {(8, 1): 'ext4', (8, 2): 'xfs', (8, 3): 'btrfs'}]
	def read_entries():
		reads.append(1)
		return [{'devnum': key, 'fstype': kind} for key, kind in tables[min(len(reads), len(tables)) - 1].items()]
	monkeypatch.setattr(hpcp_mod, '_MOUNT_TYPES', {})
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', read_entries)
	assert hpcp_mod._mount_type(os.makedev(8, 1)) == 'ext4' and len(reads) == 1
	assert hpcp_mod._mount_type(os.makedev(8, 1)) == 'ext4' and len(reads) == 1
	assert hpcp_mod._mount_type(os.makedev(8, 2)) == 'xfs' and len(reads) == 2  # mounted since: found on the reread
	assert hpcp_mod._mount_type(os.makedev(9, 9)) is None and len(reads) == 3
	assert hpcp_mod._mount_type(os.makedev(9, 9)) is None and len(reads) == 3  # the miss is kept


@root_only
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='loop mounts are Linux')
def test_a_root_copy_onto_an_ext4_full_for_users_gets_one_needless_warning(tmp_path):
	# No root rule: free space is f_bavail, so a root run onto an ext filesystem that is full for users (its reserved blocks
	# left) is warned about once, needlessly, and the copy fits into the reserve: exit 0. This pins that cost.
	import shutil
	mkfs = shutil.which('mkfs.ext4')
	if not mkfs:
		pytest.skip('needs mkfs.ext4')
	image, mount = tmp_path / 'ext.img', tmp_path / 'ext'
	with open(str(image), 'wb') as f:
		f.truncate(64 * _MIB)
	if subprocess.run([mkfs, '-q', '-F', '-m', '10', str(image)], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode:
		pytest.skip('mkfs.ext4 refused')
	mount.mkdir()
	result = subprocess.run(['mount', '-o', 'loop', str(image), str(mount)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
	if result.returncode:
		pytest.skip('mount failed: ' + result.stderr.strip())
	try:
		vfs = os.statvfs(str(mount))
		# Filled as a user would: up to f_bavail, by an unprivileged writer (root would go on into the reserve).
		with open(str(mount / 'fill'), 'wb') as f:
			f.write(b'f' * (vfs.f_bavail * vfs.f_frsize))
			f.flush()
			os.fsync(f.fileno())
		while os.statvfs(str(mount)).f_bavail > 0:
			with open(str(mount / 'fill'), 'ab') as f:
				f.write(b'f' * (os.statvfs(str(mount)).f_bavail * vfs.f_frsize or vfs.f_frsize))
		vfs = os.statvfs(str(mount))
		assert vfs.f_bavail == 0 and vfs.f_bfree * vfs.f_frsize > 4 * _MIB, vfs
		write(tmp_path / 'a/x', os.urandom(2 * _MIB))
		rc, output = run([str(tmp_path / 'a/x'), '-d', str(mount / 'out') + '/'], tmp_path)
		assert rc == 0 and read_bytes(mount / 'out/x') == read_bytes(tmp_path / 'a/x'), output[-3000:]
		live, repeated = _warnings(output, str(mount))
		assert len(live) == 1 and repeated == live, output[-3000:]
	finally:
		subprocess.run(['umount', '-l', str(mount)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# ---- rule 9: a stop is a cleanup ----

_SLOW_CP = '''#!/bin/sh
# cp, slowed: write a part of each copy into a temporary name (by its name, or the held file hpcp made there, through
# /proc/self/fd/N), say so, and wait for the go file (or 60 s) before the real cp.
case "$1 $2" in --version*|*/hpcp-held-probe-*) exec cp "$@";; esac
for last; do :; done
name=$(basename "$(readlink "$last" 2>/dev/null || echo "$last")")
case "$name" in
.hpcp_tmp_*)
	head -c 100 /etc/passwd > "$last"
	touch "$MARKS/$name"
	waited=0
	while [ ! -e "$GO" ] && [ $waited -lt 600 ]; do sleep 0.1; waited=$((waited + 1)); done;;
esac
exec cp "$@"
'''


_SLOW_CP_IN_PLACE = '''#!/bin/sh
# cp, slowed, for a rewrite in place: write a part of the copy into what it is given (the held file, /proc/self/fd/N), say
# so, and wait for the go file (or 60 s) before the real cp. --version and the held-file probe (cp of a temporary file
# named hpcp-held-probe-*) go straight to cp.
case "$1 $2" in --version*|*/hpcp-held-probe-*) exec cp "$@";; esac
for last; do :; done
head -c 100 /etc/passwd > "$last"
touch "$MARKS/$$"
waited=0
while [ ! -e "$GO" ] && [ $waited -lt 600 ]; do sleep 0.1; waited=$((waited + 1)); done
exec cp "$@"
'''


def _slow_run(tmp_path, args, preexec=None, script=_SLOW_CP, env=None):
	"""hpcp copying four files over old copies, each cp holding a part (beside its landing, or in it with _SLOW_CP_IN_PLACE)
	until tmp_path/go exists. env: more variables for the run."""
	cp = tmp_path / 'slowcp'
	cp.write_text(script)
	os.chmod(str(cp), 0o755)
	marks = tmp_path / 'marks'
	os.makedirs(str(marks))
	for name in 'abcd':
		write(tmp_path / 'src' / name, 'new ' + name)
		write(tmp_path / 'out' / name, 'old ' + name)
	env = dict(os.environ, CP_PATH=str(cp), MARKS=str(marks), GO=str(tmp_path / 'go'), **(env or {}))
	task = subprocess.Popen([sys.executable, HPCP, 'src/', '-d', 'out/'] + args, cwd=str(tmp_path), env=env,
		stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=preexec)
	return task, marks


def _wait_for(condition, timeout=60):
	deadline = time.monotonic() + timeout
	while not condition():
		assert time.monotonic() < deadline, 'timed out'
		time.sleep(0.05)


def _group_gone(pgid):
	'''No live process left in group pgid. One that ended and that nothing reaped (a zombie) counts as gone: a container whose
	PID 1 reaps no orphan keeps a killed run's ended workers so (GitHub's job container runs `tail -f /dev/null`).'''
	try:
		os.killpg(pgid, 0)
	except ProcessLookupError:
		return True
	except PermissionError:
		return False
	if not os.path.isdir('/proc/self'):
		return False
	for entry in os.listdir('/proc'):
		if not entry.isdigit():
			continue
		try:
			with open('/proc/%s/stat' % entry) as handle:
				# after the command's closing parenthesis: state, ppid, pgrp
				state, _, group = handle.read().rsplit(')', 1)[1].split()[:3]
		except (OSError, IndexError, ValueError):
			continue
		if int(group) == pgid and state != 'Z':
			return False
	return True


def _each_old_or_new(tmp_path):
	return all(read(tmp_path / 'out' / name) in ('old ' + name, 'new ' + name) for name in 'abcd')


@posix_only
@pytest.mark.parametrize('how, signum, code', [
	('group', signal.SIGTERM, -signal.SIGTERM),
	('main', signal.SIGTERM, -signal.SIGTERM),
	('group', signal.SIGHUP, -signal.SIGHUP),
	('group', signal.SIGINT, 130),
	('main', signal.SIGKILL, -signal.SIGKILL),
], ids=['term-group', 'term-main', 'hup-group', 'int-group', 'kill-main'])
@pytest.mark.parametrize('serial', [False, True], ids=['parallel', 'serial'])
def test_a_stop_leaves_no_temporary_file_and_ends_as_the_signal_asked(tmp_path, how, signum, code, serial):
	# verify/g.py: 9.84's prototype left complete or partial .hpcp_tmp_ files on SIGTERM and SIGINT. A SIGKILL of the main
	# process is cleaned up by the workers' watch (SIGUSR2); a serial run killed so has nobody left to clean up.
	if serial and signum == signal.SIGKILL:
		pytest.skip('a serial run killed with SIGKILL leaves its temporary file: the approved known limitation')
	task, marks = _slow_run(tmp_path, (['-s'] if serial else ['-j', '4']) + ['-wb'])
	try:
		_wait_for(lambda: len(os.listdir(str(marks))) >= (1 if serial else 4))
		if how == 'group':
			os.killpg(task.pid, signum)
		else:
			os.kill(task.pid, signum)
		output = task.communicate(timeout=60)[0]
		_wait_for(lambda: _group_gone(task.pid), timeout=30)
	finally:
		if task.poll() is None:
			os.killpg(task.pid, signal.SIGKILL)
	assert task.returncode == code, output[-2000:]
	assert temporaries(tmp_path / 'out') == [] and _each_old_or_new(tmp_path), output[-2000:]


@posix_only
@pytest.mark.parametrize('how, signum, code', [
	('group', signal.SIGTERM, -signal.SIGTERM),
	('main', signal.SIGTERM, -signal.SIGTERM),
	('group', signal.SIGHUP, -signal.SIGHUP),
	('group', signal.SIGINT, 130),
	('main', signal.SIGKILL, -signal.SIGKILL),
], ids=['term-group', 'term-main', 'hup-group', 'int-group', 'kill-main'])
@pytest.mark.parametrize('serial', [False, True], ids=['parallel', 'serial'])
def test_a_stop_during_a_rewrite_in_place_leaves_no_landing_torn(tmp_path, how, signum, code, serial):
	# A stop while cp writes into the held file: cp is stopped, and the torn file goes with its name (_remove_torn), so each
	# landing is old (not reached yet), new, or gone, never a part; the old copies being written are lost (the approved known
	# limitation), the sources kept. A SIGKILL of a parallel run's main process is handled by the workers' watch; a serial run
	# killed so leaves its landing torn (the same known limitation).
	if serial and signum == signal.SIGKILL:
		pytest.skip('a serial run killed with SIGKILL leaves its landing torn: the approved known limitation')
	task, marks = _slow_run(tmp_path, ['-s'] if serial else ['-j', '4'], script=_SLOW_CP_IN_PLACE)
	try:
		_wait_for(lambda: len(os.listdir(str(marks))) >= (1 if serial else 4))
		if how == 'group':
			os.killpg(task.pid, signum)
		else:
			os.kill(task.pid, signum)
		output = task.communicate(timeout=60)[0]
		_wait_for(lambda: _group_gone(task.pid), timeout=30)
	finally:
		if task.poll() is None:
			os.killpg(task.pid, signal.SIGKILL)
	assert task.returncode == code, output[-2000:]
	gone = [name for name in 'abcd' if not os.path.lexists(str(tmp_path / 'out' / name))]
	assert len(gone) == (1 if serial else 4), (gone, output[-2000:])
	assert all(read(tmp_path / 'out' / name) in ('old ' + name, 'new ' + name) for name in 'abcd' if name not in gone)
	assert all(read(tmp_path / 'src' / name) == 'new ' + name for name in 'abcd') and temporaries(tmp_path / 'out') == []


@posix_only
def test_a_nohup_run_keeps_going_on_a_hangup(tmp_path):
	# nohup starts the run with SIGHUP ignored: it stays ignored, in the workers too, as in 9.84.
	task, marks = _slow_run(tmp_path, ['-j', '4', '-wb'], preexec=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN))
	try:
		_wait_for(lambda: len(os.listdir(str(marks))) >= 4)
		os.killpg(task.pid, signal.SIGHUP)
		time.sleep(1)
		(tmp_path / 'go').write_text('')
		output = task.communicate(timeout=60)[0]
	finally:
		if task.poll() is None:
			os.killpg(task.pid, signal.SIGKILL)
	assert task.returncode == 0 and all(read(tmp_path / 'out' / name) == 'new ' + name for name in 'abcd'), output[-2000:]
	assert temporaries(tmp_path / 'out') == []


@posix_only
@pytest.mark.skipif(not hasattr(signal, 'pthread_kill'), reason='pthread_kill')
def test_a_run_started_with_sigint_ignored_cleans_up_when_its_parent_is_killed(tmp_path):
	# A background job of a non-interactive sh on EL7 starts with SIGINT ignored. Only the watch's SIGUSR2 can stop the
	# workers' cp once the main process is gone: _thread.interrupt_main would run SIG_IGN and raise TypeError on 3.6.
	task, marks = _slow_run(tmp_path, ['-j', '4', '-wb'], preexec=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
	try:
		_wait_for(lambda: len(os.listdir(str(marks))) >= 4)
		os.kill(task.pid, signal.SIGKILL)
		task.communicate(timeout=60)
		_wait_for(lambda: _group_gone(task.pid), timeout=30)
	finally:
		if task.poll() is None:
			os.killpg(task.pid, signal.SIGKILL)
	assert temporaries(tmp_path / 'out') == [] and _each_old_or_new(tmp_path)


def test_stop_handlers_keep_an_ignored_signal_ignored(hpcp_mod, monkeypatch):
	installed = {}
	monkeypatch.setattr(hpcp_mod.signal, 'getsignal', lambda signum: signal.SIG_IGN if signum == signal.SIGHUP else signal.SIG_DFL)
	monkeypatch.setattr(hpcp_mod.signal, 'signal', lambda signum, handler: installed.__setitem__(signum, handler))
	hpcp_mod._install_stops()
	assert installed == {signal.SIGINT: hpcp_mod._stop, signal.SIGTERM: hpcp_mod._stop}


@posix_only
def test_a_stop_prints_no_traceback_from_workers_that_ran_no_task(tmp_path):
	# Python 3.6 to 3.13 fork the pool's workers, and a worker has _stop before its first task: each idle one printed a
	# KeyboardInterrupt traceback from the pool's queue (four here: four files, eight workers).
	task, marks = _slow_run(tmp_path, ['-j', '8', '-wb'])
	try:
		_wait_for(lambda: len(os.listdir(str(marks))) >= 4)
		os.kill(task.pid, signal.SIGTERM)
		output = task.communicate(timeout=60)[0]
		_wait_for(lambda: _group_gone(task.pid), timeout=30)
	finally:
		if task.poll() is None:
			os.killpg(task.pid, signal.SIGKILL)
	assert task.returncode == -signal.SIGTERM and b'Traceback' not in output, output[-2000:]
	assert temporaries(tmp_path / 'out') == [] and _each_old_or_new(tmp_path), output[-2000:]


def test_a_stop_in_a_task_s_setup_ends_the_worker(hpcp_mod, monkeypatch):
	# 9.85's first build: a stop in _init_worker raised KeyboardInterrupt outside _run_task's handler; the pool took it for the
	# task's error and kept the worker, which then ignored every later stop, and the pool's shutdown waited for it for good
	# (an early Ctrl-C hung about 1 run in 20 on EL8, Python 3.6).
	ended = []
	def interrupted():
		hpcp_mod._STOPPED, hpcp_mod._STOPPED_PID = signal.SIGTERM, os.getpid()  # what _stop records
		raise KeyboardInterrupt
	monkeypatch.setattr(hpcp_mod, '_STOPPED', None)
	monkeypatch.setattr(hpcp_mod, '_STOPPED_PID', None)
	monkeypatch.setattr(hpcp_mod, '_init_worker', interrupted)
	monkeypatch.setattr(hpcp_mod, '_end_by', ended.append)
	hpcp_mod._run_task({}, lambda: None)
	assert ended == [signal.SIGTERM] and not hpcp_mod._IN_TASK


def test_a_worker_the_pool_kept_after_its_stop_ends_at_its_next_task(hpcp_mod, monkeypatch):
	# Python 3.7+: a stop landing while the pool's worker loop sends a task's result back (_sendback_result) raises SystemExit
	# there, which the loop sends back as that task's error and keeps the worker. Its next task ran whole, every stop during it
	# ignored as already handled (the pass-on, the pool's terminate), and only then did the worker end.
	ended, ran = [], []
	class Ended(Exception):
		pass
	def end_by(signum):
		ended.append(signum)
		raise Ended
	monkeypatch.setattr(hpcp_mod, '_STOPPED', signal.SIGTERM)
	monkeypatch.setattr(hpcp_mod, '_STOPPED_PID', os.getpid())
	monkeypatch.setattr(hpcp_mod, '_end_by', end_by)
	with pytest.raises(Ended):
		hpcp_mod._run_task({}, lambda: ran.append(True))
	assert ended == [signal.SIGTERM] and ran == [] and not hpcp_mod._IN_TASK


@pytest.mark.parametrize('where', ['start', 'end'])
def test_a_stop_at_a_task_s_start_or_end_ends_the_worker(hpcp_mod, monkeypatch, where):
	# Between tasks _stop raises SystemExit, which the pool's worker loop takes for the task's error (except BaseException,
	# Python 3.6 to 3.14) and goes on: the worker outlived its stop, and the parent's _task_result raised SystemExit.
	ended = []
	monkeypatch.setattr(hpcp_mod, '_STOPPED', None)
	monkeypatch.setattr(hpcp_mod, '_STOPPED_PID', None)
	monkeypatch.setattr(hpcp_mod, '_MAIN_PID', os.getpid() + 1)
	monkeypatch.setattr(hpcp_mod, '_worker_initialized', True)
	monkeypatch.setattr(hpcp_mod, '_IN_TASK', False)
	monkeypatch.setattr(hpcp_mod, '_end_by', ended.append)
	stop = lambda *args: hpcp_mod._stop(signal.SIGTERM, None)
	if where == 'start':
		monkeypatch.setattr(hpcp_mod, '_init_worker', stop)
		hpcp_mod._run_task({}, lambda: None)
	else:
		class Errors(list):
			def __delitem__(self, key):
				list.__delitem__(self, key)
				stop()
		monkeypatch.setattr(hpcp_mod, 'ERRORS', Errors())
		hpcp_mod._run_task({}, lambda: None)
	assert ended == [signal.SIGTERM] and not hpcp_mod._IN_TASK


# Python reports the dropped KeyboardInterrupt, as it would in the worker; pytest turns that report into a warning.
@pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning')
def test_a_stop_whose_keyboardinterrupt_python_drops_still_ends_the_worker(hpcp_mod, monkeypatch):
	# Python 3.14, 6da5ecf: the pool's SIGTERM reached a worker's _stop inside a weakref callback (WeakSet._remove), where
	# Python reports the KeyboardInterrupt and drops it; the task went on, the pool's second SIGTERM was ignored as a later
	# stop, and the pool waited for the worker for good.
	import weakref
	ended = []
	monkeypatch.setattr(hpcp_mod, '_STOPPED', None)
	monkeypatch.setattr(hpcp_mod, '_STOPPED_PID', None)
	monkeypatch.setattr(hpcp_mod, '_MAIN_PID', os.getpid() + 1)
	monkeypatch.setattr(hpcp_mod, '_worker_initialized', True)
	monkeypatch.setattr(hpcp_mod, '_end_by', ended.append)
	monkeypatch.setattr(hpcp_mod, '_pass_stop_on', lambda signum: None)
	class Thing:
		pass
	def task():
		thing = Thing()
		keep = weakref.ref(thing, lambda ref: hpcp_mod._stop(signal.SIGTERM, None))
		del thing  # the callback runs here; its KeyboardInterrupt is reported and dropped
		return 'went on'
	hpcp_mod._run_task({}, task)
	assert ended == [signal.SIGTERM] and not hpcp_mod._IN_TASK


def test_a_stop_while_a_worker_ends_does_not_keep_it(hpcp_mod, monkeypatch):
	# 6da5ecf on EL8: a worker stopped in its task was flushing in _end_by when the pool's SIGTERM came; _stop raised
	# SystemExit out of the ending, the pool's worker loop kept the worker, and the pool waited for it for good.
	monkeypatch.setattr(hpcp_mod, '_ENDING', False)
	monkeypatch.setattr(hpcp_mod, '_STOPPED', None)
	monkeypatch.setattr(hpcp_mod, '_STOPPED_PID', None)
	monkeypatch.setattr(hpcp_mod, '_MAIN_PID', os.getpid() + 1)
	monkeypatch.setattr(hpcp_mod, '_IN_TASK', False)
	seen = []
	def flush():
		seen.append(hpcp_mod._stop(signal.SIGTERM, None))
	monkeypatch.setattr(hpcp_mod.sys.stdout, 'flush', flush)
	monkeypatch.setattr(hpcp_mod.signal, 'signal', lambda signum, handler: None)
	monkeypatch.setattr(hpcp_mod, '_block_stops', lambda: None)
	monkeypatch.setattr(hpcp_mod.os, 'kill', lambda pid, signum: None)
	monkeypatch.setattr(hpcp_mod.os, '_exit', lambda code: None)
	hpcp_mod._end_by(signal.SIGINT)
	assert seen == [None]


_END_BY = '''
import os, signal, sys
sys.path.insert(0, %(root)r)
import hpcp
hpcp._install_stops()
real = signal.signal
def switch(signum, handler):
	if handler is signal.SIG_DFL:
		with open(%(seen)r, 'w') as f:
			f.write(repr(signum in signal.pthread_sigmask(signal.SIG_BLOCK, [])))
	return real(signum, handler)
signal.signal = switch
hpcp._end_by(signal.SIGINT)
'''


@posix_only
def test_a_stop_cannot_land_while_the_ending_switches_to_the_default_action(tmp_path):
	# test_an_early_stop_prints_no_traceback, about one run in 150 on 3.14: a worker in _end_by got the pool's pass-on of the
	# Ctrl-C inside signal.signal(SIG_DFL), after Python's check for pending stops; Python then found it pending with the
	# default action set and printed 'Exception ignored ... Signal 2 ignored due to race condition' with a traceback. The
	# window is inside Python's C code, so this checks the stop is blocked there, and that the process still ends by it.
	seen = tmp_path / 'seen'
	code = subprocess.run([sys.executable, '-c', _END_BY % {'root': os.path.dirname(HPCP), 'seen': str(seen)}], timeout=60).returncode
	assert (code, seen.read_text()) == (-signal.SIGINT, 'True')


def test_a_second_stop_after_a_task_s_cleanup_ends_the_worker(hpcp_mod, monkeypatch):
	# The first stop (SIGTERM) runs the task's cleanup; a second one (SIGINT) landing after _IN_TASK = False raised SystemExit
	# out of _run_task in 6da5ecf, in place of the KeyboardInterrupt, and the pool kept the worker.
	ended = []
	monkeypatch.setattr(hpcp_mod, '_STOPPED', None)
	monkeypatch.setattr(hpcp_mod, '_STOPPED_PID', None)
	monkeypatch.setattr(hpcp_mod, '_MAIN_PID', os.getpid() + 1)
	monkeypatch.setattr(hpcp_mod, '_worker_initialized', True)
	monkeypatch.setattr(hpcp_mod, '_IN_TASK', False)
	monkeypatch.setattr(hpcp_mod, '_end_by', ended.append)
	monkeypatch.setattr(hpcp_mod, '_pass_stop_on', lambda signum: None)
	class Errors(list):
		def __delitem__(self, key):
			list.__delitem__(self, key)
			hpcp_mod._stop(signal.SIGINT, None)
	monkeypatch.setattr(hpcp_mod, 'ERRORS', Errors())
	def task():
		hpcp_mod._stop(signal.SIGTERM, None)
	hpcp_mod._run_task({}, task)
	# The second stop ends the worker at once (_stop); this stub of _end_by returns, so _run_task's own _end_by follows.
	assert ended and set(ended) == {signal.SIGTERM} and not hpcp_mod._IN_TASK


def test_a_forked_worker_sets_itself_up_at_its_start_only_in_a_run(tmp_path):
	# 9.85's worker setup runs at a worker's start, from multiprocessing's after-fork hook that main() registers: a worker
	# that never ran a task (or that its pool never counted) still had no parent watch, and outlived the run. An import of
	# hpcp alone must not register it: another program's multiprocessing children would get hpcp's stop handlers.
	if 'fork' not in __import__('multiprocessing').get_all_start_methods():
		pytest.skip('no fork start method')
	script = tmp_path / 'probe.py'
	script.write_text('''
import multiprocessing, sys
sys.path.insert(0, %r)
import hpcp
def child(queue):
	queue.put(hpcp._worker_initialized)
def run():
	context = multiprocessing.get_context('fork')
	queue = context.Queue()
	process = context.Process(target=child, args=(queue,))
	process.start()
	got = queue.get(timeout=60)
	process.join()
	return got
before = run()
hpcp._register_worker_start()
print(before, run())
''' % os.path.dirname(HPCP))
	task = subprocess.run([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
	assert task.stdout.decode().split()[-2:] == ['False', 'True'], task.stdout


@pytest.mark.parametrize('initialized', [False, True], ids=['no-task-yet', 'between-tasks'])
def test_an_idle_worker_ends_on_its_own_stop_after_its_parent_s(hpcp_mod, monkeypatch, initialized):
	# A worker forked after its parent's stop has the parent's _STOPPED: that must not make it ignore a stop of its own.
	other = os.getpid() + 1
	monkeypatch.setattr(hpcp_mod, '_STOPPED', signal.SIGINT)
	monkeypatch.setattr(hpcp_mod, '_STOPPED_PID', other)
	monkeypatch.setattr(hpcp_mod, '_MAIN_PID', other)
	monkeypatch.setattr(hpcp_mod, '_worker_initialized', initialized)
	monkeypatch.setattr(hpcp_mod, '_IN_TASK', False)
	with pytest.raises(SystemExit):
		hpcp_mod._stop(signal.SIGTERM, None)


# ---- 9.87 group 1: stops end, and end cleanly (entries 16, 17, 18) ----

_WRAP = '''import os, random, signal, struct, sys, time, weakref
sys.path.insert(0, %(root)r)
import hpcp
# What a test puts here runs in every process of the run: forked workers inherit it, and forkserver and spawn ones import
# this file as __mp_main__.
%(inject)s
if __name__ == '__main__':
	%(before)s
	sys.argv = [%(hpcp)r] + sys.argv[1:]
	hpcp.main()
'''


def _wrapped(tmp_path, args, inject='', before='pass', env=None, cwd=None):
	"""hpcp.main() through a wrapper script whose top level runs inject; in a session of its own."""
	script = tmp_path / 'wrap.py'
	script.write_text(_WRAP % {'root': os.path.dirname(HPCP), 'inject': inject, 'before': before, 'hpcp': HPCP})
	return subprocess.Popen([sys.executable, str(script)] + args, cwd=str(cwd or tmp_path), env=dict(os.environ, **(env or {})),
		stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)


def _ended(task, timeout=60):
	"""(return code, output, seconds it took from now); 'timeout' after 60 s, its session killed: a hang fails, never holds,
	and keeps what the run printed before it was killed."""
	started = time.monotonic()
	try:
		output = task.communicate(timeout=timeout)[0]
	except subprocess.TimeoutExpired:
		os.killpg(task.pid, signal.SIGKILL)
		return 'timeout', task.communicate()[0], time.monotonic() - started
	return task.returncode, output, time.monotonic() - started


def _files(root, count, prefix='f', content='x'):
	for i in range(count):
		write(root / ('%s%03d' % (prefix, i)), content)


@posix_only
def test_a_stop_inside_futures_wait_ends_the_run(tmp_path):
	# r985d/probe/futlock.py: the parent's stop raised inside concurrent.futures.wait's _AcquireFutures.__enter__, after it
	# took the first future's lock, left that lock held; the pool's shutdown then waited on it for good (9.86: hangs). In a
	# pool the parent's stop is only recorded there, and the loop raises it once the wait returns.
	_files(tmp_path / 'src', 8)
	inject = '''
from concurrent.futures import _base
armed = [True]
def enter(self):
	for n, future in enumerate(sorted(self.futures, key=id)):
		future._condition.acquire()
		if n == 0 and armed[0] and os.getpid() == hpcp._MAIN_PID:
			armed[0] = False
			with open(os.environ['MARK'], 'w') as f:
				f.write('%r' % time.time())
			os.kill(os.getpid(), signal.SIGINT)
_base._AcquireFutures.__enter__ = enter
'''
	task = _wrapped(tmp_path, ['src/', '-d', 'out/', '-j', '2'], inject, env={'MARK': str(tmp_path / 'mark')})
	code, output, _ = _ended(task)
	ended = time.time()
	assert code == 130, output[-3000:]
	# Within the loop's next pass (1 s), the workers' end (they had the stop at once), and slack for a loaded host.
	assert ended - float((tmp_path / 'mark').read_text()) < 1 + _GRACE + 2 + 3, output[-3000:]
	assert b'Traceback' not in output, output[-3000:]


_GRACE = 4  # hpcp._WATCH_GRACE


@posix_only
@pytest.mark.skipif(not hasattr(signal, 'pthread_sigmask'), reason='pthread_sigmask')
def test_a_worker_stuck_between_tasks_is_killed_after_the_grace(tmp_path):
	# A worker that sticks after its task (as on the pool's queue lock: inferred from 9.85's hang dumps), its stops blocked
	# so nothing but a kill ends it. 9.86's pool shutdown waited for it for good; 9.87 kills it 4 s after the stop: it holds
	# nothing of a file (its slot reads 'between tasks').
	_files(tmp_path / 'src', 8)
	_files(tmp_path / 'out', 8, content='old')
	inject = '''
real_run_task = hpcp._run_task
def run_task(*args, **kwargs):
	result = real_run_task(*args, **kwargs)
	try:
		fd = os.open(os.environ['STUCK'], os.O_CREAT | os.O_EXCL | os.O_WRONLY)
	except FileExistsError:
		return result
	signal.pthread_sigmask(signal.SIG_BLOCK, hpcp._STOPS)
	os.write(fd, str(os.getpid()).encode())
	os.close(fd)
	while True:
		time.sleep(60)
hpcp._run_task = run_task
'''
	stuck = tmp_path / 'stuck'
	task = _wrapped(tmp_path, ['src/', '-d', 'out/', '-j', '4', '-wb'], inject, env={'STUCK': str(stuck)})
	try:
		_wait_for(lambda: stuck.exists() and stuck.read_text())
		pid = int(stuck.read_text())
		os.killpg(task.pid, signal.SIGINT)
		code, output, took = _ended(task)
	finally:
		if task.poll() is None:
			os.killpg(task.pid, signal.SIGKILL)
	# The grace, the kill's 1 s join, and slack for a loaded host.
	assert code == 130 and took < _GRACE + 1 + 5, (took, output[-3000:])
	# Killed by the run, not by its own parent watch, which would end it 4.5 s after the run.
	_wait_for(lambda: _dead(pid), timeout=1)
	assert temporaries(tmp_path / 'out') == [] and b'Traceback' not in output, output[-3000:]


def _dead(pid):
	"""pid has exited (gone, or a zombie not reaped yet)."""
	try:
		with open('/proc/%d/stat' % pid) as f:
			return f.read().rsplit(')', 1)[1].split()[0] == 'Z'
	except FileNotFoundError:
		return True
	except OSError:
		pass
	try:
		os.kill(pid, 0)
	except ProcessLookupError:
		return True
	return False


@posix_only
@pytest.mark.skipif(not hasattr(signal, 'pthread_sigmask'), reason='pthread_sigmask')
def test_a_worker_in_a_long_cleanup_is_waited_for_not_killed(tmp_path):
	# A stop while a worker fsyncs a beside temporary file (-wb), made to take 8 s with the stops held back, as a real fsync
	# is not cut by a signal: the run waits for it past the grace, says so once, and its cleanup removes the temporary file
	# (the approved entries: SIGINT, SIGTERM and SIGHUP remove it).
	write(tmp_path / 'src' / 'a', 'new')
	write(tmp_path / 'out' / 'a', 'old')
	inject = '''
def slow(path):
	if os.path.basename(path).startswith('.hpcp_tmp_'):
		mask = signal.pthread_sigmask(signal.SIG_BLOCK, hpcp._STOPS)
		open(os.environ['MARK'], 'w').close()
		time.sleep(8)
		open(os.environ['MARK'] + '.done', 'w').close()
		signal.pthread_sigmask(signal.SIG_SETMASK, mask)
real_fsync, real_os_fsync = hpcp._fsync_file, os.fsync
def fsync_file(path, *args):
	slow(path)
	return real_fsync(path, *args)
def os_fsync(fd):
	# the beside route's new file, made and held by hpcp, is fsynced through its descriptor
	slow(os.readlink('/proc/self/fd/%d' % fd) if os.path.isdir('/proc/self/fd') else '')
	return real_os_fsync(fd)
hpcp._fsync_file, os.fsync = fsync_file, os_fsync
'''
	mark = tmp_path / 'mark'
	task = _wrapped(tmp_path, ['src/', '-d', 'out/', '-j', '2', '-wb'], inject, env={'MARK': str(mark)})
	try:
		_wait_for(mark.exists)
		os.killpg(task.pid, signal.SIGINT)
		code, output, took = _ended(task)
	finally:
		if task.poll() is None:
			os.killpg(task.pid, signal.SIGKILL)
	assert code == 130, output[-3000:]
	assert output.count(b'Waiting for 1 worker(s) still finishing a file') == 1, output[-3000:]
	assert (tmp_path / 'mark.done').exists() and took > _GRACE, (took, output[-3000:])
	assert temporaries(tmp_path / 'out') == [] and read(tmp_path / 'out' / 'a') == 'old', output[-3000:]


@posix_only
def test_a_worker_killed_while_it_sends_a_big_result_ends_the_run(tmp_path):
	# A worker killed after the first 16 KiB of a 200 KiB result reached the pipe (a listing of a wide directory, -pfl): the
	# pool's manager thread waits for the rest for good, as the parent holds the pipe's write end too, and no future ever
	# completes (9.86: hangs). The pool's watch ends the run: 191, and the move removes nothing.
	wide = tmp_path / 'src' / 'wide'
	os.makedirs(str(wide))
	for i in range(3000):
		open(str(wide / ('%04d-' % i + 'n' * 60)), 'w').close()
	write(tmp_path / 'src' / 'top', 'top')
	inject = '''
import multiprocessing.connection
real_send = multiprocessing.connection.Connection._send_bytes
def send_bytes(self, buf):
	if len(buf) > 100000 and os.getpid() != hpcp._MAIN_PID:
		self._send(struct.pack('!i', len(buf)))
		self._send(buf[:16384])
		os.kill(os.getpid(), signal.SIGKILL)
	return real_send(self, buf)
multiprocessing.connection.Connection._send_bytes = send_bytes
'''
	task = _wrapped(tmp_path, ['src/', '-d', 'out/', '-j', '2', '-pfl', '-rf'], inject)
	code, output, took = _ended(task)
	assert code == 191, output[-3000:]
	assert b'a pool worker died and its pool stopped answering' in output, output[-3000:]
	# The watch's pass (1 s) and grace, then the exit's grace for a worker stuck on the lock the dead one held, a 1 s join,
	# and slack for a loaded host.
	assert took < 1 + _GRACE + _GRACE + 1 + 5, (took, output[-3000:])
	assert len(os.listdir(str(wide))) == 3000 and read(tmp_path / 'src' / 'top') == 'top'


_DROP_STOP = '''
import weakref
class Thing:
	pass
def drop_a_stop():
	"""Python drops a KeyboardInterrupt raised in a weakref callback (every Python, 3.6 to 3.14): _stop's raise in there."""
	def callback(ref):
		os.kill(os.getpid(), signal.SIGINT)
		for _ in range(1000):
			pass
	thing = Thing()
	keep = weakref.ref(thing, callback)
	del thing
'''


@posix_only
@pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning')
@pytest.mark.parametrize('how', ['move', 'rme'])
def test_a_dropped_keyboardinterrupt_in_the_parent_is_acted_on(tmp_path, how):
	# Entry 17: the parent's stop, its KeyboardInterrupt dropped by Python outside a pool (here as the links are made): 9.86
	# went on, a move removing its sources (exit 0), -rme to its prompt. 9.87 acts on it at the next step: 130, nothing removed,
	# no prompt.
	_files(tmp_path / 'src', 6)
	os.symlink('f000', str(tmp_path / 'src' / 'link'))
	write(tmp_path / 'out' / 'extra', 'extra')
	inject = _DROP_STOP + '''
real_links = hpcp.create_sym_links
def create_sym_links(*args, **kwargs):
	drop_a_stop()
	return real_links(*args, **kwargs)
hpcp.create_sym_links = create_sym_links
'''
	args = ['src/', '-d', 'out/', '-j', '2'] + (['-rf'] if how == 'move' else ['-rme'])
	code, output, _ = _ended(_wrapped(tmp_path, args, inject))
	assert code == 130, output[-3000:]
	assert sorted(os.listdir(str(tmp_path / 'src'))) == ['f%03d' % i for i in range(6)] + ['link']
	assert b'Do you want to' not in output and read(tmp_path / 'out' / 'extra') == 'extra', output[-3000:]


@posix_only
def test_a_broken_pool_from_a_stop_ends_130_not_191(tmp_path):
	# Entry 17 on 3.14 (r985d/probe/lostki.py): the parent's stop landed in a weakref callback as a pool ran; its workers ended
	# on the pass-on, the pool read as broken, and the run ended 191, 'Copy process crashed'. Inside a pool the stop is only
	# recorded, and the loop acts on it: 130.
	_files(tmp_path / 'src', 40)
	inject = _DROP_STOP + '''
calls = [0]
real_submit = hpcp._submit_task
def submit_task(executor, func, *args, **kwargs):
	calls[0] += 1
	if calls[0] == 5 and os.getpid() == hpcp._MAIN_PID:
		drop_a_stop()
	return real_submit(executor, func, *args, **kwargs)
hpcp._submit_task = submit_task
'''
	code, output, _ = _ended(_wrapped(tmp_path, ['src/', '-d', 'out/', '-j', '2'], inject))
	assert code == 130 and b'Copy process crashed' not in output, output[-3000:]


@posix_only
def test_a_second_stop_is_ignored_and_names_ctrl_backslash(tmp_path):
	# Later stops stay ignored, so the cleanup is never cut (a -di image stays mounted otherwise), and the first one says once
	# how to end at once.
	inject = '''
real_clean_up = hpcp.clean_up
def clean_up(*args, **kwargs):
	open(os.environ['MARK'], 'w').close()
	time.sleep(6)
	real_clean_up(*args, **kwargs)
	open(os.environ['MARK'] + '.done', 'w').close()
hpcp.clean_up = clean_up
'''
	cp = tmp_path / 'slowcp'
	cp.write_text(_SLOW_CP_IN_PLACE)
	os.chmod(str(cp), 0o755)
	marks, mark = tmp_path / 'marks', tmp_path / 'mark'
	os.makedirs(str(marks))
	_files(tmp_path / 'src', 4)
	_files(tmp_path / 'out', 4, content='old')
	task = _wrapped(tmp_path, ['src/', '-d', 'out/', '-j', '2'], inject,
		env={'CP_PATH': str(cp), 'MARKS': str(marks), 'GO': str(tmp_path / 'go'), 'MARK': str(mark)})
	try:
		_wait_for(lambda: len(os.listdir(str(marks))) >= 1)
		os.killpg(task.pid, signal.SIGINT)
		_wait_for(mark.exists)
		os.kill(task.pid, signal.SIGINT)
		time.sleep(0.5)
		os.kill(task.pid, signal.SIGTERM)
		code, output, _ = _ended(task)
	finally:
		if task.poll() is None:
			os.killpg(task.pid, signal.SIGKILL)
	assert code == 130 and (tmp_path / 'mark.done').exists(), output[-3000:]
	assert output.count(b'Stopping: cleaning up first') == 1 and b'Ctrl-\\ ends hpcp now' in output, output[-3000:]


_SLOWER_CP = '''#!/bin/sh
# cp, each copy 0.05 s slower: a run long enough to be stopped early in it.
case "$1" in --version) exec cp "$@";; esac
sleep 0.05
exec cp "$@"
'''


def _start_methods():
	import multiprocessing
	return [method for method in ('fork', 'forkserver', 'spawn') if method in multiprocessing.get_all_start_methods()]


# Python 3.6 and 3.7 have no sys.unraisablehook: a raise Python drops (a generator the collector closes, a weakref callback, a
# __del__) is printed by the interpreter itself. limitations.md, "On Python 3.6 and 3.7 an early Ctrl-C can print an
# `Exception ignored ... KeyboardInterrupt` block" (F8, about 1 early stop in 2,000). Only hpcp's own stop, raised by _stop
# after it recorded it, so the run still acts on it (130). The block can follow a progress bar frame on its line.
_DROPPED_STOP = re.compile(rb'Exception ignored in: [^\n]*\nTraceback \(most recent call last\):\n(?:  [^\n]*\n)*?'
	rb'  File "[^"\n]*hpcp\.py", line \d+, in _stop\n    raise KeyboardInterrupt\nKeyboardInterrupt(?::[^\n]*)?\n')

# Item (b), 9.90's deferral (docs/limitations.md "On Python 3.6 and 3.7 an early stop with forkserver or spawn can print a worker's
# EOFError"): a stop's SystemExit lands in an idle worker between a queued job's length and its payload, and the next worker reads
# the payload as a length: 2 early stops in 3,800 on 3.6 in 9.87, 0 in 3,800 in 9.88's review; old, and the run still ends 130.
# Only that worker's traceback is accepted: _process_worker's call_queue.get ending in EOFError. The line can follow a bar frame.
_OLD_EOF = re.compile(rb'Process [^\n]*:\nTraceback \(most recent call last\):\n(?:  [^\n]*\n)*?'
	rb'  File "[^"\n]*process\.py", line \d+, in _process_worker\n    call_item = call_queue\.get\(block=True\)\n'
	rb'(?:  [^\n]*\n)*?EOFError: Ran out of input\n')


def _without_old_blocks(output):
	"""output with the known blocks of Python 3.6 and 3.7 (_DROPPED_STOP, _OLD_EOF) taken out."""
	return _OLD_EOF.sub(b'', _DROPPED_STOP.sub(b'', output))


def _expected_old_blocks(output):
	"""What an early stop's output may not hold: it, less the known blocks on 3.6 and 3.7; whole on 3.8 and later."""
	return _without_old_blocks(output) if sys.version_info < (3, 8) else output


_EOF_BLOCK = (b'1 F Synced | 0 Scheduled |\xe2\x96\x88|100.0% | 0sProcess Process-2:\n'
	b'Traceback (most recent call last):\n'
	b'  File "/usr/local/lib/python3.6/multiprocessing/process.py", line 258, in _bootstrap\n    self.run()\n'
	b'  File "/usr/local/lib/python3.6/multiprocessing/process.py", line 93, in run\n    self._target(*self._args, **self._kwargs)\n'
	b'  File "/usr/local/lib/python3.6/concurrent/futures/process.py", line 169, in _process_worker\n'
	b'    call_item = call_queue.get(block=True)\n'
	b'  File "/usr/local/lib/python3.6/multiprocessing/queues.py", line 113, in get\n    return _ForkingPickler.loads(res)\n'
	b'EOFError: Ran out of input\n')


def test_only_the_old_eoferror_block_is_taken_out_of_an_early_stop_s_output():
	# the block as the 9.88 review saw it (3.6 forkserver and spawn), after a bar frame, is accepted; no other traceback is
	assert _without_old_blocks(b'Copying\n' + _EOF_BLOCK + b'Done\n') == b'Copying\n1 F Synced | 0 Scheduled |\xe2\x96\x88|100.0% | 0sDone\n'
	for other in (_EOF_BLOCK.replace(b'EOFError: Ran out of input', b'BrokenPipeError: [Errno 32] Broken pipe'),
			_EOF_BLOCK.replace(b'call_queue.get(block=True)', b'result_queue.put(item)'),
			b'Traceback (most recent call last):\n  File "x.py", line 1, in <module>\nEOFError: Ran out of input\n'):
		assert b'Traceback' in _without_old_blocks(b'Copying\n' + other), other


@posix_only
@pytest.mark.parametrize('batch', range(4))
@pytest.mark.parametrize('method', _start_methods())
def test_an_early_stop_prints_no_traceback(tmp_path, method, batch):
	# Entry 18: under forkserver (3.14's default) and spawn a worker started with Python's own SIGINT handler, and hpcp's went
	# in only later: about one early Ctrl-C in three printed a KeyboardInterrupt traceback on 3.14. A worker now starts with
	# the stops blocked and lets them in once set up. 10 runs a batch, 40 a start method; each stopped at a random 0 to 0.3 s
	# after hpcp.main() was called, by SIGINT to the run's process group, as a terminal's Ctrl-C. On 3.6 and 3.7 Python's own
	# block for hpcp's dropped stop is accepted (_DROPPED_STOP, a known limitation there). 9.88: on 3.6 and 3.7 with
	# forkserver or spawn (skipped in 9.87), the semaphore tracker they start at the run's first SemLock starts with the stops
	# blocked, as 3.8's resource tracker blocks them itself: before, a Ctrl-C there printed the tracker's traceback (13 and 28
	# of 200 runs on 3.7 in the 9.87 docs round).
	cp = tmp_path / 'slowcp'
	cp.write_text(_SLOWER_CP)
	os.chmod(str(cp), 0o755)
	for d in range(10):
		_files(tmp_path / 'src' / ('d%d' % d), 20)
	ready = tmp_path / 'ready'
	before = "import multiprocessing; multiprocessing.set_start_method(%r); open(os.environ['READY'], 'w').close()" % method
	for run in range(10):
		shutil.rmtree(str(tmp_path / 'out'), ignore_errors=True)
		with contextlib.suppress(FileNotFoundError):
			os.unlink(str(ready))
		task = _wrapped(tmp_path, ['src/', '-d', 'out/', '-j', '8'], before=before, env={'CP_PATH': str(cp), 'READY': str(ready)})
		try:
			_wait_for(ready.exists)
			time.sleep(random.uniform(0, 0.3))
			os.killpg(task.pid, signal.SIGINT)
			code, output, _ = _ended(task)
		finally:
			if task.poll() is None:
				os.killpg(task.pid, signal.SIGKILL)
		left = _expected_old_blocks(output)
		if code != 130 or b'Traceback' in left:
			print(output.decode(errors='replace'))  # whole, in the failure's captured stdout: an assertion message is cut short
		assert code == 130 and b'Traceback' not in left, (run, output[-3000:])


_STOP_IN = '''
import os, signal, sys, weakref
sys.path.insert(0, %(root)r)
import hpcp
hpcp._MAIN_PID = os.getpid()
hpcp._install_stops()
def stop_in(name):
	def trace(frame, event, arg):
		if event == 'call' and frame.f_code.co_name == name:
			sys.settrace(None)
			os.kill(os.getpid(), signal.SIGINT)
	return trace
class Going:
	def __del__(self):
		pass
class Item:
	pass
if sys.argv[1] == 'del':
	sys.settrace(stop_in('__del__'))
	Going()
else:
	items, item = weakref.WeakSet(), Item()
	items.add(item)
	sys.settrace(stop_in('_remove'))
	del item
sys.settrace(None)
try:
	hpcp._check_stop()
except KeyboardInterrupt:
	print('acted on')
'''


@posix_only
@pytest.mark.parametrize('where', ['del', 'weakset'])
def test_a_stop_in_a_call_python_drops_prints_nothing_and_is_acted_on(tmp_path, where):
	# Python drops a raise in a __del__ or a weakref callback (a thread's, a pool worker's going), and 3.6 and 3.7, which have
	# no sys.unraisablehook, printed hpcp's stop there as 'Exception ignored' with its traceback: about 1 early Ctrl-C in
	# 100 under a loaded host (test_an_early_stop_prints_no_traceback, 3.6). There the stop is only recorded, and acted on.
	script = tmp_path / 'stop_in.py'
	script.write_text(_STOP_IN % {'root': os.path.dirname(HPCP)})
	done = subprocess.run([sys.executable, str(script), where], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
	assert done.returncode == 0 and done.stdout.strip() == b'acted on', done.stdout


_SUBMIT_HELD = '''
import os, signal, sys, threading, time
sys.path.insert(0, %(root)r)
import hpcp
case = sys.argv[1]
if case == 'ignored':
	signal.signal(signal.SIGHUP, signal.SIG_IGN)  # as nohup starts a run: _install_stops leaves it so
hpcp._MAIN_PID = os.getpid()
hpcp._install_stops()
hpcp._register_worker_start()
signum = signal.SIGHUP if case == 'ignored' else signal.SIGINT
sent = []
def send():
	if case != 'recorded':
		signal.pthread_sigmask(signal.SIG_BLOCK, [signum])  # every thread blocks it: it stays pending
	time.sleep(0.3)
	sent.append(time.monotonic())
	os.kill(os.getpid(), signum)
def release(lock):
	# the dead holder's lock, let go of by another process after 1 s
	signal.pthread_sigmask(signal.SIG_BLOCK, [signum])
	time.sleep(1)
	pid = os.fork()
	if pid == 0:
		lock.release()
		os._exit(0)
	os.waitpid(pid, 0)
try:
	with hpcp._pool(2) as pool:
		pool.submit(os.getpid).result(timeout=30)
		lock = pool.executor._result_queue._wlock
		pid = os.fork()
		if pid == 0:
			lock.acquire()  # a worker that ended holding the result queue's write lock
			os._exit(0)
		os.waitpid(pid, 0)
		threading.Thread(target=send, daemon=True).start()
		if case == 'ignored':
			threading.Thread(target=release, args=(lock,), daemon=True).start()
		pool.submit(os.getpid).result(timeout=30)
	print('returned', flush=True)
except KeyboardInterrupt:
	print('KeyboardInterrupt %%.1f s after the signal' %% (time.monotonic() - sent[0]), flush=True)
# Python 3.6's exit wakes each pool's manager thread through that lock: not waited on here
os._exit(0)
'''


@posix_only
@pytest.mark.skipif(sys.version_info >= (3, 7), reason="Python 3.6's own submit")
@pytest.mark.parametrize('case', ['pending', 'recorded', 'ignored'])
def test_a_stop_ends_a_3_6_submit_that_waits_on_a_dead_worker_s_lock(tmp_path, case):
	# Python 3.6's submit wakes the pool's manager thread under the result queue's write lock, which a worker ended inside its
	# acquire leaves held: 9.87's parent waited on it for good, its stops blocked, and a second Ctrl-C printed nothing (about 3
	# early stops in 2,000). hpcp's own submit takes the lock in 0.1 s steps and acts on a stop _stop handles: pending (every
	# thread blocks it, as a real run's do while a submit runs) or recorded (another thread took it). An ignored one (nohup's
	# SIGHUP) stays pending while blocked and is no stop: the submit waits for the lock and returns.
	script = tmp_path / 'submit_held.py'
	script.write_text(_SUBMIT_HELD % {'root': os.path.dirname(HPCP)})
	try:
		done = subprocess.run([sys.executable, str(script), case], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
	except subprocess.TimeoutExpired:
		pytest.fail('the submit waited on the held lock for good (60 s)')
	output = done.stdout.decode(errors='replace')
	if case == 'ignored':
		assert done.returncode == 0 and output.strip() == 'returned', output
	else:
		said = re.match(r'KeyboardInterrupt ([0-9.]+) s after the signal$', output.strip())
		assert done.returncode == 0 and said and float(said.group(1)) < 5, output


def test_a_3_6_submit_without_a_write_lock_is_the_stock_one():
	# Python 3.6 on Windows makes the result queue with no write lock (_wlock None: a pipe message is written whole there), so
	# its submit cannot hang on one; hpcp's own submit took the lock unconditionally and raised AttributeError on every pooled
	# task (9.87 used the stock submit). Faked: no Windows machine here.
	import hpcp
	class _Queue:
		_wlock = None
	class _Executor:
		_result_queue = _Queue()
		def submit(self, fn, *args, **kwargs):
			return ('stock', fn, args, kwargs)
	assert hpcp._submit_36(_Executor(), None, max, (1, 2), {'key': len}) == ('stock', max, (1, 2), {'key': len})


# ---- a Ctrl-C while Python imports hpcp ----

_IMPORT_STOP = '''
import os, signal, sys
sys.path.insert(0, %(root)r)
VERSION_LINE = %(version_line)d
seen = {'line': None, 'sent': None}
def module_lines(frame, event, arg):
	if event == 'line':
		seen['line'] = frame.f_lineno
	return module_lines
def trace(frame, event, arg):
	code = frame.f_code
	if event != 'call':
		return None
	if code.co_name == '<module>' and os.path.basename(code.co_filename) == 'hpcp.py':
		return module_lines
	# importlib's weak-reference callbacks (a module lock's cb; 3.12+ also KeyedRef.remove), while hpcp's import header runs
	if (seen['sent'] is None and seen['line'] is not None and seen['line'] < VERSION_LINE and code.co_name in ('cb', 'remove')
			and code.co_filename.startswith('<frozen importlib')):
		seen['sent'] = (code.co_name, seen['line'])
		os.kill(os.getpid(), signal.SIGINT)
	return None
sys.settrace(trace)
try:
	import hpcp
finally:
	sys.settrace(None)
	print('sent in', seen['sent'], flush=True)
print('continued', flush=True)
'''


def _version_line():
	with open(HPCP, encoding='utf-8') as f:
		return next(number for number, line in enumerate(f, 1) if line.startswith('version = '))


@posix_only
def test_a_ctrl_c_in_an_import_callback_is_not_dropped():
	# Python's own SIGINT handler raised a Ctrl-C that came while Python imported hpcp inside importlib's weak-reference
	# callbacks, which drop it ('Exception ignored in: ...cb'), and the run went on, with -rf to its removal (1 in 3,000 early
	# Ctrl-Cs through pip's launcher, 3.7 and 3.14). SIGINT is held for hpcp's imports, and raised once they are done.
	done = subprocess.run([sys.executable, '-c', _IMPORT_STOP % {'root': os.path.dirname(HPCP), 'version_line': _version_line()}],
		stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
	out, err = done.stdout.decode(errors='replace'), done.stderr.decode(errors='replace')
	assert 'sent in None' not in out, (out, err)  # the callback ran inside the header, and the Ctrl-C was sent there
	assert 'continued' not in out and 'KeyboardInterrupt' in err and 'Exception ignored' not in err, (out, err)


_IMPORT_MASKS = '''
import signal, sys
sys.path.insert(0, %(root)r)
case = sys.argv[1]
if case == 'blocked':
	signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGINT])
elif case != 'plain':
	sys.modules[case] = None  # an import of hpcp's that fails
	sys.modules['multiCMD'] = None  # the embedded one is used
try:
	import hpcp
	print('imported')
except ImportError:
	print('ImportError')
print('SIGINT blocked' if signal.SIGINT in signal.pthread_sigmask(signal.SIG_BLOCK, []) else 'SIGINT open')
'''


def _embed_codec():
	with open(HPCP, encoding='utf-8') as f:
		return re.search(r'exec\((\w+)\.decompress\(', f.read()).group(1)


@posix_only
@pytest.mark.parametrize('case', ['plain', 'blocked', 'uuid', 'codec'])
def test_the_import_mask_is_released_on_every_path(case):
	# The mask covers hpcp's import header only, released by a finally on every path: a failed import (one of its imports,
	# or the embedded multiCMD's codec, missing) leaves SIGINT open, as it was; one blocked before the import stays blocked.
	if case == 'codec':
		case = _embed_codec()
	done = subprocess.run([sys.executable, '-c', _IMPORT_MASKS % {'root': os.path.dirname(HPCP)}, case],
		stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
	expected = {'plain': 'imported\nSIGINT open', 'blocked': 'imported\nSIGINT blocked'}.get(case, 'ImportError\nSIGINT open')
	assert done.stdout.decode(errors='replace').strip() == expected, done.stdout


_IMPORT_POPENS = '''
import os, subprocess, sys
sys.path.insert(0, %(root)r)
started = []
real = subprocess.Popen.__init__
def spy(self, args, *rest, **kwargs):
	# whose module's import started it: the innermost module frame (uuid's own ldconfig on 3.6 is uuid's)
	frame = sys._getframe(1)
	while frame is not None and frame.f_code.co_name != '<module>':
		frame = frame.f_back
	started.append((args, frame and os.path.basename(frame.f_code.co_filename)))
	return real(self, args, *rest, **kwargs)
subprocess.Popen.__init__ = spy
import hpcp
print([args for args, whose in started if whose == 'hpcp.py'])
'''


@posix_only
def test_importing_hpcp_runs_no_subprocess():
	# On Python 3.6 to 3.8 platform.machine() runs `uname -p` (platform.uname() computes the processor there), which
	# _fat_ioctl_numbers called as hpcp's module ran: a fork and exec each import, and a Ctrl-C was dropped in its Popen.__del__
	# once the import mask had ended. os.uname() asks the kernel.
	done = subprocess.run([sys.executable, '-c', _IMPORT_POPENS % {'root': os.path.dirname(HPCP)}],
		stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
	assert done.returncode == 0 and done.stdout.decode(errors='replace').strip() == '[]', done.stdout


@posix_only
@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM], ids=['int', 'term'])
def test_a_stopped_run_leaves_no_multiprocessing_files(tmp_path, signum):
	# A stopped run ends by os._exit (or its signal), past multiprocessing's exit work: its temporary directory (a fork
	# server's socket) and its named semaphores (forkserver, spawn) are removed first, so nothing is left in TMPDIR and the
	# resource tracker has nothing to warn about.
	import tempfile
	# Short: a fork server's socket is made in it, and an AF_UNIX path holds at most 107 bytes (pytest's tmp_path can be longer).
	tmpdir = tempfile.mkdtemp(prefix='hpcp-')
	try:
		task, marks = _slow_run(tmp_path, ['-j', '4', '-wb'], env={'TMPDIR': tmpdir})
		try:
			_wait_for(lambda: len(os.listdir(str(marks))) >= 4)
			os.killpg(task.pid, signum)
			code, output, _ = _ended(task)
			_wait_for(lambda: _group_gone(task.pid), timeout=30)
		finally:
			if task.poll() is None:
				os.killpg(task.pid, signal.SIGKILL)
		assert code == (130 if signum == signal.SIGINT else -signum), output[-3000:]
		assert os.listdir(tmpdir) == [] and b'leaked' not in output, (os.listdir(tmpdir), output[-3000:])
	finally:
		shutil.rmtree(tmpdir, ignore_errors=True)


def test_every_pool_goes_through_one_door(tmp_path, hpcp_mod, reset_hpcp_globals):
	# One door (_pool) for every process pool, one wait (_wait): what S1 and S2 do holds for each. In the parent, inside a pool
	# region, nothing but the loop runs (no cp, no prompt): conftest's guard checks every in-process run of the suite, and
	# these here.
	with open(HPCP, encoding='utf-8') as f:  # hpcp.py is UTF-8 whatever the locale (LANG unset: ASCII on 3.6)
		source = f.read()
	assert source.count('ProcessPoolExecutor(') == 1 and source.count('concurrent.futures.wait(') == 1
	assert 'as_completed(' not in source
	for d in range(4):
		_files(tmp_path / 'src' / ('d%d' % d), 100)
	os.symlink('f000', str(tmp_path / 'src' / 'd0' / 'link'))
	common = dict(single_thread=False, max_workers=2, files_per_job=4, batch=True)
	# -pfl lists in a pool, the directories are synced and the files copied in pools, -rme lists the destination (nothing
	# extra: no prompt), a move checks its copies (_looked_at) and removes in a pool.
	assert hpcp_mod.hpcp([str(tmp_path / 'src') + os.sep], dest_paths=[str(tmp_path / 'out') + os.sep], parallel_file_listing=True,
		remove_extra=True, **common) == 0
	assert hpcp_mod.hpcp([str(tmp_path / 'src') + os.sep], dest_paths=[str(tmp_path / 'moved') + os.sep], remove=True,
		remove_force=True, **common) == 0
	assert len(os.listdir(str(tmp_path / 'moved' / 'd1'))) == 100 and not os.path.exists(str(tmp_path / 'src' / 'd1'))


# ---- entry 6: a move of a link and a path through it ----

def _linked(root):
	write(root / 'b/x', 'X')
	write(root / 'b/y', 'Y')
	write(root / 'b/s/z', 'Z')
	write(root / 'b/d/w', 'W')
	os.symlink('b', str(root / 'lb'))


@posix_only
@pytest.mark.parametrize('batch', range(20))
def test_a_move_of_a_link_and_a_path_through_it_removes_both_on_every_run(tmp_path, batch):
	# The entry's form: 9.84 exited 192 in about a third of the runs, and in every run left b/x, which the operator named.
	# Only a directory holds what lies below it, and a named link goes last. Twenty batches of ten runs.
	for attempt in range(10):
		root = tmp_path / str(attempt)
		_linked(root)
		rc, output = run(['-sfl', '-', 'lb/x', '-d', 'out/', '-rf', '-j', '8'], root, b'lb/\n')
		assert rc == 0 and not os.path.lexists(str(root / 'lb')) and not os.path.exists(str(root / 'b/x')), output[-2000:]
		assert read(root / 'out/x') == 'X' and os.path.islink(str(root / 'out/lb')) and read(root / 'b/y') == 'Y'


@posix_only
def test_a_move_of_a_link_and_a_directory_through_it_moves_that_directory(tmp_path):
	# A second form, found by the designer: 9.84 exited 192 in 9 of 10 runs and moved none of b/d's files.
	for attempt in range(10):
		root = tmp_path / str(attempt)
		_linked(root)
		rc, output = run(['lb', 'lb/d/', '-d', 'out/', '-rf', '-j', '8'], root)
		assert rc == 0 and not os.path.lexists(str(root / 'lb')) and not os.path.exists(str(root / 'b/d')), output[-2000:]
		assert read(root / 'out/w') == 'W' and read(root / 'b/y') == 'Y' and read(root / 'b/x') == 'X'


# ---- entry 3: a FIFO, socket or device named as a source ----

@posix_only
@pytest.mark.skipif(not hasattr(__import__('socket'), 'AF_UNIX'), reason='unix sockets')
@pytest.mark.parametrize('flags', [[], ['-s'], ['-rf']], ids=['parallel', 'serial', 'move'])
def test_a_named_socket_and_fifo_are_copied_as_cp_copies_them(tmp_path, flags):
	# 9.80 to 9.84 made the node and exited 200 ("Source path type error"), and -rf kept every source (212); -s exited 0.
	import socket
	os.makedirs(str(tmp_path / 'a'))
	os.mkfifo(str(tmp_path / 'a/p'))
	sock = socket.socket(socket.AF_UNIX)
	sock.bind(str(tmp_path / 'a/s'))
	sock.close()
	write(tmp_path / 'a/f', 'F')
	rc, output = run(['a/p', 'a/s', 'a/f', '-d', 'out/'] + flags, tmp_path, timeout=60)
	assert rc == 0 and b'Source path type error' not in output, output[-2000:]
	assert stat.S_ISFIFO(os.lstat(str(tmp_path / 'out/p')).st_mode)
	if sys.platform == 'darwin':
		# macOS copies no socket (its cp skips one): noted, not copied, and left in place by a move (9.90).
		assert b'Special file skipped' in output and not os.path.lexists(str(tmp_path / 'out/s')), output[-2000:]
		assert os.path.lexists(str(tmp_path / 'a/s')) and all(os.path.lexists(str(tmp_path / 'a' / name)) != (flags == ['-rf']) for name in 'pf')
		return
	assert stat.S_ISSOCK(os.lstat(str(tmp_path / 'out/s')).st_mode)
	assert all(os.path.lexists(str(tmp_path / 'a' / name)) != (flags == ['-rf']) for name in 'psf')


@posix_only
def test_a_fifo_in_a_tree_is_still_skipped(tmp_path):
	os.makedirs(str(tmp_path / 'a'))
	os.mkfifo(str(tmp_path / 'a/p'))
	rc, output = run(['a/', '-d', 'out/'], tmp_path, timeout=60)
	assert rc == 0 and not os.path.lexists(str(tmp_path / 'out/p')) and b'Special file skipped' in output, output[-2000:]


# ---- entry 9: a copy into its own inside under another spelling ----

def _spellings_are_one(hpcp_mod, monkeypatch, answer):
	"""Every probe says the spellings it is asked about are one name (answer True), two (False), or cannot say (None): a
	destination where D and d have numbers of their own, as FUSE exFAT gives them."""
	def ask(directory, names):
		if answer is None:
			return {}, {name: 'the reason given' for name in names}
		return ({name: names[0] for name in names} if answer else {name: name for name in names}), {}
	monkeypatch.setattr(hpcp_mod, '_ask_names', ask)


@posix_only
def test_a_copy_into_its_own_inside_in_another_spelling_leaves_the_destination_out(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, capsys):
	# FUSE exFAT: d is D, so d/sub is D/sub. 9.84 walked it and nested a copy one level deeper on every run.
	monkeypatch.chdir(tmp_path)
	write(tmp_path / 'D/f', 'f')
	write(tmp_path / 'D/sub/f', 'sub')
	os.makedirs(str(tmp_path / 'd/sub'))
	_spellings_are_one(hpcp_mod, monkeypatch, True)
	for _ in range(3):
		assert hpcp_mod.hpcp(['D/'], ['d/sub/'], single_thread=True, max_workers=1, batch=True) == 0
	assert not os.path.exists(str(tmp_path / 'd/sub/sub')) and not os.path.exists(str(tmp_path / 'D/sub/sub'))
	assert 'A source holds the destination' in capsys.readouterr().out


@posix_only
def test_two_real_directories_that_differ_in_case_are_copied_as_before(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	monkeypatch.chdir(tmp_path)
	write(tmp_path / 'D/f', 'f')
	write(tmp_path / 'D/sub/f', 'sub')
	os.makedirs(str(tmp_path / 'd/sub'))
	_spellings_are_one(hpcp_mod, monkeypatch, False)
	assert hpcp_mod.hpcp(['D/'], ['d/sub/'], single_thread=True, max_workers=1, batch=True) == 0
	assert read(tmp_path / 'd/sub/sub/f') == 'sub' and read(tmp_path / 'd/sub/f') == 'f'


@posix_only
def test_a_destination_that_may_be_inside_the_source_and_cannot_be_asked_is_refused(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, capsys):
	monkeypatch.chdir(tmp_path)
	write(tmp_path / 'D/f', 'f')
	os.makedirs(str(tmp_path / 'd/sub'))
	_spellings_are_one(hpcp_mod, monkeypatch, None)
	assert hpcp_mod.hpcp(['D/'], ['d/sub/'], single_thread=True, max_workers=1, batch=True) == 241
	assert 'may lie inside the source' in capsys.readouterr().err and os.listdir(str(tmp_path / 'd/sub')) == []


# ---- entry 7: the probe by descriptors ----

@posix_only
def test_a_pair_landing_within_a_few_characters_of_path_max_is_asked_and_copied(tmp_path):
	# A landing directory of 4,066 characters: 9.84 made the probe's names at its full path, past PATH_MAX (213, "File name
	# too long"). Relative to descriptors a name costs its own length.
	write(tmp_path / 'a/X', 'upper')
	write(tmp_path / 'b/x', 'lower')
	root = str(tmp_path / 'out')
	parts = []
	while len(root) + 1 + sum(len(part) + 1 for part in parts) + 200 <= 4066:
		parts.append('d' * 200)
	parts.append('e' * (4066 - len(root) - sum(len(part) + 1 for part in parts) - 1))
	deep = os.path.join(root, *parts)
	assert len(deep) == 4066
	os.makedirs(deep)
	rc, output = run(['a/', 'b/', '-d', deep + '/'], tmp_path)
	assert rc == 0 and b'File name too long' not in output, output[-2000:]
	assert read(os.path.join(deep, 'X')) == 'upper' and read(os.path.join(deep, 'x')) == 'lower'


# ---- entry 10: the check's records cost the GC nothing ----

def test_the_check_keeps_its_jobs_where_the_gc_does_not_look(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# Lists inside each record made every one a container the cyclic GC tracked and rescanned: 1 to 2 s per million entries.
	import gc
	for name in range(20):
		write(tmp_path / 'a' / str(name), 'x')
	seen = []
	real = hpcp_mod._names_may_fold
	monkeypatch.setattr(hpcp_mod, '_names_may_fold', lambda jobs, listing: seen.extend(jobs) or real(jobs, listing))
	sources = [str(tmp_path / 'a' / str(name)) for name in range(20)] + [str(tmp_path / 'a') + os.sep]
	hpcp_mod._settle_destinations(sources, [str(tmp_path / 'out') + os.sep, str(tmp_path / 'two') + os.sep])
	# CPython untracks a tuple of untracked objects at a collection that meets it (the outer one at the next, once its inner
	# ones are); a list stays tracked for good.
	gc.collect()
	gc.collect()
	assert len(seen) == 21 and not any(gc.is_tracked(job) for job in seen)


# ---- a locale with no LANG ----

@posix_only
def test_a_non_ascii_name_copies_with_lang_unset(tmp_path):
	# LANG unset, as cron and some ssh sessions give it on EL7: Python 3.6 then holds paths in ASCII with escapes, and its
	# stdout is ASCII. (Newer Pythons are kept there by PYTHONUTF8=0 and PYTHONCOERCECLOCALE=0.) A tree, a rerun, a list, a
	# message naming such a file and a move all work, and nothing is printed that the terminal cannot take.
	root = os.fsencode(str(tmp_path))
	def p(*parts):
		return os.path.join(root, *parts)
	os.makedirs(p(b's', b'd\xc3\xa9'))
	for name, content in ((b'd\xc3\xa9/f\xc3\xafle', b'x'), (b'na\xc3\xafve', b'y'), (b'legacy\xe9', b'z')):
		with open(p(b's', name), 'wb') as f:
			f.write(content)
	os.makedirs(p(b'o', b'clash\xc3\xa9'))
	with open(p(b's', b'clash\xc3\xa9'), 'wb') as f:
		f.write(b'c')
	unset = [k for k in os.environ if k == 'LANG' or k.startswith('LC_') or k in ('PYTHONIOENCODING', 'LANGUAGE')]
	env = {'PYTHONUTF8': '0', 'PYTHONCOERCECLOCALE': '0'}
	def check(rc, output, want):
		assert rc == want and b'Traceback' not in output and b'UnicodeEncodeError' not in output, (rc, output[-3000:])
	for _ in range(2):
		rc, output = run(['s/', '-d', 'o/'], tmp_path, env=env, unset=unset)
		check(rc, output, 213)
		assert b'is a directory, not copying' in output
	for name, content in ((b'd\xc3\xa9/f\xc3\xafle', b'x'), (b'na\xc3\xafve', b'y'), (b'legacy\xe9', b'z')):
		with open(p(b'o', name), 'rb') as f:
			assert f.read() == content
	rc, output = run(['-sfl', '-', '-d', 'o2/'], tmp_path / 's', b'd\xc3\xa9/f\xc3\xafle\nlegacy\xe9\n', env=env, unset=unset)
	check(rc, output, 0)
	assert sorted(os.listdir(p(b's', b'o2'))) == [b'd\xc3\xa9', b'legacy\xe9']
	os.remove(p(b's', b'clash\xc3\xa9'))
	rc, output = run(['s/', '-d', 'o3/', '-rf', '-v'], tmp_path, env=env, unset=unset)
	check(rc, output, 0)
	with open(p(b'o3', b'd\xc3\xa9', b'f\xc3\xafle'), 'rb') as f:
		assert f.read() == b'x'
	assert not os.path.lexists(p(b's', b'na\xc3\xafve'))


# ---- what a power cut finds: a rewrite and a move are on disk before the old copy or the source goes ----

def _spy_fsync_and_renames(hpcp_mod, monkeypatch, fsync_error=None):
	"""Record ('fsync', inode) for each fsync and ('rename', inode of what is renamed) for each rename or replace, in order;
	fsync_error: an errno each fsync fails with instead."""
	events = []
	real_fsync, real_rename, real_replace = os.fsync, os.rename, os.replace
	def fsync(fd):
		events.append(('fsync', os.fstat(fd).st_ino))
		if fsync_error is not None:
			raise OSError(fsync_error, os.strerror(fsync_error))
		return real_fsync(fd)
	def renamer(real):
		def rename(src, dst, *args, **kwargs):
			events.append(('rename', os.lstat(src, dir_fd=kwargs.get('src_dir_fd')).st_ino))
			return real(src, dst, *args, **kwargs)
		return rename
	monkeypatch.setattr(hpcp_mod.os, 'fsync', fsync)
	monkeypatch.setattr(hpcp_mod.os, 'rename', renamer(real_rename))
	monkeypatch.setattr(hpcp_mod.os, 'replace', renamer(real_replace))
	return events


@posix_only
def test_a_rewrite_is_on_disk_before_it_is_renamed_over_the_old_copy(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# XFS does not flush a file renamed over another, and no filesystem orders its data before the rename: unsynced, a power
	# cut can leave the landing empty and the old copy gone.
	write(tmp_path / 'a/x', 'new content')
	write(tmp_path / 'out/x', 'old')
	monkeypatch.setattr(hpcp_mod, 'WRITE_BESIDE', True)
	monkeypatch.setattr(hpcp_mod, 'NAMED_DESTS', frozenset([str(tmp_path / 'out')]))
	events = _spy_fsync_and_renames(hpcp_mod, monkeypatch)
	hpcp_mod.copy_file(str(tmp_path / 'a/x'), [str(tmp_path / 'out/x')])
	assert read(tmp_path / 'out/x') == 'new content'
	renamed = [inode for kind, inode in events if kind == 'rename']
	assert renamed == [os.lstat(str(tmp_path / 'out/x')).st_ino]
	assert events.index(('fsync', renamed[0])) < events.index(('rename', renamed[0]))


@posix_only
def test_a_rewrite_whose_new_copy_fails_to_reach_the_disk_keeps_the_old_copy(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# An fsync that fails (EIO) is a failed rewrite: the temporary file goes, the old copy stays (never written in place over
	# it, where the same disk would fail it too), Copy failed.
	write(tmp_path / 'a/x', 'new content, longer')
	write(tmp_path / 'out/x', 'old')
	monkeypatch.setattr(hpcp_mod, 'WRITE_BESIDE', True)
	monkeypatch.setattr(hpcp_mod, 'NAMED_DESTS', frozenset([str(tmp_path / 'out')]))
	_spy_fsync_and_renames(hpcp_mod, monkeypatch, errno.EIO)
	hpcp_mod.copy_file(str(tmp_path / 'a/x'), [str(tmp_path / 'out/x')])
	assert read(tmp_path / 'out/x') == 'old' and temporaries(tmp_path) == []
	assert any('Copy failed' in error and 'the new copy cannot be written to disk' in error for error in hpcp_mod.ERRORS), hpcp_mod.ERRORS


@posix_only
@pytest.mark.parametrize('error', ['EINVAL', 'ENOTSUP', 'ENOSYS'])
def test_a_filesystem_that_cannot_fsync_still_rewrites(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, error):
	# Some FUSE filesystems answer fsync with these: the copy goes on as before 9.85.
	write(tmp_path / 'a/x', 'new content')
	write(tmp_path / 'out/x', 'old')
	monkeypatch.setattr(hpcp_mod, 'WRITE_BESIDE', True)
	monkeypatch.setattr(hpcp_mod, 'NAMED_DESTS', frozenset([str(tmp_path / 'out')]))
	_spy_fsync_and_renames(hpcp_mod, monkeypatch, getattr(errno, error))
	hpcp_mod.copy_file(str(tmp_path / 'a/x'), [str(tmp_path / 'out/x')])
	assert read(tmp_path / 'out/x') == 'new content' and temporaries(tmp_path) == [] and not hpcp_mod.ERRORS


@posix_only
def test_a_rewrite_in_place_outside_a_move_is_not_fsynced(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# As 9.84: a plain copy keeps its source, and an fsync per rewrite cost about 1 to 1.5 ms a file on EL7 (the approved
	# known limitation: a power cut soon after can leave it partial or empty). A move fsyncs it (_sync_landing).
	write(tmp_path / 'a/x', 'new content')
	write(tmp_path / 'out/x', 'old')
	inode = os.stat(str(tmp_path / 'out/x')).st_ino
	monkeypatch.setattr(hpcp_mod, 'NAMED_DESTS', frozenset([str(tmp_path / 'out')]))
	events = _spy_fsync_and_renames(hpcp_mod, monkeypatch)
	synced = []
	monkeypatch.setattr(hpcp_mod, '_fsync_file', lambda path, *args: synced.append(path))
	hpcp_mod.copy_file(str(tmp_path / 'a/x'), [str(tmp_path / 'out/x')])
	assert read(tmp_path / 'out/x') == 'new content' and os.stat(str(tmp_path / 'out/x')).st_ino == inode
	assert synced == [] and events == [] and not hpcp_mod.ERRORS


def _patch_row(hpcp_mod, monkeypatch, path, size):
	monkeypatch.setattr(hpcp_mod, 'decode_state', lambda state: {'size': size})
	return type('Row', (), {'new': 'f:%d' % size, 'path': path})()


@posix_only
def test_a_patch_s_replacement_is_on_disk_before_it_is_renamed(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	write(tmp_path / 'live/x', 'old')
	write(tmp_path / 'new/x', 'new')
	row = _patch_row(hpcp_mod, monkeypatch, 'x', 3)
	events = _spy_fsync_and_renames(hpcp_mod, monkeypatch)
	hpcp_mod._replace_files([(row, str(tmp_path / 'new/x'), str(tmp_path / 'live/x'))], str(tmp_path / 'live'), 1)
	assert read(tmp_path / 'live/x') == 'new' and temporaries(tmp_path) == []
	renamed = [inode for kind, inode in events if kind == 'rename']
	assert renamed and events.index(('fsync', renamed[-1])) < events.index(('rename', renamed[-1])), events


@posix_only
def test_a_patch_s_replacement_that_fails_to_reach_the_disk_keeps_the_old_file(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	write(tmp_path / 'live/x', 'old')
	write(tmp_path / 'new/x', 'new')
	row = _patch_row(hpcp_mod, monkeypatch, 'x', 3)
	_spy_fsync_and_renames(hpcp_mod, monkeypatch, errno.EIO)
	hpcp_mod._replace_files([(row, str(tmp_path / 'new/x'), str(tmp_path / 'live/x'))], str(tmp_path / 'live'), 1)
	assert read(tmp_path / 'live/x') == 'old' and temporaries(tmp_path) == []
	assert any(error.startswith('Copy failed: x:') for error in hpcp_mod.ERRORS), hpcp_mod.ERRORS


def _move(hpcp_mod, tmp_path, **extra):
	return hpcp_mod.hpcp([str(tmp_path / 'a') + os.sep], dest_paths=[str(tmp_path / 'out') + os.sep], single_thread=True,
		remove=True, remove_force=True, **extra)


@posix_only
def test_a_move_fsyncs_every_landing_before_any_source_goes(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# Each copy is written to disk by the worker that made it, while its source still stands: a power cut after the move
	# cannot find the sources gone and the copies in memory.
	write(tmp_path / 'a/x', 'X')
	write(tmp_path / 'a/d/y', 'Y')
	events = []
	real_fsync, real_remove = os.fsync, hpcp_mod.remove_moved_sources
	def fsync(fd):
		events.append(('fsync', os.fstat(fd).st_ino, os.path.exists(str(tmp_path / 'a/x')) and os.path.exists(str(tmp_path / 'a/d/y'))))
		return real_fsync(fd)
	def remove(*args, **kwargs):
		events.append(('remove',))
		return real_remove(*args, **kwargs)
	monkeypatch.setattr(hpcp_mod.os, 'fsync', fsync)
	monkeypatch.setattr(hpcp_mod, 'remove_moved_sources', remove)
	_move(hpcp_mod, tmp_path)
	assert read(tmp_path / 'out/x') == 'X' and read(tmp_path / 'out/d/y') == 'Y' and not os.path.exists(str(tmp_path / 'a'))
	assert events[-1] == ('remove',) and ('remove',) not in events[:-1]
	fsyncs = [event for event in events if event[0] == 'fsync']
	fsynced = {event[1] for event in fsyncs}
	assert {os.lstat(str(tmp_path / 'out/x')).st_ino, os.lstat(str(tmp_path / 'out/d/y')).st_ino} <= fsynced
	assert fsyncs and all(event[2] for event in fsyncs)


@posix_only
def test_a_move_whose_landing_fails_to_reach_the_disk_removes_no_source(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, capsys):
	# On the kernels of EL7 and EL8 a syncfs reports no write error; the fsync of the copy's own file does, while the data is
	# still dirty and its descriptor holds the inode. That landing is a failed copy: its own line, the move's gate, nothing removed.
	write(tmp_path / 'a/x', 'X')
	write(tmp_path / 'a/y', 'Y')
	real_fsync = os.fsync
	def fsync(fd):
		if os.path.basename(os.readlink('/proc/self/fd/%d' % fd)) == 'y':
			raise OSError(errno.EIO, os.strerror(errno.EIO))
		return real_fsync(fd)
	monkeypatch.setattr(hpcp_mod.os, 'fsync', fsync)
	rc = _move(hpcp_mod, tmp_path)
	assert read(tmp_path / 'a/x') == 'X' and read(tmp_path / 'a/y') == 'Y'
	err = capsys.readouterr().err
	assert f'{tmp_path / "out/y"} cannot be written to disk (Input/output error)' in err, err
	assert 'Remove skipped: not removing ' in err and 'the copy reported 1 error(s)' in err, err
	assert rc != 0
	# The copy that could not be trusted is not left to be taken for the finished one by a rerun.
	assert not os.path.lexists(str(tmp_path / 'out/y'))


@posix_only
def test_a_rerun_of_a_failed_move_copies_the_landing_again(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	write(tmp_path / 'a/y', 'Y')
	real_fsync = os.fsync
	def failing(fd):
		raise OSError(errno.EIO, os.strerror(errno.EIO))
	monkeypatch.setattr(hpcp_mod.os, 'fsync', failing)
	assert _move(hpcp_mod, tmp_path) != 0 and read(tmp_path / 'a/y') == 'Y'
	hpcp_mod.ERRORS.clear()
	monkeypatch.setattr(hpcp_mod.os, 'fsync', real_fsync)
	assert _move(hpcp_mod, tmp_path) == 0
	assert read(tmp_path / 'out/y') == 'Y' and not os.path.exists(str(tmp_path / 'a'))


@posix_only
def test_a_move_fsyncs_a_landing_an_earlier_copy_left(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# hpcp a/ -d out/ , then the same with -rm: the copy is there (same size and time, so not copied again) and may still be in
	# memory, so the move fsyncs it before it removes the source.
	write(tmp_path / 'a/x', 'X')
	assert hpcp_mod.hpcp([str(tmp_path / 'a') + os.sep], dest_paths=[str(tmp_path / 'out') + os.sep], single_thread=True) == 0
	fsynced = []
	real_fsync = os.fsync
	# files only: the move fsyncs its directories too (9.87's M2)
	monkeypatch.setattr(hpcp_mod.os, 'fsync', lambda fd: (fsynced.append(os.fstat(fd).st_ino) if stat.S_ISREG(os.fstat(fd).st_mode)
		else None) or real_fsync(fd))
	assert _move(hpcp_mod, tmp_path) == 0
	assert fsynced == [os.lstat(str(tmp_path / 'out/x')).st_ino] and not os.path.exists(str(tmp_path / 'a'))


@posix_only
def test_a_copy_that_is_not_a_move_is_not_fsynced(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	write(tmp_path / 'a/x', 'X')
	fsynced = []
	monkeypatch.setattr(hpcp_mod.os, 'fsync', lambda fd: fsynced.append(fd))
	assert hpcp_mod.hpcp([str(tmp_path / 'a') + os.sep], dest_paths=[str(tmp_path / 'out') + os.sep], single_thread=True) == 0
	assert fsynced == [] and read(tmp_path / 'out/x') == 'X'


@not_root
def test_a_move_into_a_write_only_destination_goes_through(tmp_path):
	# 9.85's syncfs opened the destination directory for reading, which a 0333 directory refuses: exit 212, the sources kept.
	write(tmp_path / 'a/x', 'X')
	write(tmp_path / 'a/d/y', 'Y')
	os.mkdir(str(tmp_path / 'out'))
	os.chmod(str(tmp_path / 'out'), 0o333)
	try:
		rc, output = run(['a/x', 'a/d', '-d', 'out/', '-rf'], tmp_path)
	finally:
		os.chmod(str(tmp_path / 'out'), 0o755)
	assert rc == 0, output[-2000:]
	assert read(tmp_path / 'out/x') == 'X' and read(tmp_path / 'out/d/y') == 'Y'
	assert not os.path.exists(str(tmp_path / 'a/x')) and not os.path.exists(str(tmp_path / 'a/d'))


@not_root
@pytest.mark.parametrize('mode', [0o000, 0o044, 0o070], ids=lambda mode: '%04o' % mode)
def test_a_landing_its_owner_can_neither_read_nor_write_is_fsynced_and_keeps_its_mode(tmp_path, hpcp_mod, monkeypatch, mode):
	# A non-root run copying another user's file of mode 0044 owns a landing it cannot open: the owner gets read for the open,
	# and the exact mode is put back through the descriptor before the fsync.
	write(tmp_path / 'f', 'data')
	os.chmod(str(tmp_path / 'f'), mode)
	seen = []
	real_fsync = os.fsync
	def fsync(fd):
		seen.append(stat.S_IMODE(os.fstat(fd).st_mode))
		return real_fsync(fd)
	monkeypatch.setattr(hpcp_mod.os, 'fsync', fsync)
	try:
		hpcp_mod._fsync_file(str(tmp_path / 'f'))
		assert seen == [mode] and stat.S_IMODE(os.lstat(str(tmp_path / 'f')).st_mode) == mode
	finally:
		os.chmod(str(tmp_path / 'f'), 0o600)


@posix_only
def test_a_move_to_a_destination_that_is_a_link_to_a_file_fsyncs_the_file_it_wrote(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# What the operator named is followed (cp writes through it): the fsync goes to the file the link leads to.
	write(tmp_path / 'a/x', 'new')
	write(tmp_path / 'where/target', 'old')
	os.symlink(str(tmp_path / 'where/target'), str(tmp_path / 'link'))
	fsynced = []
	real_fsync = os.fsync
	monkeypatch.setattr(hpcp_mod.os, 'fsync', lambda fd: fsynced.append(os.fstat(fd).st_ino) or real_fsync(fd))
	rc = hpcp_mod.hpcp([str(tmp_path / 'a/x')], dest_paths=[str(tmp_path / 'link')], single_thread=True, remove=True, remove_force=True)
	assert rc == 0 and read(tmp_path / 'where/target') == 'new' and not os.path.exists(str(tmp_path / 'a/x'))
	assert os.stat(str(tmp_path / 'where/target')).st_ino in fsynced


@not_root
def test_a_landing_that_cannot_be_opened_by_anyone_here_is_a_failed_fsync(tmp_path, hpcp_mod, monkeypatch):
	# A file of another owner that this run cannot open is an error, not a silent skip.
	write(tmp_path / 'f', 'data')
	os.chmod(str(tmp_path / 'f'), 0o044)
	monkeypatch.setattr(hpcp_mod.os, 'geteuid', lambda: os.stat(str(tmp_path / 'f')).st_uid + 1)
	with pytest.raises(PermissionError):
		hpcp_mod._fsync_file(str(tmp_path / 'f'))
	assert stat.S_IMODE(os.lstat(str(tmp_path / 'f')).st_mode) == 0o044
	os.chmod(str(tmp_path / 'f'), 0o600)


@not_root
def test_a_move_whose_landing_cannot_be_opened_is_a_failed_copy_not_a_removed_source(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, capsys):
	write(tmp_path / 'a/x', 'X')
	def refuse(path, *args):
		raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), path)
	monkeypatch.setattr(hpcp_mod, '_fsync_file', refuse)
	assert _move(hpcp_mod, tmp_path) != 0
	assert read(tmp_path / 'a/x') == 'X' and 'cannot be written to disk (Permission denied)' in capsys.readouterr().err


@posix_only
def test_a_fifo_made_in_a_landing_s_place_is_not_opened_to_fsync_it(tmp_path, hpcp_mod):
	# Nothing to write, and opening a FIFO would wait for a writer.
	os.mkfifo(str(tmp_path / 'fifo'))
	hpcp_mod._fsync_file(str(tmp_path / 'fifo'))


@posix_only
def test_a_move_writes_its_destination_to_disk_end_to_end(tmp_path):
	# The real fsync in a run: the move ends as before.
	write(tmp_path / 'a/x', 'X')
	write(tmp_path / 'a/d/y', 'Y')
	rc, output = run(['a/', '-d', 'out/', '-rf'], tmp_path)
	assert rc == 0 and read(tmp_path / 'out/x') == 'X' and read(tmp_path / 'out/d/y') == 'Y' and not os.path.exists(str(tmp_path / 'a')), output[-2000:]


def _losetup(path):
	result = subprocess.run(['losetup', '-f', '--show', str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
	if result.returncode != 0:
		pytest.skip('no loop device: ' + result.stderr.strip())
	return result.stdout.strip()


@root_only
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='loop devices are Linux')
def test_a_move_onto_a_block_device_writes_it_and_keeps_every_source(tmp_path, hpcp_mod, reset_hpcp_globals):
	# A block device named by -d is written through, as cp does; a device is another kind of entry than the file it holds,
	# so the move's check finds no landing for the source and keeps it (213). Nothing to fsync before a removal that never
	# comes, which is why the device is not fsynced.
	with open(str(tmp_path / 'disk.img'), 'wb') as f:
		f.truncate(8 << 20)
	device = _losetup(tmp_path / 'disk.img')
	try:
		write(tmp_path / 'a/x', 'device content')
		rc = hpcp_mod.hpcp([str(tmp_path / 'a/x')], dest_paths=[device], single_thread=True, remove=True, remove_force=True)
		assert rc == 213 and read(tmp_path / 'a/x') == 'device content'
		with open(device, 'rb') as d:
			assert d.read(14) == b'device content'
	finally:
		subprocess.run(['losetup', '-d', device], stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@root_only
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='mounts are Linux')
def test_a_move_fsyncs_a_landing_on_a_filesystem_mounted_below_the_destination(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# 9.85's syncfs was asked only about the destination root's filesystem: a landing on another one below it went unsynced.
	os.makedirs(str(tmp_path / 'out/sub'))
	result = subprocess.run(['mount', '-t', 'tmpfs', 'hpcp-test', str(tmp_path / 'out/sub')], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
	if result.returncode != 0:
		pytest.skip('mount failed: ' + result.stderr.strip())
	try:
		write(tmp_path / 'a/sub/f', 'F')
		write(tmp_path / 'a/g', 'G')
		fsynced = []
		real_fsync = os.fsync
		monkeypatch.setattr(hpcp_mod.os, 'fsync', lambda fd: fsynced.append(os.fstat(fd).st_ino) or real_fsync(fd))
		assert _move(hpcp_mod, tmp_path) == 0
		assert os.lstat(str(tmp_path / 'out/sub/f')).st_dev != os.lstat(str(tmp_path / 'out/g')).st_dev
		assert os.lstat(str(tmp_path / 'out/sub/f')).st_ino in fsynced and os.lstat(str(tmp_path / 'out/g')).st_ino in fsynced
		assert not os.path.exists(str(tmp_path / 'a/sub/f'))
	finally:
		subprocess.run(['umount', '-l', str(tmp_path / 'out/sub')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@root_only
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='loop devices and mounts are Linux')
def test_a_move_into_an_image_fsyncs_each_landing_in_it_before_the_source_goes(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# -di mounts the image on a loop device; the fsync of a file in it is flushed down to the image file by the loop driver
	# (REQ_FLUSH becomes an fsync of the backing file), and one that fails is a failed copy as anywhere.
	if subprocess.run(['sh', '-c', 'command -v mkfs.ext4'], stdout=subprocess.PIPE).returncode:
		pytest.skip('needs mkfs.ext4')
	image = str(tmp_path / 'x.img')
	with open(image, 'wb') as f:
		f.truncate(64 << 20)
	if subprocess.run(['mkfs.ext4', '-q', '-F', image], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode:
		pytest.skip('mkfs.ext4 refused')
	write(tmp_path / 'a/x', 'one')
	write(tmp_path / 'a/d/y', 'two')
	fsynced = []
	real_fsync, real_remove = os.fsync, hpcp_mod.remove_moved_sources
	def fsync(fd):
		fsynced.append(os.readlink('/proc/self/fd/%d' % fd))
		return real_fsync(fd)
	def remove(*args, **kwargs):
		# each file landed in it (its directories are fsynced too, since 9.87's M2)
		files = sorted(os.path.basename(path) for path in fsynced if os.path.isfile(path))
		assert files == ['x', 'y'] and os.path.exists(str(tmp_path / 'a/x')), fsynced
		return real_remove(*args, **kwargs)
	monkeypatch.setattr(hpcp_mod.os, 'fsync', fsync)
	monkeypatch.setattr(hpcp_mod, 'remove_moved_sources', remove)
	rc = hpcp_mod.hpcp([str(tmp_path / 'a') + os.sep], dest_image=image, single_thread=True, remove=True, remove_force=True)
	assert rc == 0 and not os.path.exists(str(tmp_path / 'a'))


@posix_only
def test_a_move_fsyncs_the_directories_of_its_links_before_a_source_goes(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# A link has no data to fsync and is made after every file: across filesystems a power cut could otherwise find the
	# source link removed and the new one not yet on disk.
	write(tmp_path / 'a/x', 'X')
	os.makedirs(str(tmp_path / 'a/links'))
	os.symlink('../x', str(tmp_path / 'a/links/l'))
	events = []
	real_fsync, real_remove = os.fsync, hpcp_mod.remove_moved_sources
	def fsync(fd):
		events.append(('fsync', os.fstat(fd).st_ino))
		return real_fsync(fd)
	def remove(*args, **kwargs):
		events.append(('remove',))
		return real_remove(*args, **kwargs)
	monkeypatch.setattr(hpcp_mod.os, 'fsync', fsync)
	monkeypatch.setattr(hpcp_mod, 'remove_moved_sources', remove)
	assert _move(hpcp_mod, tmp_path) == 0 and not os.path.exists(str(tmp_path / 'a'))
	assert os.readlink(str(tmp_path / 'out/links/l')) == '../x'
	before = {event[1] for event in events[:events.index(('remove',))]}
	assert os.lstat(str(tmp_path / 'out/links')).st_ino in before


@posix_only
def test_a_move_whose_link_directory_fails_to_reach_the_disk_removes_no_source(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, capsys):
	write(tmp_path / 'a/x', 'X')
	os.symlink('x', str(tmp_path / 'a/l'))
	real_fsync = os.fsync
	def fsync(fd):
		if stat.S_ISDIR(os.fstat(fd).st_mode) and os.readlink('/proc/self/fd/%d' % fd) == str(tmp_path / 'out'):
			raise OSError(errno.EIO, os.strerror(errno.EIO))
		return real_fsync(fd)
	monkeypatch.setattr(hpcp_mod.os, 'fsync', fsync)
	assert _move(hpcp_mod, tmp_path) != 0
	assert read(tmp_path / 'a/x') == 'X' and os.path.islink(str(tmp_path / 'a/l'))
	err = capsys.readouterr().err
	assert 'cannot be written to disk (Input/output error)' in err and 'Remove skipped' in err, err


@not_root
def test_a_directory_its_owner_cannot_read_is_fsynced_and_keeps_its_mode(tmp_path, hpcp_mod, monkeypatch):
	os.makedirs(str(tmp_path / 'd'))
	os.chmod(str(tmp_path / 'd'), 0o333)
	seen = []
	real_fsync = os.fsync
	monkeypatch.setattr(hpcp_mod.os, 'fsync', lambda fd: seen.append(stat.S_IMODE(os.fstat(fd).st_mode)) or real_fsync(fd))
	try:
		hpcp_mod._fsync_file(str(tmp_path / 'd'))
		assert seen == [0o333] and stat.S_IMODE(os.lstat(str(tmp_path / 'd')).st_mode) == 0o333
	finally:
		os.chmod(str(tmp_path / 'd'), 0o755)


@not_root
def test_a_link_moved_into_a_write_only_directory_still_moves(tmp_path):
	# 0333: the directory cannot be listed, only written; the owner is given read for the fsync's open.
	write(tmp_path / 'a/x', 'X')
	os.symlink('x', str(tmp_path / 'a/l'))
	os.mkdir(str(tmp_path / 'out'))
	os.chmod(str(tmp_path / 'out'), 0o333)
	try:
		rc, output = run(['a/x', 'a/l', '-d', 'out/', '-rf'], tmp_path)
		assert stat.S_IMODE(os.lstat(str(tmp_path / 'out')).st_mode) == 0o333
	finally:
		os.chmod(str(tmp_path / 'out'), 0o755)
	assert rc == 0, output[-2000:]
	assert os.readlink(str(tmp_path / 'out/l')) == 'x' and not os.path.lexists(str(tmp_path / 'a/l'))


@posix_only
def test_a_move_fsyncs_the_directory_a_link_to_a_directory_named_by_d_leads_to(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# -d out/ where out is a link to a directory: the links are made in the directory behind it, and that is what is fsynced.
	write(tmp_path / 'a/x', 'X')
	os.symlink('x', str(tmp_path / 'a/l'))
	os.makedirs(str(tmp_path / 'real'))
	os.symlink('real', str(tmp_path / 'out'))
	events = []
	real_fsync, real_remove = os.fsync, hpcp_mod.remove_moved_sources
	def fsync(fd):
		events.append(('fsync', os.fstat(fd).st_ino))
		return real_fsync(fd)
	def remove(*args, **kwargs):
		events.append(('remove',))
		return real_remove(*args, **kwargs)
	monkeypatch.setattr(hpcp_mod.os, 'fsync', fsync)
	monkeypatch.setattr(hpcp_mod, 'remove_moved_sources', remove)
	assert _move(hpcp_mod, tmp_path) == 0 and not os.path.exists(str(tmp_path / 'a'))
	assert os.readlink(str(tmp_path / 'real/l')) == 'x'
	before = {event[1] for event in events[:events.index(('remove',))]}
	assert os.lstat(str(tmp_path / 'real')).st_ino in before


# ---- review r986r: a node source, a landing changed after the look, and -co's carry-over ----

@posix_only
@pytest.mark.parametrize('flags, kind', [([], 'fifo'), (['-co'], 'fifo'), ([], 'socket')], ids=['plain', 'content-only', 'socket'])
def test_a_node_source_replaces_a_file_at_its_landing(tmp_path, monkeypatch, flags, kind):
	# cp -a makes a FIFO or socket anew, and cannot unlink a held file's /proc name: such a source goes beside, as in 9.85 (exit 0).
	if kind == 'socket' and sys.platform == 'darwin':
		pytest.skip('macOS copies no socket (9.90: a named one is skipped with a note)')
	os.makedirs(str(tmp_path / 'a'))
	if kind == 'fifo':
		os.mkfifo(str(tmp_path / 'a/p'))
	else:
		monkeypatch.chdir(tmp_path)  # a socket's path is short (108 bytes at most)
		import socket
		with contextlib.closing(socket.socket(socket.AF_UNIX)) as sock:
			sock.bind('a/p')
	write(tmp_path / 'out/p', 'old')
	rc, output = run(['a/p', '-d', 'out/'] + flags, tmp_path)
	assert rc == 0 and (stat.S_ISFIFO if kind == 'fifo' else stat.S_ISSOCK)(os.lstat(str(tmp_path / 'out/p')).st_mode), output[-3000:]


@posix_only
def test_a_named_landing_that_turns_into_a_fifo_after_the_look_is_never_written_by_name(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# A regular file at the look; a FIFO someone holds open at the open; a link to a victim when cp would open the name.
	write(tmp_path / 'a/f', 'SECRET')
	write(tmp_path / 'out/f', 'old')
	write(tmp_path / 'victim', 'victim')
	landing, readers = str(tmp_path / 'out/f'), []
	claim, runner = hpcp_mod._claim_device, hpcp_mod.run_command_in_multicmd_with_path_check
	def swap_to_fifo(target, st, named, src_path):
		if target == landing and not readers:
			os.rename(landing, landing + '.away')
			os.mkfifo(landing)
			readers.append(os.open(landing, os.O_RDONLY | os.O_NONBLOCK))
		return claim(target, st, named, src_path)
	def swap_to_link(command, *args, **kwargs):
		if command[-1] == landing and stat.S_ISFIFO(os.lstat(landing).st_mode):
			os.unlink(landing)
			os.symlink(str(tmp_path / 'victim'), landing)
		return runner(command, *args, **kwargs)
	monkeypatch.setattr(hpcp_mod, '_claim_device', swap_to_fifo)
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', swap_to_link)
	monkeypatch.setattr(hpcp_mod, 'NAMED_DESTS', frozenset({landing}))
	try:
		hpcp_mod._cp_once(str(tmp_path / 'a/f'), landing, '-af', True)
	finally:
		for fd in readers:
			os.close(fd)
	assert read(tmp_path / 'victim') == 'victim' and read(landing) == 'SECRET' and not os.path.islink(landing)


@posix_only
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='O_PATH and /proc/self/fd')
@pytest.mark.parametrize('swap', ['link', 'hard link'])
def test_content_only_carries_nothing_through_what_is_put_at_the_temporary_name(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, swap):
	# -co -wb: once cp has written the temporary file, a user who can write the directory puts a link (or another file's
	# second name) there. 9.86's carry-over refused it; 9.87's goes through the file hpcp made and holds, and the rename finds
	# the name no longer leads to it: 190, the old copy kept, and the victim keeps its mode.
	write(tmp_path / 'a/f', 'new')
	write(tmp_path / 'out/f', 'old')
	write(tmp_path / 'victim', 'victim')
	os.chmod(str(tmp_path / 'out/f'), 0o646)
	os.chmod(str(tmp_path / 'victim'), 0o600)
	runner = hpcp_mod.run_command_in_multicmd_with_path_check
	def swap_in(command, *args, **kwargs):
		temp = os.readlink(command[-1]) if command[-1].startswith('/proc/self/fd/') else command[-1]
		result = runner(command, *args, **kwargs)
		if '.hpcp_tmp_' in temp:
			os.unlink(temp)
			(os.symlink if swap == 'link' else os.link)(str(tmp_path / 'victim'), temp)
		return result
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', swap_in)
	hpcp_mod.CONTENT_ONLY = hpcp_mod.WRITE_BESIDE = True
	hpcp_mod.copy_file(str(tmp_path / 'a/f'), [str(tmp_path / 'out/f')])
	assert stat.S_IMODE(os.stat(str(tmp_path / 'victim')).st_mode) == 0o600 and read(tmp_path / 'out/f') == 'old'
	assert any('cannot be given to the new copy' in error or 'something else was put at its name' in error
		for error in hpcp_mod.ERRORS), hpcp_mod.ERRORS


def _protected_symlinks():
	try:
		with open('/proc/sys/fs/protected_symlinks') as f:
			return f.read().strip() == '1'
	except OSError:
		return False


@posix_only
@root_only
@pytest.mark.skipif(not _protected_symlinks(), reason='fs.protected_symlinks is off here')
def test_a_named_link_the_kernel_refuses_to_follow_is_kept(tmp_path):
	# Another user's link in a sticky world-writable directory is refused to the follower (fs.protected_symlinks): 9.84's cp
	# got EACCES (190, nothing written); 9.85 resolved the link itself and wrote through. A root run keeps it, exit 190.
	write(tmp_path / 'a/x', 'SECRET')
	write(tmp_path / 'victim', 'victim')
	os.makedirs(str(tmp_path / 'out'))
	os.chmod(str(tmp_path / 'out'), 0o1777)
	os.symlink(str(tmp_path / 'victim'), str(tmp_path / 'out/drop'))
	os.lchown(str(tmp_path / 'out/drop'), 65534, 65534)
	rc, output = run(['a/x', '-d', 'out/drop'], tmp_path)
	assert rc == 190 and read(tmp_path / 'victim') == 'victim' and os.path.islink(str(tmp_path / 'out/drop')), output[-3000:]
	assert b'cannot be followed (Permission denied)' in output, output[-3000:]


@posix_only
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='the probe is Linux-only')
@pytest.mark.parametrize('answers, asked, left', [([124, 0, 1], [None, 'c', 'c'], [1]), ([1, 0], [None, None], [0]), ([127, 1], [None, None], [])],
	ids=['timeout-then-yes', 'cp-refuses', 'could-not-start-then-no'])
def test_the_held_file_probe_keeps_only_cp_s_own_answer(hpcp_mod, monkeypatch, answers, asked, left):
	# A probe that timed out (124) or could not start cp (127) is no answer: the next rewrite asks again. cp's own answer (0, the
	# held file then holding the source's byte: the route; 1: no route) is kept for the worker's life.
	hpcp_mod.cp_supports_sparse()  # the real cp is asked before the stub stands in for the probe's
	monkeypatch.setattr(hpcp_mod, '_HELD', [None, None])
	answers = list(answers)
	def run_commands(*args, **kwargs):
		code = answers.pop(0)
		if code == 0:
			fd = kwargs['pass_fds'][0]
			os.pwrite(fd, b'c', 0)
			os.ftruncate(fd, 1)
		return [type('Task', (), {'returncode': code})()]
	monkeypatch.setattr(hpcp_mod.multiCMD, 'run_commands', run_commands)
	assert [hpcp_mod._held_route() for _ in asked] == asked and answers == left


@posix_only
@not_root
def test_a_node_source_that_cannot_go_beside_keeps_the_old_file_and_says_why(tmp_path):
	# out 0555 takes no new name: a FIFO source goes beside by default too (cp -a makes it anew), so both runs keep out/p, 190,
	# and -wb does not claim that without it the file would be written in place.
	os.makedirs(str(tmp_path / 'a'))
	os.mkfifo(str(tmp_path / 'a/p'))
	write(tmp_path / 'out/p', 'old')
	os.chmod(str(tmp_path / 'out/p'), 0o666)
	os.chmod(str(tmp_path / 'out'), 0o555)
	try:
		rc, output = run(['a/p', '-d', 'out/'], tmp_path)
		assert rc == 190 and read(tmp_path / 'out/p') == 'old' and b'a/p is no regular file, which cp makes anew' in output, output[-2000:]
		rc, output = run(['a/p', '-d', 'out/', '-wb'], tmp_path)
	finally:
		os.chmod(str(tmp_path / 'out'), 0o755)
	assert rc == 190 and read(tmp_path / 'out/p') == 'old' and b'-wb writes beside it; ' in output, output[-2000:]
	assert b'it would be written in place' not in output, output[-2000:]


_STOP_AS_STOPS_GO_IN = '''
import os, signal, sys
sys.path.insert(0, %(root)r)
import hpcp
real = signal.signal
def switch(signum, handler):
	result = real(signum, handler)
	if signum == signal.SIGTERM and handler is hpcp._stop:
		os.kill(os.getpid(), signal.SIGINT)
	return result
signal.signal = switch
sys.argv = ['hpcp', '-V']
hpcp.main()
'''


@posix_only
def test_a_stop_as_the_stops_go_in_ends_quietly(tmp_path):
	# A Ctrl-C landing once SIGINT's handler is in, while main() installs the next one: _stop raised it inside signal.signal,
	# outside main()'s try, and Python printed a traceback and exited 1 (1 early Ctrl-C in 2,400 on 3.7 in the review's probe).
	script = tmp_path / 'stops.py'
	script.write_text(_STOP_AS_STOPS_GO_IN % {'root': os.path.dirname(HPCP)})
	done = subprocess.run([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
	assert done.returncode == 130 and b'Traceback' not in done.stdout, done.stdout
