"""9.85: nothing at a landing is written into or followed. A file goes to a new name beside its landing and is renamed
over it once whole; what is there (an old copy, a FIFO, a device node, a link) is replaced, never opened or written
through; only what the operator named (-d) is followed. One fallback writes in place, and only a regular file whose
source just read whole. A stop runs the cleanup, so no temporary file is left."""
import errno
import os
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
small_fs = pytest.mark.skipif(not SMALL_FS, reason='HPCP_TEST_SMALL_FS: a 5 MiB filesystem of its own')


def run(args, cwd, stdin=b'', env=None, timeout=120):
	"""The CLI in a session of its own: (return code, stdout and stderr as bytes); ('timeout', output) when it hung. A cp
	that hangs (on a FIFO, in 9.84) holds the pipe in a process group of its own: the run's group is killed, and the output is
	read only as far as is there."""
	task = subprocess.Popen([sys.executable, HPCP] + args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
		cwd=str(cwd), env=dict(os.environ, **(env or {})), start_new_session=True)
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


def temporaries(root):
	"""Every .hpcp_tmp_ name below root."""
	return [os.path.join(path, name) for path, dirs, names in os.walk(str(root)) for name in names + dirs if name.startswith('.hpcp_tmp_')]


# ---- entry 1: a source that fails on read keeps the old copy ----

def _cp_writing_a_part(hpcp_mod, monkeypatch):
	"""Every cp writes a part to the name it is given and fails, as a source failing on read in the middle does."""
	def cp(command, *args, **kwargs):
		with open(command[-1], 'w') as target:
			target.write('par')
		raise RuntimeError('cp: error reading: Input/output error')
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', cp)


@posix_only
@pytest.mark.parametrize('content_only', [False, True], ids=['plain', 'content-only'])
def test_a_source_that_fails_on_read_keeps_the_old_copy_and_its_inode(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, content_only):
	# 9.84: cp truncated the old copy, and its part was removed: the last good copy gone, exit 190.
	write(tmp_path / 'a/x', 'the new and longer content')
	write(tmp_path / 'out/x', 'the last good copy')
	inode = os.stat(str(tmp_path / 'out/x')).st_ino
	monkeypatch.setattr(hpcp_mod, 'CONTENT_ONLY', content_only)
	monkeypatch.setattr(hpcp_mod, 'NAMED_DESTS', frozenset([str(tmp_path / 'out')]))
	# The source reads whole when asked first (a source that fails only on the second read), and the temporary file failed
	# partway: in place only when the old copy is at least the source's size, which it is not.
	_cp_writing_a_part(hpcp_mod, monkeypatch)
	assert hpcp_mod.copy_file(str(tmp_path / 'a/x'), [str(tmp_path / 'out/x')])[0] == 0
	assert read(tmp_path / 'out/x') == 'the last good copy' and os.stat(str(tmp_path / 'out/x')).st_ino == inode
	assert temporaries(tmp_path / 'out') == []
	assert any(error.startswith('Copy failed') and 'leaving' in error for error in hpcp_mod.ERRORS)


@posix_only
@not_root
@pytest.mark.skipif(not os.path.exists('/proc/self/mem'), reason='/proc/self/mem: a file that opens and fails on read')
@pytest.mark.parametrize('flags', [[], ['-co']], ids=['plain', 'content-only'])
def test_a_source_that_fails_on_read_in_a_directory_that_takes_no_new_name_keeps_the_old_copy(tmp_path, flags):
	# The review's A3: out 0555 with a writable out/mem. 9.84 (191) and v1's fallback left out/mem at 0 bytes.
	write(tmp_path / 'out/mem', 'the last good copy')
	os.chmod(str(tmp_path / 'out/mem'), 0o666)
	os.chmod(str(tmp_path / 'out'), 0o555)
	try:
		rc, output = run(['/proc/self/mem', '-d', 'out/'] + flags, tmp_path)
	finally:
		os.chmod(str(tmp_path / 'out'), 0o755)
	assert rc == 190 and read(tmp_path / 'out/mem') == 'the last good copy', output[-2000:]
	assert b'leaving ' in output and b'failed on read' in output


@posix_only
@not_root
@pytest.mark.parametrize('flags', [[], ['-co']], ids=['plain', 'content-only'])
def test_a_growing_rewrite_in_a_directory_that_takes_no_new_name_goes_in_place(tmp_path, flags):
	# Rule 3's clause: no temporary file could be made, nothing was written, and the free space holds the source.
	write(tmp_path / 'a/x', 'a longer new content')
	write(tmp_path / 'out/x', 'old')
	os.chmod(str(tmp_path / 'out/x'), 0o666)
	os.chmod(str(tmp_path / 'out'), 0o555)
	try:
		rc, output = run(['a/x', '-d', 'out/'] + flags, tmp_path)
	finally:
		os.chmod(str(tmp_path / 'out'), 0o755)
	assert rc == 0 and read(tmp_path / 'out/x') == 'a longer new content', output[-2000:]


@posix_only
@pytest.mark.skipif(not hasattr(os, 'link'), reason='hard links')
def test_a_landing_with_other_names_is_never_written_in_place(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# Beside failed: a write in place would change every name of it. Both names keep the old content, 190.
	write(tmp_path / 'a/x', 'new')
	write(tmp_path / 'out/x', 'old')
	os.link(str(tmp_path / 'out/x'), str(tmp_path / 'out/other'))
	real = hpcp_mod.run_command_in_multicmd_with_path_check
	def cp(command, *args, **kwargs):
		if os.path.basename(command[-1]).startswith('.hpcp_tmp_'):
			raise RuntimeError('cp: cannot create regular file: Permission denied')
		return real(command, *args, **kwargs)
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', cp)
	hpcp_mod.copy_file(str(tmp_path / 'a/x'), [str(tmp_path / 'out/x')])
	assert read(tmp_path / 'out/x') == 'old' and read(tmp_path / 'out/other') == 'old'
	assert any(error.startswith('Copy failed') and 'other names' in error for error in hpcp_mod.ERRORS)


@posix_only
def test_a_rename_that_is_refused_goes_in_place_at_once(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# The temporary file was whole: one cp beside, then the write in place (a sticky directory, a bind-mounted file).
	write(tmp_path / 'a/x', 'a longer new content')
	write(tmp_path / 'out/x', 'old')
	cps = []
	real = hpcp_mod.run_command_in_multicmd_with_path_check
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', lambda command, *a, **k: cps.append(command[-1]) or real(command, *a, **k))
	real_rename = os.rename
	def rename(source, target, **kwargs):
		if os.path.basename(str(source)).startswith('.hpcp_tmp_'):
			raise PermissionError(errno.EPERM, 'Operation not permitted', target)
		return real_rename(source, target, **kwargs)
	monkeypatch.setattr(hpcp_mod.os, 'rename', rename)
	assert hpcp_mod.copy_file(str(tmp_path / 'a/x'), [str(tmp_path / 'out/x')])[0] > 0
	assert read(tmp_path / 'out/x') == 'a longer new content' and temporaries(tmp_path / 'out') == []
	assert [os.path.basename(path).startswith('.hpcp_tmp_') for path in cps if path.startswith(str(tmp_path / 'out'))] == [True, False]


@posix_only
@root_only
def test_a_sticky_directory_keeps_9_84_s_outcome_for_another_user_s_file(tmp_path, tmp_path_factory):
	# The re-review's stickyprobe: a root-owned sticky directory, another user's 0666 file, a growing source, run as nobody.
	# The rename over it is refused (EPERM): written in place, as 9.84 did; the content is new and the owner kept.
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
	write(tmp_path / 'out/x', 'old')
	os.chown(str(tmp_path / 'out/x'), 1000, 1000)
	os.chmod(str(tmp_path / 'out/x'), 0o666)
	for flags in ([], ['-co']):
		task = subprocess.run([os.path.realpath(sys.executable), HPCP, 'a/x', '-d', 'out/'] + flags, cwd=str(tmp_path), stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT, timeout=120, preexec_fn=lambda: (os.setgid(65534), os.setuid(65534)))
		assert read(tmp_path / 'out/x') == 'a longer new content' and os.stat(str(tmp_path / 'out/x')).st_uid == 1000, task.stdout[-2000:]
		write(tmp_path / 'out/x', 'old')


# ---- what is at a landing: replaced, never opened or followed ----

@posix_only
@pytest.mark.parametrize('flags', [[], ['-co']], ids=['plain', 'content-only'])
def test_a_lone_fifo_at_a_landing_is_replaced_and_never_opened(tmp_path, flags):
	# 9.84 opened it for writing and waited for a reader: the run never ended.
	write(tmp_path / 'a/x', 'A')
	os.makedirs(str(tmp_path / 'out'))
	os.mkfifo(str(tmp_path / 'out/x'))
	rc, output = run(['a/', '-d', 'out/'] + flags, tmp_path, timeout=60)
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
@pytest.mark.parametrize('kind', ['fifo', 'link'])
def test_what_cannot_be_replaced_is_refused_and_never_opened(tmp_path, kind):
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
		rc, output = run(['a/x', '-d', 'out/'], tmp_path, timeout=60)
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
@pytest.mark.parametrize('flags', [[], ['-co']], ids=['plain', 'content-only'])
def test_a_link_to_a_device_at_a_landing_is_replaced(tmp_path, flags):
	# 9.84 wrote into /dev/null through it and failed on its times, 190, on every run.
	write(tmp_path / 'a/x', 'A')
	os.makedirs(str(tmp_path / 'out'))
	os.symlink('/dev/null', str(tmp_path / 'out/x'))
	for _ in range(2):
		rc, output = run(['a/', '-d', 'out/'] + flags, tmp_path)
		assert rc == 0 and not os.path.islink(str(tmp_path / 'out/x')) and read(tmp_path / 'out/x') == 'A', output[-2000:]


@posix_only
@root_only
@pytest.mark.parametrize('flags', [[], ['-co']], ids=['plain', 'content-only'])
def test_a_device_node_at_a_landing_is_replaced_never_written(tmp_path, flags):
	# A character node of /dev/null's numbers in the tree: 9.84 wrote the file into the device.
	write(tmp_path / 'a/x', 'A')
	os.makedirs(str(tmp_path / 'out'))
	os.mknod(str(tmp_path / 'out/x'), 0o666 | stat.S_IFCHR, os.stat('/dev/null').st_rdev)
	rc, output = run(['a/', '-d', 'out/'] + flags, tmp_path)
	assert rc == 0 and stat.S_ISREG(os.lstat(str(tmp_path / 'out/x')).st_mode) and read(tmp_path / 'out/x') == 'A', output[-2000:]


@posix_only
def test_a_link_at_a_file_s_landing_is_replaced_with_its_target_left_alone(tmp_path):
	write(tmp_path / 'a/x', 'A')
	write(tmp_path / 't/f', 'kept')
	os.chmod(str(tmp_path / 't/f'), 0o640)
	os.makedirs(str(tmp_path / 'out'))
	os.symlink('../t/f', str(tmp_path / 'out/x'))
	os.chmod(str(tmp_path / 'a/x'), 0o600)
	rc, output = run(['a/', '-d', 'out/'], tmp_path)
	assert rc == 0 and read(tmp_path / 'out/x') == 'A' and not os.path.islink(str(tmp_path / 'out/x')), output[-2000:]
	assert read(tmp_path / 't/f') == 'kept' and stat.S_IMODE(os.stat(str(tmp_path / 't/f')).st_mode) == 0o640


@posix_only
def test_a_named_link_is_followed_and_its_target_rewritten_beside_itself(tmp_path):
	write(tmp_path / 'a/x', 'A')
	write(tmp_path / 't/f', 'old')
	os.makedirs(str(tmp_path / 'out'))
	os.symlink('../t/f', str(tmp_path / 'out/x'))
	rc, output = run(['a/x', '-d', 'out/x'], tmp_path)
	assert rc == 0 and os.path.islink(str(tmp_path / 'out/x')) and read(tmp_path / 't/f') == 'A', output[-2000:]
	assert temporaries(tmp_path) == []


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
	hpcp_mod._dest_directory(d)
	assert os.path.isdir(d) and not os.path.islink(d)


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
def test_a_rewrite_carries_the_source_s_xattrs_only(tmp_path):
	# A new file beside the old one: a stale user.* xattr of the old copy does not survive (in place it did).
	write(tmp_path / 'a/x', 'new')
	write(tmp_path / 'out/x', 'old')
	try:
		os.setxattr(str(tmp_path / 'out/x'), 'user.stale', b'1')
		os.setxattr(str(tmp_path / 'a/x'), 'user.kept', b'2')
	except OSError:
		pytest.skip('no user xattrs here')
	rc, output = run(['a/', '-d', 'out/'], tmp_path)
	assert rc == 0, output[-2000:]
	assert _user_xattrs(tmp_path / 'out/x') == ['user.kept']


# ---- -co, option A': decide first, then carry exactly what was there ----

@posix_only
def test_content_only_carries_the_old_file_s_mode_and_xattrs_to_a_new_inode(tmp_path):
	write(tmp_path / 'a/x', 'new and longer')
	write(tmp_path / 'out/x', 'old')
	os.chmod(str(tmp_path / 'out/x'), 0o604)
	try:
		os.setxattr(str(tmp_path / 'out/x'), 'user.mine', b'kept')
		os.setxattr(str(tmp_path / 'a/x'), 'user.source', b'not carried')
	except OSError:
		pytest.skip('no user xattrs here')
	inode = os.stat(str(tmp_path / 'out/x')).st_ino
	rc, output = run(['a/', '-d', 'out/', '-co'], tmp_path)
	assert rc == 0 and read(tmp_path / 'out/x') == 'new and longer', output[-2000:]
	st = os.stat(str(tmp_path / 'out/x'))
	assert st.st_ino != inode and stat.S_IMODE(st.st_mode) == 0o604
	assert _user_xattrs(tmp_path / 'out/x') == ['user.mine'] and os.getxattr(str(tmp_path / 'out/x'), 'user.mine') == b'kept'


def _acl(*entries):
	"""A POSIX ACL as its xattr holds it: version 2, then (tag, perm, id) each, little-endian (u16, u16, u32)."""
	import struct
	return struct.pack('<I', 2) + b''.join(struct.pack('<HHI', tag, perm, ident) for tag, perm, ident in entries)

# Tags: user_obj, user, group_obj, mask, other. 0xffffffff is "no id".
_NO_ID = 0xffffffff


@posix_only
def test_content_only_carries_the_old_acl_and_drops_one_the_directory_gave(tmp_path):
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
	rc, output = run(['a/', '-d', 'out/', '-co'], tmp_path)
	assert rc == 0 and read(tmp_path / 'out/x') == 'new' and read(tmp_path / 'out/y') == 'new', output[-2000:]
	assert 'system.posix_acl_access' not in os.listxattr(str(tmp_path / 'out/x'))
	assert os.getxattr(str(tmp_path / 'out/y'), 'system.posix_acl_access') == before


def shutil_which(name):
	import shutil
	return shutil.which(name)


@posix_only
@root_only
def test_content_only_drops_a_file_capability_as_a_write_in_place_does(tmp_path):
	# VM 902: a write in place strips security.capability (setuid kept); carrying it would give new content a capability.
	if not shutil_which('setcap'):
		pytest.skip('setcap')
	write(tmp_path / 'a/x', 'new')
	write(tmp_path / 'out/x', 'old')
	os.chmod(str(tmp_path / 'out/x'), 0o4755)
	if subprocess.run(['setcap', 'cap_net_raw+ep', str(tmp_path / 'out/x')]).returncode:
		pytest.skip('no file capabilities here')
	rc, output = run(['a/', '-d', 'out/', '-co'], tmp_path)
	assert rc == 0 and read(tmp_path / 'out/x') == 'new', output[-2000:]
	assert 'security.capability' not in os.listxattr(str(tmp_path / 'out/x'))
	assert stat.S_IMODE(os.stat(str(tmp_path / 'out/x')).st_mode) == 0o4755


@posix_only
@not_root
def test_content_only_drops_setuid_for_a_non_root_owner_as_the_kernel_does(tmp_path):
	# suidprobe: the owner writing in place lost 4700 -> 700 and 6755 -> 755, kept 2745.
	for name, mode, kept in (('u', 0o4700, 0o700), ('ug', 0o6755, 0o755), ('g', 0o2745, 0o2745)):
		write(tmp_path / 'a' / name, 'new')
		write(tmp_path / 'out' / name, 'old')
		os.chmod(str(tmp_path / 'out' / name), mode)
	rc, output = run(['a/', '-d', 'out/', '-co'], tmp_path)
	assert rc == 0, output[-2000:]
	for name, mode, kept in (('u', 0o4700, 0o700), ('ug', 0o6755, 0o755), ('g', 0o2745, 0o2745)):
		assert stat.S_IMODE(os.stat(str(tmp_path / 'out' / name)).st_mode) == kept and read(tmp_path / 'out' / name) == 'new'


@posix_only
def test_content_only_writes_in_place_where_an_xattr_cannot_be_carried(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# trusted.*, security.ima, a filesystem's own: no new file can carry it, so the old inode is written, after the source
	# read whole.
	write(tmp_path / 'a/x', 'new')
	write(tmp_path / 'out/x', 'old')
	inode = os.stat(str(tmp_path / 'out/x')).st_ino
	monkeypatch.setattr(hpcp_mod, 'CONTENT_ONLY', True)
	real = os.listxattr
	monkeypatch.setattr(hpcp_mod.os, 'listxattr', lambda path, **kwargs: ['trusted.overlay'] if str(path).endswith('out/x') else real(path, **kwargs))
	assert hpcp_mod._co_carry(str(tmp_path / 'out/x'), os.lstat(str(tmp_path / 'out/x'))) is None
	hpcp_mod.copy_file(str(tmp_path / 'a/x'), [str(tmp_path / 'out/x')])
	assert read(tmp_path / 'out/x') == 'new' and os.stat(str(tmp_path / 'out/x')).st_ino == inode


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
def test_content_only_writes_another_user_s_file_in_place(tmp_path, hpcp_mod, monkeypatch):
	write(tmp_path / 'out/x', 'old')
	st = os.lstat(str(tmp_path / 'out/x'))
	other = os.stat_result((st.st_mode, st.st_ino, st.st_dev, st.st_nlink, st.st_uid + 1, st.st_gid) + tuple(st)[6:])
	assert hpcp_mod._co_carry(str(tmp_path / 'out/x'), other) is None
	foreign_group = os.stat_result((st.st_mode, st.st_ino, st.st_dev, st.st_nlink, st.st_uid, 4242) + tuple(st)[6:])
	if 4242 not in os.getgroups() and os.getegid() != 4242:
		assert hpcp_mod._co_carry(str(tmp_path / 'out/x'), foreign_group) is None


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


# ---- rule 3's space cases, on a small filesystem of its own ----

def _fill(out, free):
	"""Fill out's filesystem so that free bytes are left."""
	fill = os.path.join(out, 'fill')
	if os.path.exists(fill):
		os.unlink(fill)
	vfs = os.statvfs(out)
	write(fill, b'f' * (vfs.f_bavail * vfs.f_frsize - free))


@posix_only
@small_fs
def test_the_space_cases_keep_every_old_copy(tmp_path):
	# One test, as they share the one small filesystem. Scaled from the review's cases to 5 MiB:
	# C1: old 1 MiB, source 3 MiB, 2 MiB free. 9.84 truncated the old copy and lost it. The temporary file runs out of room,
	#     and growing in place could too: 190, the old copy kept.
	# C2: old 2 MiB, source 3 MiB, 1.5 MiB free. 9.84 grew it in place (0); the lossless form keeps it, 190 (proposed known
	#     limitation 13).
	# C3: four 1 MiB rewrites at once with 1.5 MiB free, -j 4, twenty runs. 9.84 landed all four; one that finds no room
	#     beside goes in place in the forced pass, the old copy being as large as the source.
	out = os.path.join(SMALL_FS, 'cases-%d' % os.getpid())
	subprocess.run(['rm', '-rf', out])
	os.makedirs(out)
	try:
		for old, new, free in ((1 << 20, 3 << 20, 2 << 20), (2 << 20, 3 << 20, 3 << 19)):
			subprocess.run(['rm', '-f', os.path.join(out, 'fill'), os.path.join(out, 'x')])
			write(os.path.join(out, 'x'), b'o' * old)
			_fill(out, free)
			write(tmp_path / 'a/x', b'n' * new)
			rc, output = run([str(tmp_path / 'a/x'), '-d', out + '/'], tmp_path)
			assert rc == 190 and read(os.path.join(out, 'x')) == 'o' * old and temporaries(out) == [], output[-2000:]
		os.unlink(os.path.join(out, 'x'))
		for run_number in range(20):
			os.unlink(os.path.join(out, 'fill'))
			for name in 'abcd':
				write(os.path.join(out, name), b'o' * (1 << 20))
				write(tmp_path / 'src' / name, (name * (1 << 20)).encode())
			_fill(out, 3 << 19)
			rc, output = run([str(tmp_path / 'src') + '/', '-d', out + '/', '-j', '4'], tmp_path)
			assert rc == 0 and all(read(os.path.join(out, name)) == name * (1 << 20) for name in 'abcd'), (run_number, output[-2000:])
			assert temporaries(out) == []
	finally:
		subprocess.run(['rm', '-rf', out])


# ---- rule 9: a stop is a cleanup ----

_SLOW_CP = '''#!/bin/sh
# cp, slowed: write a part of each copy into a temporary name, say so, and wait for the go file (or 60 s) before the real cp.
case "$1" in --version) exec cp "$@";; esac
for last; do :; done
case "$(basename "$last")" in
.hpcp_tmp_*)
	head -c 100 /etc/passwd > "$last"
	touch "$MARKS/$(basename "$last")"
	waited=0
	while [ ! -e "$GO" ] && [ $waited -lt 600 ]; do sleep 0.1; waited=$((waited + 1)); done;;
esac
exec cp "$@"
'''


def _slow_run(tmp_path, args, preexec=None):
	"""hpcp copying four files over old copies, each cp holding a part beside its landing until tmp_path/go exists."""
	cp = tmp_path / 'slowcp'
	cp.write_text(_SLOW_CP)
	os.chmod(str(cp), 0o755)
	marks = tmp_path / 'marks'
	os.makedirs(str(marks))
	for name in 'abcd':
		write(tmp_path / 'src' / name, 'new ' + name)
		write(tmp_path / 'out' / name, 'old ' + name)
	env = dict(os.environ, CP_PATH=str(cp), MARKS=str(marks), GO=str(tmp_path / 'go'))
	task = subprocess.Popen([sys.executable, HPCP, 'src/', '-d', 'out/'] + args, cwd=str(tmp_path), env=env,
		stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=preexec)
	return task, marks


def _wait_for(condition, timeout=60):
	deadline = time.monotonic() + timeout
	while not condition():
		assert time.monotonic() < deadline, 'timed out'
		time.sleep(0.05)


def _group_gone(pgid):
	try:
		os.killpg(pgid, 0)
	except ProcessLookupError:
		return True
	except PermissionError:
		return False
	return False


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
		pytest.skip('a serial run killed with SIGKILL leaves its temporary file: the proposed known limitation')
	task, marks = _slow_run(tmp_path, ['-s'] if serial else ['-j', '4'])
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
def test_a_nohup_run_keeps_going_on_a_hangup(tmp_path):
	# nohup starts the run with SIGHUP ignored: it stays ignored, in the workers too, as in 9.84.
	task, marks = _slow_run(tmp_path, ['-j', '4'], preexec=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN))
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
	task, marks = _slow_run(tmp_path, ['-j', '4'], preexec=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
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
	assert stat.S_ISFIFO(os.lstat(str(tmp_path / 'out/p')).st_mode) and stat.S_ISSOCK(os.lstat(str(tmp_path / 'out/s')).st_mode)
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
