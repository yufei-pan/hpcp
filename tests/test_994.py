'''hpcp 9.94. Each test fails on 9.93 (v9.93) unless it says it guards what 9.93 already gets right. Every swap is forced in
this process, at the call it is timed by: no sleeps, no random sweep.'''
import errno
import os
import stat

import pytest

import hpcp

posix_only = pytest.mark.skipif(os.name != 'posix', reason='POSIX')
root_only = pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() != 0, reason='needs root')
not_root = pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() == 0, reason='root writes every directory')

# Another user, in a root run: the victim's owner, and the owner a root -co gives the rewritten file.
OTHER, VICTIM = 1234, 4321


def _write(path, text):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'w') as fh:
		fh.write(text)


def _read(path):
	with open(path) as fh:
		return fh.read()


def _user_xattrs(directory):
	'''Whether a file in directory takes a user.* xattr (Linux; not tmpfs before 6.6, not every overlay).'''
	probe = os.path.join(directory, 'xattr-probe')
	_write(probe, '')
	try:
		os.setxattr(probe, 'user.hpcp-probe', b'1')
		return True
	except (AttributeError, OSError):
		return False
	finally:
		os.unlink(probe)


def _carry_route(monkeypatch, route):
	'''How _carry_over reaches a new file that cp made by name (no held route). proc: Linux with /proc, an O_PATH descriptor
	reached by its /proc/self/fd name. read: Linux without /proc, macOS and the BSDs, a descriptor opened for reading
	(forced off /proc here).'''
	if route == 'proc':
		if not hpcp._PATH_FDS:
			pytest.skip('no O_PATH with /proc here')
	else:
		monkeypatch.setattr(hpcp, '_PATH_FDS', False)


class _Victim:
	'''A file a link put at the carry's name leads to: another user's in a root run (VICTIM, 0600), the runner's otherwise.
	untouched(): its mode, owner, group, content and user.* xattrs are as they were.'''

	def __init__(self, path):
		self.path = path
		_write(path, 'not yours')
		os.chmod(path, 0o600)
		if os.geteuid() == 0:
			os.chown(path, VICTIM, VICTIM)
		self.before = self.state()

	def state(self):
		st = os.lstat(self.path)
		names = sorted(os.listxattr(self.path)) if hasattr(os, 'listxattr') else []
		return stat.S_IMODE(st.st_mode), st.st_uid, st.st_gid, st.st_nlink > 0 and _read(self.path), names

	def untouched(self):
		return self.state() == self.before


def _old_file(path):
	'''The file a -co rewrite replaces: 0666 and the runner's (a mode the victim lacks); in a root run 06755 and OTHER's.'''
	_write(path, 'old')
	if os.geteuid() == 0:
		os.chown(path, OTHER, OTHER)
		os.chmod(path, 0o6755)
	else:
		os.chown(path, -1, os.getegid())  # a new file's group (BSD: a new file takes its directory's)
		os.chmod(path, 0o666)
	return os.lstat(path)


def _plant(what, temp, victim):
	'''Put what at temp, the carry's name: a symbolic or hard link to victim, a FIFO, or (root) another user's own file.'''
	if os.path.lexists(temp):
		os.unlink(temp)
	if what == 'symlink':
		os.symlink(victim.path, temp)
	elif what == 'hardlink':
		os.link(victim.path, temp)
	elif what == 'fifo':
		os.mkfifo(temp, 0o600)
	else:
		_write(temp, 'theirs')
		os.chown(temp, VICTIM, VICTIM)
		os.chmod(temp, 0o644)


PLANTED = ['symlink', 'hardlink', 'fifo']


@posix_only
@pytest.mark.parametrize('what', PLANTED)
@pytest.mark.parametrize('route', ['proc', 'read'])
def test_994_what_is_put_at_the_carry_s_name_gets_no_mode_owner_or_xattr(tmp_path, monkeypatch, route, what):
	# -co's carry (_carry_over) of a new file cp made by name (no held route: Linux without /proc, a BSD, a cp other than GNU
	# cp): 9.93 gave it the old file's owner and mode by name, os.chown and os.chmod following a link, so a link another user
	# put at that name passed them to the file it leads to (a 0600 file became 0666; in a root run another user's file became
	# OTHER's 6755). 9.94 opens the name without following a link and refuses what is not a regular file with one name. On
	# the proc route 9.93's O_PATH check refused both links already (a guard); a FIFO took the mode there.
	_carry_route(monkeypatch, route)
	victim = _Victim(str(tmp_path / 'v' / 'victim'))
	if what == 'fifo':
		victim = None
	st = _old_file(str(tmp_path / 'out' / 'x'))
	temp = str(tmp_path / 'out' / '.hpcp_tmp_0123456789abcdef')
	if what == 'fifo':
		_plant(what, temp, None)
		before = os.lstat(temp)
	else:
		_plant(what, temp, victim)
	carried = {'user.hpcp': b'carried'} if what == 'hardlink' and _user_xattrs(str(tmp_path)) else {}
	with pytest.raises(OSError) as raised:
		hpcp._carry_over(st, temp, carried)
	assert raised.value.errno == errno.EEXIST and raised.value.strerror == 'it was replaced since cp made it', raised.value
	if victim is not None:
		assert victim.untouched(), (victim.before, victim.state())
	else:
		now = os.lstat(temp)
		assert (stat.S_IMODE(now.st_mode), now.st_uid, now.st_gid) == (stat.S_IMODE(before.st_mode), before.st_uid, before.st_gid)


@posix_only
@root_only
@pytest.mark.parametrize('route', ['proc', 'read'])
def test_994_another_user_s_file_at_the_carry_s_name_keeps_its_owner_and_mode(tmp_path, monkeypatch, route):
	# In a root run, another user who can write the directory renames a file of their own to the new file's name once cp has
	# made it: 9.93 (both routes) gave it the old file's owner and mode (OTHER's 6755) and renamed it over the old file, so
	# their content became a set-user-ID program of another owner. 9.94 refuses a file that is neither the runner's nor
	# already the old file's owner's.
	_carry_route(monkeypatch, route)
	st = _old_file(str(tmp_path / 'out' / 'x'))
	temp = str(tmp_path / 'out' / '.hpcp_tmp_0123456789abcdef')
	_plant('theirs', temp, None)
	with pytest.raises(OSError) as raised:
		hpcp._carry_over(st, temp, {})
	assert raised.value.errno == errno.EEXIST, raised.value
	now = os.lstat(temp)
	assert (stat.S_IMODE(now.st_mode), now.st_uid, now.st_gid) == (0o644, VICTIM, VICTIM)


def _co_beside(tmp_path, monkeypatch, route, plant=None, source='regular'):
	'''A -co -wb rewrite of out/x (_old_file) from x through copy_file, where cp makes the new file by name (no held route)
	and _carry_over reaches it by route (_carry_route). plant(temp) runs once cp has written the new file, before the carry.
	source: 'regular', or 'fifo' (cp -a makes the new node). The run's ERRORS.'''
	_carry_route(monkeypatch, route)
	if source == 'fifo':
		os.mkfifo(str(tmp_path / 'x'), 0o644)
	else:
		_write(str(tmp_path / 'x'), 'new')
	st = _old_file(str(tmp_path / 'out' / 'x'))
	monkeypatch.setattr(hpcp, 'CONTENT_ONLY', True)
	monkeypatch.setattr(hpcp, 'WRITE_BESIDE', True)
	monkeypatch.setattr(hpcp, '_held_route', lambda: None)
	real = hpcp.run_command_in_multicmd_with_path_check
	given = []

	def cp_then_plant(command, *args, **kwargs):
		result = real(command, *args, **kwargs)
		given.append(command[-1])
		if plant is not None:
			plant(command[-1])
		return result
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check', cp_then_plant)
	hpcp.copy_file(str(tmp_path / 'x'), [str(tmp_path / 'out' / 'x')])
	assert given and all(os.path.basename(name).startswith('.hpcp_tmp_') for name in given), given  # copy_file tries again
	return st, list(hpcp.ERRORS)


@posix_only
@pytest.mark.parametrize('what', PLANTED)
@pytest.mark.parametrize('route', ['proc', 'read'])
def test_994_a_rewrite_whose_new_file_was_swapped_keeps_the_old_copy_and_the_link_s_file(tmp_path, monkeypatch, reset_hpcp_globals, route, what):
	# The same through copy_file: a link (or a FIFO) put at the new file's name after cp wrote it. 9.93's read route gave the
	# victim the old file's mode and owner and renamed the link over out/x; 9.94 keeps out/x as it was (190, naming why), the
	# victim untouched, and nothing left at the temporary name.
	victim = _Victim(str(tmp_path / 'v' / 'victim'))
	st, errors = _co_beside(tmp_path, monkeypatch, route, lambda temp: _plant(what, temp, victim))
	assert victim.untouched(), (victim.before, victim.state())
	now = os.lstat(str(tmp_path / 'out' / 'x'))
	assert stat.S_ISREG(now.st_mode) and _read(str(tmp_path / 'out' / 'x')) == 'old' and (now.st_dev, now.st_ino) == (st.st_dev, st.st_ino)
	assert any(error.startswith('Copy failed') and 'it was replaced since cp made it' in error for error in errors), errors
	assert sorted(os.listdir(str(tmp_path / 'out'))) == ['x']


@posix_only
@pytest.mark.parametrize('source', ['regular', 'fifo'])
@pytest.mark.parametrize('route', ['proc', 'read'])
def test_994_a_rewrite_through_the_carry_s_descriptor_keeps_its_metadata(tmp_path, monkeypatch, reset_hpcp_globals, route, source):
	# A -co rewrite beside, with nothing put at the new file's name, gives it the old file's mode and owner (the runner's 0666;
	# in a root run OTHER's 6755) and a user.* xattr, as 9.93 does: it guards what 9.93 gets right. A FIFO source is made anew
	# by cp -a, and the FIFO gets them (opened for reading where there is no /proc; a FIFO takes no user.* xattr on Linux).
	xattrs = source == 'regular' and _user_xattrs(str(tmp_path))
	if xattrs:
		_write(str(tmp_path / 'out' / 'x'), 'old')
		os.setxattr(str(tmp_path / 'out' / 'x'), 'user.kept', b'yes')
	st, errors = _co_beside(tmp_path, monkeypatch, route, source=source)
	assert not errors, errors
	now = os.lstat(str(tmp_path / 'out' / 'x'))
	assert now.st_ino != st.st_ino
	assert stat.S_ISFIFO(now.st_mode) if source == 'fifo' else _read(str(tmp_path / 'out' / 'x')) == 'new'
	assert (stat.S_IMODE(now.st_mode), now.st_uid, now.st_gid) == (stat.S_IMODE(st.st_mode), st.st_uid, st.st_gid)
	if xattrs:
		assert os.getxattr(str(tmp_path / 'out' / 'x'), 'user.kept') == b'yes'
	assert sorted(os.listdir(str(tmp_path / 'out'))) == ['x']


def _cli(args, cwd):
	'''The CLI in its own session (no terminal to prompt on): (return code, stdout and stderr as text).'''
	import subprocess
	import sys
	task = subprocess.run([sys.executable, hpcp.__file__] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
		cwd=str(cwd), timeout=600, start_new_session=True)
	return task.returncode, task.stdout.decode('utf-8', 'replace')


@posix_only
@pytest.mark.parametrize('case', ['link-to-the-file-s-directory', 'a-file-destination'])
def test_994_a_destination_that_is_a_file_gets_no_mounted_dir_hint(tmp_path, case):
	# hpcp a/x ldir/ -d a/x (ldir a link to a) and hpcp y z -d f (f a plain file): with more than one source the destination
	# gets a trailing separator, its making meets the file, and 9.93 printed "Destination path ... may be a mounted dir" and
	# "Continuing without creating dest folder..." though nothing there is a directory. 9.94 leaves it to what comes after
	# (213 at the landing check on Linux, 202 on macOS; 190 for the copy into a file, or 202 on macOS, whose mkdir of f/
	# says ENOTDIR, which the destination loop takes for a destination it cannot make).
	for name, text in (('a/x', 'A'), ('y', 'Y'), ('z', 'Z'), ('f', 'F')):
		_write(str(tmp_path / name), text)
	os.symlink('a', str(tmp_path / 'ldir'))
	args = ['a/x', 'ldir/', '-d', 'a/x'] if case == 'link-to-the-file-s-directory' else ['y', 'z', '-d', 'f']
	rc, output = _cli(args, tmp_path)
	assert 'may be a mounted dir' not in output and 'Continuing without creating dest folder' not in output, output[-2000:]
	assert rc in ((202, 213) if case == 'link-to-the-file-s-directory' else (190, 202) if __import__('sys').platform == 'darwin' else (190,)), output[-2000:]
	assert _read(str(tmp_path / 'a/x')) == 'A' and _read(str(tmp_path / 'f')) == 'F'


@pytest.mark.skipif(not (3, 9) <= __import__('sys').version_info < (3, 12), reason='3.9 to 3.11: a forkserver or spawn pool '
	'adds its workers in submit, and its manager thread walks them without the lock (3.12 takes it; 3.6 to 3.8 start all at once)')
def test_994_a_pool_s_worker_table_can_be_walked_while_a_worker_is_added(reset_hpcp_globals):
	# On EL7 (Python 3.11, spawn) an early Ctrl-C printed the pool manager thread's `RuntimeError: dictionary changed size
	# during iteration` in 8 of 12 batches of the early-stop test (9.88 to 9.93): a worker died, and the manager thread's
	# terminate_broken walked the workers (get_n_children_alive, then p.terminate() and p.join()) while the main thread's
	# submit added one. Forced here at that instant: an add inside the walk. 9.93's table is a plain dict.
	with hpcp._pool(1) as pool:
		processes = pool.executor._processes
		processes[-1] = None
		walked = []
		for process in processes.values():  # the manager thread's walk
			walked.append(process)
			processes[-2] = None  # the main thread's submit, adding a worker meanwhile
		assert walked == [None] and sorted(processes) == [-2, -1]
		del processes[-1], processes[-2]


@posix_only
@pytest.mark.parametrize('sync', ['-nds', 'synced'])
def test_994_an_ncd_refusal_another_destination_takes_is_said(tmp_path, sync):
	# hpcp src/ -d A/ -d B/ -ncd -nds -s, B/sub a link: the door refuses to make a directory in the link's place (-ncd), A
	# takes src/sub's files, and 9.93 said nothing of B (exit 0, no line): -nds has no directory sync to say it. 9.94 notes it
	# once a process for each refused directory (two files below it, one note). With a directory sync (synced) its own 213 line
	# says it, and in this serial run no file's note repeats it (a guard: 9.93 printed the 213 line too).
	for name in ('f', 'g'):
		_write(str(tmp_path / 'src' / 'sub' / name), name)
	os.makedirs(str(tmp_path / 'A' / 'sub'))
	os.makedirs(str(tmp_path / 'B'))
	os.makedirs(str(tmp_path / 'elsewhere'))
	os.symlink(str(tmp_path / 'elsewhere'), str(tmp_path / 'B' / 'sub'))
	args = ['src/', '-d', 'A/', '-d', 'B/', '-ncd', '-s'] + (['-nds'] if sync == '-nds' else [])
	rc, output = _cli(args, tmp_path)
	notes = [line for line in output.splitlines() if line.startswith('Note: ') and 'B/sub is a link' in line]
	assert _read(str(tmp_path / 'A' / 'sub' / 'f')) == 'f' and _read(str(tmp_path / 'A' / 'sub' / 'g')) == 'g'
	assert os.listdir(str(tmp_path / 'elsewhere')) == []
	if sync == '-nds':
		assert rc == 0 and len(notes) == 1, output[-3000:]
		assert '-ncd makes no directory in its place; nothing is written below it, and the other destinations take its files' in notes[0]
	else:
		assert rc == 213 and not notes and 'B/sub is a link' in output, output[-3000:]


def _tfs_tree(tmp_path):
	_write(str(tmp_path / 'src' / 'f'), 'f')


@posix_only
def test_994_tfs_asks_a_destination_s_type_before_making_it(tmp_path):
	# hpcp src/ -d new/ -tfs TYPE, new/ missing and its parent of another type: 9.93 made new/, copied and exited 0 (the type
	# was asked only of a destination that was there). 9.94 asks new/'s nearest existing parent first: 227, new/ not made.
	_tfs_tree(tmp_path)
	rc, output = _cli(['src/', '-d', 'new/', '-tfs', 'nosuchfs'], tmp_path)
	assert rc == 227 and 'Destination fs type mismatch: Expected nosuchfs, got ' in output, output[-2000:]
	assert not os.path.lexists(str(tmp_path / 'new'))


@posix_only
def test_994_tfs_takes_a_destination_of_its_type_made_or_there(tmp_path):
	# The type hpcp reads for tmp_path, given to -tfs: a missing destination is made and copied into, and so is one that is
	# there (it guards what 9.93 gets right on Linux; on macOS 9.93 read no type and skipped the check).
	_tfs_tree(tmp_path)
	kind = hpcp.get_fs_type_from_path(str(tmp_path))
	assert kind, 'no type read for %s' % tmp_path
	os.makedirs(str(tmp_path / 'there'))
	for dest in ('new/', 'there/'):
		rc, output = _cli(['src/', '-d', dest, '-tfs', kind.upper()], tmp_path)
		assert rc == 0 and _read(str(tmp_path / dest / 'f')) == 'f', output[-2000:]


@posix_only
@pytest.mark.skipif(__import__('sys').platform == 'darwin', reason='macOS reads the type by statfs, not df')
def test_994_tfs_refuses_a_destination_whose_type_cannot_be_read(tmp_path):
	# A df that prints no type (DF_PATH, as a df other than GNU's or a dead mount can): 9.93 read '' and skipped the check
	# (exit 0, the copy made), as macOS's df always did. 9.94 leaves the destination out, naming why: 227.
	_tfs_tree(tmp_path)
	os.makedirs(str(tmp_path / 'out'))
	df = tmp_path / 'df'
	df.write_text('#!/bin/sh\nexit 0\n')
	os.chmod(str(df), 0o755)
	import subprocess
	import sys
	task = subprocess.run([sys.executable, hpcp.__file__, 'src/', '-d', 'out/', '-tfs', 'ext4'], stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT, cwd=str(tmp_path), timeout=600, start_new_session=True, env=dict(os.environ, DF_PATH=str(df)))
	output = task.stdout.decode('utf-8', 'replace')
	assert task.returncode == 227 and 'cannot be read (df -PT printed no type)' in output, output[-2000:]
	assert os.listdir(str(tmp_path / 'out')) == []


@pytest.mark.skipif(__import__('sys').platform != 'darwin', reason='macOS')
def test_994_macos_reads_a_destination_s_type(tmp_path):
	# macOS's df takes -T as a list of types to show and printed nothing, so 9.93's -tfs read '' and checked nothing. statfs's
	# f_fstypename is what mount(8) prints for the same filesystem.
	import subprocess
	kind = hpcp.get_fs_type_from_path(str(tmp_path))
	mounts = subprocess.run(['mount'], stdout=subprocess.PIPE, check=True).stdout.decode('utf-8', 'replace')
	assert kind and ('(%s,' % kind in mounts or '(%s)' % kind in mounts), (kind, mounts)


_CLEANUP_STOPPED = '''
import os, signal, sys
sys.path.insert(0, %(root)r)
import hpcp
hpcp._MAIN_PID = os.getpid()
hpcp._install_stops()
here = %(here)r
first = ['sh', '-c', 'kill -%(signal)s %%d; sleep 0.5; touch %%s/one' %% (os.getpid(), here)]
try:
	hpcp.clean_up([], [], [first, ['touch', os.path.join(here, 'two')]])
	print('ended', hpcp._STOPPED)
except KeyboardInterrupt:
	print('cut')
'''


@posix_only
@pytest.mark.parametrize('name', ['INT', 'TERM', 'HUP'])
def test_994_a_first_stop_during_the_cleanup_lets_it_finish(tmp_path, name):
	# A run that mounted an image or attached a loop device, stopped for the first time while clean_up() runs (here inside its
	# delayed commands' step: the first command signals hpcp): 9.93 raised the stop there, so the step was cut (its command
	# stopped) and every step after it skipped, and mounts, loop devices or a -dd clone's mkfs UUID could stay. 9.94 records
	# it, as a later stop: both commands run to their end, and the stop is there for main() to end the run by.
	import subprocess
	import sys
	code = _CLEANUP_STOPPED % {'root': os.path.dirname(hpcp.__file__), 'here': str(tmp_path), 'signal': name}
	task = subprocess.run([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60,
		start_new_session=True)
	output = task.stdout.decode('utf-8', 'replace')
	assert output.strip().splitlines()[-1] == 'ended %d' % getattr(__import__('signal'), 'SIG' + name), output
	assert os.path.exists(str(tmp_path / 'one')) and os.path.exists(str(tmp_path / 'two')), output


@posix_only
@pytest.mark.skipif(not os.path.isdir('/proc/self/fd'), reason='the stand-in names an open file by /proc/self/fd')
def test_994_a_rewrite_s_fat_bits_go_to_the_held_directory_not_through_a_swapped_one(tmp_path, monkeypatch, reset_hpcp_globals):
	# A -co rewrite beside of a vfat file reads the old file's FAT attribute bits and sets them on the new one after the
	# rename. Without /proc (forced) 9.93 did both by the landing's logical path: out/d swapped for a link to victim/ once cp
	# had written the new file led them to victim/x. 9.94 opens the name in the directory the door holds, as the rename.
	# The ioctl is stood in for (no vfat here): the old file reads read-only (0x01), the new one 0.
	import fcntl
	monkeypatch.setattr(hpcp, '_PATH_FDS', False)
	_write(str(tmp_path / 'x'), 'new')
	_write(str(tmp_path / 'out' / 'd' / 'x'), 'old')
	_write(str(tmp_path / 'victim' / 'x'), 'not yours')
	monkeypatch.setattr(hpcp, 'CONTENT_ONLY', True)
	monkeypatch.setattr(hpcp, 'WRITE_BESIDE', True)
	monkeypatch.setattr(hpcp, 'NAMED_DESTS', frozenset([str(tmp_path / 'out')]))
	calls = []

	def ioctl(fd, request, arg=0, *rest):
		where = os.readlink('/proc/self/fd/%d' % fd)
		calls.append((request, where))
		if request == hpcp._FAT_IOCTL_GET_ATTRIBUTES:
			return __import__('struct').pack('I', 1 if len(calls) == 1 else 0)
		return 0
	monkeypatch.setattr(fcntl, 'ioctl', ioctl)
	real = hpcp.run_command_in_multicmd_with_path_check

	def cp_then_swap(command, *args, **kwargs):
		result = real(command, *args, **kwargs)
		os.rename(str(tmp_path / 'out' / 'd'), str(tmp_path / 'out' / 'd.away'))
		os.symlink(str(tmp_path / 'victim'), str(tmp_path / 'out' / 'd'))
		return result
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check', cp_then_swap)
	hpcp.copy_file(str(tmp_path / 'x'), [str(tmp_path / 'out' / 'd' / 'x')])
	assert calls and not [where for _, where in calls if 'victim' in where], calls
	assert [where for request, where in calls if request == hpcp._FAT_IOCTL_SET_ATTRIBUTES] == [str(tmp_path / 'out' / 'd.away' / 'x')], calls


def _held(directory):
	'''The door's hold on directory, as it holds one where there is no /proc (a read descriptor).'''
	return os.open(directory, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))


@posix_only
@pytest.mark.skipif(not os.path.isdir('/proc/self/fd'), reason='the stand-in names an open file by /proc/self/fd')
def test_994_an_identical_fifo_s_xattrs_go_through_a_descriptor_where_there_is_no_proc(tmp_path, monkeypatch):
	# A rerun settles an identical FIFO at a landing (_settle_identical). Without /proc (forced) 9.93 listed, removed and set
	# its xattrs by the landing's logical path, so out/d swapped for a link to victim/ after the door held it led them to
	# victim/fifo, and set its mode by its name below the held directory (following a link put there). 9.94 opens the FIFO
	# for reading without following a link (it opens without waiting for a writer) and does all of it through that
	# descriptor. The xattr calls are stood in for: a FIFO takes no user.* xattr on Linux.
	monkeypatch.setattr(hpcp, '_PATH_FDS', False)
	os.makedirs(str(tmp_path / 'out' / 'd'))
	os.makedirs(str(tmp_path / 'victim'))
	os.mkfifo(str(tmp_path / 'src'), 0o640)
	os.mkfifo(str(tmp_path / 'out' / 'd' / 'fifo'), 0o600)
	os.mkfifo(str(tmp_path / 'victim' / 'fifo'), 0o600)
	src_st = os.lstat(str(tmp_path / 'src'))
	look = os.lstat(str(tmp_path / 'out' / 'd' / 'fifo'))
	base = _held(str(tmp_path / 'out' / 'd'))
	set_on = []

	def where(target):
		return os.readlink('/proc/self/fd/%d' % target) if isinstance(target, int) else target

	def listxattr(target, **kwargs):
		return ['user.k'] if where(target) == str(tmp_path / 'src') else []

	def getxattr(target, name, **kwargs):
		return b'v'

	def setxattr(target, name, value, **kwargs):
		set_on.append(where(target))

	def removexattr(target, name, **kwargs):
		set_on.append(where(target))
	monkeypatch.setattr(hpcp, '_XATTRS', (listxattr, getxattr, setxattr, removexattr))
	try:
		os.rename(str(tmp_path / 'out' / 'd'), str(tmp_path / 'out' / 'd.away'))
		os.symlink(str(tmp_path / 'victim'), str(tmp_path / 'out' / 'd'))
		assert hpcp._settle_identical(str(tmp_path / 'src'), src_st, str(tmp_path / 'out' / 'd' / 'fifo'), look, base)
	finally:
		os.close(base)
	assert set_on == [str(tmp_path / 'out' / 'd.away' / 'fifo')], set_on
	assert stat.S_IMODE(os.lstat(str(tmp_path / 'out' / 'd.away' / 'fifo')).st_mode) == 0o640
	assert stat.S_IMODE(os.lstat(str(tmp_path / 'victim' / 'fifo')).st_mode) == 0o600


def _fifo_takes_xattrs(directory):
	probe = os.path.join(directory, 'fifo-probe')
	os.mkfifo(probe)
	try:
		hpcp._XATTRS[2](probe, 'user.hpcp-probe', b'1', follow_symlinks=False)
		return True
	except (OSError, TypeError):
		return False
	finally:
		os.unlink(probe)


@posix_only
def test_994_an_identical_fifo_gets_its_source_s_xattrs(tmp_path):
	# Where a FIFO takes an xattr (macOS: user names; Linux: none as nobody), a rerun gives an identical FIFO the source's.
	# 9.93 gave a node none on macOS (no xattr call there takes a dir_fd, so they would have gone by the full path); 9.94
	# sets them through the descriptor it opened the FIFO by. Skipped where a FIFO takes none.
	if hpcp._XATTRS is None or not _fifo_takes_xattrs(str(tmp_path)):
		pytest.skip('a FIFO takes no xattr here')
	os.makedirs(str(tmp_path / 'out'))
	os.mkfifo(str(tmp_path / 'src'), 0o640)
	os.mkfifo(str(tmp_path / 'out' / 'fifo'), 0o640)
	hpcp._XATTRS[2](str(tmp_path / 'src'), 'user.kept', b'yes', follow_symlinks=False)
	base = _held(str(tmp_path / 'out'))
	try:
		assert hpcp._settle_identical(str(tmp_path / 'src'), os.lstat(str(tmp_path / 'src')), str(tmp_path / 'out' / 'fifo'),
			os.lstat(str(tmp_path / 'out' / 'fifo')), base)
	finally:
		os.close(base)
	assert hpcp._XATTRS[1](str(tmp_path / 'out' / 'fifo'), 'user.kept', follow_symlinks=False) == b'yes'
