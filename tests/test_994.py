'''hpcp 9.94. Each test fails on 9.93 (v9.93) unless it says it guards what 9.93 already gets right. Every swap is forced in
this process, at the call it is timed by, and no sweep is random. (The first-stop tests have a child wait 0.5 s for a command
to be running; a stop that lands late still passes.)'''
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
	# victim untouched, and nothing left at the temporary name. [proc-symlink] and [proc-hardlink] guard what 9.93 gets right
	# (its O_PATH check refused both links); the FIFO ones and the read route fail on 9.93.
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


@posix_only
@pytest.mark.parametrize('call', ['chown', 'chmod'])
@pytest.mark.parametrize('route', ['proc', 'read'])
def test_994_a_link_put_at_the_new_file_s_name_inside_the_carry_gets_neither_call(tmp_path, monkeypatch, route, call):
	# The carry checks the file it opened and then acts through that descriptor, never its name: a link put at the name
	# between the check and the chown (or the chmod; a user who can write the directory, in the microseconds there) must get
	# nothing, and the file the carry checked gets all of it. A carry that checks by descriptor and then acts by name passes
	# every test that plants the link before the carry (the three above). Forced here, at the call. As nobody the chown
	# half cannot show (the group the victim has is the runner's already): the root runs do. [proc-chown] and [proc-chmod]
	# guard what 9.93 gets right (its /proc route acted through the O_PATH descriptor's /proc/self/fd name); the read route
	# fails on 9.93.
	_carry_route(monkeypatch, route)
	victim = _Victim(str(tmp_path / 'v' / 'victim'))
	st = _old_file(str(tmp_path / 'out' / 'x'))
	temp = str(tmp_path / 'out' / '.hpcp_tmp_0123456789abcdef')
	_write(temp, 'what cp wrote')
	os.chmod(temp, 0o600)
	real = getattr(os, call)
	swapped = []

	def swap_then_call(path, *args, **kwargs):
		if not swapped:
			swapped.append(path)
			os.rename(temp, temp + '.away')
			os.symlink(victim.path, temp)
		return real(path, *args, **kwargs)
	monkeypatch.setattr(os, call, swap_then_call)
	hpcp._carry_over(st, temp, {})
	monkeypatch.undo()
	assert swapped, 'the carry never called os.%s' % call
	assert victim.untouched(), (victim.before, victim.state())
	now = os.lstat(temp + '.away')
	assert (stat.S_IMODE(now.st_mode), now.st_uid, now.st_gid) == (stat.S_IMODE(st.st_mode), st.st_uid, st.st_gid)


@posix_only
@pytest.mark.parametrize('route', ['proc', 'read'])
def test_994_the_carry_opens_the_new_file_in_the_directory_the_door_holds(tmp_path, monkeypatch, reset_hpcp_globals, route):
	# A run the door holds (NAMED_DESTS), cp making the new file by name (no held route). Once cp has written out/d/.hpcp_tmp_X,
	# out/d is swapped for a link to victim/, which holds a regular one-name file of that name and of the runner's (what the
	# carry's check lets through). The carry opens the name relative to the directory the door holds (out/d.away): the victim's
	# file is untouched. Opened by the logical path (out/d/...), which leads to the victim now, it would pass the check and get
	# the old file's mode, and in a root run its owner. [proc] guards what 9.93 gets right (it gave the carry the name below
	# the held directory's /proc/self/fd entry); [read] fails on 9.93.
	_carry_route(monkeypatch, route)
	_write(str(tmp_path / 'x'), 'new')
	st = _old_file(str(tmp_path / 'out' / 'd' / 'x'))
	os.makedirs(str(tmp_path / 'victim'))
	monkeypatch.setattr(hpcp, 'CONTENT_ONLY', True)
	monkeypatch.setattr(hpcp, 'WRITE_BESIDE', True)
	monkeypatch.setattr(hpcp, 'NAMED_DESTS', frozenset([str(tmp_path / 'out')]))
	monkeypatch.setattr(hpcp, '_held_route', lambda: None)
	real = hpcp.run_command_in_multicmd_with_path_check
	victims = []

	def cp_then_swap(command, *args, **kwargs):
		result = real(command, *args, **kwargs)
		victims.append(_Victim(str(tmp_path / 'victim' / os.path.basename(command[-1]))))
		os.rename(str(tmp_path / 'out' / 'd'), str(tmp_path / 'out' / 'd.away'))
		os.symlink(str(tmp_path / 'victim'), str(tmp_path / 'out' / 'd'))
		return result
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check', cp_then_swap)
	hpcp.copy_file(str(tmp_path / 'x'), [str(tmp_path / 'out' / 'd' / 'x')])
	assert victims and victims[0].untouched(), (victims[0].before, victims[0].state())
	assert not hpcp.ERRORS, hpcp.ERRORS
	now = os.lstat(str(tmp_path / 'out' / 'd.away' / 'x'))
	assert _read(str(tmp_path / 'out' / 'd.away' / 'x')) == 'new'
	assert (stat.S_IMODE(now.st_mode), now.st_uid, now.st_gid) == (stat.S_IMODE(st.st_mode), st.st_uid, st.st_gid)


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
	# says ENOTDIR, which the destination loop takes for a destination it cannot make). Each platform's own code (9.97: the
	# link case took either code on both; docs/limitations.md, "`hpcp a/x ldir/ -d a/x`: exit 213 on Linux, 202 on macOS").
	for name, text in (('a/x', 'A'), ('y', 'Y'), ('z', 'Z'), ('f', 'F')):
		_write(str(tmp_path / name), text)
	os.symlink('a', str(tmp_path / 'ldir'))
	args = ['a/x', 'ldir/', '-d', 'a/x'] if case == 'link-to-the-file-s-directory' else ['y', 'z', '-d', 'f']
	rc, output = _cli(args, tmp_path)
	assert 'may be a mounted dir' not in output and 'Continuing without creating dest folder' not in output, output[-2000:]
	assert rc == (202 if __import__('sys').platform == 'darwin' else 213 if case == 'link-to-the-file-s-directory' else 190), output[-2000:]
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
@pytest.mark.parametrize('workers', ['serial', 'pool'])
@pytest.mark.parametrize('sync', ['-nds', 'synced'])
def test_994_an_ncd_refusal_another_destination_takes_is_said_under_nds_only(tmp_path, sync, workers):
	# hpcp src/ -d A/ -d B/ -ncd [-nds], B/sub a link: the door refuses to make a directory in the link's place (-ncd) and A takes
	# src/sub's files. 9.93 said nothing of B where -nds (no directory sync to say it); 9.94 as built printed a Note once a
	# process even beside a directory sync's own 213 line (9 of them with a pool, 9.93: the 213 line alone), and the note said "the other
	# destinations take its files", which is false where they cannot. Now: under -nds a note, once a process for each refused
	# directory (one serial, at most one a worker), without that clause; with a directory sync its 213 line and no note (a guard
	# of what 9.93 does: it fails on 9.94 as built with a pool).
	for index in range(40):
		_write(str(tmp_path / 'src' / 'sub' / ('f%d' % index)), str(index))
	os.makedirs(str(tmp_path / 'A' / 'sub'))
	os.makedirs(str(tmp_path / 'B'))
	os.makedirs(str(tmp_path / 'elsewhere'))
	os.symlink(str(tmp_path / 'elsewhere'), str(tmp_path / 'B' / 'sub'))
	args = ['src/', '-d', 'A/', '-d', 'B/', '-ncd'] + (['-s'] if workers == 'serial' else ['-j', '4']) + (['-nds'] if sync == '-nds' else [])
	rc, output = _cli(args, tmp_path)
	notes = [line for line in output.splitlines() if line.startswith('Note: ')]
	assert sorted(os.listdir(str(tmp_path / 'A' / 'sub'))) == sorted('f%d' % index for index in range(40))
	assert os.listdir(str(tmp_path / 'elsewhere')) == []
	if sync == '-nds':
		assert rc == 0 and 'B/sub is a link' in output, output[-3000:]
		# (9.97: once a run; 9.94 to 9.96 said it once for each worker that met it, up to the 4 workers here.)
		assert len(notes) == 1, output[-3000:]
		for note in notes:
			assert '-ncd makes no directory in its place; nothing is written below it (name it with -d ' in note, note
			assert 'other destinations' not in note, note
	else:
		assert rc == 213 and not notes and 'B/sub is a link' in output, output[-3000:]


@posix_only
def test_994_an_ncd_refusal_under_nds_is_no_promise_that_another_destination_takes_the_files(tmp_path):
	# The note's old clause "and the other destinations take its files" was false where the other destination cannot either:
	# here C/sub is missing and -ncd makes none, so every file fails (213, nothing copied). The note claims nothing of the
	# others. A guard of what 9.93 does (it said nothing); 9.94 as built said the false clause.
	for index in range(10):
		_write(str(tmp_path / 'src' / 'sub' / ('f%d' % index)), str(index))
	os.makedirs(str(tmp_path / 'B'))
	os.makedirs(str(tmp_path / 'C'))
	os.makedirs(str(tmp_path / 'elsewhere'))
	os.symlink(str(tmp_path / 'elsewhere'), str(tmp_path / 'B' / 'sub'))
	rc, output = _cli(['src/', '-d', 'B/', '-d', 'C/', '-ncd', '-nds', '-s'], tmp_path)
	assert rc == 213 and 'the other destinations take' not in output, output[-3000:]
	# Nothing copied: not below C (no sub made), not through B/sub's link (still a link, its target empty).
	assert os.listdir(str(tmp_path / 'C')) == [] and os.listdir(str(tmp_path / 'elsewhere')) == [], output[-3000:]
	assert os.listdir(str(tmp_path / 'B')) == ['sub'] and os.path.islink(str(tmp_path / 'B' / 'sub')), output[-3000:]


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
for name in ('SIGINT', 'SIGTERM', 'SIGHUP'):
	signal.signal(getattr(signal, name), signal.SIG_DFL)  # a caller that ignores them (nohup, a background job) would keep them so
hpcp._MAIN_PID = os.getpid()
hpcp._install_stops()
here = %(here)r
first = ['sh', '-c', 'kill -%(signal)s %%d; sleep 0.5; touch %%s/one' %% (os.getpid(), here)]
run = hpcp.run_commands_in_multicmd_with_path_check

def run_then_stop_again(*args, **kwargs):
	# A later stop, still in the cleanup: sent by this process once the delayed commands have run, so it is handled apart from
	# the first (two signals pending at once would be handled once).
	result = run(*args, **kwargs)
	os.kill(os.getpid(), signal.SIG%(signal)s)
	return result

hpcp.run_commands_in_multicmd_with_path_check = run_then_stop_again
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
	# it, as a later stop: both commands run to their end, and the stop is there for main() to end the run by. A second stop
	# follows once the commands have run (still in the cleanup): it is recorded too, and says nothing more.
	import subprocess
	import sys
	code = _CLEANUP_STOPPED % {'root': os.path.dirname(hpcp.__file__), 'here': str(tmp_path), 'signal': name}
	task = subprocess.run([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60,
		start_new_session=True)
	output = task.stdout.decode('utf-8', 'replace')
	assert output.strip().splitlines()[-1] == 'ended %d' % getattr(__import__('signal'), 'SIG' + name), output
	assert os.path.exists(str(tmp_path / 'one')) and os.path.exists(str(tmp_path / 'two')), output
	# The first stop says so (it landed in the cleanup, which may take minutes: an e2fsck; later ones were the only ones to),
	# once: the second says nothing more.
	assert output.count('Stopping: cleaning up first') == 1 and 'Ctrl-\\ ends hpcp now' in output, output


_DD_NESTED_CLEANUP_STOPPED = '''
import os, signal, sys, tempfile
from types import SimpleNamespace
sys.path.insert(0, %(root)r)
import hpcp
for name in ('SIGINT', 'SIGTERM', 'SIGHUP'):
	signal.signal(getattr(signal, name), signal.SIG_DFL)  # a caller that ignores them (nohup, a background job) would keep them so
hpcp._MAIN_PID = os.getpid()
hpcp._install_stops()
here = %(here)r
roots, copied, sent = [], [], []

def mount_dir():
	path = os.path.join(here, 'mount%%d' %% len(roots))
	os.mkdir(path)
	roots.append(path)
	if len(roots) == 1:
		with open(os.path.join(path, 'data.txt'), 'w') as fh:  # partition 1's source holds a file; partition 2's mounts nothing
			fh.write('partition data')
	return path

def run(command, **kwargs):
	return [] if command[0] == 'mount' else real_run(command, **kwargs)

real_run = hpcp.run_command_in_multicmd_with_path_check
name = '_clean_up' if hasattr(hpcp, '_clean_up') else 'clean_up'  # 9.93 has no _clean_up
real_clean = getattr(hpcp, name)

def clean_with_a_stop(*args):
	# The first cleanup is the first partition's nested run's: the stop lands in it (the lists are empty there).
	if not sent:
		sent.append(1)
		os.kill(os.getpid(), signal.SIGINT)
	return real_clean(*args)

hpcp.validate_dd_source_path = lambda *a, **k: '/dev/fake-source'
hpcp.create_dd_dest_part_table = lambda *a, **k: ({'/dev/fake-source': {}, '1': {'size': 4096}, '2': {'size': 4096}}, [], None)
hpcp.get_partitions = lambda device: [device + '1', device + '2']
hpcp.is_device = lambda path: False  # the fake partitions: a mount that failed goes to dd_partition, not to a stat of them
hpcp.os.geteuid = lambda: 0
hpcp.tempfile = SimpleNamespace(mkdtemp=mount_dir)
hpcp.os.path.ismount = lambda path: path in roots[:2]
hpcp.run_command_in_multicmd_with_path_check = run
setattr(hpcp, name, clean_with_a_stop)
hpcp.dd_partition = lambda *a, **k: copied.append(a[2])
rc = hpcp.hpcp(['/dev/fake-source'], dest_image=os.path.join(here, 'dest.img'), dd=True, single_thread=True, max_workers=1,
	no_directory_sync=True)
print('rc', rc, 'mounted', len(roots), 'dd copied partitions', copied)
'''


@pytest.mark.skipif(not __import__('sys').platform.startswith('linux'), reason='-dd is Linux')
def test_994_a_stop_in_a_partition_s_cleanup_ends_the_dd_run_before_the_next_partition(tmp_path):
	# -dd copies each partition by a nested hpcp(). Ctrl-C landing in its cleanup (its lists are empty there, so the window
	# is short) is only recorded and the nested run returns 0: 9.94 as built went on to the next partition's mounts, and where
	# that could not be mounted dd_partition copied it whole, before anything checked the stop. 9.93 raised out of the nested
	# cleanup at once, so this guards what 9.93 gets right. The disk setup and the mounts are stood in for (as test_dd_errors;
	# is_device too, so a run that went on would reach dd_partition for the second partition, as described, not fail at a stat
	# of its fake path); the stop is a real SIGINT.
	import subprocess
	import sys
	code = _DD_NESTED_CLEANUP_STOPPED % {'root': os.path.dirname(hpcp.__file__), 'here': str(tmp_path)}
	task = subprocess.run([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120,
		start_new_session=True)
	output = task.stdout.decode('utf-8', 'replace')
	assert 'Keyboard Interrupt' in output and 'dd copied partitions []' in output, output[-3000:]
	assert os.listdir(str(tmp_path)).count('mount2') == 0, output[-3000:]  # the second partition's mounts were never made
	assert os.path.exists(str(tmp_path / 'mount1' / 'data.txt')), output[-3000:]  # the first partition's copy was whole


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


def _reader_seen(path, _open=os.open, _close=os.close):
	'''Whether something has the FIFO at path open for reading, asked as a writer would be let in: a write open that does not
	wait answers ENXIO where nobody reads. (By the real open and close, which a test may be standing in for.)'''
	try:
		fd = _open(path, os.O_WRONLY | os.O_NONBLOCK)
	except OSError as e:
		if e.errno == errno.ENXIO:
			return False
		raise
	_close(fd)
	return True


@posix_only
@pytest.mark.parametrize('landing_mode', [0o600, 0o222])
@pytest.mark.parametrize('held', [True, False])
@pytest.mark.parametrize('proc', [True, False])
def test_994_a_rerun_never_opens_an_identical_fifo_for_reading(tmp_path, monkeypatch, proc, held, landing_mode):
	# A guard: 9.93 never opened an identical FIFO landing, and 9.94 as built did where there is no /proc (a read open that
	# does not wait): a program blocked in open(landing, O_WRONLY) was let in, and what it wrote went into a pipe nobody
	# reads and was lost (measured on macOS 26.6.2; on Linux without /proc the writer died of SIGPIPE). On macOS even an
	# O_EVTONLY open does it. Here the writer is told from a probe: before and after each call the settle makes (each open,
	# close, stat, chown, chmod and utime), a non-blocking write open must still find no reader, so an open for reading is
	# seen even where it is closed before the next call. (A FIFO's owner and mode are set by name without /proc.)
	if proc and not hpcp._PATH_FDS:
		pytest.skip('no O_PATH with /proc here')
	monkeypatch.setattr(hpcp, '_PATH_FDS', proc)
	os.makedirs(str(tmp_path / 'out'))
	source, landing = str(tmp_path / 'src'), str(tmp_path / 'out' / 'fifo')
	os.mkfifo(source, 0o640)
	os.mkfifo(landing, 0o600)
	os.chmod(landing, landing_mode)
	os.utime(source, (1, 1))
	look, src_st = os.lstat(landing), os.lstat(source)
	base = _held(str(tmp_path / 'out')) if held else str(tmp_path / 'out')
	seen = []  # (the settle's calls only: the test's own are made before and after)
	for name in ('open', 'close', 'stat', 'lstat', 'fstat', 'chown', 'chmod', 'utime'):
		real = getattr(os, name)

		def probing(*args, _real=real, **kwargs):
			seen.append(_reader_seen(landing))
			try:
				return _real(*args, **kwargs)
			finally:
				seen.append(_reader_seen(landing))
		monkeypatch.setattr(os, name, probing)
		for group in ('supports_dir_fd', 'supports_fd', 'supports_follow_symlinks'):
			if real in getattr(os, group):  # hpcp asks them (os.chmod in os.supports_follow_symlinks): the same answer
				monkeypatch.setattr(os, group, getattr(os, group) | {probing})
	try:
		assert hpcp._settle_identical(source, src_st, landing, look, base)
	finally:
		monkeypatch.undo()
		if held:
			os.close(base)
	assert seen and not any(seen), seen
	assert stat.S_IMODE(os.lstat(landing).st_mode) == 0o640 and os.lstat(landing).st_mtime == 1  # it was settled


# ---- The 9.94 final review's mutants that no test caught (9.97): each test guards what 9.93 gets right ----


@posix_only
@pytest.mark.parametrize('what', PLANTED)
@pytest.mark.parametrize('content_only', [True, False], ids=['-co', 'plain'])
def test_994_what_is_put_at_a_held_new_file_s_name_is_not_renamed_over_the_landing(tmp_path, monkeypatch, reset_hpcp_globals, content_only, what):
	# A rewrite beside (-wb) where cp wrote the new file through the descriptor hpcp made it by (_held_route): its rename is
	# by name, so the name is first checked to be still the file hpcp holds. Forced at the fsync of that descriptor, the last
	# call before the check: the new file renamed aside and a link to the victim (or a FIFO) put at its name, by a user who
	# can write out/. The old copy stays (190, naming why), the victim untouched, nothing left at the temporary name. Without
	# the check (the review's M1) the link or the FIFO was renamed over out/x. copy_file tries again: each attempt meets it.
	if not hpcp._held_route():
		pytest.skip('cp writes no held file here')
	victim = _Victim(str(tmp_path / 'v' / 'victim'))
	out = str(tmp_path / 'out')
	_write(str(tmp_path / 'x'), 'new')
	_write(os.path.join(out, 'x'), 'old')
	st = os.lstat(os.path.join(out, 'x'))
	monkeypatch.setattr(hpcp, 'CONTENT_ONLY', content_only)
	monkeypatch.setattr(hpcp, 'WRITE_BESIDE', True)
	real, planted = os.fsync, []

	def plant_then_fsync(fd):
		held = os.fstat(fd)
		for name in os.listdir(out):
			temp = os.path.join(out, name)
			if name.startswith('.hpcp_tmp_') and not name.endswith('.away') and os.lstat(temp).st_ino == held.st_ino:
				os.rename(temp, temp + '.away')
				_plant(what, temp, victim)
				planted.append(name)
		return real(fd)
	monkeypatch.setattr(os, 'fsync', plant_then_fsync)
	hpcp.copy_file(str(tmp_path / 'x'), [os.path.join(out, 'x')])
	errors = list(hpcp.ERRORS)
	monkeypatch.undo()
	assert planted, 'no fsync of a held new file'
	assert victim.untouched(), (victim.before, victim.state())
	now = os.lstat(os.path.join(out, 'x'))
	assert stat.S_ISREG(now.st_mode) and (now.st_dev, now.st_ino) == (st.st_dev, st.st_ino), (now, st)
	assert _read(os.path.join(out, 'x')) == 'old'
	assert any(error.startswith('Copy failed') and 'something else was put at its name' in error for error in errors), errors
	assert sorted(os.listdir(out)) == sorted(['x'] + [name + '.away' for name in planted])


def _identical(tmp_path):
	'''A source and its landing out/x that the skip check found identical, the landing still to be settled: the source 0640
	with an old time (in a root run OTHER's), the landing 0600. (source, its lstat, landing, the landing's lstat, the victim,
	its time).'''
	source, landing = str(tmp_path / 'src'), str(tmp_path / 'out' / 'x')
	for path in (source, landing):
		_write(path, 'same')
	os.chmod(source, 0o640)
	os.utime(source, (1, 1))
	if os.geteuid() == 0:
		os.chown(source, OTHER, OTHER)
	os.chmod(landing, 0o600)
	victim = _Victim(str(tmp_path / 'v' / 'victim'))
	return source, os.lstat(source), landing, os.lstat(landing), victim, os.lstat(victim.path).st_mtime_ns


@posix_only
@pytest.mark.parametrize('route', ['proc', 'read'])
def test_994_a_settle_whose_landing_is_swapped_at_its_open_sets_nothing(tmp_path, monkeypatch, route):
	# A rerun's skip check found out/x identical to the source; the settle (_settle_identical) then gives it the source's
	# mode, times (and owner) through a descriptor it opens at the name, checked against the skip check's look. Forced at that
	# open: out/x renamed aside and a hard link to the victim put at its name (a user who can write out/; a symbolic link the
	# open's O_NOFOLLOW refuses already). The settle answers False with nothing set, for copy_file to look again. Without the
	# check (the review's M5) the victim got the source's mode and times, and in a root run its owner.
	_carry_route(monkeypatch, route)
	source, src_st, landing, look, victim, victim_time = _identical(tmp_path)
	base = _held(os.path.dirname(landing))
	real, swapped = os.open, []

	def swap_then_open(path, *args, **kwargs):
		if not swapped and os.path.basename(path) == 'x':
			swapped.append(path)
			os.rename(landing, landing + '.away')
			os.link(victim.path, landing)
		return real(path, *args, **kwargs)
	monkeypatch.setattr(os, 'open', swap_then_open)
	try:
		settled = hpcp._settle_identical(source, src_st, landing, look, base)
	finally:
		monkeypatch.undo()
		os.close(base)
	assert swapped, 'the settle opened nothing'
	assert settled is False
	assert victim.untouched() and os.lstat(victim.path).st_mtime_ns == victim_time, (victim.before, victim.state())
	now = os.lstat(landing + '.away')
	assert (stat.S_IMODE(now.st_mode), now.st_uid, now.st_gid, now.st_mtime_ns) == (0o600, look.st_uid, look.st_gid, look.st_mtime_ns)


@posix_only
@pytest.mark.parametrize('call', ['chown', 'chmod', 'utime'])
@pytest.mark.parametrize('route', ['proc', 'read'])
def test_994_a_link_put_at_the_landing_inside_the_settle_gets_none_of_its_calls(tmp_path, monkeypatch, route, call):
	# The settle checks the descriptor it opened against the look, then acts through it (Linux with /proc: its /proc/self/fd
	# name, which leads to that file and no further; elsewhere the descriptor), never by out/x's name: a link put there
	# between the check and the chown, the chmod or the utime (a user who can write out/) gets nothing, and the file looked at
	# gets it all. Forced at the call. A chmod by the landing's name (the review's M13, on the /proc route) followed the link.
	# The chown only in a root run: as nobody the source's owner and group are the landing's already, so there is none.
	if call == 'chown' and os.geteuid() != 0:
		pytest.skip('the settle gives no owner as nobody: the source is the runner\'s')
	_carry_route(monkeypatch, route)
	source, src_st, landing, look, victim, victim_time = _identical(tmp_path)
	base = _held(os.path.dirname(landing))
	real, swapped = getattr(os, call), []

	def swap_then_call(path, *args, **kwargs):
		if not swapped:
			swapped.append(path)
			os.rename(landing, landing + '.away')
			os.symlink(victim.path, landing)
		return real(path, *args, **kwargs)
	monkeypatch.setattr(os, call, swap_then_call)
	try:
		assert hpcp._settle_identical(source, src_st, landing, look, base)
	finally:
		monkeypatch.undo()
		os.close(base)
	assert swapped, 'the settle never called os.%s' % call
	assert victim.untouched() and os.lstat(victim.path).st_mtime_ns == victim_time, (victim.before, victim.state())
	now = os.lstat(landing + '.away')
	assert (stat.S_IMODE(now.st_mode), now.st_uid, now.st_gid, now.st_mtime_ns) == (0o640, src_st.st_uid, src_st.st_gid, src_st.st_mtime_ns)


@posix_only
def test_994_a_rewrite_gives_the_new_file_the_old_file_s_fat_bits(tmp_path, monkeypatch, reset_hpcp_globals):
	# A -co rewrite beside of a vfat file reads the old file's FAT attribute bits before the rename (here read-only and
	# archive, 0x21) and sets them on the new file after it, which vfat gave archive alone (0x20). The ioctl is stood in for
	# (no vfat here), each file's bits kept by its number. One SET, of the old file's bits, on the new file: a SET of other
	# bits (the review's M6) fails.
	import fcntl
	import struct
	_write(str(tmp_path / 'x'), 'new')
	_write(str(tmp_path / 'out' / 'x'), 'old')
	old = os.lstat(str(tmp_path / 'out' / 'x'))
	bits, sets, real = {old.st_ino: 0x21}, [], fcntl.ioctl

	def ioctl(fd, request, arg=0, *rest):
		number = os.fstat(fd).st_ino
		if request == hpcp._FAT_IOCTL_GET_ATTRIBUTES:
			return struct.pack('I', bits.get(number, 0x20))
		if request == hpcp._FAT_IOCTL_SET_ATTRIBUTES:
			bits[number] = struct.unpack('I', arg)[0]
			sets.append((number, bits[number]))
			return 0
		return real(fd, request, arg, *rest)
	monkeypatch.setattr(fcntl, 'ioctl', ioctl)
	monkeypatch.setattr(hpcp, 'CONTENT_ONLY', True)
	monkeypatch.setattr(hpcp, 'WRITE_BESIDE', True)
	hpcp.copy_file(str(tmp_path / 'x'), [str(tmp_path / 'out' / 'x')])
	assert not hpcp.ERRORS, hpcp.ERRORS
	now = os.lstat(str(tmp_path / 'out' / 'x'))
	assert now.st_ino != old.st_ino and _read(str(tmp_path / 'out' / 'x')) == 'new'
	assert sets == [(now.st_ino, 0x21)], sets
