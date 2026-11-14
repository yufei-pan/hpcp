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


def _co_beside(tmp_path, monkeypatch, route, plant=None):
	'''A -co -wb rewrite of out/x (_old_file) from x through copy_file, where cp makes the new file by name (no held route)
	and _carry_over reaches it by route (_carry_route). plant(temp) runs once cp has written the new file, before the carry.
	The run's ERRORS.'''
	_carry_route(monkeypatch, route)
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
@pytest.mark.parametrize('route', ['proc', 'read'])
def test_994_a_rewrite_through_the_carry_s_descriptor_keeps_its_metadata(tmp_path, monkeypatch, reset_hpcp_globals, route):
	# A -co rewrite beside, with nothing put at the new file's name, gives it the old file's mode and owner (the runner's 0666;
	# in a root run OTHER's 6755) and a user.* xattr, as 9.93 does: it guards what 9.93 gets right.
	xattrs = _user_xattrs(str(tmp_path))
	if xattrs:
		_write(str(tmp_path / 'out' / 'x'), 'old')
		os.setxattr(str(tmp_path / 'out' / 'x'), 'user.kept', b'yes')
	st, errors = _co_beside(tmp_path, monkeypatch, route)
	assert not errors, errors
	now = os.lstat(str(tmp_path / 'out' / 'x'))
	assert _read(str(tmp_path / 'out' / 'x')) == 'new' and now.st_ino != st.st_ino
	assert (stat.S_IMODE(now.st_mode), now.st_uid, now.st_gid) == (stat.S_IMODE(st.st_mode), st.st_uid, st.st_gid)
	if xattrs:
		assert os.getxattr(str(tmp_path / 'out' / 'x'), 'user.kept') == b'yes'
	assert sorted(os.listdir(str(tmp_path / 'out'))) == ['x']
