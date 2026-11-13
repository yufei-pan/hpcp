'''hpcp 9.91: the name probe's undo trusted what it found at its own name (docs/lessons-full.md, 2026-11-13). Each test fails on
9.90 (b4dc027) unless it says it guards a case 9.90 already gets right (the 'nothing' cases are the controls: no swap). Every
swap is forced at the call it is timed by, in this process (no sleeps, no random sweep): another user who can write the
directory acts once.'''
import contextlib
import errno
import os
import stat

import pytest

import hpcp

posix_only = pytest.mark.skipif(os.name != 'posix', reason='POSIX')
not_root = pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() == 0, reason='root may do it')

PROBE = '.hpcp-case-probe-'
# What the directory that must be kept holds: files, a link, and a subdirectory (an unlink of the undo cannot remove it).
KEPT = ['X', 'keep1', 'keep2', 'lnk', 'sub']


def _fill(directory):
	'''KEPT in directory, as the runner's own; in a root run another user's, mode 0700 (anyone's files are what is lost there).'''
	os.makedirs(os.path.join(directory, 'sub'))
	for name in ('keep1', 'keep2', 'X'):
		with open(os.path.join(directory, name), 'w') as fh:
			fh.write('mine')
	os.symlink('keep1', os.path.join(directory, 'lnk'))
	if os.geteuid() == 0:
		for name in ('keep1', 'keep2', 'X', 'sub', '.'):
			os.chown(os.path.join(directory, name), 1234, 1234)
		os.chmod(directory, 0o700)


def _names(directory):
	return sorted(os.listdir(directory))


def _macs_route(monkeypatch):
	'''macOS's route, whatever the platform: no O_PATH and no /proc/self/fd, so a read descriptor holds the directory (9.90 had its
	probe by path there: _PROBE_BY_FD, which is False on macOS, then gated it; 9.91's gate is _ASK_BY_FD, True there).'''
	if not hpcp._ASK_BY_FD:
		pytest.skip('no descriptor calls here: the probe goes by path')
	monkeypatch.setattr(hpcp, '_PATH_FDS', False)
	monkeypatch.setattr(hpcp, '_PROBE_BY_FD', False)
	monkeypatch.setattr(hpcp, '_PARENT_FLAGS', os.O_RDONLY | os.O_DIRECTORY | getattr(os, 'O_CLOEXEC', 0))


def _can_lchmod_by_path():
	'''Whether chmod takes a path and says not to follow a link (macOS; glibc 2.32's fchmodat through /proc on Python 3.11 and
	later; not Python 3.6 on Linux): what gives a directory with no owner read or search its owner bits (_owner_read).'''
	import tempfile
	with tempfile.TemporaryDirectory() as work:
		try:
			os.chmod(work, 0o700, follow_symlinks=False)
			return True
		except (OSError, NotImplementedError, ValueError):
			return False


@contextlib.contextmanager
def _swap(out, what, victim, moved):
	'''One swap while the probe of out runs, timed by the call it follows. dir-early: right after the probe directory is made,
	moved (a directory of the runner's own, in out) is renamed over it. link-early: the same moment, the probe renamed away and a
	link to victim put at its name. link-late: the same, after the first name is made, so the undo's listing is the next call.
	The probe's name once it is known: state['probe'].'''
	real_mkdir, real_listdir, real_write = os.mkdir, os.listdir, os.write
	state = {'done': what == 'nothing', 'probe': None, 'made': False}

	def swap(name):
		assert not state['done']
		state['done'] = True
		path = os.path.join(out, name)
		if what == 'dir-early':
			os.rename(moved, path)
		else:
			os.rename(path, path + '.away')
			os.symlink(victim, path)

	def mkdir(path, *args, **kwargs):
		result = real_mkdir(path, *args, **kwargs)
		name = os.path.basename(str(path))
		if name.startswith(PROBE) and state['probe'] is None:
			state['probe'] = name
			if what in ('dir-early', 'link-early'):
				swap(name)
		return result

	def write(fd, data):
		state['made'] = True
		return real_write(fd, data)

	def listdir(path='.'):
		if what == 'link-late' and state['made'] and not state['done']:
			swap(state['probe'])
		return real_listdir(path)

	os.mkdir, os.listdir, os.write = mkdir, listdir, write
	try:
		yield state
	finally:
		os.mkdir, os.listdir, os.write = real_mkdir, real_listdir, real_write


@posix_only
@pytest.mark.parametrize('what', ['nothing', 'dir-early', 'link-early', 'link-late'])
@pytest.mark.parametrize('route', ['native', 'macos'])
def test_991_a_probe_replaced_by_a_directory_or_a_link_costs_nobody_a_file(tmp_path, monkeypatch, route, what):
	# The name probe makes a hidden directory in the destination, makes the names in it, and lists it and unlinks everything
	# listed. Another user who can write the destination (not sticky) renames what stands at the probe's name and puts another
	# there: a directory of the runner's own (or, in a root run, anyone's) between its mkdir and its open was taken for the probe
	# and emptied of its files (Linux's descriptor route, 9.89 and 9.90); a link put there (macOS's route, the probe by path) sent
	# the undo's listing and unlinks wherever it led. 9.91 holds the probe by a descriptor on both, and a directory it opens must
	# be empty, or it is not the one just made: nothing is made in it and nothing of it removed, and the pair cannot be asked.
	# 'native' is the platform's own route (9.90 on Linux: the descriptor, which already refuses a link: guarded); 'macos' is
	# macOS's, on any platform. Nothing of what is put at the probe's name may be lost.
	if route == 'macos':
		_macs_route(monkeypatch)
	out, victim, moved = str(tmp_path / 'out'), str(tmp_path / 'victim'), str(tmp_path / 'out' / 'E')
	os.makedirs(out)
	os.chmod(out, 0o777)
	_fill(victim)
	_fill(moved)
	with _swap(out, what, victim, moved) as state:
		answers, why = hpcp._ask_names(out, ['X', 'x'])
	assert state['probe'], 'the probe was never made'
	assert _names(victim) == KEPT, 'the directory a link at the probe led to lost files: %r' % _names(victim)
	kept = os.path.join(out, state['probe']) if what == 'dir-early' else moved
	assert _names(kept) == KEPT, 'the directory put over the probe lost files: %r' % _names(kept)
	if what == 'nothing':
		assert not why and set(answers) == {'X', 'x'}, (answers, why)
		assert _names(out) == ['E'], _names(out)  # the probe is gone
	elif what == 'dir-early':
		# Not the probe: both names cannot be asked, and the reason says what happened and what to do.
		assert answers == {} and set(why) == {'X', 'x'}, (answers, why)
		reason = why['X']
		assert 'nothing can be made in' in reason and 'replaced by another' in reason and 'run again' in reason, reason


@posix_only
def test_991_a_rerun_after_a_replaced_probe_asks_the_directory(tmp_path):
	# The refusal is of that one try: the swapped-in directory put back where it came from, the next question is answered.
	out, victim, moved = str(tmp_path / 'out'), str(tmp_path / 'victim'), str(tmp_path / 'out' / 'E')
	os.makedirs(out)
	_fill(victim)
	_fill(moved)
	with _swap(out, 'dir-early', victim, moved) as state:
		answers, why = hpcp._ask_names(out, ['X', 'x'])
	assert answers == {} and set(why) == {'X', 'x'}, (answers, why)
	os.rename(os.path.join(out, state['probe']), moved)
	answers, why = hpcp._ask_names(out, ['X', 'x'])
	assert not why and set(answers) == {'X', 'x'}, (answers, why)
	assert _names(out) == ['E'] and _names(moved) == KEPT


@posix_only
@not_root
def test_991_a_directory_the_runner_may_write_and_not_read_is_still_asked(tmp_path, monkeypatch):
	# Guards what 9.90 does on macOS: there the directory is held by a read descriptor, which a directory the runner may write and
	# search and not read (0311) refuses. It is held by O_SEARCH where Python has one (macOS), and asked by path where it has
	# none (this route on Linux; docs/limitations.md, "The name probe by path"). The pair is answered and the probe is gone; the
	# directory keeps its mode and its times (put back through the O_SEARCH descriptor on macOS, where a failure would be silent).
	_macs_route(monkeypatch)
	directory = str(tmp_path / 'd')
	os.mkdir(directory)
	os.utime(directory, (1500000000, 1500000000))
	os.chmod(directory, 0o311)
	try:
		answers, why = hpcp._ask_names(directory, ['X', 'x'])
		now = stat.S_IMODE(os.stat(directory).st_mode)
	finally:
		os.chmod(directory, 0o755)
	assert not why and set(answers) == {'X', 'x'}, (answers, why)
	assert now == 0o311, oct(now)
	assert os.stat(directory).st_mtime == 1500000000, 'the directory did not keep its times'
	assert _names(directory) == []  # the probe's own directory is gone


@posix_only
@not_root
@pytest.mark.parametrize('mode', [0o311, 0o207, 0o007, 0o047])
def test_991_a_link_put_at_the_probe_in_a_directory_the_runner_may_not_read_costs_nobody_a_file(tmp_path, monkeypatch, mode):
	# macOS's route: a read descriptor cannot hold a directory the runner may write and not read, and 9.90 and 9.91's first build
	# asked it by path, so a link put at the probe's name before the undo sent the undo's listing and unlinks where it led. The
	# probe holds it too: one with search (0311, another user's drop box) by O_SEARCH (macOS has it; Linux, where this route is
	# only stood in for, has none: skipped), and one of the runner's own with neither read nor search (0207, 0007, 0047) by a read
	# descriptor once the probe has given it its owner bits (the door's _owner_read: chmod by name without following a link).
	_macs_route(monkeypatch)
	if mode & 0o100 and not hasattr(os, 'O_SEARCH'):
		pytest.skip('no O_SEARCH here: asked by path (docs/limitations.md, "The name probe by path")')
	if not mode & 0o100 and not _can_lchmod_by_path():
		pytest.skip('no chmod by name that does not follow a link here: the probe is refused (docs/limitations.md)')
	out, victim = str(tmp_path / 'out'), str(tmp_path / 'victim')
	os.mkdir(out)
	_fill(victim)
	os.utime(out, (1500000000, 1500000000))
	os.chmod(out, mode)
	try:
		with _swap(out, 'link-late', victim, None) as state:
			answers, why = hpcp._ask_names(out, ['X', 'x'])
	finally:
		os.chmod(out, 0o755)
	assert state['done'], 'the link was never put at the probe'
	assert _names(victim) == KEPT, 'the directory a link at the probe led to lost files: %r' % _names(victim)
	assert not why and set(answers) == {'X', 'x'}, (answers, why)
	assert os.stat(out).st_mtime == 1500000000, 'the directory asked did not keep its times'


@posix_only
@pytest.mark.parametrize('route', ['native', 'macos'])
def test_991_an_empty_directory_of_anothers_put_over_the_probe_is_left_as_it_is(tmp_path, monkeypatch, route):
	# "Empty when opened" is not "the one just made": an EMPTY directory renamed over the probe between its mkdir and its open
	# passed 9.91's first check. The names were then made in it, whoever owned it could put a name there first and so decide
	# the answer, the undo removed what its owner made in it meanwhile, and the probe's rmdir removed the directory itself. The
	# first file the probe makes must have the directory's owner (as the mount shows both); where it has not, that file goes,
	# and nothing else of the directory is touched: not what its owner makes in it meanwhile, not the directory.
	# As root the directory is uid 1234's. Without root no test can make another's directory: os.fstat then says so of it.
	if route == 'macos':
		_macs_route(monkeypatch)
	out = str(tmp_path / 'out')
	moved = os.path.join(out, 'E')
	os.makedirs(moved)
	os.chmod(out, 0o777)
	its = os.stat(moved)
	if os.geteuid() == 0:
		os.chown(moved, 1234, 1234)
	else:
		real_fstat = os.fstat

		def fstat(fd):
			st = real_fstat(fd)
			if (st.st_dev, st.st_ino) == (its.st_dev, its.st_ino):
				return os.stat_result(tuple(st[:4]) + (st.st_uid + 1,) + tuple(st[5:10]))
			return st
		monkeypatch.setattr(os, 'fstat', fstat)
	real_listdir = os.listdir
	late = []

	def listdir(path='.'):
		names = real_listdir(path)
		if isinstance(path, int) and not late:
			# Right after the probe's own look at what it opened: its owner makes a file in it.
			probe = [name for name in real_listdir(out) if name.startswith(PROBE)][0]
			late.append(os.path.join(out, probe, 'late'))
			with open(late[0], 'w') as fh:
				fh.write('its owner made this')
		return names
	monkeypatch.setattr(os, 'listdir', listdir)
	with _swap(out, 'dir-early', None, moved) as state:
		answers, why = hpcp._ask_names(out, ['X', 'x'])
	kept = os.path.join(out, state['probe'])
	assert os.path.isdir(kept) and os.stat(kept).st_ino == its.st_ino, 'the directory put over the probe is gone'
	assert _names(kept) == ['late'], _names(kept)
	assert answers == {} and set(why) == {'X', 'x'}, (answers, why)
	assert 'replaced by another' in why['X'] and 'run again' in why['X'], why['X']


@posix_only
@pytest.mark.parametrize('route', ['native', 'macos', 'no-proc'])
def test_991_the_probe_sets_the_times_of_the_directory_it_holds_not_of_what_stands_at_its_name(tmp_path, monkeypatch, route):
	# The probe's last call puts the directory's times back. 9.91's first build and every version before it did that by path:
	# with the directory renamed away and a link put at its name while the probe ran (forced at the probe's rmdir), the link's
	# target got those times, in a root run any file's. They go through the descriptor the probe holds; 'no-proc' is Linux's
	# O_PATH route without /proc, which the first fix round still did by path.
	if route == 'macos':
		_macs_route(monkeypatch)
	elif route == 'no-proc':
		if not hpcp._PROBE_BY_FD:
			pytest.skip('no O_PATH here')
		monkeypatch.setattr(hpcp, '_PATH_FDS', False)
	top, victim = str(tmp_path / 'top'), str(tmp_path / 'victim-file')
	out = os.path.join(top, 'out')
	os.makedirs(out)
	with open(victim, 'w') as fh:
		fh.write('data')
	os.utime(victim, (1000000000, 1000000000))
	os.utime(out, (1500000000, 1500000000))
	real_rmdir, swapped = os.rmdir, []

	def rmdir(path, *args, **kwargs):
		result = real_rmdir(path, *args, **kwargs)
		if not swapped and PROBE in str(path):
			swapped.append(True)
			os.rename(out, out + '.away')
			os.symlink(victim, out)
		return result
	monkeypatch.setattr(os, 'rmdir', rmdir)
	answers, why = hpcp._ask_names(out, ['X', 'x'])
	assert swapped, 'the probe never removed its directory'
	assert not why and set(answers) == {'X', 'x'}, (answers, why)
	assert os.stat(victim).st_mtime == 1000000000, 'the file a link at the directory\'s name led to got its times'
	assert os.stat(out + '.away').st_mtime == 1500000000, 'the directory asked did not keep its times'


@posix_only
@pytest.mark.parametrize('what', ['fifo', 'link'])
@pytest.mark.parametrize('route', ['native', 'macos'])
def test_991_a_fifo_or_a_link_at_a_name_in_the_probe_is_never_waited_for_or_followed(tmp_path, monkeypatch, route, what):
	# A name that is there already when the probe makes it, once two names are made, is opened to read which made name holds it
	# (_holder). Every version to 9.91's first build opened it with O_RDONLY alone: a FIFO put there by whoever may write the
	# probe (on macOS an inheritable ACL entry of the destination gives its owner that; until 9.91 an empty directory of his own
	# renamed over the probe did) was waited for, for ever, with SIGINT, SIGTERM and SIGHUP blocked, and a link was followed. The
	# open neither waits nor follows; such a name says nothing, and is one "it cannot say which" holds. Forced right after the
	# second name is made (P, Q and R: no filesystem folds them together, so two are made everywhere). An open that would wait
	# or follow is refused here instead of made, so the old code fails, not hangs.
	if route == 'macos':
		_macs_route(monkeypatch)
	out, zero = str(tmp_path / 'out'), str(tmp_path / 'zero')
	os.mkdir(out)
	with open(zero, 'w') as fh:
		fh.write('0')
	real_open = os.open
	planted, flags_seen = [], []

	def open_(path, flags, *args, **kwargs):
		if planted and not flags & os.O_CREAT and os.path.basename(str(path)) == 'R':
			flags_seen.append(flags)
			if not (flags & os.O_NONBLOCK and flags & os.O_NOFOLLOW):
				raise OSError(errno.EIO, 'this open would wait for a FIFO or follow a link', str(path))
		fd = real_open(path, flags, *args, **kwargs)
		if flags & os.O_CREAT and os.path.basename(str(path)) == 'Q' and not planted:
			probe = [name for name in os.listdir(out) if name.startswith(PROBE)][0]
			planted.append(os.path.join(out, probe, 'R'))
			if what == 'fifo':
				os.mkfifo(planted[0])
			else:
				os.symlink(zero, planted[0])
		return fd
	monkeypatch.setattr(os, 'open', open_)
	answers, why = hpcp._ask_names(out, ['P', 'Q', 'R'])
	assert planted and flags_seen, 'the name that was there already was never opened'
	assert all(flags & os.O_NONBLOCK and flags & os.O_NOFOLLOW for flags in flags_seen), [oct(flags) for flags in flags_seen]
	assert 'R' in why and 'cannot say which' in why['R'], (answers, why)
	assert answers.get('P') == 'P' and answers.get('Q') == 'Q', (answers, why)
	assert _names(out) == []  # the probe is gone, with what was put in it


@posix_only
@pytest.mark.parametrize('what', ['names-there', 'first-create-fails'])
@pytest.mark.parametrize('route', ['native', 'macos'])
def test_991_the_undo_removes_nothing_it_lists_before_a_file_made_in_the_probe_passed_the_check(tmp_path, monkeypatch, route, what):
	# The undo lists the probe and removes what it finds. Until a file the probe made there has passed the owner check, what
	# stands at the probe's name may be another's directory that was empty when opened. 9.91's first round still listed and
	# emptied it when no file was made: every name was there already (made by someone else right after the probe's look; a
	# root run then removed what a third user had put in that directory meanwhile), or the first create failed (no space). What
	# was put there (forced right after the probe's look) stays, and so does the directory while it holds anything.
	if route == 'macos':
		_macs_route(monkeypatch)
	out = str(tmp_path / 'out')
	os.mkdir(out)
	real_listdir, real_open = os.listdir, os.open
	state = {'probe': None}

	def listdir(path='.'):
		names = real_listdir(path)
		if isinstance(path, int) and state['probe'] is None:
			state['probe'] = [name for name in real_listdir(out) if name.startswith(PROBE)][0]
			for name in ['precious'] + (['X', 'x'] if what == 'names-there' else []):
				with open(os.path.join(out, state['probe'], name), 'w') as fh:
					fh.write('someone else made this')
		return names

	def open_(path, flags, *args, **kwargs):
		if what == 'first-create-fails' and state['probe'] and flags & os.O_CREAT and os.path.basename(str(path)) in ('X', 'x'):
			raise OSError(errno.ENOSPC, 'No space left on device', str(path))
		return real_open(path, flags, *args, **kwargs)
	monkeypatch.setattr(os, 'listdir', listdir)
	monkeypatch.setattr(os, 'open', open_)
	answers, why = hpcp._ask_names(out, ['X', 'x'])
	assert state['probe'], 'the probe was never looked at'
	kept = os.path.join(out, state['probe'])
	assert os.path.isdir(kept) and 'precious' in real_listdir(kept), 'what was put in the probe was removed'
	assert answers == {} and set(why) == {'X', 'x'}, (answers, why)


@posix_only
@pytest.mark.parametrize('route', ['native', 'macos'])
def test_991_a_probe_whose_files_show_an_owner_other_than_the_runner_is_answered(tmp_path, monkeypatch, route):
	# Guards the form of the owner check (9.90 had none). A mount that shows its own owner for everything (vfat or exFAT with
	# uid=, NFS root_squash in a root run) gives the probe and its files an owner other than geteuid(). The check asks whether
	# the first file made got the probe's owner, as the mount shows both, so the pair is answered there; a check against
	# geteuid() would refuse it on every run. Stood in for by an os.fstat that shows one owner higher for all but the directory.
	if route == 'macos':
		_macs_route(monkeypatch)
	out = str(tmp_path / 'out')
	os.mkdir(out)
	its, real_fstat = os.stat(out), os.fstat

	def fstat(fd):
		st = real_fstat(fd)
		if (st.st_dev, st.st_ino) == (its.st_dev, its.st_ino):
			return st
		return os.stat_result(tuple(st[:4]) + (st.st_uid + 1,) + tuple(st[5:10]))
	monkeypatch.setattr(os, 'fstat', fstat)
	answers, why = hpcp._ask_names(out, ['X', 'x'])
	assert not why and set(answers) == {'X', 'x'}, (answers, why)
	assert _names(out) == []  # the probe is gone


@posix_only
@pytest.mark.parametrize('route', ['native', 'macos'])
def test_991_a_directory_a_link_at_the_asked_name_leads_to_keeps_its_own_times(tmp_path, monkeypatch, route):
	# The directory asked is found by its path (docs/limitations.md, "The name probe by path"). 9.91's first round took its times
	# by path before it opened it and put them back on what it held, so a link put at its name in between gave the directory it
	# led to (in a root run any directory) the times of the one asked. They are taken from what is held. Forced at that open.
	if route == 'macos':
		_macs_route(monkeypatch)
	out, other = str(tmp_path / 'top' / 'out'), str(tmp_path / 'other')
	os.makedirs(out)
	os.mkdir(other)
	os.utime(out, (1500000000, 1500000000))
	os.utime(other, (1100000000, 1100000000))
	real_open, swapped = os.open, []

	def open_(path, flags, *args, **kwargs):
		if not swapped and str(path) == out:
			swapped.append(True)
			os.rename(out, out + '.away')
			os.symlink(other, out)
		return real_open(path, flags, *args, **kwargs)
	monkeypatch.setattr(os, 'open', open_)
	answers, why = hpcp._ask_names(out, ['X', 'x'])
	assert swapped, 'the directory asked was never opened'
	assert not why and set(answers) == {'X', 'x'}, (answers, why)
	assert os.stat(other).st_mtime == 1100000000, 'the directory a link at the name led to got the times of the one asked'
	assert _names(other) == []  # the probe made there is gone
