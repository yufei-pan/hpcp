'''hpcp 9.91: the name probe's undo trusted what it found at its own name (docs/lessons-full.md, 2026-11-13). Each test fails on
9.90 (b4dc027) unless it says it guards a case 9.90 already gets right (the 'nothing' cases are the controls: no swap). Every
swap is forced at the call it is timed by, in this process (no sleeps, no random sweep): another user who can write the
directory acts once. 9.93 made the probe's names directories and its undo rmdir alone: the tests that timed a swap by a
file's write or the undo's listing are timed by the probe's mkdir now; 9.93's own tests are in test_993.py.'''
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
def _swap(out, what, victim, moved, then=None):
	'''One swap while the probe of out runs, timed by the call it follows. dir-early: right after the probe directory is made,
	moved (a directory of the runner's own, in out) is renamed over it. link-early: the same moment, the probe renamed away and a
	link to victim put at its name. link-late: the same, right after the first name is made in the probe, so the calls after
	it (the next name, the undo) meet the link (9.91: right before the undo's listing). then(probe), if given, runs right after
	that first name too. The probe's name once it is known: state['probe'].'''
	real_mkdir = os.mkdir
	state = {'done': what == 'nothing', 'probe': None, 'named': False}

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
		elif state['probe'] is not None and not state['named']:
			state['named'] = True
			if then:
				then(state['probe'])
			if what == 'link-late':
				swap(state['probe'])
		return result

	os.mkdir = mkdir
	try:
		yield state
	finally:
		os.mkdir = real_mkdir


@posix_only
@pytest.mark.parametrize('what', ['nothing', 'dir-early', 'link-early', 'link-late'])
@pytest.mark.parametrize('route', ['native', 'macos'])
def test_991_a_probe_replaced_by_a_directory_or_a_link_costs_nobody_a_file(tmp_path, monkeypatch, route, what):
	# The name probe makes a hidden directory in the destination, makes the names in it, and (to 9.92) listed it and unlinked
	# everything listed. Another user who can write the destination (not sticky) renames what stands at the probe's name and puts
	# another there: a directory of the runner's own (or, in a root run, anyone's) between its mkdir and its open was taken for
	# the probe and emptied of its files (Linux's descriptor route, 9.89 and 9.90); a link put there (macOS's route, the probe by
	# path) sent the undo's listing and unlinks wherever it led. 9.91 refused a directory that was not empty when opened. 9.93
	# makes only directories and removes only with rmdir: a directory put there early is asked (what it holds decides the
	# answer: docs/limitations.md), and a link is refused by the open's own error (both routes here hold the probe by a
	# descriptor; by path a link is followed: test_993.py). 'native' is the platform's own route;
	# 'macos' is macOS's, on any platform. Nothing of what is put at the probe's name may be lost.
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
	assert state['done'], 'the swap never ran'
	assert _names(victim) == KEPT, 'the directory a link at the probe led to lost files: %r' % _names(victim)
	kept = os.path.join(out, state['probe']) if what == 'dir-early' else moved
	assert _names(kept) == KEPT, 'the directory put over the probe lost files: %r' % _names(kept)
	if what == 'nothing':
		assert not why and set(answers) == {'X', 'x'}, (answers, why)
		assert _names(out) == ['E'], _names(out)  # the probe is gone
	elif what == 'dir-early':
		# The directory put there is asked: its X is a name the probe did not make, so which made name holds it cannot be said.
		assert 'X' in why and 'cannot say which' in why['X'] and 'replaced by another' not in why['X'], (answers, why)
	elif what == 'link-early':
		# Not followed (the open's O_NOFOLLOW): nothing is asked in the directory the link leads to, and both names are refused.
		assert not answers and set(why) == {'X', 'x'}, (answers, why)


@posix_only
def test_991_a_rerun_after_a_replaced_probe_asks_the_directory(tmp_path):
	# The swapped-in directory put back where it came from, the next question is answered.
	out, victim, moved = str(tmp_path / 'out'), str(tmp_path / 'victim'), str(tmp_path / 'out' / 'E')
	os.makedirs(out)
	_fill(victim)
	_fill(moved)
	with _swap(out, 'dir-early', victim, moved) as state:
		answers, why = hpcp._ask_names(out, ['X', 'x'])
	assert 'X' in why and _names(os.path.join(out, state['probe'])) == KEPT, (answers, why)
	os.rename(os.path.join(out, state['probe']), moved)
	answers, why = hpcp._ask_names(out, ['X', 'x'])
	assert not why and set(answers) == {'X', 'x'}, (answers, why)
	assert _names(out) == ['E'] and _names(moved) == KEPT


@posix_only
@not_root
def test_991_a_directory_the_runner_may_write_and_not_read_is_still_asked(tmp_path, monkeypatch):
	# Guards what 9.90 does on macOS: there the directory is held by a read descriptor, which a directory the runner may write and
	# search and not read (0311) refuses. It is held by O_SEARCH where Python has one (macOS), and asked by path where it has
	# none (this route on Linux), where the undo's rmdir removes nothing but an empty directory. The pair is answered and the
	# probe is gone; the directory keeps its mode and its times (put back through the O_SEARCH descriptor on macOS, where a
	# failure would be silent).
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
		pytest.skip('no O_SEARCH here: asked by path, where only an empty directory can go')
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
def test_991_an_empty_directory_of_anothers_put_over_the_probe_keeps_what_its_owner_makes_in_it(tmp_path, monkeypatch, route):
	# "Empty when opened" is not "the one just made": an EMPTY directory renamed over the probe between its mkdir and its open
	# passed 9.91's first check, the names were made in it, and the undo removed what its owner made in it meanwhile, then the
	# directory. 9.93's undo is rmdir of the names it made and of the probe's name: what the owner makes in it stays, and so
	# does the directory while it holds anything (an empty one may go: whoever put it there could remove it). As root the
	# directory is uid 1234's; without root it is the runner's own, which changes nothing here (no owner is asked).
	if route == 'macos':
		_macs_route(monkeypatch)
	out = str(tmp_path / 'out')
	moved = os.path.join(out, 'E')
	os.makedirs(moved)
	os.chmod(out, 0o777)
	its = os.stat(moved)
	if os.geteuid() == 0:
		os.chown(moved, 1234, 1234)

	def owner_makes(probe):
		# Right after the probe's first name: its owner makes a file in it.
		with open(os.path.join(out, probe, 'late'), 'w') as fh:
			fh.write('its owner made this')
	with _swap(out, 'dir-early', None, moved, then=owner_makes) as state:
		answers, why = hpcp._ask_names(out, ['X', 'x'])
	assert state['named'], 'no name was made in the directory put over the probe'
	kept = os.path.join(out, state['probe'])
	assert os.path.isdir(kept) and os.stat(kept).st_ino == its.st_ino, 'the directory put over the probe is gone'
	assert _names(kept) == ['late'], _names(kept)


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
	# A name that is there already when the probe makes it, once two names are made, is looked at to learn which made name holds
	# it (_holder). Every version to 9.91's first build opened it with O_RDONLY alone: a FIFO put there by whoever may write the
	# probe was waited for, for ever, with SIGINT, SIGTERM and SIGHUP blocked, and a link was followed; 9.91 opened it with
	# O_NONBLOCK and O_NOFOLLOW. 9.93 opens nothing in the probe: it compares numbers by a stat that does not follow a link (a
	# link to the made P, followed, would give P's number), and such a name is one "it cannot say which" holds. Forced right
	# after the second name is made (P, Q and R: no filesystem folds them together, so two are made everywhere). What was put
	# there stays (the undo is rmdir: the probe is left holding it, for whoever put it there).
	if route == 'macos':
		_macs_route(monkeypatch)
	out = str(tmp_path / 'out')
	os.mkdir(out)
	real_open, real_mkdir = os.open, os.mkdir
	planted, opened = [], []

	def open_(path, flags, *args, **kwargs):
		if planted and os.path.basename(str(path)) == 'R':
			opened.append(flags)
			raise OSError(errno.EIO, 'this open could wait for a FIFO or follow a link', str(path))
		return real_open(path, flags, *args, **kwargs)

	def mkdir(path, *args, **kwargs):
		result = real_mkdir(path, *args, **kwargs)
		if os.path.basename(str(path)) == 'Q' and not planted:
			probe = [name for name in os.listdir(out) if name.startswith(PROBE)][0]
			planted.append(os.path.join(out, probe, 'R'))
			if what == 'fifo':
				os.mkfifo(planted[0])
			else:
				os.symlink('P', planted[0])
		return result
	monkeypatch.setattr(os, 'open', open_)
	monkeypatch.setattr(os, 'mkdir', mkdir)
	answers, why = hpcp._ask_names(out, ['P', 'Q', 'R'])
	assert planted, 'R was never put in the probe'
	assert not opened, 'the name that was there already was opened: %r' % [oct(flags) for flags in opened]
	assert 'R' in why and 'cannot say which' in why['R'], (answers, why)
	assert answers.get('P') == 'P' and answers.get('Q') == 'Q', (answers, why)
	probe = os.path.dirname(planted[0])
	assert _names(out) == [os.path.basename(probe)] and _names(probe) == ['R'], (_names(out), _names(probe))


@posix_only
@pytest.mark.parametrize('route', ['native', 'macos'])
def test_991_a_directory_a_link_at_the_asked_name_leads_to_keeps_its_own_times(tmp_path, monkeypatch, route):
	# The directory asked is found by its path (docs/limitations.md, the name probe's answer). 9.91's first round took its times
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
