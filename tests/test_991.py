'''hpcp 9.91: the name probe's undo trusted what it found at its own name (docs/lessons-full.md, 2026-11-13). Each test fails on
9.90 (b4dc027) unless it says it guards a case 9.90 already gets right. Every swap is forced at the call it is timed by, in this
process (no sleeps, no random sweep): another user who can write the directory (not sticky) acts once.'''
import contextlib
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
	monkeypatch.setattr(hpcp, '_PATH_FDS', False)
	monkeypatch.setattr(hpcp, '_PROBE_BY_FD', False)
	monkeypatch.setattr(hpcp, '_PARENT_FLAGS', os.O_RDONLY | os.O_DIRECTORY | getattr(os, 'O_CLOEXEC', 0))


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
def test_991_a_directory_the_runner_may_write_and_not_read_is_still_asked_by_path(tmp_path, monkeypatch):
	# Guards what 9.90 does on macOS: there the directory is held by a read descriptor, which a directory the runner may write and
	# search and not read (0311) refuses, so it is asked by path as before (docs/limitations.md, "The name probe by path"). The
	# pair is answered and the probe is gone; the directory keeps its mode.
	_macs_route(monkeypatch)
	directory = str(tmp_path / 'd')
	os.mkdir(directory)
	os.chmod(directory, 0o311)
	try:
		answers, why = hpcp._ask_names(directory, ['X', 'x'])
		now = stat.S_IMODE(os.stat(directory).st_mode)
	finally:
		os.chmod(directory, 0o755)
	assert not why and set(answers) == {'X', 'x'}, (answers, why)
	assert now == 0o311, oct(now)
	assert _names(directory) == []  # the probe's own directory is gone
