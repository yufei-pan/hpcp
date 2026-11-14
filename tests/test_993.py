'''hpcp 9.93: the name probe makes only directories and removes only with rmdir (docs/lessons-full.md, 2026-11-13, "An undo
that cannot destroy needs no proof"). Each test fails on 9.92 (a22d721) unless it says it guards what 9.92 already gets
right. Every swap is forced before or after the call it is timed by, in this process: no sleeps, no random sweep.

A test that stands in for an os call adds its stand-in to the sets that held the real one (os.supports_dir_fd,
os.supports_follow_symlinks: _stand_in): hpcp asks them as it calls (os.utime in os.supports_dir_fd), and would go
another way than it goes for real.'''
import errno
import os
import types

import pytest

import hpcp

posix_only = pytest.mark.skipif(os.name != 'posix', reason='POSIX')

PROBE = '.hpcp-case-probe-'
# What the probe may never call: anything that removes, renames or writes a file, or makes something that is no directory.
NEVER = ('unlink', 'remove', 'rename', 'replace', 'removedirs', 'write', 'truncate', 'ftruncate', 'symlink', 'link', 'mkfifo',
	'mknod')
# Every call the probe makes (and the ones above, to catch them): what the counter numbers.
COUNTED = ('open', 'close', 'mkdir', 'rmdir', 'stat', 'lstat', 'fstat', 'utime', 'listdir', 'scandir', 'chmod') + NEVER


def _stand_in(monkeypatch, name, call):
	'''os.NAME replaced by call, which the sets of the calls that take dir_fd, a descriptor or follow_symlinks then hold where
	they held the real one.'''
	real = getattr(os, name)
	monkeypatch.setattr(os, name, call)
	for group in ('supports_dir_fd', 'supports_fd', 'supports_follow_symlinks'):
		if real in getattr(os, group):
			monkeypatch.setattr(os, group, getattr(os, group) | {call})


def _route(monkeypatch, route):
	'''fd: the platform's own descriptor route. path: by path throughout (Windows; forced). macos: no O_PATH and no /proc, a
	read descriptor holds the directory (macOS's route, on any platform).'''
	if route != 'path' and not hpcp._ASK_BY_FD:
		pytest.skip('no descriptor calls here: the probe goes by path')
	if route == 'path':
		monkeypatch.setattr(hpcp, '_ASK_BY_FD', False)
	elif route == 'macos':
		monkeypatch.setattr(hpcp, '_PATH_FDS', False)
		monkeypatch.setattr(hpcp, '_PROBE_BY_FD', False)
		monkeypatch.setattr(hpcp, '_PARENT_FLAGS', os.O_RDONLY | os.O_DIRECTORY | getattr(os, 'O_CLOEXEC', 0))


class _Counter:
	'''Numbers every os call in COUNTED while on, and runs before(k, name) and after(k, name) around call k. Hooks run with
	the counter off; they call the real functions (self.real).'''

	def __init__(self, monkeypatch):
		self.real = {name: getattr(os, name) for name in COUNTED if hasattr(os, name)}
		self.calls = []
		self.on = False
		self.before = self.after = None
		for name, real in self.real.items():
			_stand_in(monkeypatch, name, self._wrap(name, real))

	def _hook(self, hook, k, name):
		if hook:
			self.on = False
			try:
				hook(k, name)
			finally:
				self.on = True

	def _wrap(self, name, real):
		def call(*args, **kwargs):
			if not self.on:
				return real(*args, **kwargs)
			k = len(self.calls)
			self.calls.append((name, args, kwargs))
			self._hook(self.before, k, name)
			try:
				return real(*args, **kwargs)
			finally:
				self._hook(self.after, k, name)
		return call

	def ask(self, directory, names):
		self.on = True
		try:
			return hpcp._ask_names(directory, names)
		finally:
			self.on = False


def _folding(monkeypatch, lookup='number'):
	'''A directory that folds case, stood in for: names made, removed and looked up in a probe directory are lowercased (so
	'X' and 'x' are one directory, 'x'). lookup: 'number' (stat finds it, one number for both spellings, as most filesystems);
	'own' (each spelling stats with a number of its own: exFAT through FUSE); 'miss' (stat finds no name there: vfat with
	iocharset=utf8 looks up only the spelling it made).'''
	real = {name: getattr(os, name) for name in ('mkdir', 'rmdir', 'stat')}
	def in_probe(path, kwargs):
		if not isinstance(path, str):
			return None
		directory, name = os.path.split(path)
		if not directory and kwargs.get('dir_fd') is not None:
			directory = _fd_path(kwargs['dir_fd'])
		return name if os.path.basename(directory).startswith(PROBE) else None

	def lowered(name, call):
		def standing_in(path, *args, **kwargs):
			spelled = in_probe(path, kwargs)
			if spelled is None:
				return call(path, *args, **kwargs)
			if name == 'stat' and lookup == 'miss':
				raise FileNotFoundError(errno.ENOENT, 'No such file or directory', path)
			result = call(os.path.join(os.path.dirname(path), spelled.lower()), *args, **kwargs)
			if name == 'stat' and lookup == 'own':
				ino = (hash(spelled) & 0xffffffff) | 1
				result = os.stat_result((result.st_mode, ino, result.st_dev) + tuple(result[3:10]))
			return result
		return standing_in
	for name in ('mkdir', 'rmdir', 'stat'):
		_stand_in(monkeypatch, name, lowered(name, real[name]))


def _fd_path(fd):
	if os.path.isdir('/proc/self/fd'):
		return os.readlink('/proc/self/fd/%d' % fd)
	import fcntl
	return fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(1024)).split(b'\0')[0].decode('utf-8', 'surrogateescape')


def _owned(path):
	'''In a root run another user's, mode 0700 (anyone's files are what is lost there); the runner's own otherwise.'''
	if os.geteuid() == 0:
		for root, dirs, files in os.walk(path):
			for name in dirs + files:
				os.lchown(os.path.join(root, name), 1234, 1234)
		os.chown(path, 1234, 1234)
		os.chmod(path, 0o700)


def _fill(directory):
	'''A directory of another's holding files: X and x (the asked names, where the filesystem keeps them apart), keep, a
	link, and a subdirectory with a file. {relative path: content} of its files.'''
	os.makedirs(os.path.join(directory, 'sub'))
	held = {}
	for name in ('X', 'x', 'keep', os.path.join('sub', 'f')):
		try:
			with open(os.path.join(directory, name), 'x') as fh:
				fh.write('theirs: ' + name)
			held[name] = 'theirs: ' + name
		except FileExistsError:  # x on a filesystem that folds: it is X
			pass
	os.symlink('keep', os.path.join(directory, 'lnk'))
	_owned(directory)
	return held


def _held(directory):
	found = {}
	for root, dirs, files in os.walk(directory):
		for name in files:
			path = os.path.join(root, name)
			if not os.path.islink(path):
				with open(path) as fh:
					found[os.path.relpath(path, directory)] = fh.read()
	return found


@posix_only
@pytest.mark.parametrize('names', [['X', 'x'], ['a', 'A', 'K', 'k', 'K']], ids=['pair', 'group'])
@pytest.mark.parametrize('route', ['fd', 'path', 'macos'])
def test_993_the_probe_makes_directories_and_removes_with_rmdir_and_nothing_else(tmp_path, monkeypatch, route, names):
	# 9.83 to 9.92 made each name as a file (O_CREAT | O_EXCL), wrote its place into it, and the undo unlinked: on every route
	# the probe could remove a file, and only the checks around the undo (9.91) said which. 9.93 makes each name with mkdir and
	# undoes with rmdir: no call that can remove, rename or write a file, and no open that can make or write one. The group, in
	# a directory that folds (stood in for), also finds which made name holds a spelling (a stat of each).
	_route(monkeypatch, route)
	if len(names) > 2:
		_folding(monkeypatch)
	counter = _Counter(monkeypatch)
	out = str(tmp_path / 'out')
	os.mkdir(out)
	answers, why = counter.ask(out, names)
	assert not why and set(answers) == set(names), (answers, why)
	called = [name for name, args, kwargs in counter.calls]
	assert not set(called) & set(NEVER), called
	writes = os.O_CREAT | os.O_WRONLY | os.O_RDWR | os.O_TRUNC | os.O_APPEND
	opens = [args[1] for name, args, kwargs in counter.calls if name == 'open']
	assert not [flags for flags in opens if flags & writes], [oct(flags) for flags in opens]
	# One rmdir for each name made (one for each answer) and one for the probe.
	assert 'mkdir' in called and called.count('rmdir') == len(set(answers.values())) + 1, called
	assert os.listdir(out) == []


@posix_only
@pytest.mark.parametrize('what', ['their-directory', 'a-link-to-it', 'an-empty-directory-then-a-file'])
@pytest.mark.parametrize('route', ['fd', 'path', 'macos'])
def test_993_whatever_stands_at_the_probes_name_keeps_its_files(tmp_path, monkeypatch, route, what):
	# Another user who may write the directory asked (not sticky) puts something at the probe's name, before any one call of the
	# probe (the probe's own directory, once made, renamed aside first): a directory of his holding files, the asked names among
	# them; a link to such a directory elsewhere (by path, the Windows route, the probe's calls go where it leads); an empty
	# directory he makes a file in right after that call. 9.92 listed what stood at the name and unlinked it (by path, through
	# the link; on any route, a directory put there after its check). 9.93's undo is rmdir alone: every file stays, wherever
	# it is, and so does a directory while it holds anything. Each call k of an honest probe, and one more. The fd and macos
	# routes guard what 9.92 gets right through its checks (empty when opened, the owner of its first file); by path it fails.
	_route(monkeypatch, route)
	monkeypatch.setattr(hpcp.uuid, 'uuid4', lambda: types.SimpleNamespace(hex='0' * 32))
	probe_name = '%s%d-%s' % (PROBE, os.getpid(), '0' * 32)
	counter = _Counter(monkeypatch)
	real = counter.real
	honest = str(tmp_path / 'honest')
	os.mkdir(honest)
	assert counter.ask(honest, ['X', 'x'])[1] == {}
	total = len(counter.calls)
	assert total >= 5, counter.calls
	planted = 0
	for k in range(total + 1):
		top = str(tmp_path / ('k%d' % k))
		out, victim = os.path.join(top, 'out'), os.path.join(top, 'victim')
		os.makedirs(out)
		os.chmod(out, 0o777)
		with open(os.path.join(out, 'mine'), 'w') as fh:
			fh.write('the runner\'s')
		theirs = os.path.join(out, 'T')
		held = _fill(victim if what == 'a-link-to-it' else theirs) if what != 'an-empty-directory-then-a-file' else {}
		if what == 'an-empty-directory-then-a-file':
			os.mkdir(theirs)
			_owned(theirs)
		at = os.path.join(out, probe_name)
		state = {'planted': False, 'late': False}

		def before(i, name):
			if i != k or state['planted']:
				return
			state['planted'] = True
			if os.path.lexists(at):
				real['rename'](at, at + '.away')
			if what == 'a-link-to-it':
				os.symlink(victim, at)
			else:
				real['rename'](theirs, at)

		def after(i, name):
			if what == 'an-empty-directory-then-a-file' and i == k and state['planted'] and os.path.isdir(at) and not os.path.islink(at):
				with open(os.path.join(at, 'late'), 'w') as fh:
					fh.write('theirs: late')
				state['late'] = True
		counter.calls = []
		counter.before, counter.after = before, after
		try:
			counter.ask(out, ['X', 'x'])
		finally:
			counter.before = counter.after = None
		where = victim if what == 'a-link-to-it' else at if state['planted'] else theirs
		if what == 'an-empty-directory-then-a-file':
			if state['late']:
				assert _held(at) == {'late': 'theirs: late'}, (k, _held(at))
		else:
			assert _held(where) == held, (k, counter.calls[:k + 1], _held(where))
			assert os.path.islink(os.path.join(where, 'lnk')), k
		with open(os.path.join(out, 'mine')) as fh:
			assert fh.read() == 'the runner\'s', k
		planted += state['planted']
	assert planted >= total, (planted, total)


@pytest.mark.parametrize('lookup', ['number', 'own', 'miss'], ids=['by-number', 'a-number-for-each-spelling', 'a-lookup-that-misses'])
def test_993_a_group_in_a_directory_that_folds_finds_each_holder(tmp_path, monkeypatch, lookup):
	# Five spellings, two names, in a directory that folds (stood in for: mkdir, rmdir and stat lowercase what is made in a
	# probe). Which made name holds a spelling that is there already is read from the numbers (9.92 read it from the file the
	# name held): equal numbers prove one directory, so no further probe is made. Where each spelling stats with a number of its
	# own (exFAT through FUSE) or the lookup misses (vfat with iocharset=utf8), numbers prove nothing, and each made name is
	# asked alone, as 9.92 did: the same answers, and nothing left.
	_folding(monkeypatch, lookup)
	asked = []
	ask = hpcp._ask_names
	monkeypatch.setattr(hpcp, '_ask_names', lambda directory, names: asked.append(list(names)) or ask(directory, names))
	names = ['a', 'A', 'K', 'k', 'K']
	answers, why = hpcp._ask_names(str(tmp_path), names)
	assert (answers, why) == ({'a': 'a', 'A': 'a', 'K': 'K', 'k': 'K', 'K': 'K'}, {}), (answers, why)
	if lookup == 'number':
		assert asked == [names], asked
	else:
		assert len(asked) > 1 and asked[0] == names, asked
	assert os.listdir(str(tmp_path)) == []


@pytest.mark.parametrize('which', ['first', 'second'])
@pytest.mark.parametrize('number,fix', [(errno.ENOSPC, 'free some space there'), (errno.EDQUOT, 'free some of the quota')], ids=['nospace', 'quota'])
def test_993_a_name_that_cannot_be_made_for_space_or_quota_says_the_fix_and_leaves_nothing(tmp_path, monkeypatch, number, fix, which):
	# A full disk or a quota at its limit refuses the mkdir of a name in the probe: both names get the line that names the fix,
	# and the probe goes with what it made. (9.91 and 9.92: a quota error at a probe file's close ended the run with exit 70, or
	# left the probe and that file.)
	real, made = os.mkdir, []

	def mkdir(path, *args, **kwargs):
		if isinstance(path, str) and not os.path.basename(path).startswith(PROBE):
			made.append(path)
			if len(made) == (1 if which == 'first' else 2):
				raise OSError(number, os.strerror(number), path)
		return real(path, *args, **kwargs)
	_stand_in(monkeypatch, 'mkdir', mkdir)
	directory = str(tmp_path)
	answers, why = hpcp._ask_names(directory, ['X', 'x'])
	line = 'nothing can be made in %s to ask it (%s): %s and run again' % (directory, os.strerror(number), fix)
	assert made and (answers, why) == ({}, {'X': line, 'x': line}), (made, answers, why)
	assert os.listdir(directory) == []
