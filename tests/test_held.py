'''9.87 group 5: cp writes a file hpcp holds through the descriptor it inherits (H: _held_route, /proc/self/fd/N on Linux),
a first copy made by hpcp and written so (L1), and the beside route's new file made so (L2). multiCMD 1.59's pass_fds through
the embedded copy (E) is in test_embedded_multicmd.py.'''
import contextlib
import errno
import io
import itertools
import json
import os
import shutil
import stat
import struct
import subprocess
import sys
import tempfile

import pytest

import hpcp

HPCP = hpcp.__file__
posix_only = pytest.mark.skipif(os.name != 'posix', reason='POSIX')
linux_only = pytest.mark.skipif(not sys.platform.startswith('linux'), reason='the held route on Linux: /proc/self/fd')
root_only = pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() != 0, reason='needs root')
not_root = pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() == 0, reason='root writes every directory')

_RUNS = itertools.count()
_HELD_PREFIX = '/proc/self/fd/'


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
def fresh(monkeypatch, reset_hpcp_globals):
	'''Each test asks cp anew (its --version, the held route) from the environment it sets, and leaves nothing behind.'''
	monkeypatch.setattr(hpcp, '_binPaths', {})
	monkeypatch.setattr(hpcp, '_badOverrides', {})
	monkeypatch.setattr(hpcp, '_SPARSE', [None, False])
	monkeypatch.setattr(hpcp, '_HELD', [None, None])
	monkeypatch.setattr(hpcp, '_HELD_GIVEN', None)
	hpcp.check_path.cache_clear()
	yield hpcp
	hpcp.check_path.cache_clear()


def _copy_file(monkeypatch, src, dest):
	'''copy_file of one source to one landing, its folder named by the operator; (size, ...) as copy_file returns.'''
	monkeypatch.setattr(hpcp, 'NAMED_DESTS', frozenset([os.path.dirname(os.path.abspath(str(dest)))]))
	monkeypatch.setattr(hpcp, 'RUN_ID', 'held-%d' % next(_RUNS))
	with contextlib.redirect_stdout(io.StringIO()):
		return hpcp.copy_file(str(src), [str(dest)])


def _named(target):
	'''What cp's destination argument leads to: the held file's name for /proc/self/fd/N, else itself. Asked while cp runs.'''
	return os.readlink(target) if target.startswith(_HELD_PREFIX) else target


def _record_cp(monkeypatch, after=None):
	'''Each cp the copy runs, as (its destination argument, what that leads to, pass_fds); after(command, named) runs once cp
	has.'''
	seen = []
	real = hpcp.run_command_in_multicmd_with_path_check
	def cp(command, *args, **kwargs):
		named = _named(command[-1])
		seen.append((command[-1], named, tuple(kwargs.get('pass_fds', ()))))
		result = real(command, *args, **kwargs)
		if after is not None:
			after(command, named)
		return result
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check', cp)
	return seen


def _probe_dir(tmp_path, monkeypatch):
	'''A temporary directory of the test's own for the probe's files, to see that none is left.'''
	path = str(tmp_path / 'probe-tmp')
	os.makedirs(path)
	monkeypatch.setattr(tempfile, 'tempdir', path)
	return path


def _held_or_skip():
	'''Skip where cp cannot write a held file. Before 9.87 (no _held_route) the test runs, to fail as 9.86 does.'''
	route = getattr(hpcp, '_held_route', None)
	if route is not None and route() is None:
		pytest.skip('cp cannot write a held file here')


def _gnu_cp_or_skip():
	if hpcp.cp_supports_sparse() is not True:
		pytest.skip('cp here is not GNU cp')


# ---- the source: cp is never given another process's descriptors

def test_hpcp_names_no_other_process_s_descriptors():
	# 9.86 gave cp /proc/PID/fd/N, which a /proc mounted hidepid, a worker /proc refuses, and an ancestor PID namespace's /proc
	# (where PID is another process) kept from cp; cp's own /proc/self/fd/N has none of these.
	with open(hpcp.__file__, encoding='utf-8') as handle:
		source = handle.read()
	assert '/proc/%d/fd' not in source and '/proc/%s/fd' not in source and "/proc/' + str(" not in source


# ---- H: the probe

@linux_only
def test_the_probe_answers_c_with_gnu_cp_and_leaves_nothing(tmp_path, fresh, monkeypatch):
	_gnu_cp_or_skip()
	probes = _probe_dir(tmp_path, monkeypatch)
	assert hpcp._held_route() == 'c' and os.listdir(probes) == []


_GNU_VERSION = 'if [ "$1" = --version ]; then echo "cp (GNU coreutils) 9.1"; exit 0; fi\n'
# A cp that opens what it is given without truncating it (as a dup does on macOS) and writes the source from its start.
_NO_TRUNCATE = '''for last; do :; done
eval source=\\${$(($# - 1))}
cat "$source" 1<> "$last"
'''


def _fake_cp(directory, monkeypatch, body):
	path = os.path.join(str(directory), 'fakecp')
	with open(path, 'w') as handle:
		handle.write('#!/bin/sh\n' + _GNU_VERSION + body)
	os.chmod(path, 0o755)
	monkeypatch.setenv('CP_PATH', path)
	return path


@linux_only
def test_a_cp_that_leaves_an_old_tail_gets_the_held_file_emptied_first(exec_dir, tmp_path, fresh, monkeypatch):
	# The probe's held file then holds cb: hpcp empties the held file right before each cp, so a rewrite from a longer old
	# copy to a shorter source is exact, and the inode stays.
	_fake_cp(exec_dir, monkeypatch, _NO_TRUNCATE)
	probes = _probe_dir(tmp_path, monkeypatch)
	assert hpcp._held_route() == 'cb' and os.listdir(probes) == []
	src = write(tmp_path / 'a/x', 'new')
	landing = write(tmp_path / 'out/x', 'a longer old copy')
	inode = os.stat(landing).st_ino
	_copy_file(monkeypatch, src, landing)
	assert read(landing) == 'new' and os.stat(landing).st_ino == inode and not hpcp.ERRORS, hpcp.ERRORS
	_copy_file(monkeypatch, src, tmp_path / 'out/y')  # a first copy: the new file is empty already
	assert read(tmp_path / 'out/y') == 'new' and not hpcp.ERRORS, hpcp.ERRORS


@linux_only
@pytest.mark.parametrize('body', ['exit 1\n', 'exit 0\n'], ids=['fails', 'writes-elsewhere'])
def test_no_route_for_a_cp_that_does_not_write_the_held_file(exec_dir, tmp_path, fresh, monkeypatch, body):
	# cp's own answer: kept for the process (no second probe), and cp gets the name.
	_fake_cp(exec_dir, monkeypatch, body)
	probes = _probe_dir(tmp_path, monkeypatch)
	assert hpcp._held_route() is None and hpcp._HELD[0] == os.getpid() and os.listdir(probes) == []


@linux_only
def test_no_route_without_a_temporary_file(tmp_path, fresh, monkeypatch):
	# tempfile finds no directory to make the probe's files in (TMPDIR and its list unwritable): cp gets the name.
	_gnu_cp_or_skip()
	monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path / 'no-such-directory'))
	assert hpcp._held_route() is None and hpcp._HELD[0] == os.getpid()


@linux_only
def test_a_probe_that_times_out_keeps_no_answer(exec_dir, tmp_path, fresh, monkeypatch):
	# A cp that hangs past the probe's timeout is no answer: the next copy asks again; the probe's files go all the same.
	_fake_cp(exec_dir, monkeypatch, 'exec sleep 30\n')
	monkeypatch.setattr(hpcp, '_CP_ASK_TIMEOUT', 2)
	probes = _probe_dir(tmp_path, monkeypatch)
	assert hpcp._held_route() is None and hpcp._HELD[0] is None and os.listdir(probes) == []


# ---- the routes cp is given

@linux_only
@pytest.mark.parametrize('case', ['first', 'rewrite', 'beside'])
def test_cp_writes_what_hpcp_holds_through_its_own_descriptor(tmp_path, fresh, monkeypatch, case):
	# A first copy (L1), a rewrite in place (H) and the beside route's new file (L2): cp gets /proc/self/fd/N of the
	# descriptor it inherits, which leads to the landing, or to the new file beside it.
	_held_or_skip()
	src = write(tmp_path / 'a/x', 'new content')
	landing = str(tmp_path / 'out/x')
	if case != 'first':
		write(landing, 'old')
	monkeypatch.setattr(hpcp, 'WRITE_BESIDE', case == 'beside')
	seen = _record_cp(monkeypatch)
	_copy_file(monkeypatch, src, landing)
	assert read(landing) == 'new content' and not hpcp.ERRORS and len(seen) == 1, (seen, hpcp.ERRORS)
	target, named, fds = seen[0]
	assert target == _HELD_PREFIX + str(fds[0]), seen
	if case == 'beside':
		assert os.path.dirname(named) == str(tmp_path / 'out') and os.path.basename(named).startswith('.hpcp_tmp_'), seen
	else:
		assert named == landing, seen


@posix_only
@pytest.mark.parametrize('case', ['first', 'rewrite', 'beside'])
def test_with_no_held_route_cp_gets_the_name(tmp_path, fresh, monkeypatch, case):
	# BusyBox, uutils' cp, the BSDs, a cp that failed the probe: the landing, or the new file beside it, by name, as 9.84.
	monkeypatch.setattr(hpcp, '_held_route', lambda: None)
	src = write(tmp_path / 'a/x', 'new content')
	landing = str(tmp_path / 'out/x')
	if case != 'first':
		write(landing, 'old')
	monkeypatch.setattr(hpcp, 'WRITE_BESIDE', case == 'beside')
	seen = _record_cp(monkeypatch)
	_copy_file(monkeypatch, src, landing)
	assert read(landing) == 'new content' and not hpcp.ERRORS and len(seen) == 1, (seen, hpcp.ERRORS)
	target, _, fds = seen[0]
	assert fds == () and not target.startswith('/proc/')
	if case == 'beside':
		assert os.path.basename(target).startswith('.hpcp_tmp_'), seen
	else:
		assert target == landing, seen


@posix_only
def test_a_name_that_appears_before_a_first_copy_is_looked_at_again(tmp_path, fresh, monkeypatch):
	# Between the look and hpcp's create a file appears at the landing: the create fails (O_EXCL), the landing is looked at
	# again, and the copy takes the route that fits what is there (a rewrite in place).
	_held_or_skip()
	src = write(tmp_path / 'a/x', 'new content')
	landing = str(tmp_path / 'out/x')
	os.makedirs(str(tmp_path / 'out'))
	real = hpcp._claim_device
	def appear(target, st, named, src_path):
		if not os.path.lexists(landing):
			write(landing, 'appeared')
		return real(target, st, named, src_path)
	monkeypatch.setattr(hpcp, '_claim_device', appear)
	_copy_file(monkeypatch, src, landing)
	assert read(landing) == 'new content' and not hpcp.ERRORS, hpcp.ERRORS


@posix_only
def test_a_first_copy_whose_name_keeps_being_taken_fails_after_three_looks(tmp_path, fresh, monkeypatch):
	_held_or_skip()
	src = write(tmp_path / 'a/x', 'new content')
	landing = str(tmp_path / 'out/x')
	os.makedirs(str(tmp_path / 'out'))
	creates = []
	real = os.open
	def taken(path, flags, *args, **kwargs):
		if path == landing and flags & os.O_EXCL:
			creates.append(path)
			raise FileExistsError(errno.EEXIST, 'File exists', path)
		return real(path, flags, *args, **kwargs)
	monkeypatch.setattr(hpcp.os, 'open', taken)
	_copy_file(monkeypatch, src, landing)
	monkeypatch.setattr(hpcp.os, 'open', real)
	# 3 looks an attempt (copy_file's own passes may try again)
	assert creates and len(creates) % 3 == 0 and not os.path.lexists(landing), creates
	assert any('it kept changing while it was copied' in error for error in hpcp.ERRORS), hpcp.ERRORS


# ---- names swapped mid-copy by a user who can write the destination (swaplab.py's cases, root)

@root_only
@linux_only
@pytest.mark.parametrize('case', ['gone-then-link', 'link-at-a-first-copy', 'hard-link-at-the-beside-temp'])
def test_a_name_swapped_mid_copy_never_leads_the_copy_elsewhere(tmp_path, fresh, monkeypatch, case):
	# The destination directory is another user's (uid 1000), who swaps names while root copies into it:
	# - gone-then-link (9.86's A:gone): the landing removed after the look, a link to root's file planted before cp opens:
	#   9.86 gave cp the name, and the victim got the content. Now hpcp makes the file (O_EXCL), so the link cannot be put.
	# - link-at-a-first-copy: hpcp's new file renamed away and a link planted at its name before cp opens: cp writes the
	#   file hpcp holds; 9.86 gave cp the name, and the victim got the content.
	# - hard-link-at-the-beside-temp (9.86's G:hard, -wb): the new file beside the landing swapped for a second name of the
	#   victim once cp has written it: 9.86 removed the victim's xattr through it and renamed it over the landing. Now what
	#   follows goes through the file hpcp holds, and the rename finds the name leads elsewhere: 190, the old copy kept.
	_held_or_skip()
	victim = write(tmp_path / 'victim/secret', 'victim', 0o600 if case != 'hard-link-at-the-beside-temp' else 0o666)
	xattrs = False
	try:
		os.setxattr(victim, 'user.keep', b'1')
		xattrs = True
	except OSError:
		pass
	before = os.lstat(victim)
	src = write(tmp_path / 'src/f', 'NEW-CONTENT', 0o600)
	out = str(tmp_path / 'out')
	landing = os.path.join(out, 'f')
	os.makedirs(out)
	if case != 'link-at-a-first-copy':
		write(landing, 'old')
		os.chown(landing, 1000, 1000)
	os.chown(out, 1000, 1000)
	monkeypatch.setattr(hpcp, 'WRITE_BESIDE', case == 'hard-link-at-the-beside-temp')
	if case == 'gone-then-link':
		real = hpcp._claim_device
		def removed(target, st, named, src_path):
			if os.path.abspath(target) == landing and not os.path.lexists(landing + '.away'):
				os.rename(landing, landing + '.away')
			return real(target, st, named, src_path)
		monkeypatch.setattr(hpcp, '_claim_device', removed)
	real_cp = hpcp.run_command_in_multicmd_with_path_check
	def cp(command, *args, **kwargs):
		named = _named(command[-1])
		if case == 'gone-then-link':
			with contextlib.suppress(FileExistsError):
				os.symlink(victim, landing)
		elif case == 'link-at-a-first-copy':
			if os.path.lexists(landing):
				os.rename(landing, landing + '.away')
			os.symlink(victim, landing)
		result = real_cp(command, *args, **kwargs)
		if case == 'hard-link-at-the-beside-temp' and '.hpcp_tmp_' in named:
			os.rename(named, named + '.away')
			os.link(victim, named)
		return result
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check', cp)
	_copy_file(monkeypatch, src, landing)
	after = os.lstat(victim)
	assert read(victim) == 'victim' and (after.st_mode, after.st_uid, after.st_gid) == (before.st_mode, before.st_uid, before.st_gid)
	assert not xattrs or os.getxattr(victim, 'user.keep') == b'1'
	if case == 'gone-then-link':
		assert not os.path.islink(landing) and read(landing) == 'NEW-CONTENT' and not hpcp.ERRORS, hpcp.ERRORS
	elif case == 'link-at-a-first-copy':
		assert read(landing + '.away') == 'NEW-CONTENT' and not hpcp.ERRORS, hpcp.ERRORS
	else:
		assert read(landing) == 'old' and os.lstat(landing).st_ino != after.st_ino
		assert any('something else was put at its name' in error for error in hpcp.ERRORS), hpcp.ERRORS


# ---- L1: a first copy ends as cp by name makes it

_MODES = (0o644, 0o600, 0o755, 0o4755, 0o2755, 0o1777, 0o444, 0o000)


def _default_acl(directory):
	'''A default ACL on directory granting uid 1000 rwx (user::rwx user:1000:rwx group::r-x mask::rwx other::r-x), as
	setfacl -d writes it; False where the filesystem takes none.'''
	entries = [(0x01, 7, 0xffffffff), (0x02, 7, 1000), (0x04, 5, 0xffffffff), (0x10, 7, 0xffffffff), (0x20, 5, 0xffffffff)]
	value = struct.pack('<I', 2) + b''.join(struct.pack('<HHI', *entry) for entry in entries)
	try:
		os.setxattr(directory, 'system.posix_acl_default', value)
	except OSError as e:
		if e.errno in (errno.ENOTSUP, errno.EOPNOTSUPP):
			return False
		raise
	return True


def _outcome(path):
	try:
		st = os.lstat(path)
	except FileNotFoundError:
		return None
	acl = None
	with contextlib.suppress(OSError, AttributeError):
		acl = os.getxattr(path, 'system.posix_acl_access', follow_symlinks=False)
	return oct(stat.S_IMODE(st.st_mode)), st.st_uid, st.st_gid, acl, read(path)


@posix_only
@pytest.mark.parametrize('acl', [False, True], ids=['no-acl', 'default-acl'])
@pytest.mark.parametrize('umask', [0o022, 0o077, 0o002], ids=['022', '077', '002'])
@pytest.mark.parametrize('content_only', [False, True], ids=['plain', 'content-only'])
def test_a_first_copy_ends_as_cp_by_name_makes_it(tmp_path, fresh, monkeypatch, content_only, umask, acl):
	# 9.86 gave a first copy's name to cp (-a, or -f under -co): the reference, run here with the held route off. 9.87
	# makes the file (0600 for cp -a, which sets the source's mode through its descriptor; under -co the mode cp -f gives a
	# file it makes) and cp writes it through the held descriptor: the same mode, owner, group and ACL in every cell.
	_held_or_skip()
	monkeypatch.setattr(hpcp, 'CONTENT_ONLY', content_only)
	sources = []
	for mode in _MODES:
		path = write(tmp_path / 'src' / ('m%04o' % mode), 'mode %o' % mode)
		if os.geteuid() == 0:
			os.chown(path, 1000, 1000)  # before the mode: a chown clears set-user-ID
		os.chmod(path, mode)
		sources.append(path)
	for name in ('held', 'named'):
		os.makedirs(str(tmp_path / name))
		if acl and not _default_acl(str(tmp_path / name)):
			pytest.skip('no POSIX ACLs here')
	seen = _record_cp(monkeypatch)
	old = os.umask(umask)
	try:
		for path in sources:
			_copy_file(monkeypatch, path, tmp_path / 'held' / os.path.basename(path))
		monkeypatch.setattr(hpcp, '_held_route', lambda: None)
		for path in sources:
			_copy_file(monkeypatch, path, tmp_path / 'named' / os.path.basename(path))
	finally:
		os.umask(old)
	if not (acl and not content_only and os.geteuid() == 0):
		# (a root run's plain first copy of another user's file into a default-ACL directory is cp's by name, as in 9.86)
		assert any(target.startswith(_HELD_PREFIX) for target, _, _ in seen), seen
	cells = {os.path.basename(path): (_outcome(str(tmp_path / 'held' / os.path.basename(path))),
		_outcome(str(tmp_path / 'named' / os.path.basename(path)))) for path in sources}
	assert all(held == named for held, named in cells.values()), {name: cell for name, cell in cells.items() if cell[0] != cell[1]}


@linux_only
@root_only
def test_a_root_first_copy_is_given_its_owner_only_once_written(tmp_path, fresh, monkeypatch):
	# cp -a gives a file its owner once it has written it (9.86 by name): given first, XFS charges the data to the owner's
	# quota while root writes it, and a user at the hard limit got Disk quota exceeded where 9.86 completed (shown on a real
	# XFS with a 1 MiB limit). A file with a default ACL is still given its owner first (cp -a would narrow its entries).
	_held_or_skip()
	src = write(tmp_path / 'src/f', 'owned by someone else')
	os.chown(src, 1234, 1234)
	owners = []
	real = hpcp.run_command_in_multicmd_with_path_check
	def cp(command, *args, **kwargs):
		owners.append(os.stat(command[-1]).st_uid)
		return real(command, *args, **kwargs)
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check', cp)
	_copy_file(monkeypatch, src, tmp_path / 'out/f')
	assert owners == [0] and os.stat(str(tmp_path / 'out/f')).st_uid == 1234, owners


@linux_only
@root_only
def test_a_root_first_copy_into_a_default_acl_directory_is_written_as_root_too(tmp_path, fresh, monkeypatch):
	# There hpcp gave the new file its owner first (cp -a narrows the ACL of a file it did not make before it gives it another
	# owner), so root's write was charged to that owner's quota, as review A's finding 4 showed without an ACL. cp makes the
	# file by name there, as in 9.86: written as root, the owner given at the end, the directory's entries kept as cp keeps them.
	_held_or_skip()
	out = tmp_path / 'out'
	out.mkdir()
	if not _default_acl(str(out)):
		pytest.skip('no POSIX ACLs here')
	src = write(tmp_path / 'src/f', 'owned by someone else')
	os.chown(src, 1234, 1234)
	seen = []
	real = hpcp.run_command_in_multicmd_with_path_check
	def cp(command, *args, **kwargs):
		seen.append((command[-1], os.stat(command[-1]).st_uid if os.path.lexists(command[-1]) else None))
		return real(command, *args, **kwargs)
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check', cp)
	_copy_file(monkeypatch, src, out / 'f')
	assert seen == [(str(out / 'f'), None)] and os.stat(str(out / 'f')).st_uid == 1234, seen


@linux_only
@root_only
@pytest.mark.parametrize('owner, group, by_name', [(0, 1234, True), (1234, 0, True), (1234, 1234, True), (0, 0, False)],
	ids=['root-in-another-group', 'another-user-in-root-s-group', 'another-user-and-group', 'root-own'])
def test_a_root_first_copy_of_another_group_s_file_into_a_default_acl_directory_is_written_as_root_too(
		tmp_path, fresh, monkeypatch, owner, group, by_name):
	# The group case of the test above: giving a root:G file its group first charges root's write to G's quota (XFS enforces a
	# group's quota on a root write as an owner's), and on VM 901 a root:G file with G at its hard limit failed with Disk quota
	# exceeded (exit 190) where 9.86 completed. cp makes the file by name unless it is root's own (root:root), which keeps
	# the held route: nothing to give it, so nothing to charge differently.
	_held_or_skip()
	out = tmp_path / 'out'
	out.mkdir()
	if not _default_acl(str(out)):
		pytest.skip('no POSIX ACLs here')
	src = write(tmp_path / 'src/f', 'owned by someone else')
	os.chown(src, owner, group)
	seen = []
	real = hpcp.run_command_in_multicmd_with_path_check
	def cp(command, *args, **kwargs):
		seen.append(command[-1])
		return real(command, *args, **kwargs)
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check', cp)
	_copy_file(monkeypatch, src, out / 'f')
	made = os.stat(str(out / 'f'))
	assert (made.st_uid, made.st_gid) == (owner, group), (made.st_uid, made.st_gid)
	if by_name:
		assert seen == [str(out / 'f')], seen
	else:
		assert len(seen) == 1 and seen[0].startswith(_HELD_PREFIX), seen


@linux_only
@root_only
def test_a_stop_as_a_by_name_first_copy_s_made_file_is_closed_is_a_stop(tmp_path, fresh, monkeypatch):
	# Where cp makes the file by name (a default-ACL directory, another user's file), hpcp drops the file it made and closes it.
	# A stop whose handler raised as that close returned had the failure handler close it again: EBADF took the stop's place,
	# a false Copy failed, and the worker went on to its task's end (4 % of stops landing in _cp_made, 3.6 and 3.14).
	import signal
	_held_or_skip()
	out = tmp_path / 'out'
	out.mkdir()
	if not _default_acl(str(out)):
		pytest.skip('no POSIX ACLs here')
	src = write(tmp_path / 'src/f', 'another user\'s')
	os.chown(src, 1234, 1234)
	target = str(out / 'f')
	_as_the_main_process(monkeypatch)
	made = []
	real_open, real_close = os.open, os.close
	def opened(path, flags, *args, **kwargs):
		fd = real_open(path, flags, *args, **kwargs)
		if path == target:
			made.append(fd)
		return fd
	def closed(fd):
		real_close(fd)
		if made and fd == made[0]:
			del made[:]
			hpcp._stop(signal.SIGINT, None)
	monkeypatch.setattr(os, 'open', opened)
	monkeypatch.setattr(os, 'close', closed)
	with pytest.raises(KeyboardInterrupt):
		hpcp._cp_made(['cp', '-af'], src, target, None, None)
	monkeypatch.setattr(os, 'open', real_open)
	monkeypatch.setattr(os, 'close', real_close)
	assert not os.path.lexists(target)


@linux_only
@root_only
@pytest.mark.parametrize('content_only', [False, True], ids=['plain', 'content-only'])
def test_a_first_copy_s_selinux_label_is_cp_by_name_s(tmp_path, fresh, monkeypatch, content_only):
	# Plain: hpcp gives the new, empty file the source's label before cp writes it (cp -a sets an existing file's label by
	# name, which /proc/self/fd/N is not), so it ends with the source's label, as cp -a by name gives; -co: the label the
	# directory gives, as cp -f by name.
	try:
		with open('/sys/fs/selinux/enforce') as handle:
			if handle.read().strip() != '1':
				pytest.skip('SELinux is not enforcing')
	except OSError:
		pytest.skip('no SELinux')
	_held_or_skip()
	monkeypatch.setattr(hpcp, 'CONTENT_ONLY', content_only)
	src = write(tmp_path / 'src/f', 'labelled')
	label = b'system_u:object_r:user_home_t:s0\x00'
	os.setxattr(src, 'security.selinux', label)
	_copy_file(monkeypatch, src, tmp_path / 'held/f')
	monkeypatch.setattr(hpcp, '_held_route', lambda: None)
	_copy_file(monkeypatch, src, tmp_path / 'named/f')
	held = os.getxattr(str(tmp_path / 'held/f'), 'security.selinux')
	assert held == os.getxattr(str(tmp_path / 'named/f'), 'security.selinux'), held
	assert (held == label) is not content_only, held


@posix_only
def test_a_first_copy_of_a_sparse_file_stays_sparse(tmp_path, fresh, monkeypatch):
	src = str(tmp_path / 'a/sparse')
	os.makedirs(os.path.dirname(src))
	with open(src, 'wb') as handle:
		handle.seek(8 << 20)
		handle.write(b'x')
	if os.stat(src).st_blocks * 512 >= 1 << 20:
		pytest.skip('this filesystem makes no holes')
	seen = _record_cp(monkeypatch)
	_copy_file(monkeypatch, src, tmp_path / 'out/sparse')
	st = os.stat(str(tmp_path / 'out/sparse'))
	assert st.st_size == (8 << 20) + 1 and st.st_blocks * 512 < 1 << 20 and not hpcp.ERRORS, (st, seen, hpcp.ERRORS)


_FS_IOC_FIEMAP = 0xC020660B
_FIEMAP_EXTENT_SHARED = 0x2000


def _extent_flags(path):
	'''The FIEMAP flags of each extent of path (synced first).'''
	import fcntl
	count = 64
	buffer = bytearray(struct.pack('=QQIIII', 0, 2 ** 64 - 1, 1, 0, count, 0) + b'\0' * 56 * count)
	with open(path, 'rb') as handle:
		fcntl.ioctl(handle.fileno(), _FS_IOC_FIEMAP, buffer)
	mapped = struct.unpack_from('=I', buffer, 20)[0]
	return [struct.unpack_from('=I', buffer, 32 + 56 * i + 40)[0] for i in range(mapped)]


@linux_only
def test_a_first_copy_on_a_reflink_filesystem_is_cloned_as_cp_by_name_clones_it(tmp_path, fresh, monkeypatch):
	# coreutils 9's cp clones by default (--reflink=auto), through the held descriptor too: a same-device first copy on XFS
	# with reflink or btrfs shares its extents with the source, as cp by name made it (coreutils 8 copies, both ways).
	src = str(tmp_path / 'a/x')
	os.makedirs(os.path.dirname(src))
	with open(src, 'wb') as handle:
		handle.write(os.urandom(1 << 20))
	probe = subprocess.run([hpcp._bin('cp'), '--reflink=always', src, str(tmp_path / 'probe')], stdout=subprocess.PIPE,
		stderr=subprocess.PIPE)
	if probe.returncode != 0:
		pytest.skip('no reflink here: %s' % probe.stderr.decode(errors='replace').strip())
	_held_or_skip()
	seen = _record_cp(monkeypatch)
	_copy_file(monkeypatch, src, tmp_path / 'held/x')
	monkeypatch.setattr(hpcp, '_held_route', lambda: None)
	_copy_file(monkeypatch, src, tmp_path / 'named/x')
	held, named = (_extent_flags(str(tmp_path / name / 'x')) for name in ('held', 'named'))
	assert seen and seen[0][0].startswith(_HELD_PREFIX), seen
	if not all(flag & _FIEMAP_EXTENT_SHARED for flag in named):
		pytest.skip('cp here does not clone by default (%s)' % named)
	assert held and all(flag & _FIEMAP_EXTENT_SHARED for flag in held), held


# ---- H where 9.86 gave the name: a /proc that hides processes, an ancestor PID namespace's /proc (root, a test VM)

_SWAP_AFTER_OPEN = '''
import json, os, sys
sys.path.insert(0, sys.argv[1])
import hpcp
work = sys.argv[2]
landing, victim = work + '/out/x', work + '/victim'
inode = os.lstat(landing).st_ino
seen = []
real = hpcp.run_command_in_multicmd_with_path_check
def cp(command, *args, **kwargs):
	seen.append(command[-1])
	os.rename(landing, landing + '.held')
	os.symlink(victim, landing)  # a link put at the name after hpcp's open
	return real(command, *args, **kwargs)
hpcp.run_command_in_multicmd_with_path_check = cp
hpcp.NAMED_DESTS = frozenset([work + '/out'])
hpcp.copy_file(work + '/a/x', [landing])
print(json.dumps({'seen': seen, 'victim': open(victim).read(), 'held': open(landing + '.held').read(),
	'inode': os.lstat(landing + '.held').st_ino == inode, 'errors': hpcp.ERRORS, 'pid': os.getpid()}))
'''


def _swap_run(prefix=(), preexec=None, owner=None):
	"""_SWAP_AFTER_OPEN's result, run in a directory of /var/tmp that owner (when given) owns, with a copy of hpcp.py of its
	own (a home directory or pytest's basetemp is not open to another user); the directory removed after."""
	root = tempfile.mkdtemp(prefix='hpcp-held-', dir='/var/tmp')
	try:
		os.chmod(root, 0o755)
		work = os.path.join(root, 'work')
		made = [work, work + '/a', work + '/out', root + '/lib']
		made += [write(os.path.join(work, 'a/x'), 'new content'), write(os.path.join(work, 'out/x'), 'old'),
			write(os.path.join(work, 'victim'), 'victim'), write(os.path.join(root, 'swap.py'), _SWAP_AFTER_OPEN)]
		os.makedirs(root + '/lib')
		shutil.copy(hpcp.__file__, root + '/lib/hpcp.py')
		made.append(root + '/lib/hpcp.py')
		if owner is not None:
			for path in made:
				os.chown(path, owner, owner)
		# The interpreter itself: a virtual environment's link to it can sit where owner cannot reach (a home directory of mode 0700).
		python = os.path.realpath(sys.executable)
		task = subprocess.run(list(prefix) + [python, '-I', root + '/swap.py', root + '/lib', work], stdout=subprocess.PIPE,
			stderr=subprocess.PIPE, preexec_fn=preexec, cwd=root, timeout=120)
		assert task.returncode == 0, task.stderr.decode(errors='replace')[-3000:]
		return json.loads(task.stdout.decode().strip().splitlines()[-1])
	finally:
		shutil.rmtree(root, ignore_errors=True)


def _written_through_the_held_file(result):
	assert len(result['seen']) == 1 and result['seen'][0].startswith(_HELD_PREFIX), result
	assert result['victim'] == 'victim' and result['held'] == 'new content' and result['inode'] and not result['errors'], result


@root_only
@linux_only
@pytest.mark.skipif(os.environ.get('HPCP_TEST_PROC_REMOUNT') != '1',
	reason="remounts this machine's /proc hidepid=2: a test VM only (HPCP_TEST_PROC_REMOUNT=1)")
def test_the_held_route_works_where_proc_hides_processes():
	# hpcp run as nobody under a /proc that shows no other user's processes
	def nobody():
		os.setgroups([])
		os.setgid(65534)
		os.setuid(65534)
	subprocess.run(['mount', '-o', 'remount,hidepid=2', '/proc'], check=True)
	try:
		result = _swap_run(preexec=nobody, owner=65534)
	finally:
		subprocess.run(['mount', '-o', 'remount,hidepid=0', '/proc'], check=True)
	_written_through_the_held_file(result)


@root_only
@linux_only
def test_the_held_route_works_from_a_pid_namespace_without_its_own_proc():
	# unshare -p -f without --mount-proc: hpcp is pid 1 of its namespace, and /proc/1 is another process; 9.86 found
	# /proc/self was not its pid and gave cp the name.
	unshare = shutil.which('unshare')
	if unshare is None or subprocess.run([unshare, '-p', '-f', 'true'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode:
		pytest.skip('no unshare -p here')
	result = _swap_run(prefix=[unshare, '-p', '-f'])
	assert result['pid'] == 1, result
	_written_through_the_held_file(result)


@posix_only
@pytest.mark.parametrize('case', ['named-dangling-link', 'directory-holding-a-link'])
def test_what_appears_at_a_first_copy_s_name_is_judged_as_the_first_look_is(tmp_path, fresh, monkeypatch, case):
	# A name put at a first copy's landing between the look and hpcp's create makes the create fail (O_EXCL), and the landing
	# is looked at again. That look is judged as the first one is: a dangling link the operator named is not written through
	# (9.86's cp refused it; judged only at the first look, hpcp made the file where it points), and a directory is not
	# copied into (cp by name wrote through a link inside it, in 9.86 too).
	_held_or_skip()
	src = write(tmp_path / 'a/x', 'NEW')
	victim = write(tmp_path / 'victim/file', 'victim')
	out = str(tmp_path / 'out')
	os.makedirs(out)
	landing = os.path.join(out, 'y' if case == 'named-dangling-link' else 'x')
	real = hpcp._claim_device
	def plant(target, st, named, src_path):
		if not os.path.lexists(landing):
			if case == 'named-dangling-link':
				os.symlink('../victim/new', landing)
			else:
				os.mkdir(landing)
				os.symlink(victim, os.path.join(landing, 'x'))
		return real(target, st, named, src_path)
	monkeypatch.setattr(hpcp, '_claim_device', plant)
	monkeypatch.setattr(hpcp, 'NAMED_DESTS', frozenset([landing if case == 'named-dangling-link' else out]))
	monkeypatch.setattr(hpcp, 'RUN_ID', 'held-%d' % next(_RUNS))
	with contextlib.redirect_stdout(io.StringIO()):
		hpcp.copy_file(src, [landing])
	assert read(victim) == 'victim' and not os.path.lexists(str(tmp_path / 'victim/new')), hpcp.ERRORS
	assert hpcp.ERRORS


def _as_the_main_process(monkeypatch):
	'''hpcp's stop handler as the main process's, with no stop recorded yet and no pool worker to pass it on to.'''
	for name, value in (('_MAIN_PID', os.getpid()), ('_STOPPED', None), ('_STOPPED_PID', None), ('_STOP_ACTED', False), ('_DEFER', 0),
			('_STOP_NOTED', False), ('_pass_stop_on', lambda signum: None)):
		monkeypatch.setattr(hpcp, name, value)


@linux_only
@pytest.mark.parametrize('held', ['tmpfile', 'named'])
@pytest.mark.parametrize('handler', ['hpcp', 'python'])
def test_a_stop_anywhere_in_the_probe_leaves_nothing(tmp_path, fresh, monkeypatch, stop_alarms, handler, held):
	# 9.87's probe made its files with tempfile: gettempdir's test file and, where no file without a name can be made (macOS
	# always; a kernel or filesystem without O_TMPFILE: 'named'), mkstemp's, each removed by tempfile itself, so a stop
	# landing between a create and its unlink left the file. hpcp's own handler was deferred there (_MAKING); Python's own, in a
	# program that imports hpcp, was not. 9.88 makes them itself, each held by C as it is made (_create), and removes each.
	probes = _probe_dir(tmp_path, monkeypatch)
	monkeypatch.setattr(tempfile, 'tempdir', None)
	monkeypatch.setenv('TMPDIR', probes)
	if held == 'named':
		monkeypatch.setattr(tempfile, '_O_TMPFILE_WORKS', False, raising=False)  # 9.87's tempfile
		monkeypatch.delattr(os, 'O_TMPFILE', raising=False)
	monkeypatch.setattr(hpcp, 'cp_supports_sparse', lambda: True)
	class Done:
		returncode = 0
	monkeypatch.setattr(hpcp.multiCMD, 'run_commands', lambda *args, **kwargs: [Done()])
	def call(index, got):
		hpcp._HELD[:] = [None, None]
		tempfile.tempdir = None  # 9.87's gettempdir tries again each time
		got.extend(map(lambda _: hpcp._held_route() or True, (None,)))
	stopped = stop_alarms(handler, call, 2000)
	assert stopped and os.listdir(probes) == [], (len(stopped), os.listdir(probes)[:10])


@linux_only
def test_the_probe_s_held_file_has_no_name_where_o_tmpfile_works(tmp_path, fresh, monkeypatch):
	_gnu_cp_or_skip()
	probes = _probe_dir(tmp_path, monkeypatch)
	try:
		os.close(os.open(probes, os.O_TMPFILE | os.O_RDWR, 0o600))
	except (AttributeError, OSError):
		pytest.skip('no O_TMPFILE here')
	real = hpcp.multiCMD.run_commands
	names = []
	def run_commands(commands, *args, **kwargs):
		names.append(os.readlink(commands[0][-1]))  # cp's destination: /proc/self/fd/N of the held file, as this process holds it
		return real(commands, *args, **kwargs)
	monkeypatch.setattr(hpcp.multiCMD, 'run_commands', run_commands)
	assert hpcp._held_route() == 'c' and os.listdir(probes) == []
	assert len(names) == 1 and 'hpcp-held-probe' not in names[0] and names[0].endswith('(deleted)'), names


@linux_only
@not_root
def test_no_route_where_no_candidate_directory_takes_a_file(tmp_path, fresh, monkeypatch):
	# tempfile's order, tried by hpcp itself: TMPDIR, TEMP and TMP, then /tmp, /var/tmp and /usr/tmp, then the working directory.
	_gnu_cp_or_skip()
	shut = [str(tmp_path / name) for name in ('tmpdir', 'fixed', 'cwd')]
	for path in shut:
		os.makedirs(path, 0o555)
	monkeypatch.setattr(tempfile, 'tempdir', None)
	monkeypatch.setenv('TMPDIR', shut[0])
	monkeypatch.delenv('TEMP', raising=False)
	monkeypatch.delenv('TMP', raising=False)
	monkeypatch.setattr(hpcp, '_TEMP_DIRS', (shut[1],))
	monkeypatch.chdir(shut[2])
	assert hpcp._temp_candidates() == shut
	assert hpcp._held_route() is None and hpcp._HELD[0] == os.getpid()
	assert all(os.listdir(path) == [] for path in shut)


@posix_only
@pytest.mark.parametrize('handler', ['hpcp', 'python'])
def test_a_stop_as_a_first_copy_s_file_is_made_leaves_no_file(tmp_path, monkeypatch, stop_alarms, handler):
	# A stop's handler runs as os.open returns: raising there, `fd = os.open(...)` lost the descriptor before _cp_made knew the
	# file, and the landing stayed, empty and mode 0600. 9.87 deferred hpcp's own handler there (_MAKING); Python's own, in a
	# program that imports hpcp, left the file (about 10 % of stops landing on the create, holder.py). The descriptor is now
	# stored by C before any handler runs (_create), and the cleanup removes the file. cp is one write (_cp_held).
	# Python 3.8 to 3.10 leave a try before its return (POP_BLOCK, then RETURN_VALUE, where a handler runs): a stop there
	# leaves the copy, whole, and its descriptor open, as a stop just after the return would (3.9, four runs side by side:
	# 19 of about 190,000 stops, every one at the return). Elsewhere nothing runs a handler between the try and the return.
	import functools
	source = write(tmp_path / 'a' / 'x', 'data')
	out = tmp_path / 'out'
	out.mkdir()
	monkeypatch.setattr(hpcp, '_cp_held', lambda command, src_path, target, fd, *args, **kwargs: os.write(fd, b'data'))
	made = functools.partial(hpcp._cp_made, ['cp', '-af'], source)
	def call(index, got):
		got.extend(map(made, (str(out / ('f%d' % index)),), (None,), (None,)))
	def settle(got):
		if got:
			os.close(got[0][0])
	if sys.platform.startswith('linux'):
		before = len(os.listdir('/proc/self/fd'))
	stopped = stop_alarms(handler, call, 2000, settle)
	left = {str(out / ('f%d' % index)) for index in stopped if os.path.lexists(str(out / ('f%d' % index)))}
	whole = {path for path in left if os.lstat(path).st_size == 4}
	assert stopped and left == whole and (not left or (3, 8) <= sys.version_info < (3, 11)), (len(stopped), sorted(left), sorted(whole))
	if sys.platform.startswith('linux'):
		for fd in os.listdir('/proc/self/fd'):
			with contextlib.suppress(OSError):
				if os.readlink('/proc/self/fd/' + fd) in whole:
					os.close(int(fd))
		assert len(os.listdir('/proc/self/fd')) == before


@posix_only
@pytest.mark.parametrize('content_only', [False, True], ids=['plain', 'content-only'])
def test_a_stop_as_a_first_copy_s_create_fails_is_a_stop(tmp_path, monkeypatch, reset_hpcp_globals, content_only):
	# A create that fails (a name is there) as a stop comes: the stop is raised, and what was there stays.
	import signal
	source = write(tmp_path / 'a' / 'x', 'data')
	(tmp_path / 'out').mkdir()
	target = str(tmp_path / 'out' / 'x')
	hpcp.CONTENT_ONLY = content_only
	_as_the_main_process(monkeypatch)
	write(target, 'put there since')
	real = os.open
	def taken(path, flags, *args, **kwargs):
		if path == target:
			hpcp._stop(signal.SIGINT, None)
		return real(path, flags, *args, **kwargs)
	monkeypatch.setattr(os, 'open', taken)
	with pytest.raises(KeyboardInterrupt):
		hpcp._cp_made(['cp', '-f' if content_only else '-af'], source, target, None, None)
	monkeypatch.setattr(os, 'open', real)
	assert read(target) == 'put there since'


class _CapturingExecutor:
	def __init__(self):
		self.submitted = []

	def submit(self, func, *args, **kwargs):
		self.submitted.append((func, args, kwargs))


@linux_only
def test_a_pool_s_workers_are_handed_the_route_the_parent_asked(tmp_path, fresh, monkeypatch):
	# The parent asks cp once, as it opens a copy pool; a worker is handed the answer with its task and runs no probe of its own
	# (an early Ctrl-C met each worker's probe, and left its file in the temporary directory).
	_gnu_cp_or_skip()
	probes = _probe_dir(tmp_path, monkeypatch)
	monkeypatch.setattr(hpcp, '_pool', lambda *args, **kwargs: None)
	assert hpcp._copy_pool(2) is None
	assert hpcp._HELD[:] == [os.getpid(), 'c'] and os.listdir(probes) == []
	executor = _CapturingExecutor()
	hpcp._submit_task(executor, len, 'abc')
	(_, (settings, _, _), _), = executor.submitted
	assert settings['_HELD_GIVEN'] == ('c',)
	# the worker: a process of its own, which asks nothing
	monkeypatch.setattr(hpcp, '_HELD', [None, None])
	monkeypatch.setattr(hpcp, '_HELD_GIVEN', settings['_HELD_GIVEN'])
	def asked(*args, **kwargs):
		raise AssertionError('the worker asked cp for the route itself')
	monkeypatch.setattr(hpcp.multiCMD, 'run_commands', asked)
	monkeypatch.setattr(hpcp, 'cp_supports_sparse', asked)
	assert hpcp._held_route() == 'c'
	# no answer in the parent (cp timed out): nothing is handed, and the worker asks for itself
	monkeypatch.setattr(hpcp, '_HELD', [None, None])
	executor = _CapturingExecutor()
	hpcp._submit_task(executor, len, 'abc')
	assert executor.submitted[0][1][0]['_HELD_GIVEN'] is None


_PROBES_MADE = '''
import multiprocessing, os, sys
sys.path.insert(0, %(root)r)
import hpcp
multiprocessing.set_start_method('fork', force=True)
real = os.open
def counting(path, flags, *args, **kwargs):
	fd = real(path, flags, *args, **kwargs)
	if os.path.basename(str(path)).startswith('hpcp-held-probe-') and flags & os.O_CREAT:
		log = real(%(log)r, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
		os.write(log, b'%%d\\n' %% os.getpid())
		os.close(log)
	return fd
os.open = counting
sys.argv = ['hpcp', %(src)r, '-d', %(dst)r, '-j', '4']
hpcp.main()
'''


@linux_only
@pytest.mark.skipif('fork' not in __import__('multiprocessing').get_all_start_methods(), reason='forked workers see the patch')
def test_a_run_with_workers_makes_one_probe(tmp_path):
	# One probe a run (the parent's), where each of its workers made one: the files made are counted in every process.
	for i in range(40):
		write(tmp_path / 'src' / ('f%d' % i), 'data %d' % i)
	log = str(tmp_path / 'probes.log')
	script = tmp_path / 'probes.py'
	script.write_text(_PROBES_MADE % {'root': os.path.dirname(HPCP), 'log': log, 'src': str(tmp_path / 'src') + '/', 'dst': str(tmp_path / 'dst') + '/'})
	done = subprocess.run([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120,
		env=dict(os.environ, TMPDIR=str(tmp_path)))
	assert done.returncode == 0 and read(tmp_path / 'dst/f39') == 'data 39', done.stdout
	if hpcp._held_route() is None:
		pytest.skip('cp cannot write a held file here')
	pids = read(log).split() if os.path.exists(log) else []
	assert len(pids) == 1, (pids, done.stdout)


# ---- W1 (9.88): the door holds the directories on the way to a landing by descriptor, from the named destination down

def test_a_job_s_files_are_copied_in_path_order(tmp_path, reset_hpcp_globals, monkeypatch):
	# Each directory's subtree is one run of a job's files (the paths below it share its path and a separator), so a
	# worker's consecutive landings share their directories and the door enters each once. 9.87 took the listing's set
	# order.
	names = ['b/x', 'a.x/q', 'a/b/g', 'a/f', 'a/z', 'a/b/c/d', 'a/b.c/h', 'a-1/y', 'c']
	for name in names:
		write(tmp_path / 'src' / name, name)
	src = str(tmp_path / 'src')
	files = frozenset(os.path.join(src, name) for name in names)
	assert [f for _, _, f in hpcp.get_copy_file_job_iter([(src, [str(tmp_path / 'out')], files)])] == sorted(files)
	seen = []
	monkeypatch.setattr(hpcp, 'copy_file', lambda src_path, dests, **kwargs: seen.append(src_path) or (0, 0, {}))
	with contextlib.redirect_stdout(io.StringIO()):
		hpcp.copy_files_serial_batch([(src + '/', [str(tmp_path / 'out') + '/'])])
	assert seen == sorted(files)
	# each directory's subtree is one run: the door enters it once
	for directory in {os.path.dirname(path) for path in seen}:
		inside = [index for index, path in enumerate(seen) if path.startswith(directory + os.sep)]
		assert inside == list(range(inside[0], inside[-1] + 1)), (directory, seen)
