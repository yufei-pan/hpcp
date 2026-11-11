'''9.87 group 4: a rerun's looks (R1), the identical file's metadata (R2), the copy check (R3: its second pass by index, and by
directory on a network filesystem). In process, as the plan's probes count (probe/p1_rerun_calls.py, p3_first_and_dirs.py).'''
import collections
import contextlib
import errno
import io
import os
import random
import shutil
import stat
import struct
import subprocess
import sys
import threading
import time

import pytest

import hpcp

posix_only = pytest.mark.skipif(os.name != 'posix', reason='POSIX metadata')
linux_only = pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux: O_PATH and /proc/self/fd')
root_only = pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() != 0, reason='needs root')

_COUNTED = ('lstat', 'stat', 'open', 'chmod', 'chown', 'utime', 'listxattr', 'getxattr', 'setxattr', 'removexattr', 'statvfs',
	'readlink', 'access', 'scandir', 'mkdir', 'unlink', 'rename', 'fstat')
_SETS = ('chmod', 'chown', 'utime', 'setxattr', 'removexattr')


@contextlib.contextmanager
def _counting(monkeypatch, names=_COUNTED):
	'''A Counter of the os calls this thread makes inside the block.'''
	counts, me = collections.Counter(), threading.get_ident()
	with monkeypatch.context() as m:
		for name in names:
			real = getattr(os, name, None)
			if real is None:
				continue
			def counted(*args, _real=real, _name=name, **kwargs):
				if threading.get_ident() == me:
					counts[_name] += 1
				return _real(*args, **kwargs)
			m.setattr(os, name, counted)
		yield counts


def _now_ns():
	return int(time.time() * 10 ** 9)  # 3.6 has no time.time_ns


def _tree(root):
	'''3 directories of 5 files each. Their read times are an hour ahead, after their change times, so that a read moves no
	read time (relatime), on the copies either (cp -a keeps them): what a rerun then finds to set is only what differs.'''
	files = []
	now = _now_ns() + 3600 * 10 ** 9
	for d in range(3):
		os.makedirs(os.path.join(root, 'd%d' % d, 'e'))
		for f in range(5):
			path = os.path.join(root, 'd%d' % d, 'e', 'f%d' % f)
			with open(path, 'w') as handle:
				handle.write('x' * 100)
			os.utime(path, ns=(now, now - 100 * 10 ** 9))
			files.append(path)
	return files


def _first_copy(src, out):
	with contextlib.redirect_stdout(io.StringIO()):
		assert hpcp.hpcp([src + '/'], dest_paths=[out + '/'], max_workers=1, single_thread=True) == 0


def _copy_each(files, src, out, label):
	hpcp.NAMED_DESTS = frozenset([os.path.abspath(out)])
	hpcp.RUN_ID = label
	with contextlib.redirect_stdout(io.StringIO()):
		for path in files:
			hpcp.copy_file(path, [os.path.join(out, os.path.relpath(path, src))])


@pytest.fixture
def copied(tmp_path, reset_hpcp_globals):
	src, out = str(tmp_path / 'src'), str(tmp_path / 'out')
	files = _tree(src)
	_first_copy(src, out)
	probe = os.path.join(out, 'd0', 'e', 'f0')
	before = os.stat(probe).st_atime_ns
	with open(probe, 'rb') as handle:
		handle.read()
	if os.stat(probe).st_atime_ns != before:
		pytest.skip('a read moves the read time here (strictatime)')
	return src, out, files


# ---- R1: one look per landing

@posix_only
def test_an_identical_file_costs_few_calls(copied, monkeypatch):
	# 9.86: 22.5 os calls a file (lstat 7.4, stat 7.1, listxattr 3, chmod 2, chown 1, utime 2), besides the two hash reads.
	src, out, files = copied
	with _counting(monkeypatch) as counts:
		_copy_each(files, src, out, 'skip')
	# and a read on both sides of each extended attribute the file carries (SELinux labels every file: EL8 8.5, Docker 6.5)
	assert sum(counts.values()) / len(files) <= 8 + 2 * len(_xattrs(files[0])), counts


@posix_only
def test_a_rewrite_in_place_costs_few_calls(copied, monkeypatch):
	# 9.86: 25.6 a file.
	src, out, files = copied
	for path in files:
		with open(path, 'a') as handle:
			handle.write('y')
	with _counting(monkeypatch) as counts:
		_copy_each(files, src, out, 'rewrite')
	assert sum(counts.values()) / len(files) <= 14, counts
	assert all(open(path).read() == open(os.path.join(out, os.path.relpath(path, src))).read() for path in files)


@posix_only
def test_a_first_copy_costs_few_calls(tmp_path, reset_hpcp_globals, monkeypatch):
	# 9.86: 23.5 a file, closes counted (p3_first_and_dirs.py).
	src, out = str(tmp_path / 'src'), str(tmp_path / 'out')
	files = _tree(src)
	for path in files:
		os.makedirs(os.path.dirname(os.path.join(out, os.path.relpath(path, src))), exist_ok=True)
	with _counting(monkeypatch, _COUNTED + ('close',)) as counts:
		_copy_each(files, src, out, 'first')
	assert sum(counts.values()) / len(files) <= 23.5, counts


# ---- R2: the identical file through one checked descriptor, setting only what differs

@posix_only
def test_a_clean_rerun_sets_no_metadata(copied, monkeypatch):
	src, out, files = copied
	landings = [os.path.join(out, os.path.relpath(path, src)) for path in files]
	ctimes = [os.stat(landing).st_ctime_ns for landing in landings]
	time.sleep(0.01)
	with _counting(monkeypatch, _SETS) as counts:
		_copy_each(files, src, out, 'clean')
	assert not counts, counts
	assert [os.stat(landing).st_ctime_ns for landing in landings] == ctimes


@posix_only
def test_reruns_set_nothing_though_their_own_reads_move_the_read_times(tmp_path, reset_hpcp_globals, monkeypatch):
	# Read times in the past, as a real tree's: under relatime a read moves a read time that is not after the change time,
	# which any metadata set makes it. Chasing the read time alone, a rerun set it on every identical file (its own skip
	# check's read had moved it), a SETATTR a file on every rerun on NFS (EL8: 10,000 a rerun of 10,000 files).
	probe = str(tmp_path / 'probe')
	with open(probe, 'w') as handle:
		handle.write('x')
	old = _now_ns() - 3600 * 10 ** 9
	os.utime(probe, ns=(old, old))
	os.chmod(probe, 0o644)  # its change time is now, after its read time
	with open(probe) as handle:
		handle.read()
	if os.stat(probe).st_atime_ns == old:
		pytest.skip('a read moves no read time here (noatime)')
	src, out = str(tmp_path / 'src'), str(tmp_path / 'out')
	files = []
	for f in range(10):
		path = os.path.join(src, 'd', 'f%d' % f)
		os.makedirs(os.path.dirname(path), exist_ok=True)
		with open(path, 'w') as handle:
			handle.write('x' * 100)
		os.utime(path, ns=(old, old))
		files.append(path)
	_first_copy(src, out)
	_copy_each(files, src, out, 'rerun-1')
	for rerun in (2, 3):
		with _counting(monkeypatch, _SETS) as counts:
			_copy_each(files, src, out, 'rerun-%d' % rerun)
		assert not counts, (rerun, counts)


def _xattrs(path):
	if not hasattr(os, 'listxattr'):
		return {}
	try:
		return {name: os.getxattr(path, name, follow_symlinks=False) for name in os.listxattr(path, follow_symlinks=False)}
	except OSError as e:
		if e.errno in (errno.ENOTSUP, errno.EOPNOTSUPP):
			return {}
		raise


def _user_xattrs_work(path):
	try:
		os.setxattr(path, 'user.probe', b'1', follow_symlinks=False)
		os.removexattr(path, 'user.probe', follow_symlinks=False)
		return True
	except (OSError, AttributeError):
		return False


def _skip_986(src, dest):
	'''9.86's skip branch, the reference.'''
	st = os.stat(src, follow_symlinks=False)
	hpcp._remove_stale_xattrs(src, dest)
	shutil.copystat(src, dest, follow_symlinks=False)
	owner_kept, group_kept = hpcp._chown_like_cp(dest, st)
	os.chmod(dest, st.st_mode & ~((0 if owner_kept else stat.S_ISUID) | (0 if group_kept else stat.S_ISGID)))
	os.utime(dest, ns=(st.st_atime_ns, st.st_mtime_ns), follow_symlinks=False)


def _metadata(path, atime=True):
	st = os.lstat(path)
	return stat.S_IMODE(st.st_mode), st.st_uid, st.st_gid, st.st_atime_ns if atime else None, st.st_mtime_ns, _xattrs(path)


_CASES = ['mode', 'setuid', 'times', 'xattr-added', 'xattr-changed', 'xattr-removed', 'owner', 'acl-added', 'acl-removed',
	'label']


def _acl(entries):
	'''A system.posix_acl_access value: (tag, perm, id) entries, as setfacl writes them.'''
	return struct.pack('<I', 2) + b''.join(struct.pack('<HHI', *entry) for entry in entries)


# user::rw- user:1000:rwx group::r-- mask::rwx other::r--
_ACL = _acl([(0x01, 6, 0xffffffff), (0x02, 7, 1000), (0x04, 4, 0xffffffff), (0x10, 7, 0xffffffff), (0x20, 4, 0xffffffff)])
_LABEL = b'system_u:object_r:user_home_t:s0\x00'


def _can_set(path, name, value):
	try:
		os.setxattr(path, name, value, follow_symlinks=False)
		return True
	except OSError:
		return False


@posix_only
@pytest.mark.parametrize('case', _CASES)
def test_the_identical_file_s_metadata_matches_9_86(tmp_path, reset_hpcp_globals, case):
	src = str(tmp_path / 'src')
	with open(src, 'w') as handle:
		handle.write('same')
	ours, theirs = str(tmp_path / 'ours'), str(tmp_path / 'theirs')
	if case.startswith('xattr') and not _user_xattrs_work(src):
		pytest.skip('no user xattrs here')
	if case == 'owner' and os.geteuid() != 0:
		pytest.skip('only root gives a file another owner')
	if case.startswith('acl') and not _can_set(src, 'system.posix_acl_access', _ACL):
		pytest.skip('no POSIX ACLs here')
	if case == 'label' and not (os.path.exists('/sys/fs/selinux/enforce') and _can_set(src, 'security.selinux', _LABEL)):
		pytest.skip('no SELinux label that can be set here')
	if case.startswith('acl'):
		os.removexattr(src, 'system.posix_acl_access')
	now = _now_ns()
	os.utime(src, ns=(now, now - 50 * 10 ** 9))
	for landing in (ours, theirs):
		shutil.copyfile(src, landing)
		os.utime(landing, ns=(now, now - 50 * 10 ** 9))
		if case == 'mode':
			os.chmod(landing, 0o600)
		elif case == 'times':
			os.utime(landing, ns=(now - 10 ** 9, now - 70 * 10 ** 9))
		elif case == 'xattr-changed':
			os.setxattr(landing, 'user.a', b'old')
		elif case == 'xattr-removed':
			os.setxattr(landing, 'user.gone', b'1')
		elif case == 'owner':
			os.chown(landing, 1000, 1000)
		elif case == 'acl-removed':
			os.setxattr(landing, 'system.posix_acl_access', _ACL)
	if case == 'setuid':
		os.chmod(src, 0o4755)
	elif case in ('xattr-added', 'xattr-changed'):
		os.setxattr(src, 'user.a', b'new')
	elif case == 'acl-added':
		os.setxattr(src, 'system.posix_acl_access', _ACL)
	elif case == 'label':
		os.setxattr(src, 'security.selinux', _LABEL)
	src_st = os.lstat(src)
	_skip_986(src, theirs)
	assert hpcp._settle_identical(src, src_st, ours, os.lstat(ours)) is True
	# 9.86 set the read time on every rerun; 9.87 sets it with the modification time only (a read moves it, relatime)
	assert _metadata(ours, atime=case == 'times') == _metadata(theirs, atime=case == 'times')


@root_only
@linux_only
def test_an_identical_file_keeps_its_capability_though_its_owner_is_set(tmp_path, reset_hpcp_globals):
	# The kernel clears security.capability on every chown. 9.86 ran copystat and then the chown on every rerun, so every
	# identical file lost its capability (a copy of ping its cap_net_raw); the xattrs go after the owner, as cp -a gives them.
	src, landing = str(tmp_path / 'src'), str(tmp_path / 'landing')
	with open(src, 'w') as handle:
		handle.write('same')
	capability = struct.pack('<IIIII', 0x02000001, 1 << 13, 0, 0, 0)
	if not _can_set(src, 'security.capability', capability):
		pytest.skip('no file capabilities here')
	now = _now_ns()
	os.utime(src, ns=(now, now - 50 * 10 ** 9))
	shutil.copyfile(src, landing)
	os.utime(landing, ns=(now, now - 50 * 10 ** 9))
	os.chown(landing, 1000, 1000)  # another owner: the settle must chown it back
	os.setxattr(landing, 'security.capability', capability)  # after the chown, which cleared it
	assert hpcp._settle_identical(src, os.lstat(src), landing, os.lstat(landing)) is True
	assert os.lstat(landing).st_uid == 0 and _xattrs(landing).get('security.capability') == capability


@posix_only
def test_an_identical_file_whose_metadata_cannot_be_set_is_a_failed_copy(copied, monkeypatch):
	# A read-only mount, another user's file: what the skip gives an identical landing cannot be set. A failed copy (190)
	# that names the landing, not a crash (191) naming the descriptor's /proc/self/fd/N.
	src, out, files = copied
	path, landing = files[0], os.path.join(out, os.path.relpath(files[0], src))
	os.chmod(landing, 0o600)  # only its mode differs
	def read_only(*args, **kwargs):
		raise OSError(errno.EROFS, os.strerror(errno.EROFS), args[0])
	monkeypatch.setattr(hpcp.os, 'chmod', read_only)
	_copy_each([path], src, out, 'read-only')
	monkeypatch.undo()
	failed = [error for error in hpcp.ERRORS if error.startswith('Copy failed')]
	assert len(failed) == 1 and landing in failed[0] and '/proc/' not in failed[0], hpcp.ERRORS
	assert failed[0].endswith('cannot be set (%s)' % os.strerror(errno.EROFS)), failed


@root_only
@linux_only
@pytest.mark.parametrize('swap', ['link', 'second-name'])
def test_a_landing_swapped_after_the_look_gets_no_metadata(tmp_path, reset_hpcp_globals, monkeypatch, swap):
	# Between the skip check and the metadata, a user who can write the destination puts a link to another file, or another
	# file's second name, at the landing. 9.86 set the source's owner and mode (and times, xattrs) on that file through it.
	victim = str(tmp_path / 'victim')
	with open(victim, 'w') as handle:
		handle.write('victim')
	os.chmod(victim, 0o600)
	if _user_xattrs_work(victim):
		os.setxattr(victim, 'user.keep', b'1')
	src_dir, out = str(tmp_path / 'src'), str(tmp_path / 'out')
	os.makedirs(src_dir)
	os.makedirs(out)
	src, landing = os.path.join(src_dir, 'f'), os.path.join(out, 'f')
	for path in (src, landing):
		with open(path, 'w') as handle:
			handle.write('same')
	os.chown(src, 1000, 1000)
	os.chmod(src, 0o4755)  # after the chown, which clears set-user-ID
	st = os.lstat(src)
	os.utime(landing, ns=(st.st_atime_ns, st.st_mtime_ns))  # identical: the skip check passes
	os.chown(out, 1000, 1000)
	before = _metadata(victim)
	real = hpcp._known_identical
	def checked_then_swapped(*args, **kwargs):
		same = real(*args, **kwargs)
		if same and not os.path.lexists(landing + '.away'):
			os.rename(landing, landing + '.away')
			if swap == 'link':
				os.symlink(victim, landing)
			else:
				os.link(victim, landing)
		return same
	monkeypatch.setattr(hpcp, '_known_identical', checked_then_swapped)
	_copy_each([src], src_dir, out, 'swap')
	assert os.path.lexists(landing + '.away')  # the skip check passed, and the swap was made
	assert _metadata(victim) == before
	with open(victim) as handle:
		assert handle.read() == 'victim'
	# The landing is the source's copy now: a link below a destination is replaced, a file of two names written beside.
	assert not os.path.islink(landing) and os.lstat(landing).st_ino != os.lstat(victim).st_ino
	with open(landing) as handle:
		assert handle.read() == 'same'


# ---- R3: the copy check's second pass by index; by directory on a network filesystem

def _random_trees(root, seed):
	'''Two sources merged into one destination, with clashes planted: a landing missing, one with other content, one that is
	a link, two landings that are one file (a hard link), and both sources' files of one name.'''
	rng = random.Random(seed)
	entries = []
	for source in ('a', 'b'):
		for _ in range(rng.randrange(20, 60)):
			rel = os.path.join(*['d%d' % rng.randrange(4) for _ in range(rng.randrange(1, 3))], 'f%d' % rng.randrange(12))
			path = os.path.join(root, source, rel)
			if os.path.isdir(path) or any(os.path.isfile(part) for part in hpcp._path_and_parents(os.path.dirname(path), root)):
				continue
			os.makedirs(os.path.dirname(path), exist_ok=True)
			with open(path, 'w') as handle:
				handle.write(source + rel)
			entries.append(path)
	out = os.path.join(root, 'out')
	for path in entries:
		landing = os.path.join(out, os.path.relpath(path, os.path.join(root, path[len(root) + 1])))
		if os.path.isdir(landing) or any(os.path.isfile(part) for part in hpcp._path_and_parents(os.path.dirname(landing), out)):
			continue
		os.makedirs(os.path.dirname(landing), exist_ok=True)
		shutil.copy2(path, landing)
	landings = sorted(os.path.join(dirpath, name) for dirpath, _, names in os.walk(out) for name in names)
	rng.shuffle(landings)
	for landing in landings[:3]:
		os.remove(landing)
	for landing in landings[3:5]:
		with open(landing, 'a') as handle:
			handle.write('changed')
	if len(landings) > 7:
		os.remove(landings[5])
		os.symlink(landings[6], landings[5])
	if len(landings) > 9:
		os.remove(landings[7])
		os.link(landings[8], landings[7])
	return entries, [os.path.join(root, 'a') + os.sep, os.path.join(root, 'b') + os.sep], [out + os.sep]


@posix_only
@pytest.mark.parametrize('move', [False, True], ids=['copy', 'move'])
@pytest.mark.parametrize('seed', range(12))
def test_the_second_pass_by_index_matches_a_second_pass_over_every_entry(tmp_path, reset_hpcp_globals, monkeypatch, seed, move):
	# The same {path: why} as 9.86's second pass, which looked at every entry again: with every fingerprint made to meet
	# every other, every entry is looked at again.
	entries, src_paths, dests = _random_trees(str(tmp_path), seed)
	indexed = hpcp._not_held(list(entries), src_paths, dests, max_workers=1, move=move)
	monkeypatch.setattr(hpcp, 'hash', lambda value: 0, raising=False)
	every = hpcp._not_held(list(entries), src_paths, dests, max_workers=1, move=move)
	assert indexed == every
	assert indexed  # the planted clashes are found


@posix_only
def test_the_second_pass_stats_only_the_entries_met_twice(tmp_path, reset_hpcp_globals, monkeypatch):
	# 9.86 looked at every landing again when two may be one file; now only those whose fingerprint met another's. out/d3 is
	# a link to out/d7, so d3's and d7's 20 files each land in one file of one name.
	src, out = tmp_path / 'src', tmp_path / 'out'
	for d in range(10):
		(src / ('d%d' % d)).mkdir(parents=True)
		(out / ('d%d' % d)).mkdir(parents=True)
		for f in range(20):
			(src / ('d%d' % d) / ('f%d' % f)).write_text('%d-%d' % (d, f))
			shutil.copy2(str(src / ('d%d' % d) / ('f%d' % f)), str(out / ('d%d' % d) / ('f%d' % f)))
	shutil.rmtree(str(out / 'd3'))
	os.symlink('d7', str(out / 'd3'))
	entries = sorted(str(path) for path in src.rglob('f*'))
	looked = []
	real = os.lstat
	def counted(path, *args, **kwargs):
		if str(path).startswith(str(out) + os.sep):
			looked.append(path)
		return real(path, *args, **kwargs)
	monkeypatch.setattr(hpcp.os, 'lstat', counted)
	not_held = hpcp._not_held(entries, [str(src) + os.sep], [str(out) + os.sep], max_workers=1, move=False)
	monkeypatch.setattr(hpcp.os, 'lstat', real)
	assert len(looked) == 200 + 40, len(looked)
	assert set(not_held) == {str(src / d / ('f%d' % f)) for d in ('d3', 'd7') for f in range(20)}, not_held


@posix_only
@pytest.mark.parametrize('move', [False, True], ids=['copy', 'move'])
@pytest.mark.parametrize('seed', range(12))
def test_the_copy_check_by_directory_matches_a_look_by_path(tmp_path, reset_hpcp_globals, monkeypatch, seed, move):
	# Where the destination is on a network filesystem, each landing is lstat'ed relative to its directory, opened once a
	# chunk (_lstat_below): the same {path: why} as a look by its path, the planted clashes and links to directories too.
	entries, src_paths, dests = _random_trees(str(tmp_path), seed)
	by_path = hpcp._not_held(list(entries), src_paths, dests, max_workers=1, move=move)
	monkeypatch.setattr(hpcp, '_on_network_fs', lambda paths: True)
	looked = []
	real = hpcp._lstat_below
	def counted(path, opened):
		looked.append(path)
		return real(path, opened)
	monkeypatch.setattr(hpcp, '_lstat_below', counted)
	grouped = hpcp._not_held(list(entries), src_paths, dests, max_workers=1, move=move)
	assert grouped == by_path
	assert grouped and looked  # the planted clashes are found, by directory


@posix_only
@pytest.mark.parametrize('kind', ['nfs4', 'nfs', 'lustre', 'cifs', 'xfs', 'ext4', None])
def test_the_copy_check_goes_by_directory_only_on_a_network_filesystem(tmp_path, reset_hpcp_globals, monkeypatch, kind):
	# NFS measured (EL8, actimeo=0): 2.4 times fewer GETATTRs and 2 to 2.4 times faster by directory; locally the opens cost
	# more than the path walks they save, so a local destination is looked at by path (9.86's). The type is asked once a
	# destination (_mount_type: None off Linux, and for a mount the table does not list), never once a landing.
	src, out = tmp_path / 'src', tmp_path / 'out'
	for d in range(3):
		(src / ('d%d' % d)).mkdir(parents=True)
		(out / ('d%d' % d)).mkdir(parents=True)
		for f in range(5):
			(src / ('d%d' % d) / ('f%d' % f)).write_text('x')
			shutil.copy2(str(src / ('d%d' % d) / ('f%d' % f)), str(out / ('d%d' % d) / ('f%d' % f)))
	entries = sorted(str(path) for path in src.rglob('f*'))
	asked = []
	monkeypatch.setattr(hpcp, '_mount_type', lambda dev: asked.append(dev) or kind)
	by_name, relative = [], []
	real = os.lstat
	def counted(path, *args, **kwargs):
		if kwargs.get('dir_fd') is not None:
			relative.append(path)
		elif str(path).startswith(str(out) + os.sep):
			by_name.append(path)
		return real(path, *args, **kwargs)
	monkeypatch.setattr(hpcp.os, 'lstat', counted)
	not_held = hpcp._not_held(entries, [str(src) + os.sep], [str(out) + os.sep], max_workers=1, move=True)
	monkeypatch.setattr(hpcp.os, 'lstat', real)
	assert not_held == {} and len(asked) == 1
	if kind in hpcp._NETWORK_FS:
		assert (len(relative), by_name) == (15, [])
	else:
		assert (relative, len(by_name)) == ([], 15)
