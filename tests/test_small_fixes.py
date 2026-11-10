'''9.87 group 3: small fixes (A1 is in test_tool_path_overrides.py). Subprocess runs use the CLI as a person does.'''
import os
import subprocess
import sys

import pytest

HPCP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hpcp.py')
posix_only = pytest.mark.skipif(os.name != 'posix', reason='POSIX paths and links')


def run(args, cwd, env=None, timeout=120, stdin=subprocess.DEVNULL):
	'''The CLI in a session of its own: (return code, output as text)'''
	task = subprocess.run([sys.executable, HPCP] + args, cwd=str(cwd), stdin=stdin, stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT, env=dict(os.environ, **(env or {})), timeout=timeout, start_new_session=True)
	return task.returncode, task.stdout.decode(errors='replace')


# ---- A2: -d folded once (entry 19)

@posix_only
@pytest.mark.parametrize('dest', ['l/../out/', 'l/../out'], ids=['into', 'as'])
def test_a_destination_through_a_link_and_dotdot_lands_where_it_says(tmp_path, dest):
	# 9.86 made deep/out (the mkdir followed l before '..') while the files went to out/
	(tmp_path / 'a').mkdir()
	(tmp_path / 'a' / 'f').write_text('f')
	(tmp_path / 'deep' / 'inner').mkdir(parents=True)
	os.symlink('deep/inner', str(tmp_path / 'l'))
	rc, output = run(['a/', '-d', dest], tmp_path)
	assert rc == 0, output[-2000:]
	assert (tmp_path / 'out' / 'f').read_text() == 'f'
	assert sorted(os.listdir(str(tmp_path / 'deep'))) == ['inner'] and os.listdir(str(tmp_path / 'deep' / 'inner')) == []


# ---- A3: -siff reads its lists as bytes (entry 9)

_PATCH = '''
import os, sys
sys.path.insert(0, %(root)r)
import hpcp
if __name__ == '__main__':  # a pool's forkserver or spawn worker imports this file
	hpcp.HASH_SIZE = 65536
	step, new, base, vol, target = sys.argv[1:6]
	if step == 'make':
		rows = hpcp.compute_changes(*hpcp.build_root_maps(new, base, max_workers=1), max_workers=1)
		hpcp.build_patch_tree(rows, new, [vol], hpcp.make_patch_meta(rows, 'n', 'b', 'root'), max_workers=1)
	else:
		print('failed', hpcp.apply_patch_tree([vol], target, max_workers=1)['failed'])
'''

# Python's C locale with neither its UTF-8 mode nor locale coercion (3.7+): names are ASCII, as under Python 3.6 with LANG unset
_C = {'LC_ALL': 'C', 'LANG': 'C', 'PYTHONUTF8': '0', 'PYTHONCOERCECLOCALE': '0'}


@posix_only
@pytest.mark.parametrize('made', ['C', 'C.UTF-8'])
def test_a_patch_with_a_non_ascii_name_applies_under_the_c_locale(tmp_path, made):
	# 9.86 decoded the lists as UTF-8, and under the C locale the name could not be encoded back: UnicodeEncodeError
	name = b'new\xc3\xa9'
	new, base, vol, target = (tmp_path / part for part in ('new', 'base', 'vol', 'target'))
	for directory in (new, base, vol, target):
		directory.mkdir()
	with open(os.path.join(os.fsencode(str(new)), name), 'wb') as f:
		f.write(b'content')
	script = tmp_path / 'patch.py'
	script.write_text(_PATCH % {'root': os.path.dirname(HPCP)})
	def step(what, env):
		done = subprocess.run([sys.executable, str(script), what, str(new), str(base), str(vol), str(target)],
			stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=dict(os.environ, **env), timeout=120)
		assert done.returncode == 0, done.stdout.decode(errors='replace')[-3000:]
		return done.stdout.decode(errors='replace')
	step('make', _C if made == 'C' else dict(_C, LC_ALL='C.UTF-8', LANG='C.UTF-8'))
	assert 'failed 0' in step('apply', _C)
	with open(os.path.join(os.fsencode(str(target)), name), 'rb') as f:
		assert f.read() == b'content'


# ---- A4: hints shell-quoted (entry 8)

_NAMES = [b'plain', b'a b', b"it's", b'$HOME', b'x\xff', b'tab\there', 'café'.encode(), b'back\\slash']


@posix_only
@pytest.mark.parametrize('encoding', ['ascii', 'utf-8'])
@pytest.mark.parametrize('name', _NAMES, ids=lambda name: name.decode('ascii', 'backslashreplace'))
def test_a_shell_word_reads_back_as_the_name(monkeypatch, name, encoding):
	# 9.86 printed hints' paths as Python holds them: a space split the command, and a non-UTF-8 byte showed as \udcff
	import hpcp
	class Out:
		pass
	out = Out()
	out.encoding = encoding
	monkeypatch.setattr(sys, 'stdout', out)
	word = hpcp._shell_word(os.fsdecode(name))
	monkeypatch.undo()
	word.encode(encoding)  # the output can show it
	done = subprocess.run([b'bash', b'-c', b'printf %s ' + os.fsencode(word)], stdout=subprocess.PIPE, check=True)
	assert done.stdout == name


@posix_only
@pytest.mark.parametrize('locale', ['C', 'C.UTF-8'])
def test_the_replaced_link_note_s_command_writes_through_the_link(tmp_path, locale):
	# the Note's command, pasted into bash, copies through the link it names
	weird = b"s p'$\xff"
	root = os.fsencode(str(tmp_path))
	src = os.path.join(root, weird)
	os.makedirs(os.path.join(src, b'd'))
	with open(os.path.join(src, b'd', b'f'), 'wb') as f:
		f.write(b'f')
	os.makedirs(os.path.join(root, b'elsewhere'))
	os.makedirs(os.path.join(root, b'out'))
	os.symlink(b'../elsewhere', os.path.join(root, b'out', b'd'))
	env = {'LC_ALL': locale, 'LANG': locale, 'PYTHONUTF8': '0', 'PYTHONCOERCECLOCALE': '0'}
	task = subprocess.run([sys.executable, HPCP, os.fsdecode(src + b'/'), '-d', 'out/'], cwd=str(tmp_path), stdin=subprocess.DEVNULL,
		stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=dict(os.environ, **env), timeout=120)
	assert task.returncode == 0, task.stdout[-2000:]
	assert not os.path.exists(os.path.join(root, b'elsewhere', b'f'))
	marker = b'name it as the destination: hpcp '
	line = next(line for line in task.stdout.splitlines() if marker in line)
	command = line.split(marker, 1)[1]
	# the operator puts the link back, and runs the command the Note gave
	import shutil
	shutil.rmtree(os.path.join(root, b'out', b'd'))
	os.symlink(b'../elsewhere', os.path.join(root, b'out', b'd'))
	done = subprocess.run([b'bash', b'-c', b'"$0" "$1" ' + command, os.fsencode(sys.executable), os.fsencode(HPCP)],
		cwd=str(tmp_path), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=dict(os.environ, **env),
		timeout=120)
	assert done.returncode == 0, (command, done.stdout[-2000:])
	with open(os.path.join(root, b'elsewhere', b'f'), 'rb') as f:
		assert f.read() == b'f'


# ---- A5: worker error lines start their own line (entry 15)

not_root = pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() == 0, reason='root reads every file')


def _unreadable(tmp_path, count=12):
	src = tmp_path / 'src'
	src.mkdir()
	for i in range(count):
		(src / ('f%02d' % i)).write_text('x' * 100)
		os.chmod(str(src / ('f%02d' % i)), 0)
	# readable ones too, so the bar draws frames between the failures
	for i in range(40):
		(src / ('ok%02d' % i)).write_text('x' * 100)
	return src


@posix_only
@not_root
@pytest.mark.parametrize('workers', [['-j', '4'], ['-s']], ids=['pool', 'serial'])
def test_error_lines_start_their_own_line_in_a_log(tmp_path, workers):
	# 9.86: with stdout and stderr in one file, 9 to 11 of 12 'Copy failed' lines started a line, the rest glued to the bar's
	# line. Under -s on Python 3.6 to 3.8 the first build of the fix still lost one in about one run of four: an error printed
	# before the bar's first frame (files come in set order) stayed in stderr's buffer and landed later, on a frame's line.
	_unreadable(tmp_path)
	for run in range(6):
		log = tmp_path / 'log'
		with open(str(log), 'wb') as out:
			subprocess.run([sys.executable, HPCP, 'src/', '-d', 'out%d/' % run] + workers, cwd=str(tmp_path),
				stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, timeout=120)
		lines = log.read_bytes().split(b'\n')
		assert sum(line.startswith(b'Copy failed') for line in lines) == 12, (run, log.read_bytes()[-3000:])


@posix_only
@not_root
def test_a_separate_error_log_has_no_empty_lines(tmp_path):
	_unreadable(tmp_path)
	with open(str(tmp_path / 'out.log'), 'wb') as out, open(str(tmp_path / 'err.log'), 'wb') as err:
		subprocess.run([sys.executable, HPCP, 'src/', '-d', 'out/', '-j', '4'], cwd=str(tmp_path), stdin=subprocess.DEVNULL,
			stdout=out, stderr=err, timeout=120)
	errors = (tmp_path / 'err.log').read_bytes().split(b'\n')[:-1]
	assert errors and b'' not in errors and sum(line.startswith(b'Copy failed') for line in errors) == 12


@posix_only
@not_root
@pytest.mark.parametrize('workers', [['-j', '4'], ['-s']], ids=['pool', 'serial'])
def test_on_a_terminal_an_error_line_clears_the_bar_first(tmp_path, workers):
	import pty
	_unreadable(tmp_path)
	pid, fd = pty.fork()
	if pid == 0:
		# a terminal of 120 columns: a new pty has none, and the bar then prints whole lines
		import fcntl, struct, termios
		fcntl.ioctl(0, termios.TIOCSWINSZ, struct.pack('HHHH', 24, 120, 0, 0))
		os.chdir(str(tmp_path))
		os.execv(sys.executable, [sys.executable, HPCP, 'src/', '-d', 'out/'] + workers)
	chunks = []
	while True:
		try:
			chunk = os.read(fd, 65536)
		except OSError:
			break
		if not chunk:
			break
		chunks.append(chunk)
	os.waitpid(pid, 0)
	output = b''.join(chunks).replace(b'\r\n', b'\n')
	# (the run's last line, 'Error occurred: [...]', quotes them)
	failed = [line for line in output.split(b'\n') if b'Copy failed' in line and not line.startswith(b'Error occurred')]
	assert len(failed) == 12, output[-3000:]
	# what the terminal shows of each: from its last carriage return, after the line clear
	assert all(line.rsplit(b'\r', 1)[-1].startswith(b'Copy failed') for line in failed), failed


# ---- A6: -enes counts what the copy will write (entry 20)

SMALL_FS = os.environ.get('HPCP_TEST_SMALL_FS')


@pytest.fixture
def small_dir():
	'''A directory of its own on the 5 MiB filesystem HPCP_TEST_SMALL_FS, emptied after; one test at a time there, across
	pytest-xdist's workers (test_landings.py takes the same lock).'''
	if not SMALL_FS:
		pytest.skip('HPCP_TEST_SMALL_FS: a 5 MiB filesystem of its own')
	import fcntl
	import shutil
	with open(os.path.join(SMALL_FS, '.lock'), 'a') as lock:
		fcntl.flock(lock, fcntl.LOCK_EX)
		out = os.path.join(SMALL_FS, 'enes-%d' % os.getpid())
		shutil.rmtree(out, ignore_errors=True)
		os.makedirs(out)
		try:
			yield out
		finally:
			shutil.rmtree(out, ignore_errors=True)


@posix_only
def test_enes_counts_only_what_the_copy_will_write(tmp_path, small_dir):
	# 3 MB copied onto 5 MiB leaves about 2.2 MB free. 9.86 summed the source's 3 MB against that and skipped the rerun (228).
	(tmp_path / 'a').mkdir()
	(tmp_path / 'a' / 'x').write_bytes(os.urandom(3000000))
	out = small_dir + '/'
	rc, output = run(['a/', '-d', out], tmp_path)
	assert rc == 0, output[-2000:]
	rc, output = run(['a/', '-d', out, '-enes'], tmp_path)
	assert rc == 0 and 'Not enough space' not in output, output[-2000:]
	# A rewrite in place of a file of the same size takes nothing more; beside it (-wb) it takes the whole file.
	(tmp_path / 'a' / 'x').write_bytes(os.urandom(3000000))
	rc, output = run(['a/', '-d', out, '-enes', '-wb'], tmp_path)
	assert rc == 228 and 'Not enough space' in output, output[-2000:]
	rc, output = run(['a/', '-d', out, '-enes'], tmp_path)
	assert rc == 0 and 'Not enough space' not in output, output[-2000:]
	with open(os.path.join(small_dir, 'x'), 'rb') as landed:
		assert landed.read() == (tmp_path / 'a' / 'x').read_bytes()
	# A new source that does not fit is still skipped.
	(tmp_path / 'b').mkdir()
	(tmp_path / 'b' / 'y').write_bytes(os.urandom(3000000))
	rc, output = run(['b/', '-d', out, '-enes'], tmp_path)
	assert rc == 228 and 'Not enough space' in output and not os.path.exists(os.path.join(small_dir, 'y')), output[-2000:]
