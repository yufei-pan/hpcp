"""Sources from a list (-sfl FILE, -sfl -, implicit stdin) keep their tree; same-name clashes never overwrite silently."""
import builtins
import io
import os
import pty
import select
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

import pytest

HPCP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hpcp.py')


def run(args, cwd, stdin=b'', env=None):
	"""The CLI, stdin given as bytes (never a tty): (return code, stdout + stderr as bytes)."""
	full_env = dict(os.environ, **(env or {}))
	# Its own session: no controlling terminal, wherever the tests run, so a prompt cannot reach a person's.
	task = subprocess.run([sys.executable, HPCP] + args, input=stdin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
		cwd=str(cwd), env=full_env, timeout=600, start_new_session=True)
	return task.returncode, task.stdout


def files(root):
	"""{path below root: content} for every file, and link -> target for every symlink, below root, as bytes."""
	found = {}
	root = os.fsencode(root)
	for dirpath, dirnames, filenames in os.walk(root):
		for name in filenames + [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]:
			path = os.path.join(dirpath, name)
			rel = os.path.relpath(path, root)
			if os.path.islink(path):
				found[rel] = b'-> ' + os.readlink(path)
			else:
				with open(path, 'rb') as f:
					found[rel] = f.read()
	return found


def write(path, content='x'):
	path = os.fsencode(path)
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'wb') as f:
		f.write(content if isinstance(content, bytes) else content.encode())


@pytest.fixture
def work(tmp_path):
	"""work/src/a/x.txt ('one'), work/src/b/x.txt ('two'), work/src/a/deep/y.log ('three'); commands run in work."""
	write(tmp_path / 'src/a/x.txt', 'one')
	write(tmp_path / 'src/b/x.txt', 'two')
	write(tmp_path / 'src/a/deep/y.log', 'three')
	return tmp_path


TREE = {b'src/a/x.txt': b'one', b'src/b/x.txt': b'two', b'src/a/deep/y.log': b'three'}
FILES = b'src/a/x.txt\nsrc/b/x.txt\nsrc/a/deep/y.log\n'


# ---- 1: paths from a list keep their tree ----

def test_implicit_stdin_keeps_the_tree(work):
	rc, _ = run(['-d', 'out'], work, FILES)
	assert rc == 0 and files(work / 'out') == TREE


def test_explicit_stdin_keeps_the_tree(work):
	rc, _ = run(['-sfl', '-', '-d', 'out'], work, FILES)
	assert rc == 0 and files(work / 'out') == TREE


def test_list_file_keeps_the_tree(work):
	(work / 'list.txt').write_bytes(FILES)
	rc, _ = run(['-sfl', 'list.txt', '-d', 'out'], work)
	assert rc == 0 and files(work / 'out') == TREE


def test_real_find_from_inside_the_source_leaves_it_out(work):
	listing = subprocess.run(['find', '.', '-type', 'f'], cwd=str(work / 'src'), stdout=subprocess.PIPE).stdout
	rc, _ = run(['-d', str(work / 'out')], work / 'src', listing)
	assert rc == 0 and files(work / 'out') == {b'a/x.txt': b'one', b'b/x.txt': b'two', b'a/deep/y.log': b'three'}


def test_absolute_path_lands_without_its_leading_slash(work):
	entry = str(work / 'src/a/x.txt')
	rc, _ = run(['-d', 'out'], work, entry.encode() + b'\n')
	assert rc == 0 and files(work / 'out') == {os.fsencode(entry.lstrip('/')): b'one'}


@pytest.mark.parametrize('entry, landing', [
	(b'../x/a', b'x/a'),
	(b'../../top/x/a', b'top/x/a'),
	(b'./../x/./a', b'x/a'),
	(b'sub/../../x/a', b'x/a'),
])
def test_dotdot_never_leaves_the_destination(tmp_path, entry, landing):
	write(tmp_path / 'top/x/a', 'data')
	(tmp_path / 'top/cwd/sub').mkdir(parents=True)
	rc, output = run(['-d', str(tmp_path / 'out')], tmp_path / 'top/cwd', entry + b'\n')
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'out') == {landing: b'data'}
	assert sorted(os.listdir(str(tmp_path))) == ['out', 'top']
	assert sorted(os.listdir(str(tmp_path / 'top'))) == ['cwd', 'x']


def test_landing_is_normalised(hpcp_mod):
	landing = hpcp_mod._list_landing
	assert landing('/a/b') == 'a/b' and landing('./a//b/') == 'a/b' and landing('//a') == 'a'
	assert landing('../x/a') == 'x/a' and landing('a/../../b') == 'b'
	assert landing('.') == landing('/') == landing('..') == landing('../..') == ''
	assert landing('..a/b...') == '..a/b...'


# ---- 2: a list always means the destination is a directory ----

def test_one_entry_list_with_a_missing_destination(work):
	rc, _ = run(['-d', 'out'], work, b'src/a/x.txt\n')
	assert rc == 0 and (work / 'out').is_dir() and files(work / 'out') == {b'src/a/x.txt': b'one'}


def test_one_entry_directory_list_lands_at_its_path(work):
	rc, _ = run(['-d', 'out'], work, b'src/b\n')
	assert rc == 0 and files(work / 'out') == {b'src/b/x.txt': b'two'}


def test_destination_that_is_a_file_fails_and_stays(work):
	write(work / 'out', 'mine')
	rc, _ = run(['-d', 'out'], work, b'src/a/x.txt\n')
	assert rc != 0 and (work / 'out').read_text() == 'mine'


# ---- 3: directories in a list ----

def test_directory_in_a_list_copies_recursively(work):
	rc, _ = run(['-d', 'out'], work, b'src/a\nsrc/b/\n')
	assert rc == 0 and files(work / 'out') == TREE


def test_plain_find_gives_a_complete_tree_copying_nothing_twice(work):
	(work / 'src/empty').mkdir()
	os.symlink('a/x.txt', str(work / 'src/link'))
	listing = subprocess.run(['find', 'src'], cwd=str(work), stdout=subprocess.PIPE).stdout
	rc, output = run(['-d', 'out'], work, listing)
	assert rc == 0
	assert files(work / 'out') == {b'src/link': b'-> a/x.txt', b'src/a/x.txt': b'one', b'src/b/x.txt': b'two', b'src/a/deep/y.log': b'three'}
	assert (work / 'out/src/empty').is_dir()
	# 'src' covers every other entry: one source.
	assert b"Copying from ['src'] to" in output


def test_find_dot_with_directories(work):
	listing = subprocess.run(['find', '.'], cwd=str(work / 'src'), stdout=subprocess.PIPE).stdout
	rc, _ = run(['-d', str(work / 'out')], work / 'src', listing)
	assert rc == 0 and files(work / 'out') == {b'a/x.txt': b'one', b'b/x.txt': b'two', b'a/deep/y.log': b'three'}


def test_top_level_find_copies_directories_to_their_paths(work):
	listing = subprocess.run(['find', 'src', '-mindepth', '1', '-maxdepth', '1'], cwd=str(work), stdout=subprocess.PIPE).stdout
	rc, _ = run(['-d', 'out'], work, listing)
	assert rc == 0 and files(work / 'out') == TREE


def test_covered_entries_go_and_repeats_too(tmp_path, hpcp_mod, monkeypatch):
	(tmp_path / 'd/e').mkdir(parents=True)
	(tmp_path / 'g').mkdir()
	(tmp_path / 'd/e/f').write_text('x')
	(tmp_path / 'g/h').write_text('x')
	os.symlink('d', str(tmp_path / 'ln'))
	monkeypatch.chdir(tmp_path)
	(tmp_path / 'list').write_bytes(b'd/e/f\nd/e\n./d/e/\ng/h\ng/h\nln\nln/e\n')
	kept = hpcp_mod._without_covered_sources(hpcp_mod.load_source_list('list'))
	# A later ancestor covers an earlier entry; a repeat goes whatever its spelling; a symlink covers nothing, but one
	# with entries below it (find -L) goes: they land in a directory of its name.
	assert kept == ['d/e', 'g/h', 'ln/e']
	(tmp_path / 'list').write_bytes(b'ln\n')
	assert hpcp_mod._without_covered_sources(hpcp_mod.load_source_list('list')) == ['ln']


def test_a_positional_directory_does_not_cover_a_list_entry(work):
	rc, _ = run(['src/b', '-sfl', '-', '-d', 'out'], work, b'src/b/x.txt\n')
	assert rc == 0 and files(work / 'out') == {b'b/x.txt': b'two', b'src/b/x.txt': b'two'}


def test_parent_directories_take_their_sources_metadata(work):
	# as cp --parents: a private directory stays private
	os.chmod(str(work / 'src/a'), 0o700)
	os.chmod(str(work / 'src'), 0o750)
	rc, _ = run(['-d', 'out'], work, b'src/a/x.txt\n')
	assert rc == 0
	assert stat.S_IMODE(os.stat(str(work / 'out/src/a')).st_mode) == 0o700
	assert stat.S_IMODE(os.stat(str(work / 'out/src')).st_mode) == 0o750


def test_no_directory_sync_leaves_parents_at_defaults(work):
	os.chmod(str(work / 'src/a'), 0o700)
	rc, _ = run(['-d', 'out', '-nds'], work, b'src/a/x.txt\n')
	assert rc == 0 and stat.S_IMODE(os.stat(str(work / 'out/src/a')).st_mode) != 0o700
	assert (work / 'out/src/a/x.txt').read_text() == 'one'


def test_dangling_symlink_in_a_list_is_copied(work):
	os.symlink('nowhere', str(work / 'src/dead'))
	rc, _ = run(['-d', 'out'], work, b'src/dead\nsrc/a/x.txt\n')
	assert rc == 0 and files(work / 'out') == {b'src/dead': b'-> nowhere', b'src/a/x.txt': b'one'}


def test_symlink_to_a_directory_in_a_list_is_copied_as_a_link(work):
	os.symlink('a', str(work / 'src/ln'))
	rc, _ = run(['-d', 'out'], work, b'src/ln\n')
	assert rc == 0 and os.readlink(str(work / 'out/src/ln')) == 'a'


# ---- 4: positional arguments keep today's layout ----

def test_positional_files_stay_flat(work):
	rc, _ = run(['src/a/x.txt', 'src/a/deep/y.log', '-d', 'out'], work)
	assert rc == 0 and files(work / 'out') == {b'x.txt': b'one', b'y.log': b'three'}


def test_positional_directory_copies_its_contents(work):
	rc, _ = run(['src', '-d', 'out'], work)
	assert rc == 0 and files(work / 'out') == {b'a/x.txt': b'one', b'b/x.txt': b'two', b'a/deep/y.log': b'three'}


def test_positional_one_file_to_a_missing_destination_names_it(work):
	rc, _ = run(['src/a/x.txt', '-d', 'out'], work)
	assert rc == 0 and (work / 'out').read_text() == 'one'


def test_each_source_follows_its_own_rule(work):
	rc, _ = run(['src/b/x.txt', '-sfl', '-', '-d', 'out'], work, b'src/a/x.txt\n')
	assert rc == 0 and files(work / 'out') == {b'x.txt': b'two', b'src/a/x.txt': b'one'}


# ---- 5: same-name clashes ----

@pytest.mark.parametrize('flags', [[], ['-s'], ['-nb'], ['-s', '-nb']], ids=['parallel', 'serial', 'sequential', 'serial-sequential'])
def test_positional_files_clash_first_wins(work, flags):
	write(work / 'src/c.txt', 'other')
	rc, output = run(['src/a/x.txt', 'src/b/x.txt', 'src/c.txt', '-d', 'out'] + flags, work)
	assert rc == 213
	assert files(work / 'out') == {b'x.txt': b'one', b'c.txt': b'other'}
	assert b'Destination clash' in output and b'src/b/x.txt' in output


def test_order_decides_which_file_wins(work):
	rc, _ = run(['src/b/x.txt', 'src/a/x.txt', '-d', 'out'], work)
	assert rc == 213 and (work / 'out/x.txt').read_text() == 'two'


@pytest.mark.parametrize('flags', [[], ['-s']], ids=['parallel', 'serial'])
def test_list_entries_landing_on_one_path_clash_first_wins(tmp_path, flags):
	cwd = tmp_path / 'top/cwd'
	write(cwd / 'x/a', 'inside')
	write(tmp_path / 'top/x/a', 'outside')
	write(cwd / 'y', 'rest')
	rc, output = run(['-d', str(tmp_path / 'out')] + flags, cwd, b'../x/a\nx/a\ny\n')
	assert rc == 213 and b'Destination clash' in output
	assert files(tmp_path / 'out') == {b'x/a': b'outside', b'y': b'rest'}
	(tmp_path / 'out/x/a').unlink()
	rc, _ = run(['-d', str(tmp_path / 'out')] + flags, cwd, b'x/a\n../x/a\ny\n')
	assert rc == 213 and (tmp_path / 'out/x/a').read_text() == 'inside'


MODES = [[], ['-s'], ['-nb'], ['-s', '-nb']]
MODE_IDS = ['parallel', 'serial', 'sequential', 'serial-sequential']


def _overlay(tmp_path, count=400):
	for index in range(count):
		write(tmp_path / ('base/d%d/f%d' % (index % 7, index)), 'base')
		write(tmp_path / ('overlay/d%d/f%d' % (index % 7, index)), 'overlay')
		write(tmp_path / ('base/only_base/f%d' % index), 'base')
		write(tmp_path / ('overlay/only_overlay/f%d' % index), 'overlay')


@pytest.mark.parametrize('flags', MODES, ids=MODE_IDS)
def test_merged_directories_later_source_wins_in_every_mode(tmp_path, flags):
	# Enough files that the parallel path runs many jobs at once: 'later' cannot depend on who finishes last.
	count = 400 if not flags else 60
	_overlay(tmp_path, count)
	rc, output = run(['base/', 'overlay/', '-d', 'out'] + flags, tmp_path)
	assert rc == 0, output[-2000:]
	got = files(tmp_path / 'out')
	assert len(got) == 3 * count
	assert {content for path, content in got.items() if path.startswith(b'd')} == {b'overlay'}
	assert b'Destination clash' not in output
	assert output.count(b'Merged directories: %d file(s) replaced by a later source' % count) == 1


@pytest.mark.parametrize('flags', MODES, ids=MODE_IDS)
def test_merged_directories_order_decides(tmp_path, flags):
	_overlay(tmp_path, 40)
	rc, _ = run(['overlay/', 'base/', '-d', 'out'] + flags, tmp_path)
	assert rc == 0
	assert {content for path, content in files(tmp_path / 'out').items() if path.startswith(b'd')} == {b'base'}


@pytest.mark.parametrize('flags', MODES, ids=MODE_IDS)
def test_same_named_directories_merge_with_the_later_winning(tmp_path, flags):
	write(tmp_path / 'a/conf/same', 'from a')
	write(tmp_path / 'a/conf/only_a', 'A')
	write(tmp_path / 'b/conf/same', 'from b')
	write(tmp_path / 'b/conf/only_b', 'B')
	rc, output = run(['a/conf', 'b/conf', '-d', 'out'] + flags, tmp_path)
	assert rc == 0 and b'Destination clash' not in output
	assert files(tmp_path / 'out') == {b'conf/same': b'from b', b'conf/only_a': b'A', b'conf/only_b': b'B'}
	assert b'1 file(s) replaced' in output


@pytest.mark.parametrize('flags', MODES, ids=MODE_IDS)
def test_three_directories_the_last_wins_and_all_count(tmp_path, flags):
	for name in 'abc':
		write(tmp_path / name / 'x', name)
	rc, output = run(['a/', 'b/', 'c/', '-d', 'out'] + flags, tmp_path)
	assert rc == 0 and files(tmp_path / 'out') == {b'x': b'c'} and b'2 file(s) replaced' in output


@pytest.mark.parametrize('flags', MODES, ids=MODE_IDS)
def test_symlinks_in_merged_directories_follow_the_same_rule(tmp_path, flags):
	write(tmp_path / 'a/ln1', 'file')
	(tmp_path / 'b').mkdir()
	os.symlink('target', str(tmp_path / 'b/ln1'))
	os.symlink('target', str(tmp_path / 'a/ln2'))
	write(tmp_path / 'b/ln2', 'file')
	os.symlink('first', str(tmp_path / 'a/ln3'))
	os.symlink('second', str(tmp_path / 'b/ln3'))
	rc, output = run(['a/', 'b/', '-d', 'out'] + flags, tmp_path)
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'out') == {b'ln1': b'-> target', b'ln2': b'file', b'ln3': b'-> second'}


@pytest.mark.parametrize('flags', MODES, ids=MODE_IDS)
def test_listed_directories_merge_with_the_later_winning(tmp_path, flags):
	cwd = tmp_path / 'top/cwd'
	write(cwd / 'x/same', 'inside')
	write(tmp_path / 'top/x/same', 'outside')
	rc, output = run(['-d', str(tmp_path / 'out')] + flags, cwd, b'../x\nx\n')
	assert rc == 0 and files(tmp_path / 'out') == {b'x/same': b'inside'} and b'1 file(s) replaced' in output


@pytest.mark.parametrize('flags', MODES, ids=MODE_IDS)
def test_a_directly_named_file_and_a_directorys_file_clash_either_way(tmp_path, flags):
	write(tmp_path / 'd/x', 'from dir')
	write(tmp_path / 'd/other', 'other')
	write(tmp_path / 'f/x', 'named')
	rc, output = run(['d/', 'f/x', '-d', 'o1'] + flags, tmp_path)
	assert rc == 213 and b'Destination clash' in output
	assert files(tmp_path / 'o1') == {b'x': b'from dir', b'other': b'other'}
	rc, output = run(['f/x', 'd/', '-d', 'o2'] + flags, tmp_path)
	assert rc == 213 and b'Destination clash' in output
	assert files(tmp_path / 'o2') == {b'x': b'named', b'other': b'other'}


@pytest.mark.parametrize('flags', MODES, ids=MODE_IDS)
def test_a_move_of_merged_directories_lands_the_later_file_and_removes_nothing(tmp_path, flags):
	write(tmp_path / 'a/x', 'early')
	write(tmp_path / 'a/only_a', 'A')
	write(tmp_path / 'b/x', 'late')
	rc, output = run(['a/', 'b/', '-d', 'out', '-rf', '-rm'] + flags, tmp_path)
	assert rc == 212 and b'Remove skipped' in output and b'only copy' in output
	assert files(tmp_path / 'out') == {b'x': b'late', b'only_a': b'A'}
	assert (tmp_path / 'a/x').read_text() == 'early' and (tmp_path / 'b/x').read_text() == 'late'


def test_a_move_of_directories_that_share_no_file_removes_them(tmp_path):
	write(tmp_path / 'a/x', 'A')
	write(tmp_path / 'b/y', 'B')
	rc, output = run(['a/', 'b/', '-d', 'out', '-rf', '-rm'], tmp_path)
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'out') == {b'x': b'A', b'y': b'B'}
	assert not (tmp_path / 'a/x').exists() and not (tmp_path / 'b/y').exists()


@pytest.mark.parametrize('sequential', [False, True], ids=['batch', 'sequential'])
def test_rme_goes_on_after_a_merge_win(tmp_path, hpcp_mod, reset_hpcp_globals, at_a_tty, monkeypatch, sequential):
	monkeypatch.chdir(tmp_path)
	write(tmp_path / 'a/x', 'early')
	write(tmp_path / 'b/x', 'late')
	write(tmp_path / 'out/stale', 'gone')
	assert hpcp_mod.hpcp(['a/', 'b/'], dest_paths=[str(tmp_path / 'out')], single_thread=True, batch=not sequential, remove_extra=True) == 0
	assert files(tmp_path / 'out') == {b'x': b'late'}


def test_a_file_loses_to_a_directory_before_it(tmp_path):
	write(tmp_path / 'a/x', 'from dir')
	write(tmp_path / 'b/a/x', 'from file')
	rc, _ = run(['a', 'b/a/x', '-d', 'out/'], tmp_path)
	# 'a' lands at out/a, the file at out/x: no clash.
	assert rc == 0
	(tmp_path / 'c').mkdir()
	write(tmp_path / 'c/x', 'third')
	rc, _ = run(['a/', 'c/x', '-d', 'out2/'], tmp_path)
	assert rc == 213 and (tmp_path / 'out2/x').read_text() == 'from dir'


def test_symlinks_clash_like_files(tmp_path):
	(tmp_path / 'a').mkdir()
	(tmp_path / 'b').mkdir()
	os.symlink('first', str(tmp_path / 'a/ln'))
	os.symlink('second', str(tmp_path / 'b/ln'))
	rc, output = run(['a/ln', 'b/ln', '-d', 'out'], tmp_path)
	assert rc == 213 and b'Destination clash' in output
	assert os.readlink(str(tmp_path / 'out/ln')) == 'first'


def test_a_file_before_a_symlink_keeps_its_place(tmp_path):
	write(tmp_path / 'a/ln', 'file')
	(tmp_path / 'b').mkdir()
	os.symlink('second', str(tmp_path / 'b/ln'))
	rc, _ = run(['a/ln', 'b/ln', '-d', 'out'], tmp_path)
	assert rc == 213 and (tmp_path / 'out/ln').read_text() == 'file' and not (tmp_path / 'out/ln').is_symlink()


def test_the_same_source_twice_is_no_clash(work):
	rc, output = run(['src/a/x.txt', 'src/a/x.txt', '-d', 'out'], work)
	assert rc == 0 and b'Destination clash' not in output and (work / 'out/x.txt').read_text() == 'one'


def test_identical_content_still_clashes_as_cp_does(tmp_path):
	write(tmp_path / 'a/x', 'same')
	write(tmp_path / 'b/x', 'same')
	rc, _ = run(['a/x', 'b/x', '-d', 'out'], tmp_path)
	assert rc == 213


def test_a_move_with_a_clash_removes_no_source(work):
	rc, output = run(['src/a/x.txt', 'src/b/x.txt', '-d', 'out', '-rf', '-rm'], work)
	assert rc == 213 and b'Remove skipped' in output
	assert (work / 'src/a/x.txt').exists() and (work / 'src/b/x.txt').exists()


def test_a_missing_name_among_good_ones_is_still_207(work):
	rc, output = run(['-d', 'out'], work, b'src/a/x.txt\nnope\nsrc/b/x.txt\n')
	assert rc == 207 and b'nope' in output and files(work / 'out') == {b'src/a/x.txt': b'one', b'src/b/x.txt': b'two'}


def test_empty_stdin_has_no_sources(work):
	rc, _ = run(['-d', 'out'], work, b'')
	assert rc == 240 and not (work / 'out').exists()


# ---- 6: parsing ----

def parsed(hpcp_mod, tmp_path, data):
	path = tmp_path / 'the-list'
	path.write_bytes(data)
	return hpcp_mod.load_source_list(str(path))


def test_list_order_is_kept(hpcp_mod, tmp_path):
	assert parsed(hpcp_mod, tmp_path, b'z\na\nm\nb\n') == ['z', 'a', 'm', 'b']


def test_nul_separated_names_are_exact(hpcp_mod, tmp_path):
	assert parsed(hpcp_mod, tmp_path, b'a b \0 lead\0new\nline\0\0last') == ['a b ', ' lead', 'new\nline', 'last']


def test_lines_split_on_newline_only(hpcp_mod, tmp_path):
	# Not str.splitlines: \v, \f, \x85, U+2028 are parts of names.
	assert parsed(hpcp_mod, tmp_path, b'a\x0bb\nc\x0cd\ne\xe2\x80\xa8f\n') == ['a\x0bb', 'c\x0cd', 'e f']


def test_crlf_lines_drop_the_cr(hpcp_mod, tmp_path):
	assert parsed(hpcp_mod, tmp_path, b'a\r\nb\n\r\nc\r\n') == ['a', 'b', 'c']


def test_surrounding_whitespace_exact_name_first(hpcp_mod, tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	(tmp_path / 'trail ').write_text('x')
	(tmp_path / 'plain').write_text('x')
	assert parsed(hpcp_mod, tmp_path, b'trail \n plain \n   \n') == ['trail ', 'plain']


def test_any_bytes_decode_under_any_locale(hpcp_mod, tmp_path):
	assert parsed(hpcp_mod, tmp_path, b'latin\xe9\n') == [os.fsdecode(b'latin\xe9')]


def test_stored_lists_are_still_sets_of_stripped_lines(hpcp_mod, tmp_path):
	path = tmp_path / 'stored'
	path.write_bytes(b'./\n a.txt:abc \n\nlatin\xe9\n')
	assert hpcp_mod.load_file_list(str(path)) == frozenset(['./', 'a.txt:abc', os.fsdecode(b'latin\xe9')])


def test_list_bytes_the_filesystem_encoding_refuses_use_the_locale(hpcp_mod, tmp_path, monkeypatch):
	# Windows: os.fsdecode is strict UTF-8 there, and a list from other tools is in the locale's encoding. Windows itself
	# is not run: the decoder is stood in for.
	real = os.fsdecode
	def strict(name):
		return name.decode('utf-8') if isinstance(name, bytes) else real(name)
	monkeypatch.setattr(hpcp_mod.os, 'fsdecode', strict)
	monkeypatch.setattr(hpcp_mod.locale, 'getpreferredencoding', lambda do_setlocale=True: 'cp1252')
	assert parsed(hpcp_mod, tmp_path, b'caf\xe9\nplain\n') == ['caf\xe9', 'plain']


def test_missing_list_file_reports(work):
	rc, output = run(['-sfl', 'nope.txt', '-d', 'out'], work)
	assert rc == 240 and b'File list not found' in output


# ---- 7: odd names end to end ----

ODD_NAMES = [b'latin\xe9', b'trail ', b' lead', b'ls\xe2\x80\xa8x', b'new\nline', b'star*[x]?', b'colon:name', b'cr\rname']

@pytest.mark.parametrize('env', [{}, {'PYTHONIOENCODING': 'utf-8:strict'}, {'LC_ALL': 'C'}, {'LC_ALL': 'C', 'PYTHONUTF8': '0', 'PYTHONCOERCECLOCALE': '0'}],
	ids=['default', 'strict', 'c-locale', 'c-locale-plain'])
@pytest.mark.parametrize('flags', [[], ['-s'], ['-v']], ids=['parallel', 'serial', 'verbose'])
def test_odd_names_copy_and_print_end_to_end(tmp_path, env, flags):
	for name in ODD_NAMES:
		write(os.path.join(os.fsencode(str(tmp_path)), b's', name), name)
	listing = subprocess.run(['find', 's', '-type', 'f', '-print0'], cwd=str(tmp_path), stdout=subprocess.PIPE).stdout
	rc, output = run(['-d', 'out'] + flags, tmp_path, listing, env)
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'out') == {os.path.join(b's', name): name for name in ODD_NAMES}
	assert b'Traceback' not in output


@pytest.mark.parametrize('name', [b'latin\xe9', b'trail ', b'ls\xe2\x80\xa8x', b'cr\rname'], ids=['latin1', 'trailing-space', 'u2028', 'inner-cr'])
def test_odd_names_work_on_newline_lists_too(tmp_path, name):
	write(os.path.join(os.fsencode(str(tmp_path)), b's', name), name)
	rc, output = run(['-d', 'out'], tmp_path, b's/' + name + b'\n', {'PYTHONIOENCODING': 'utf-8:strict'})
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'out') == {b's/' + name: name}


def test_ascii_stdout_takes_what_it_cannot_show(hpcp_mod, monkeypatch):
	# Python 3.6 in the C locale: ASCII, surrogateescape. The progress bar's block characters ended the copy (rc 70).
	buffer = io.BytesIO()
	monkeypatch.setattr(sys, 'stdout', io.TextIOWrapper(buffer, encoding='ascii', errors='surrogateescape'))
	monkeypatch.setattr(hpcp_mod, '_STRICT_STDOUT', None)
	hpcp_mod._escape_unencodable_stdout()
	print('\u2582 ' + os.fsdecode(b'latin\xe9'))
	sys.stdout.flush()
	assert b'\\u2582' in buffer.getvalue()


def test_missing_odd_name_is_reported_not_a_crash(tmp_path):
	write(tmp_path / 'good', 'x')
	rc, output = run(['-d', 'out'], tmp_path, b'gone\xe9\ngood\n', {'PYTHONIOENCODING': 'utf-8:strict'})
	assert rc == 207 and b'Traceback' not in output and (tmp_path / 'out/good').exists()


def test_clash_with_odd_names_is_reported_not_a_crash(tmp_path):
	base = os.fsencode(str(tmp_path))
	write(os.path.join(base, b'a', b'latin\xe9'), 'one')
	write(os.path.join(base, b'b', b'latin\xe9'), 'two')
	rc, output = run(['-d', 'out', '-sfl', '-'], tmp_path, b'a\n', {'PYTHONIOENCODING': 'utf-8:strict'})
	assert rc == 0
	task = subprocess.run([sys.executable, HPCP, os.path.join(b'a', b'latin\xe9').decode('utf-8', 'surrogateescape'),
		os.path.join(b'b', b'latin\xe9').decode('utf-8', 'surrogateescape'), '-d', 'o2'], cwd=str(tmp_path),
		stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONIOENCODING='utf-8:strict'), timeout=120)
	assert task.returncode == 213 and b'Traceback' not in task.stdout


# ---- modes: -rme ----

@pytest.fixture
def at_a_tty(monkeypatch):
	"""-rme asks first: answer yes, as an operator at a terminal."""
	monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
	monkeypatch.setattr(builtins, 'input', lambda *a: 'y')


def hpcp_list(hpcp_mod, tmp_path, entries, dest='out', positional=(), **options):
	(tmp_path / 'the-list').write_bytes(entries)
	return hpcp_mod.hpcp(list(positional), dest_paths=[str(tmp_path / dest)] if dest else [], source_file_list=str(tmp_path / 'the-list'), single_thread=True, **options)


def test_rme_with_listed_files_deletes_nothing(work, hpcp_mod, reset_hpcp_globals, at_a_tty, monkeypatch):
	monkeypatch.chdir(work)
	write(work / 'out/x.txt', 'an earlier flat copy')
	write(work / 'out/other/keep', 'not from this list')
	assert hpcp_list(hpcp_mod, work, b'src/a/x.txt\n', remove_extra=True) == 0
	assert files(work / 'out') == {b'x.txt': b'an earlier flat copy', b'other/keep': b'not from this list', b'src/a/x.txt': b'one'}


def test_rme_cleans_only_below_listed_directories(work, hpcp_mod, reset_hpcp_globals, at_a_tty, monkeypatch):
	monkeypatch.chdir(work)
	write(work / 'out/src/a/stale', 'gone')
	write(work / 'out/src/a/stale_dir/f', 'gone')
	write(work / 'out/src/b/not_listed', 'stays')
	write(work / 'out/other/keep', 'stays')
	write(work / 'out/flat.txt', 'stays')
	assert hpcp_list(hpcp_mod, work, b'src/a\n', remove_extra=True) == 0
	assert files(work / 'out') == {b'src/a/x.txt': b'one', b'src/a/deep/y.log': b'three', b'src/b/not_listed': b'stays', b'other/keep': b'stays', b'flat.txt': b'stays'}
	assert not (work / 'out/src/a/stale_dir').exists()
	assert (work / 'out/other').is_dir()


def test_rme_listed_directory_keeps_a_listed_file_nested_in_its_copy(tmp_path, hpcp_mod, reset_hpcp_globals, at_a_tty, monkeypatch):
	# '../x' and 'x/y' share out/x: each is its own source, and what either brought is not extra.
	cwd = tmp_path / 'top/cwd'
	write(tmp_path / 'top/x/own', 'A')
	write(cwd / 'x/y/f', 'B')
	monkeypatch.chdir(cwd)
	write(tmp_path / 'out/x/stale', 'gone')
	(tmp_path / 'top/cwd/the-list').write_bytes(b'../x\nx/y/f\n')
	assert hpcp_mod.hpcp([], dest_paths=[str(tmp_path / 'out')], source_file_list='the-list', single_thread=True, remove_extra=True) == 0
	assert files(tmp_path / 'out') == {b'x/own': b'A', b'x/y/f': b'B'}


def test_rme_with_a_positional_too_stays_scoped(work, hpcp_mod, reset_hpcp_globals, at_a_tty, monkeypatch):
	monkeypatch.chdir(work)
	write(work / 'out/b/stale', 'gone')
	write(work / 'out/keep', 'stays')
	assert hpcp_list(hpcp_mod, work, b'src/a/x.txt\n', positional=['src/b'], remove_extra=True) == 0
	assert files(work / 'out') == {b'b/x.txt': b'two', b'src/a/x.txt': b'one', b'keep': b'stays'}


def test_rme_with_the_list_on_stdin_and_no_terminal_refuses_saying_why(work):
	write(work / 'out/src/a/stale', 'stays')
	rc, output = run(['-d', 'out', '-rme'], work, b'src/a\n')
	assert rc == 147 and (work / 'out/src/a/stale').exists() and b'no terminal can be opened' in output


def test_rme_with_a_list_of_files_says_it_checked_nothing(work, hpcp_mod, reset_hpcp_globals, at_a_tty, monkeypatch, capsys):
	monkeypatch.chdir(work)
	write(work / 'out/keep', 'stays')
	assert hpcp_list(hpcp_mod, work, b'src/a/x.txt\n', remove_extra=True) == 0
	output = capsys.readouterr().out
	assert output.count('-rme checked nothing') == 1 and 'printf "src\\n" | hpcp -d out -rme' in output
	assert (work / 'out/keep').exists()


def test_rme_says_nothing_of_the_kind_when_a_directory_is_named(work, hpcp_mod, reset_hpcp_globals, at_a_tty, monkeypatch, capsys):
	monkeypatch.chdir(work)
	assert hpcp_list(hpcp_mod, work, b'src/a\nsrc/b/x.txt\n', remove_extra=True) == 0
	assert '-rme checked nothing' not in capsys.readouterr().out


def test_rme_says_nothing_of_the_kind_after_a_failed_copy(work, hpcp_mod, reset_hpcp_globals, at_a_tty, monkeypatch, capsys):
	monkeypatch.chdir(work)
	write(work / 'other/a/x.txt', 'zzz')
	(work / 'the-list').write_bytes(b'src/a/x.txt\n')
	rc = hpcp_mod.hpcp(['src/b/x.txt', 'other/a/x.txt'], dest_paths=[str(work / 'out')], source_file_list='the-list', single_thread=True, remove_extra=True)
	assert rc != 0 and '-rme checked nothing' not in capsys.readouterr().out


def test_rme_says_nothing_of_the_kind_without_a_list(work, hpcp_mod, reset_hpcp_globals, at_a_tty, monkeypatch, capsys):
	monkeypatch.chdir(work)
	assert hpcp_mod.hpcp(['src/a/'], dest_paths=[str(work / 'out') + os.sep], single_thread=True, remove_extra=True) == 0
	assert '-rme checked nothing' not in capsys.readouterr().out


def test_rme_positional_alone_still_judges_the_whole_destination(work, hpcp_mod, reset_hpcp_globals, at_a_tty, monkeypatch):
	monkeypatch.chdir(work)
	write(work / 'out/stale', 'gone')
	write(work / 'out/stale_dir/f', 'gone')
	assert hpcp_mod.hpcp(['src'], dest_paths=[str(work / 'out')], single_thread=True, remove_extra=True) == 0
	assert files(work / 'out') == {b'a/x.txt': b'one', b'b/x.txt': b'two', b'a/deep/y.log': b'three'}
	assert not (work / 'out/stale_dir').exists()


def test_rme_after_a_clash_deletes_nothing(work, hpcp_mod, reset_hpcp_globals, at_a_tty, monkeypatch):
	monkeypatch.chdir(work)
	write(work / 'out/src/stale', 'stays')
	(work / 'other/a').mkdir(parents=True)
	write(work / 'other/a/x.txt', 'zzz')
	rc = hpcp_mod.hpcp(['src/a/x.txt', 'other/a/x.txt'], dest_paths=[str(work / 'out')], single_thread=True, remove_extra=True)
	assert rc != 0 and (work / 'out/src/stale').exists()


def test_rme_listed_dir_with_exclude_keeps_the_excluded_copy(work, hpcp_mod, reset_hpcp_globals, at_a_tty, monkeypatch):
	monkeypatch.chdir(work)
	write(work / 'out/src/a/deep/old', 'stays')
	write(work / 'out/src/a/stale', 'gone')
	assert hpcp_list(hpcp_mod, work, b'src/a\n', remove_extra=True, exclude=['*/deep']) == 0
	got = files(work / 'out')
	assert b'src/a/deep/old' in got and b'src/a/stale' not in got and b'src/a/x.txt' in got


def test_expected_paths_are_linear_in_the_sources(hpcp_mod, tmp_path, reset_hpcp_globals):
	sources = ['/s/d%d/f%05d' % (i % 100, i) for i in range(30000)]
	start = time.process_time()
	got = hpcp_mod._expected_dest_paths(sources, sources, ['/o/'])
	assert time.process_time() - start < 3
	assert len(got) == 30000 and '/o/f00007' in got


def test_nested_sources_each_keep_their_copy(work, hpcp_mod, reset_hpcp_globals, at_a_tty, monkeypatch):
	# Both copied a/deep/y.log, each to its own place: neither copy is extra.
	monkeypatch.chdir(work)
	assert hpcp_mod.hpcp(['src/a', 'src/a/deep'], dest_paths=[str(work / 'out')], single_thread=True, remove_extra=True) == 0
	assert (work / 'out/a/deep/y.log').exists() and (work / 'out/deep/y.log').exists()


# ---- modes: -rm ----

def test_rm_with_a_list_removes_the_listed_paths(work):
	(work / 'keep').write_text('stays')
	rc, _ = run(['-rf', '-rm'], work, b'src/a\nsrc/b/x.txt\n')
	assert rc == 0
	assert not (work / 'src/a').exists() and not (work / 'src/b/x.txt').exists()
	assert (work / 'src/b').is_dir() and (work / 'keep').exists()


def test_rm_with_a_nul_list_removes_exact_names(tmp_path):
	base = os.fsencode(str(tmp_path))
	write(os.path.join(base, b'latin\xe9'), 'x')
	write(os.path.join(base, b'trail '), 'x')
	write(os.path.join(base, b'trail'), 'stays')
	rc, output = run(['-rf', '-rm'], tmp_path, b'latin\xe9\0trail \0', {'PYTHONIOENCODING': 'utf-8:strict'})
	assert rc == 0, output[-2000:]
	assert sorted(os.listdir(base)) == [b'trail']


def test_rm_with_a_list_refuses_without_a_terminal_or_rf(work):
	rc, _ = run(['-rm'], work, b'src/a\n')
	assert rc == 147 and (work / 'src/a/x.txt').exists()


def test_rm_of_a_dangling_link_in_a_list(tmp_path):
	os.symlink('nowhere', str(tmp_path / 'dead'))
	rc, _ = run(['-rf', '-rm'], tmp_path, b'dead\n')
	assert rc == 0 and not os.path.lexists(str(tmp_path / 'dead'))


def test_move_with_a_list_lands_the_tree_and_removes_the_sources(work):
	rc, output = run(['-rf', '-rm', '-d', 'out'], work, FILES)
	assert rc == 0, output[-2000:]
	assert files(work / 'out') == TREE
	assert not (work / 'src/a/x.txt').exists() and not (work / 'src/b/x.txt').exists()


def test_move_onto_itself_with_a_list_is_refused(work):
	# A listed directory that holds the destination: moving it would put the destination inside itself.
	rc, output = run(['-rf', '-rm', '-d', 'src/out'], work, b'src\n')
	assert rc == 241 and (work / 'src/a/x.txt').read_text() == 'one' and files(work / 'src/out') == {}


# ---- prompts when stdin held the list (a real terminal) ----

def in_a_terminal(command, cwd, answer):
	"""Run a shell pipeline with a pty as its controlling terminal, its output to a file so that only the prompt reaches
	the terminal or the log. Answer the first '(y/n)'. (exit status, what the terminal showed.)"""
	log = os.path.join(str(cwd), 'log')
	pid, fd = pty.fork()
	if pid == 0:
		os.chdir(str(cwd))
		os.execvp('sh', ['sh', '-c', command + ' >log 2>&1'])
	shown = b''
	answered = False
	deadline = time.monotonic() + 90
	try:
		while time.monotonic() < deadline:
			ready, _, _ = select.select([fd], [], [], 0.2)
			if not ready:
				# A prompt on stdin at the terminal goes to stdout, which is the log here.
				if not answered and os.path.exists(log) and b'(y/n)' in open(log, 'rb').read():
					os.write(fd, answer + b'\n')
					answered = True
				continue
			try:
				chunk = os.read(fd, 4096)
			except OSError:
				break
			if not chunk:
				break
			shown += chunk
			if not answered and b'(y/n)' in shown:
				os.write(fd, answer + b'\n')
				answered = True
		else:
			os.kill(pid, signal.SIGKILL)
			pytest.fail('no end in 90 s; the terminal showed %r' % shown)
	finally:
		os.close(fd)
	return os.WEXITSTATUS(os.waitpid(pid, 0)[1]), shown


def hpcp_command(args):
	return ' '.join(shlex.quote(part) for part in [sys.executable, HPCP] + args)


@pytest.mark.parametrize('answer, kept', [(b'y', False), (b'n', True)], ids=['yes', 'no'])
def test_rme_with_the_list_on_stdin_asks_the_terminal(work, answer, kept):
	write(work / 'out/src/a/stale', 'x')
	rc, shown = in_a_terminal("printf 'src/a\\n' | " + hpcp_command(['-d', 'out', '-rme']), work, answer)
	assert b'delete them? (y/n)' in shown
	assert rc == (147 if kept else 0)
	assert (work / 'out/src/a/stale').exists() == kept
	assert (work / 'out/src/a/x.txt').exists()


@pytest.mark.parametrize('answer, kept', [(b'y', False), (b'n', True)], ids=['yes', 'no'])
def test_rm_with_the_list_on_stdin_asks_the_terminal(work, answer, kept):
	rc, shown = in_a_terminal("printf 'src/a\\n' | " + hpcp_command(['-rm']), work, answer)
	assert b'continue? (y/n)' in shown and rc == (147 if kept else 0)
	assert (work / 'src/a/x.txt').exists() == kept


@pytest.mark.parametrize('answer, kept', [(b'y', False), (b'n', True)], ids=['yes', 'no'])
def test_move_with_the_list_on_stdin_asks_the_terminal(work, answer, kept):
	rc, shown = in_a_terminal("printf 'src/a/x.txt\\n' | " + hpcp_command(['-rm', '-d', 'out']), work, answer)
	assert b'continue? (y/n)' in shown and rc == (147 if kept else 0)
	assert (work / 'src/a/x.txt').exists() == kept
	assert (work / 'out/src/a/x.txt').read_text() == 'one'


def test_explicit_dash_on_stdin_asks_the_terminal_too(work):
	write(work / 'out/src/a/stale', 'x')
	rc, shown = in_a_terminal("printf 'src/a\\n' | " + hpcp_command(['-sfl', '-', '-d', 'out', '-rme']), work, b'y')
	assert b'(y/n)' in shown and rc == 0 and not (work / 'out/src/a/stale').exists()


def test_a_list_file_with_stdin_at_the_terminal_asks_as_ever(work):
	(work / 'the-list').write_bytes(b'src/a\n')
	write(work / 'out/src/a/stale', 'x')
	rc, shown = in_a_terminal(hpcp_command(['-sfl', 'the-list', '-d', 'out', '-rme']) + ' </dev/tty', work, b'y')
	assert rc == 0 and not (work / 'out/src/a/stale').exists()


def test_stdin_not_the_list_and_not_a_terminal_refuses_as_ever(work):
	# A terminal exists, but stdin was not the list: no asking it.
	(work / 'the-list').write_bytes(b'src/a\n')
	write(work / 'out/src/a/stale', 'x')
	rc, shown = in_a_terminal(hpcp_command(['-sfl', 'the-list', '-d', 'out', '-rme']) + ' </dev/null', work, b'y')
	assert rc == 147 and b'(y/n)' not in shown and (work / 'out/src/a/stale').exists()
	assert b'stdin is not a tty' in (work / 'log').read_bytes()


# ---- modes: excludes, -do, several destinations, -fl, -cfl, -dfl, hash modes ----

def test_exclude_drops_a_listed_file_without_touching_the_rest(work):
	rc, _ = run(['-d', 'out', '-e', '*/deep/*'], work, FILES)
	assert rc == 0 and files(work / 'out') == {b'src/a/x.txt': b'one', b'src/b/x.txt': b'two'}


def test_exclude_prunes_below_a_directory_that_covers_entries(work):
	listing = subprocess.run(['find', 'src'], cwd=str(work), stdout=subprocess.PIPE).stdout
	rc, _ = run(['-d', 'out', '-e', '*/deep'], work, listing)
	assert rc == 0 and files(work / 'out') == {b'src/a/x.txt': b'one', b'src/b/x.txt': b'two'}


def test_directory_only_makes_the_tree_without_files(work):
	rc, _ = run(['-d', 'out', '-do'], work, b'src/a\n')
	assert rc == 0 and (work / 'out/src/a/deep').is_dir() and files(work / 'out') == {}


def test_several_destinations_spread_list_sources(work):
	rc, _ = run(['-d', 'o1', '-d', 'o2'], work, FILES)
	assert rc == 0
	got = files(work / 'o1')
	got.update(files(work / 'o2'))
	assert got == TREE


@pytest.mark.parametrize('flags', [['-hs', '0'], ['-fh'], ['-co']], ids=['no-hash', 'full-hash', 'content-only'])
def test_hash_modes_skip_or_copy_as_ever(work, flags):
	rc, _ = run(['-d', 'out'] + flags, work, FILES)
	assert rc == 0 and files(work / 'out') == TREE
	rc, _ = run(['-d', 'out'] + flags, work, FILES)
	assert rc == 0 and files(work / 'out') == TREE


def rows_of(path):
	return {tuple(line.split('\t')[:2]) if '\t' in line else line for line in path.read_text().splitlines()}


def test_compare_with_a_destination_keys_a_list_by_its_tree(work, hpcp_mod, reset_hpcp_globals, monkeypatch):
	monkeypatch.chdir(work)
	assert run(['-d', 'out'], work, b'src/a/x.txt\nsrc/b\n')[0] == 0
	write(work / 'src/a/x.txt', 'changed!')
	write(work / 'src/b/new', 'new')
	diff = work / 'diff.txt'
	hpcp_list(hpcp_mod, work, b'src/a/x.txt\nsrc/b\n', compare_file_list=True, diff_file_list=str(diff))
	rows = {(row.op, row.path) for row in hpcp_mod.read_changes(str(diff))}
	assert ('M', 'src/a/x.txt') in rows and ('A', 'src/b/new') in rows
	# './' is the destination itself, which no list entry names.
	assert not any(op == 'D' and path != './' for op, path in rows), rows


def test_stored_list_of_a_list_keeps_its_tree_and_all_clashing_names(work, hpcp_mod, reset_hpcp_globals, monkeypatch):
	monkeypatch.chdir(work)
	stored = work / 'stored.txt'
	(work / 'the-list').write_bytes(b'src/a/x.txt\nsrc/b/x.txt\nsrc/a/deep\n.\n')
	hpcp_mod.hpcp([], source_file_list='the-list', target_file_list=str(stored), append_hash_to_file_list=False, single_thread=True)
	names = stored.read_text().splitlines()
	assert 'src/a/x.txt' in names and 'src/b/x.txt' in names and 'src/a/deep/y.log' in names


def test_stored_list_of_listed_files_keeps_each_name(work, hpcp_mod, reset_hpcp_globals, monkeypatch):
	monkeypatch.chdir(work)
	stored = work / 'stored.txt'
	(work / 'the-list').write_bytes(b'src/a/x.txt\nsrc/b/x.txt\nsrc/a/deep\n')
	hpcp_mod.hpcp([], source_file_list='the-list', target_file_list=str(stored), append_hash_to_file_list=False, single_thread=True)
	assert set(stored.read_text().splitlines()) == {'src/', 'src/a/', 'src/b/', 'src/a/x.txt', 'src/b/x.txt', 'src/a/deep/', 'src/a/deep/y.log'}


def test_stored_list_with_hashes_compares_against_itself(work, hpcp_mod, reset_hpcp_globals, monkeypatch):
	monkeypatch.chdir(work)
	stored = work / 'stored.txt'
	diff = work / 'diff.txt'
	(work / 'the-list').write_bytes(b'src/a/x.txt\nsrc/b\n')
	hpcp_mod.hpcp([], source_file_list='the-list', target_file_list=str(stored), single_thread=True)
	hpcp_mod.hpcp([], source_file_list='the-list', target_file_list=str(stored), compare_file_list=True, diff_file_list=str(diff), single_thread=True)
	assert not hpcp_mod.read_changes(str(diff))


def test_stored_list_holds_a_name_that_is_not_utf8(tmp_path, hpcp_mod):
	base = os.fsencode(str(tmp_path))
	write(os.path.join(base, b's', b'latin\xe9'), 'x')
	rc, output = run(['s/', '-fl', 'stored', '-nhfl'], tmp_path, b'', {'PYTHONIOENCODING': 'utf-8:strict'})
	assert rc == 0, output[-2000:]
	assert b'latin\xe9' in (tmp_path / 'stored').read_bytes()
	rc, output = run(['s/', '-fl', 'stored', '-cfl', '-nhfl'], tmp_path, b'', {'PYTHONIOENCODING': 'utf-8:strict'})
	assert rc == 0, output[-2000:]


def test_stored_list_rme_with_a_listed_source_deletes_nothing(work, hpcp_mod, reset_hpcp_globals, at_a_tty, monkeypatch):
	# The stored names keep the list's tree, so that comparison would take everything for extra.
	monkeypatch.chdir(work)
	stored = work / 'stored.txt'
	(work / 'the-list').write_bytes(b'src/a\n')
	hpcp_mod.hpcp([], source_file_list='the-list', target_file_list=str(stored), append_hash_to_file_list=False, single_thread=True)
	hpcp_mod.hpcp([], source_file_list='the-list', target_file_list=str(stored), remove_extra=True, single_thread=True)
	assert (work / 'src/a/x.txt').exists() and (work / 'src/a/deep/y.log').exists()


def test_stored_list_on_stdin_rme_keeps_a_name_with_surrounding_whitespace(tmp_path):
	# stdin can be read once: both spellings have to come from that one read. Here no terminal: refused, nothing deleted.
	write(tmp_path / 'r/trail ', 'x')
	write(tmp_path / 'r/plain', 'x')
	rc, output = run(['r', '-fl', '-', '-rme'], tmp_path, b'r/\nr/plain\nr/trail \n')
	assert b'No extra files found' in output and sorted(os.listdir(str(tmp_path / 'r'))) == ['plain', 'trail ']


def test_stored_list_rme_keeps_a_name_with_surrounding_whitespace(tmp_path, hpcp_mod, reset_hpcp_globals, at_a_tty, monkeypatch):
	monkeypatch.chdir(tmp_path)
	write(tmp_path / 'r/trail ', 'x')
	write(tmp_path / 'r/plain', 'x')
	write(tmp_path / 'r/stale', 'x')
	(tmp_path / 'stored').write_bytes(b'r/\nr/plain\nr/trail \n')
	hpcp_mod.hpcp(['r'], target_file_list='stored', remove_extra=True, single_thread=True)
	assert sorted(os.listdir('r')) == ['plain', 'trail ']


def test_auto_diff_name_of_a_long_list_is_short(work, hpcp_mod, reset_hpcp_globals, monkeypatch):
	monkeypatch.chdir(work)
	assert run(['-d', 'out'], work, FILES)[0] == 0
	hpcp_list(hpcp_mod, work, FILES, diff_file_list='auto')
	assert [name for name in os.listdir(str(work)) if name.startswith('DIFF_source_list_TO_out_')]


def test_list_output_stays_quiet(tmp_path):
	for index in range(50):
		write(tmp_path / ('s/f%d' % index))
	listing = subprocess.run(['find', 's', '-type', 'f'], cwd=str(tmp_path), stdout=subprocess.PIPE).stdout
	rc, output = run(['-d', 'out'], tmp_path, listing)
	assert rc == 0 and b'Getting file list' not in output and b'not writable' not in output and b's/f7' not in output
	assert b'Copying from 50 sources' in output


def test_help_and_readme_tell_the_layout():
	help_text = subprocess.run([sys.executable, HPCP, '-h'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60).stdout
	assert b'find src -type f | hpcp' in help_text and b'-print0' in help_text
	assert b'cd src && find . | hpcp -d ../out' in b' '.join(help_text.split())
	with open(os.path.join(os.path.dirname(HPCP), 'README.md'), encoding='utf-8') as f:
		readme = f.read()
	assert '## Upgrading to 9.81' in readme and 'find . ' in readme and '-print0' in readme


# ---- fixes after the review of 9.81 ----

def find_in(directory, *args):
	return subprocess.run(['find'] + list(args), cwd=str(directory), stdout=subprocess.PIPE).stdout


# F1: a list entry inside the destination stays where it is.

def test_nightly_archive_run_twice_keeps_each_days_log(tmp_path):
	data = tmp_path / 'data'
	write(data / 'app/app.log', 'day1')
	write(data / 'other.log', 'other')
	for day in (b'day1', b'day2'):
		write(data / 'app/app.log', day)
		rc, output = run(['-d', 'archive/', '-rf'], data, find_in(data, '.', '-name', '*.log'))
		assert rc == 0, output[-2000:]
	assert files(data / 'archive') == {b'app/app.log': b'day2', b'other.log': b'other'}
	assert not (data / 'archive/archive').exists()
	assert b'inside the destination' in output and b'not copied or moved' in output


def test_entries_inside_the_destination_stay_and_the_rest_lands(tmp_path):
	write(tmp_path / 'a/x', 'NEW')
	write(tmp_path / 'out/a/x', 'OLD')
	rc, output = run(['-d', 'out', '-rf'], tmp_path, b'out/a/x\na/x\n')
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'out') == {b'a/x': b'NEW'} and not (tmp_path / 'out/out').exists()


def test_a_listed_directory_inside_the_destination_stays_too(tmp_path):
	write(tmp_path / 'a/x', 'N')
	write(tmp_path / 'out/a/x', 'O')
	write(tmp_path / 'out/a/y', 'Y')
	rc, output = run(['-d', 'out', '-rf'], tmp_path, b'a\nout/a\n')
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'out') == {b'a/x': b'N', b'a/y': b'Y'} and not (tmp_path / 'out/out').exists()


@pytest.mark.parametrize('flags', MODES, ids=MODE_IDS)
def test_a_listed_directory_that_holds_the_destination_leaves_it_out(tmp_path, flags):
	write(tmp_path / 'src/a', 'one')
	write(tmp_path / 'src/sub/b', 'two')
	(tmp_path / 'src/empty').mkdir()
	for _ in range(2):
		rc, output = run(['-d', 'out'] + flags, tmp_path / 'src', find_in(tmp_path / 'src', '.'))
		assert rc == 0, output[-2000:]
		assert files(tmp_path / 'src/out') == {b'a': b'one', b'sub/b': b'two'}
		assert not (tmp_path / 'src/out/out').exists() and (tmp_path / 'src/out/empty').is_dir()


def test_a_listed_directory_that_holds_the_destination_by_a_symlink(tmp_path):
	write(tmp_path / 'real/src/a', 'one')
	os.symlink('real', str(tmp_path / 'link'))
	for _ in range(2):
		rc, output = run(['-d', str(tmp_path / 'real/src/out')], tmp_path, b'link/src\n')
		assert rc == 0, output[-2000:]
		assert files(tmp_path / 'real/src/out') == {b'link/src/a': b'one'}
		assert sorted(os.listdir(str(tmp_path / 'real/src/out/link/src'))) == ['a']


def test_a_list_of_everything_below_the_working_directory_with_no_destination_skips_it_all(tmp_path):
	write(tmp_path / 'src/a/x', 'one')
	os.symlink('a', str(tmp_path / 'src/l'))
	before = files(tmp_path)
	rc, output = run([], tmp_path, find_in(tmp_path, 'src'))
	assert rc == 0, output[-2000:]
	assert files(tmp_path) == before and b'inside the destination' in output


def test_a_move_of_a_listed_directory_that_holds_the_destination_is_refused(tmp_path):
	write(tmp_path / 'src/a', 'one')
	rc, output = run(['-d', 'out', '-rf'], tmp_path / 'src', find_in(tmp_path / 'src', '.'))
	assert rc == 241 and (tmp_path / 'src/a').read_text() == 'one' and files(tmp_path / 'src/out') == {}


def test_a_skipped_directory_entry_is_no_listed_directory_for_rme(work, hpcp_mod, reset_hpcp_globals, at_a_tty, monkeypatch, capsys):
	monkeypatch.chdir(work)
	write(work / 'out/gone/stale', 'stays')
	assert hpcp_list(hpcp_mod, work, b'out/gone\nsrc/a/x.txt\n', remove_extra=True) == 0
	output = capsys.readouterr().out
	assert (work / 'out/gone/stale').exists() and '-rme checked nothing' in output and 'inside the destination' in output


# A source that is another source's destination.

def test_a_source_that_another_source_would_overwrite_is_a_clash(tmp_path):
	write(tmp_path / 'a/sub/x', 'from a')
	write(tmp_path / 'out/sub/x', 'mine')
	rc, output = run(['a/', 'out/sub/x', '-d', 'out/'], tmp_path)
	# Refused whole (_settle_destinations), before anything is copied: nothing of the rest is copied either.
	assert rc == 213 and b'Destination clash' in output and b'is the source file' in output
	assert files(tmp_path) == {b'a/sub/x': b'from a', b'out/sub/x': b'mine'}


def test_a_move_that_would_overwrite_a_source_is_refused_before_any_copy(tmp_path):
	write(tmp_path / 'a/sub/x', 'from a')
	write(tmp_path / 'a/y', 'y')
	write(tmp_path / 'out/sub/x', 'mine')
	rc, output = run(['a/', 'out/sub/x', '-d', 'out/', '-rm', '-rf'], tmp_path)
	assert rc == 213 and not (tmp_path / 'out/y').exists()
	assert (tmp_path / 'a/sub/x').read_text() == 'from a' and (tmp_path / 'out/sub/x').read_text() == 'mine'


# F2: a source met again wins back what it lost.

@pytest.mark.parametrize('flags', MODES, ids=MODE_IDS)
def test_the_last_of_a_repeated_directory_wins(tmp_path, flags):
	write(tmp_path / 'a/f', 'A')
	write(tmp_path / 'b/f', 'B')
	write(tmp_path / 'a/only_a', 'a')
	for sources, winner in ((['a/', 'b/', 'a/'], b'A'), (['a/', 'b/', './a/'], b'A'), (['b/', 'a/', 'b/'], b'B'), (['a/', 'b/', 'a/', 'b/'], b'B')):
		shutil.rmtree(str(tmp_path / 'out'), ignore_errors=True)
		rc, output = run(sources + ['-d', 'out'] + flags, tmp_path)
		assert rc == 0, (sources, output[-2000:])
		assert files(tmp_path / 'out') == {b'f': winner, b'only_a': b'a'}, sources
		assert b'1 file(s) replaced' in output, sources


@pytest.mark.parametrize('flags', MODES, ids=MODE_IDS)
def test_four_directories_with_the_first_repeated(tmp_path, flags):
	for name in 'abc':
		write(tmp_path / name / 'f', name.upper())
	rc, output = run(['a/', 'b/', 'c/', 'a/', '-d', 'out'] + flags, tmp_path)
	assert rc == 0 and files(tmp_path / 'out') == {b'f': b'A'} and b'2 file(s) replaced' in output


def test_a_repeated_directory_keeps_the_existing_file_with_rme(tmp_path, hpcp_mod, reset_hpcp_globals, at_a_tty, monkeypatch):
	monkeypatch.chdir(tmp_path)
	write(tmp_path / 'a/f', 'A')
	write(tmp_path / 'b/f', 'B')
	write(tmp_path / 'out/f', 'old')
	write(tmp_path / 'out/stale', 'gone')
	assert hpcp_mod.hpcp(['a/', 'b/', 'a/'], dest_paths=[str(tmp_path / 'out')], single_thread=True, remove_extra=True) == 0
	assert files(tmp_path / 'out') == {b'f': b'A'}


# F3, F4: -fl STORED -rme.

def stored_rme(directory, stored='S', extra=()):
	"""hpcp src -fl STORED -rme, answering yes: (rc, what the terminal showed)."""
	return in_a_terminal(hpcp_command(['src', '-fl', stored, '-rme'] + list(extra)), directory, b'y')


def test_a_stored_list_is_not_split_on_nul(tmp_path):
	write(tmp_path / 'src/one', '1')
	write(tmp_path / 'src/two', '2')
	assert run(['src', '-nhfl', '-fl', 'S'], tmp_path)[0] == 0
	with open(str(tmp_path / 'S'), 'ab') as f:
		f.write(b'\0')
	stored_rme(tmp_path)
	assert files(tmp_path / 'src') == {b'one': b'1', b'two': b'2'}


def test_a_stored_list_with_digests_deletes_only_what_it_does_not_know(tmp_path):
	write(tmp_path / 'src/one', '1')
	write(tmp_path / 'src/sub/two', '2')
	os.symlink('one', str(tmp_path / 'src/lnk'))
	assert run(['src', '-fl', 'S'], tmp_path)[0] == 0
	write(tmp_path / 'src/stale', 'x')
	rc, shown = stored_rme(tmp_path)
	assert rc == 0 and files(tmp_path / 'src') == {b'one': b'1', b'sub/two': b'2', b'lnk': b'-> one'}


def test_a_stored_list_keeps_a_name_with_a_colon(tmp_path):
	write(tmp_path / 'src/a:b', '1')
	write(tmp_path / 'src/c:d:e', '2')
	assert run(['src', '-fl', 'S'], tmp_path)[0] == 0
	write(tmp_path / 'src/stale', 'x')
	stored_rme(tmp_path)
	assert files(tmp_path / 'src') == {b'a:b': b'1', b'c:d:e': b'2'}


def test_a_missing_stored_list_removes_nothing(tmp_path):
	write(tmp_path / 'src/one', '1')
	rc, shown = stored_rme(tmp_path, 'MISSING')
	assert rc == 208 and files(tmp_path / 'src') == {b'one': b'1'} and b'(y/n)' not in shown


def test_a_stored_list_cannot_tell_a_line_break_in_a_name_from_a_separator(tmp_path):
	base = os.fsencode(str(tmp_path))
	write(os.path.join(base, b'src', b'new\nline'), 'n')
	write(os.path.join(base, b'src', b'ends\r'), 'c')
	write(tmp_path / 'src/plain', 'p')
	assert run(['src', '-nhfl', '-fl', 'S'], tmp_path)[0] == 0
	write(tmp_path / 'src/stale', 'x')
	rc, shown = stored_rme(tmp_path)
	assert files(tmp_path / 'src') == {b'new\nline': b'n', b'ends\r': b'c', b'plain': b'p'}


# F5: a file meeting a directory at one destination path.

def _clash_in_every_mode(tmp_path, flags, sources, dest='out'):
	rc, output = run(sources + ['-d', dest] + flags, tmp_path)
	assert rc == 213, output[-2000:]
	assert b'Destination clash' in output
	return output


@pytest.mark.parametrize('flags', MODES, ids=MODE_IDS)
def test_a_file_onto_an_existing_directory_is_a_clash_and_nothing_nests(tmp_path, flags):
	write(tmp_path / 'a/x', 'file')
	write(tmp_path / 'a/y', 'y')
	write(tmp_path / 'out/x/keep', 'k')
	os.chmod(str(tmp_path / 'a/x'), 0o640)
	_clash_in_every_mode(tmp_path, flags, ['a/'])
	assert files(tmp_path / 'out') == {b'x/keep': b'k', b'y': b'y'}
	assert (tmp_path / 'a/x').read_text() == 'file' and stat.S_IMODE(os.stat(str(tmp_path / 'a/x')).st_mode) == 0o640


def test_a_move_of_a_file_onto_a_directory_removes_nothing(tmp_path):
	write(tmp_path / 'a/x', 'file')
	write(tmp_path / 'a/y', 'y')
	write(tmp_path / 'out/x/keep', 'k')
	rc, output = run(['a/', '-d', 'out', '-rm', '-rf'], tmp_path)
	assert rc == 213 and (tmp_path / 'a/x').read_text() == 'file' and (tmp_path / 'a/y').exists()
	assert files(tmp_path / 'out') == {b'x/keep': b'k', b'y': b'y'}


def test_rme_after_a_file_met_a_directory_is_skipped(tmp_path, hpcp_mod, reset_hpcp_globals, at_a_tty, monkeypatch):
	monkeypatch.chdir(tmp_path)
	write(tmp_path / 'a/x', 'file')
	write(tmp_path / 'out/x/keep', 'k')
	write(tmp_path / 'out/stale', 's')
	assert hpcp_mod.hpcp(['a/'], dest_paths=[str(tmp_path / 'out')], single_thread=True, remove=True, remove_force=True, remove_extra=True) != 0
	assert (tmp_path / 'a/x').read_text() == 'file' and (tmp_path / 'out/x/keep').exists() and (tmp_path / 'out/stale').exists()


@pytest.mark.parametrize('flags', MODES, ids=MODE_IDS)
def test_a_file_and_a_directory_of_two_sources_keep_the_directory(tmp_path, flags):
	write(tmp_path / 'a/x', 'file')
	write(tmp_path / 'b/x/y', 'deep')
	write(tmp_path / 'b/z', 'z')
	_clash_in_every_mode(tmp_path, flags, ['a/', 'b/'])
	assert files(tmp_path / 'out') == {b'x/y': b'deep', b'z': b'z'}
	shutil.rmtree(str(tmp_path / 'out'), ignore_errors=True)
	_clash_in_every_mode(tmp_path, flags, ['b/', 'a/'], 'out')
	assert files(tmp_path / 'out') == {b'x/y': b'deep', b'z': b'z'}


def test_list_entries_that_make_a_file_and_a_directory_of_one_path(tmp_path):
	write(tmp_path / 'top/d/x/y', 'deep')
	(tmp_path / 'top/cwd').mkdir()
	write(tmp_path / 'top/cwd/d/x', 'file')
	rc, output = run(['-d', str(tmp_path / 'out')], tmp_path / 'top/cwd', b'd/x\n../d/x/y\n')
	assert rc == 213 and b'Destination clash' in output
	assert files(tmp_path / 'out') == {b'd/x/y': b'deep'}


def test_a_list_entry_file_onto_an_existing_directory(tmp_path):
	write(tmp_path / 'a/x', 'file')
	write(tmp_path / 'out/a/x/keep', 'k')
	rc, output = run(['-d', 'out', '-rf'], tmp_path, b'a/x\n')
	assert rc == 213 and (tmp_path / 'a/x').read_text() == 'file' and files(tmp_path / 'out') == {b'a/x/keep': b'k'}


def test_a_symlink_onto_a_directory_is_still_refused(tmp_path):
	os.makedirs(str(tmp_path / 'a'))
	os.symlink('t', str(tmp_path / 'a/x'))
	write(tmp_path / 'out/x/keep', 'k')
	rc, output = run(['a/', '-d', 'out'], tmp_path)
	assert rc == 162 and files(tmp_path / 'out') == {b'x/keep': b'k'}
	rc, output = run(['a/', '-d', 'out', '-rm', '-rf'], tmp_path)
	assert rc == 212 and os.path.islink(str(tmp_path / 'a/x'))


# F6: only directories this run makes take the source's metadata.

def test_an_existing_parent_keeps_its_mode(tmp_path):
	write(tmp_path / 'overlay/etc/app/c.conf', 'conf')
	os.chmod(str(tmp_path / 'overlay/etc'), 0o700)
	os.chmod(str(tmp_path / 'overlay/etc/app'), 0o700)
	os.makedirs(str(tmp_path / 'root/etc'))
	os.chmod(str(tmp_path / 'root/etc'), 0o755)
	rc, output = run(['-d', '../root'], tmp_path / 'overlay', find_in(tmp_path / 'overlay', 'etc', '-type', 'f'))
	assert rc == 0, output[-2000:]
	assert stat.S_IMODE(os.stat(str(tmp_path / 'root/etc')).st_mode) == 0o755
	assert stat.S_IMODE(os.stat(str(tmp_path / 'root/etc/app')).st_mode) == 0o700


def test_a_parent_that_is_a_symlink_leaves_its_target_alone(tmp_path):
	write(tmp_path / 'src/d/x', 'x')
	os.chmod(str(tmp_path / 'src/d'), 0o700)
	os.makedirs(str(tmp_path / 'target'))
	os.chmod(str(tmp_path / 'target'), 0o711)
	os.makedirs(str(tmp_path / 'out/src'))
	os.symlink('../../target', str(tmp_path / 'out/src/d'))
	rc, output = run(['-d', 'out'], tmp_path, b'src/d/x\n')
	assert rc == 0, output[-2000:]
	assert stat.S_IMODE(os.stat(str(tmp_path / 'target')).st_mode) == 0o711 and (tmp_path / 'target/x').read_text() == 'x'


def test_a_parent_that_is_a_file_keeps_its_mode(tmp_path):
	write(tmp_path / 'src/d/x', 'x')
	os.chmod(str(tmp_path / 'src/d'), 0o755)
	write(tmp_path / 'out/src/d', 'file')
	os.chmod(str(tmp_path / 'out/src/d'), 0o644)
	rc, output = run(['-d', 'out'], tmp_path, b'src/d/x\n')
	assert rc != 0 and stat.S_IMODE(os.stat(str(tmp_path / 'out/src/d')).st_mode) == 0o644 and (tmp_path / 'out/src/d').read_text() == 'file'


def test_a_listed_directory_still_copies_its_metadata(tmp_path):
	write(tmp_path / 'src/d/x', 'x')
	os.chmod(str(tmp_path / 'src/d'), 0o750)
	os.makedirs(str(tmp_path / 'out/src/d'))
	os.chmod(str(tmp_path / 'out/src/d'), 0o700)
	rc, _ = run(['-d', 'out'], tmp_path, b'src/d\n')
	assert rc == 0 and stat.S_IMODE(os.stat(str(tmp_path / 'out/src/d')).st_mode) == 0o750


@pytest.mark.root
@pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() != 0, reason='needs root')
def test_an_existing_parent_keeps_its_owner(tmp_path):
	write(tmp_path / 'overlay/etc/app/c.conf', 'conf')
	for path in ('overlay/etc', 'overlay/etc/app'):
		os.chown(str(tmp_path / path), 1000, 1000)
		os.chmod(str(tmp_path / path), 0o700)
	os.makedirs(str(tmp_path / 'root/etc'))
	os.chmod(str(tmp_path / 'root/etc'), 0o755)
	rc, output = run(['-d', '../root'], tmp_path / 'overlay', find_in(tmp_path / 'overlay', 'etc', '-type', 'f'))
	assert rc == 0, output[-2000:]
	st = os.stat(str(tmp_path / 'root/etc'))
	assert (st.st_uid, stat.S_IMODE(st.st_mode)) == (0, 0o755)
	st = os.stat(str(tmp_path / 'root/etc/app'))
	assert (st.st_uid, stat.S_IMODE(st.st_mode)) == (1000, 0o700)


@pytest.mark.root
@pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() != 0, reason='needs root')
def test_a_mounted_parent_and_a_symlinked_parent_keep_their_metadata_as_root(tmp_path):
	write(tmp_path / 'src/m/x', 'x')
	write(tmp_path / 'src/s/y', 'y')
	os.chown(str(tmp_path / 'src/m'), 1000, 1000)
	os.chmod(str(tmp_path / 'src/m'), 0o700)
	os.chown(str(tmp_path / 'src/s'), 1000, 1000)
	os.chmod(str(tmp_path / 'src/s'), 0o700)
	os.makedirs(str(tmp_path / 'out/src/m'))
	os.makedirs(str(tmp_path / 'target'))
	os.chmod(str(tmp_path / 'target'), 0o711)
	os.symlink('../../target', str(tmp_path / 'out/src/s'))
	subprocess.run(['mount', '-t', 'tmpfs', '-o', 'mode=1777', 'tmpfs', str(tmp_path / 'out/src/m')], check=True)
	try:
		rc, output = run(['-d', 'out'], tmp_path, b'src/m/x\nsrc/s/y\n')
		assert rc == 0, output[-2000:]
		st = os.stat(str(tmp_path / 'out/src/m'))
		assert (st.st_uid, stat.S_IMODE(st.st_mode)) == (0, 0o1777)
		st = os.stat(str(tmp_path / 'target'))
		assert (st.st_uid, stat.S_IMODE(st.st_mode)) == (0, 0o711)
	finally:
		subprocess.run(['umount', str(tmp_path / 'out/src/m')])


# F7, F9: the advice the tool gives works.

def test_rme_advice_to_list_the_directory_itself_works(work, hpcp_mod, reset_hpcp_globals, at_a_tty, monkeypatch, capsys):
	monkeypatch.chdir(work)
	write(work / 'out/src/gone/x', 'gone')
	write(work / 'out/src/stray', 'gone')
	write(work / 'out/src/a/stray', 'gone')
	write(work / 'out/elsewhere', 'stays')
	assert hpcp_list(hpcp_mod, work, b'src/a\nsrc/b/x.txt\n', remove_extra=True) == 0
	output = capsys.readouterr().out
	assert (work / 'out/src/stray').exists() and not (work / 'out/src/a/stray').exists() and 'checked nothing' not in output
	assert hpcp_list(hpcp_mod, work, b'src\n', remove_extra=True) == 0
	assert files(work / 'out') == {b'src/a/x.txt': b'one', b'src/b/x.txt': b'two', b'src/a/deep/y.log': b'three', b'elsewhere': b'stays'}
	assert not (work / 'out/src/gone').exists()


def test_the_readme_gives_the_same_advice():
	with open(os.path.join(os.path.dirname(HPCP), 'README.md'), encoding='utf-8') as f:
		readme = f.read()
	assert "mindepth" not in readme and "printf 'src\\n' | hpcp -d out -rme" in readme
	assert 'cd src && find . | hpcp -d ../out' in ' '.join(readme.split())


# F8: -do copies no file.

def test_directory_only_with_two_file_sources_is_no_clash(tmp_path):
	write(tmp_path / 'a/x', 'A')
	write(tmp_path / 'b/x', 'B')
	rc, output = run(['a/x', 'b/x', '-d', 'out', '-do'], tmp_path)
	assert rc == 0 and b'Destination clash' not in output and not (tmp_path / 'out/x').exists()


# F10: sources whose destinations cannot meet are not settled.

def test_settling_reads_no_listing_when_destinations_cannot_meet(hpcp_mod, monkeypatch):
	def fail(*args, **kwargs):
		raise AssertionError('listed')
	monkeypatch.setattr(hpcp_mod, 'get_file_list', fail)
	monkeypatch.setattr(hpcp_mod.os.path, 'isdir', lambda path: path.startswith('/data/source'))
	assert hpcp_mod._settle_destinations(['/data/source_a', '/data/source_b'], ['/o/out/'], None, False, None, 4, False)[:2] == ({}, 0)


def test_settling_lists_when_destinations_overlap(hpcp_mod, monkeypatch):
	listings = {'/data/source_a/': (frozenset(['/data/source_a/f']), frozenset(), 0, frozenset(['/data/source_a/'])),
		'/data/source_b/': (frozenset(['/data/source_b/f']), frozenset(), 0, frozenset(['/data/source_b/']))}
	monkeypatch.setattr(hpcp_mod, 'get_file_list', lambda path, **kwargs: listings[path])
	monkeypatch.setattr(hpcp_mod.os.path, 'isdir', lambda path: path.startswith('/data/source'))
	dropped, replaced, _ = hpcp_mod._settle_destinations(['/data/source_a', '/data/source_b'], ['/o/out'], None, False, None, 4, False)
	assert (dropped, replaced) == ({('/data/source_a/', ('/o/out',)): {'/data/source_a/f'}}, 1)


def test_roots_overlap(hpcp_mod):
	overlap = hpcp_mod._roots_overlap
	assert not overlap(['/o/a', '/o/b', '/o/a-b', '/o/c.d'])
	assert overlap(['/o/a', '/o/b', '/o/a']) and overlap(['/o/a-b', '/o/a', '/o/a/c']) and overlap(['/o/a/', '/o/a/c/d'])
	assert not overlap([]) and not overlap(['/o'])


# F11: a link whose destination is the link itself survives.

@pytest.mark.parametrize('flags', MODES, ids=MODE_IDS)
def test_links_landing_on_themselves_survive(tmp_path, flags):
	write(tmp_path / 'src/sub/f', 'one')
	os.symlink('f', str(tmp_path / 'src/sub/l'))
	os.symlink('/nonexist', str(tmp_path / 'src/d'))
	os.symlink('sub', str(tmp_path / 'src/dirlink'))
	expected = files(tmp_path / 'src')
	for stdin, args, cwd in ((find_in(tmp_path, 'src', '!', '-type', 'd'), [], tmp_path),
			(find_in(tmp_path / 'src', '.', '-type', 'l'), ['-d', '.'], tmp_path / 'src'),
			(b'src/dirlink\n', [], tmp_path)):
		rc, output = run(args + flags, cwd, stdin)
		assert rc == 0, output[-2000:]
		assert files(tmp_path / 'src') == expected
	for args, cwd in ((['src/d', '-d', 'src/'], tmp_path), (['src/sub/l', '-d', 'src/sub/'], tmp_path), (['.', '-d', './'], tmp_path / 'src'),
			(['sub/l', '-d', 'sub/'], tmp_path / 'src')):
		rc, output = run(args + flags, cwd)
		assert rc == 0, (args, output[-2000:])
		assert files(tmp_path / 'src') == expected, args


def test_a_link_copied_to_another_place_still_works(tmp_path):
	os.makedirs(str(tmp_path / 'src'))
	os.symlink('f', str(tmp_path / 'src/l'))
	rc, _ = run(['src', '-d', 'out'], tmp_path)
	assert rc == 0 and os.readlink(str(tmp_path / 'out/l')) == 'f'
	os.remove(str(tmp_path / 'out/l'))
	os.symlink('other', str(tmp_path / 'out/l'))
	rc, _ = run(['src', '-d', 'out'], tmp_path)
	assert rc == 0 and os.readlink(str(tmp_path / 'out/l')) == 'f'


# ---- the second review of 9.81 ----

# R1: a named file into an existing directory lands inside it (9.80 did), in every mode.

def _dir_and_file(tmp_path):
	write(tmp_path / 'a/x', 'X')
	os.makedirs(str(tmp_path / 'd'))
	return tmp_path


@pytest.mark.parametrize('flags', MODES, ids=MODE_IDS)
@pytest.mark.parametrize('how', ['positional', 'dash-d', 'dot', 'dotdot', 'link'])
def test_a_named_file_lands_inside_an_existing_directory(tmp_path, flags, how):
	_dir_and_file(tmp_path)
	cwd, args = tmp_path, None
	if how == 'positional':
		args = ['a/x', 'd']
	elif how == 'dash-d':
		args = ['a/x', '-d', 'd']
	elif how == 'dot':
		cwd, args = tmp_path / 'd', ['../a/x', '-d', '.']
	elif how == 'dotdot':
		os.makedirs(str(tmp_path / 'd/sub'))
		cwd, args = tmp_path / 'd/sub', ['../../a/x', '-d', '..']
	else:
		os.symlink('d', str(tmp_path / 'lnk'))
		args = ['a/x', '-d', 'lnk']
	rc, output = run(args + flags, cwd)
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'd').get(b'x') == b'X'
	assert files(tmp_path / 'a') == {b'x': b'X'}
	if how == 'link':
		assert os.path.islink(str(tmp_path / 'lnk')) and not os.path.exists(str(tmp_path / 'lnk/lnk'))


def test_a_named_file_lands_inside_an_existing_directory_from_the_library(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	_dir_and_file(tmp_path)
	monkeypatch.chdir(tmp_path)
	assert hpcp_mod.hpcp(['a/x'], ['d'], single_thread=True) == 0
	assert files(tmp_path / 'd') == {b'x': b'X'}


@pytest.mark.parametrize('flags', MODES, ids=MODE_IDS)
def test_a_named_file_moves_into_an_existing_directory(tmp_path, flags):
	_dir_and_file(tmp_path)
	rc, output = run(['a/x', '-d', 'd', '-rm', '-rf'] + flags, tmp_path)
	assert rc == 0, output[-2000:]
	assert files(tmp_path) == {b'd/x': b'X'}


def test_rme_keeps_the_file_it_just_copied_into_an_existing_directory(tmp_path):
	_dir_and_file(tmp_path)
	write(tmp_path / 'd/stale', 'S')
	rc, shown = in_a_terminal(hpcp_command(['a/x', '-d', 'd', '-rme']), tmp_path, b'y')
	# 9.81 deleted the other files of d; a named file says nothing of them (see the third review, below).
	assert rc == 0 and files(tmp_path / 'd') == {b'x': b'X', b'stale': b'S'}


@pytest.mark.parametrize('flags', MODES, ids=MODE_IDS)
def test_a_named_file_still_renames_and_overwrites_a_file(tmp_path, flags):
	_dir_and_file(tmp_path)
	write(tmp_path / 'f', 'old')
	assert run(['a/x', '-d', 'new'] + flags, tmp_path)[0] == 0
	assert run(['a/x', '-d', 'f'] + flags, tmp_path)[0] == 0
	assert files(tmp_path) == {b'a/x': b'X', b'new': b'X', b'f': b'X'}


@pytest.mark.parametrize('flags', MODES, ids=MODE_IDS)
def test_a_named_file_onto_a_directory_of_its_name_is_still_a_clash(tmp_path, flags):
	_dir_and_file(tmp_path)
	write(tmp_path / 'd/x/inner', 'I')
	rc, output = run(['a/x', '-d', 'd'] + flags, tmp_path)
	assert rc == 213 and files(tmp_path / 'd') == {b'x/inner': b'I'}


def test_a_symlink_source_lands_inside_an_existing_directory(tmp_path):
	_dir_and_file(tmp_path)
	os.symlink('x', str(tmp_path / 'a/l'))
	rc, output = run(['a/l', '-d', 'd'], tmp_path)
	assert rc == 0 and files(tmp_path / 'd') == {b'l': b'-> x'}


# R3: a link spelling of a file entry is judged by its real parent directory.

@pytest.mark.parametrize('flags', MODES, ids=MODE_IDS)
def test_a_symlink_spelling_of_an_entry_inside_the_destination_stays(tmp_path, flags):
	os.makedirs(str(tmp_path / 'archive/a'))
	write(tmp_path / 'a/x', 'NEW')
	write(tmp_path / 'archive/a/x', 'OLD')
	os.symlink('archive', str(tmp_path / 'cur'))
	rc, output = run(['-d', 'archive', '-rf'] + flags, tmp_path, b'cur/a/x\na/x\n')
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'archive') == {b'a/x': b'NEW'}


def test_find_dash_l_run_twice_does_not_nest(tmp_path):
	write(tmp_path / 'logs/one.log', '1')
	write(tmp_path / 'logs/sub/two.log', '2')
	os.makedirs(str(tmp_path / 'archive'))
	os.symlink('../archive', str(tmp_path / 'logs/arch'))
	for _ in range(2):
		listing = subprocess.run(['find', '-L', '.', '-name', '*.log'], cwd=str(tmp_path / 'logs'), stdout=subprocess.PIPE).stdout
		rc, output = run(['-d', '../archive/', '-rf'], tmp_path / 'logs', listing)
		assert rc == 0, output[-2000:]
	assert files(tmp_path / 'archive') == {b'one.log': b'1', b'sub/two.log': b'2'}


# R4: the scope of -rme is decided before entries inside the destination are dropped.

def test_rme_with_a_positional_and_an_entry_inside_the_destination_stays_scoped(tmp_path):
	write(tmp_path / 'p/f', 'P')
	write(tmp_path / 'out/keep', 'K')
	write(tmp_path / 'out/stale', 'S')
	write(tmp_path / 'out/d/z', 'Z')
	(tmp_path / 'L').write_bytes(b'out/keep\n')
	rc, shown = in_a_terminal(hpcp_command(['p', '-sfl', 'L', '-d', 'out/', '-rme']), tmp_path, b'y')
	assert rc == 0 and files(tmp_path / 'out') == {b'keep': b'K', b'stale': b'S', b'd/z': b'Z', b'p/f': b'P'}


# R5: a symlink source is not removed by a move onto its own place.

def _link_in_a_directory(tmp_path, kind='file'):
	write(tmp_path / 'src/t', 'T')
	os.makedirs(str(tmp_path / 'src/sub/dir'))
	target = {'file': '../t', 'dangling': '/nonexist', 'dir': 'dir'}[kind]
	os.symlink(target, str(tmp_path / 'src/sub/l'))
	return tmp_path


@pytest.mark.parametrize('flags', MODES, ids=MODE_IDS)
@pytest.mark.parametrize('kind', ['file', 'dangling', 'dir'])
def test_a_move_of_a_link_onto_its_own_place_keeps_it(tmp_path, flags, kind):
	_link_in_a_directory(tmp_path, kind)
	before = files(tmp_path)
	rc, output = run(['src/sub/l', '-d', 'src/sub/', '-rf'] + flags, tmp_path)
	assert rc == 241, output[-2000:]
	assert os.path.islink(str(tmp_path / 'src/sub/l')) and files(tmp_path) == before


def test_a_move_of_a_link_onto_its_own_place_by_a_linked_directory_keeps_it(tmp_path):
	_link_in_a_directory(tmp_path)
	os.symlink('src/sub', str(tmp_path / 'viadir'))
	rc, output = run(['src/sub/l', '-d', 'viadir/', '-rf'], tmp_path)
	assert rc == 241 and os.path.islink(str(tmp_path / 'src/sub/l'))


def test_a_move_of_a_link_onto_its_own_place_asks_nothing_and_keeps_it(tmp_path):
	_link_in_a_directory(tmp_path)
	rc, shown = in_a_terminal(hpcp_command(['src/sub/l', '-d', 'src/sub/', '-rm']), tmp_path, b'y')
	assert rc == 241 and os.path.islink(str(tmp_path / 'src/sub/l'))


def test_a_move_of_a_link_elsewhere_still_moves_it(tmp_path):
	_link_in_a_directory(tmp_path)
	rc, output = run(['src/sub/l', '-d', 'out/', '-rf', '-rm'], tmp_path)
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'out') == {b'l': b'-> ../t'} and not os.path.lexists(str(tmp_path / 'src/sub/l'))


def test_a_move_of_a_hard_linked_link_still_moves_it(tmp_path):
	_link_in_a_directory(tmp_path)
	os.makedirs(str(tmp_path / 'out'))
	os.link(str(tmp_path / 'src/sub/l'), str(tmp_path / 'out/l'), follow_symlinks=False)
	rc, output = run(['src/sub/l', '-d', 'out/', '-rf', '-rm'], tmp_path)
	assert rc == 0, output[-2000:]
	assert os.path.islink(str(tmp_path / 'out/l')) and not os.path.lexists(str(tmp_path / 'src/sub/l'))


# Call 1: a directory source meeting a file at the destination changes nothing about that file.

@pytest.mark.parametrize('flags', MODES, ids=MODE_IDS)
def test_a_directory_onto_a_file_is_a_clash_and_leaves_its_mode(tmp_path, flags):
	write(tmp_path / 'a/x/in', 'in')
	write(tmp_path / 'out/x', 'KEEP')
	os.chmod(str(tmp_path / 'out/x'), 0o600)
	rc, output = run(['a/x', '-d', 'out/'] + flags, tmp_path)
	assert rc == 213, output[-2000:]
	assert files(tmp_path / 'out') == {b'x': b'KEEP'} and stat.S_IMODE(os.stat(str(tmp_path / 'out/x')).st_mode) == 0o600


def test_a_directory_onto_a_link_to_a_private_file_leaves_the_file(tmp_path):
	write(tmp_path / 'a/x/in', 'in')
	write(tmp_path / 'private', 'KEEP')
	os.chmod(str(tmp_path / 'private'), 0o600)
	os.makedirs(str(tmp_path / 'out'))
	os.symlink('../private', str(tmp_path / 'out/x'))
	rc, output = run(['a/x', '-d', 'out/'], tmp_path)
	assert rc == 213, output[-2000:]
	assert stat.S_IMODE(os.stat(str(tmp_path / 'private')).st_mode) == 0o600 and files(tmp_path / 'out') == {b'x': b'-> ../private'}


# -rf with every entry inside the destination.

@pytest.mark.parametrize('flags', MODES, ids=MODE_IDS)
def test_a_move_of_entries_all_inside_the_destination_keeps_them_and_says_so(tmp_path, flags):
	write(tmp_path / 'out/one', '1')
	rc, output = run(['-d', 'out', '-rf'] + flags, tmp_path, b'out/one\n')
	assert rc == 0 and files(tmp_path) == {b'out/one': b'1'} and b'inside the destination' in output


# R2: a relative entry is copied when the working directory lies strictly inside the destination.

def _overlay_dir(tmp_path):
	write(tmp_path / 't/overlay/etc/conf', 'C')
	write(tmp_path / 't/overlay/etc/sub/more', 'M')
	return tmp_path / 't/overlay'


@pytest.mark.parametrize('flags', MODES, ids=MODE_IDS)
def test_entries_below_a_working_directory_inside_the_destination_are_copied(tmp_path, flags):
	overlay = _overlay_dir(tmp_path)
	listing = subprocess.run(['find', 'etc', '-type', 'f'], cwd=str(overlay), stdout=subprocess.PIPE).stdout
	rc, output = run(['-d', '..'] + flags, overlay, listing)
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 't') == {b'overlay/etc/conf': b'C', b'overlay/etc/sub/more': b'M', b'etc/conf': b'C', b'etc/sub/more': b'M'}


def test_entries_are_copied_to_an_absolute_parent_of_the_working_directory(tmp_path):
	overlay = _overlay_dir(tmp_path)
	rc, output = run(['-d', str(tmp_path / 't')], overlay, b'etc/conf\n')
	assert rc == 0 and files(tmp_path / 't').get(b'etc/conf') == b'C'


def test_entries_are_copied_to_a_destination_reached_through_a_link(tmp_path):
	overlay = _overlay_dir(tmp_path)
	os.symlink('t', str(tmp_path / 'viat'))
	rc, output = run(['-d', str(tmp_path / 'viat')], overlay, b'etc/conf\n')
	assert rc == 0 and files(tmp_path / 't').get(b'etc/conf') == b'C'


def test_entries_are_moved_to_a_parent_of_the_working_directory(tmp_path):
	overlay = _overlay_dir(tmp_path)
	rc, output = run(['-d', '..', '-rf'], overlay, b'etc/conf\n')
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 't') == {b'etc/conf': b'C', b'overlay/etc/sub/more': b'M'}


def test_a_listed_directory_is_copied_to_a_parent_of_the_working_directory(tmp_path):
	overlay = _overlay_dir(tmp_path)
	rc, output = run(['-d', '..'], overlay, b'etc\n')
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 't').get(b'etc/sub/more') == b'M' and files(tmp_path / 't').get(b'etc/conf') == b'C'


def test_entries_that_land_on_themselves_still_stay(tmp_path):
	overlay = _overlay_dir(tmp_path)
	before = files(tmp_path)
	for args, stdin in ((['-d', '.'], b'etc/conf\n'), ([], b'etc/conf\n'), (['-d', '.', '-rf'], b'etc\n')):
		rc, output = run(args, overlay, stdin)
		assert rc == 0 and files(tmp_path) == before, (args, output[-2000:])


def test_find_dot_into_a_directory_below_still_leaves_that_directory_out(tmp_path):
	write(tmp_path / 'src/a', '1')
	write(tmp_path / 'src/out/old', 'o')
	listing = subprocess.run(['find', '.', '-type', 'f'], cwd=str(tmp_path / 'src'), stdout=subprocess.PIPE).stdout
	rc, output = run(['-d', 'out'], tmp_path / 'src', listing)
	assert rc == 0 and files(tmp_path / 'src/out') == {b'a': b'1', b'old': b'o'}


def test_absolute_entries_inside_the_destination_stay(tmp_path):
	overlay = _overlay_dir(tmp_path)
	before = files(tmp_path)
	rc, output = run(['-d', str(tmp_path / 't')], overlay, (str(overlay / 'etc/conf') + '\n').encode())
	assert rc == 0 and files(tmp_path) == before


@pytest.mark.root
@pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() != 0, reason='needs root: it writes below /')
def test_a_relative_entry_is_copied_to_the_root_directory_and_an_absolute_one_stays(tmp_path):
	write(tmp_path / 'hpcp_r2_probe/f', 'F')
	try:
		rc, output = run(['-d', '/'], tmp_path, b'hpcp_r2_probe/f\n')
		assert rc == 0, output[-2000:]
		assert files('/hpcp_r2_probe') == {b'f': b'F'}
		shutil.rmtree('/hpcp_r2_probe')
		rc, output = run(['-d', '/'], tmp_path, (str(tmp_path / 'hpcp_r2_probe/f') + '\n').encode())
		assert rc == 0 and not os.path.exists('/hpcp_r2_probe') and b'inside the destination' in output
	finally:
		shutil.rmtree('/hpcp_r2_probe', ignore_errors=True)


# Stored list: a list that shares no name with the tree is nothing known, as a missing list.

def _nothing_known(directory, said=True):
	"""The run's output (in_a_terminal writes it to log) asked nothing, and said the list matches nothing (or did not)."""
	log = (directory / 'log').read_bytes()
	return b'(y/n)' not in log and (b'matches nothing' in log) == said


def _tree_and_list(tmp_path, content):
	write(tmp_path / 'src/one', '1')
	write(tmp_path / 'src/sub/two', '2')
	(tmp_path / 'S').write_bytes(content)
	return files(tmp_path / 'src')


@pytest.mark.parametrize('content', [b'', b'\0', b'\n', b'unrelated:abc\nother\n'], ids=['empty', 'nul', 'newline', 'other-names'])
def test_a_stored_list_that_matches_nothing_deletes_nothing(tmp_path, content):
	before = _tree_and_list(tmp_path, content)
	rc, shown = stored_rme(tmp_path)
	assert rc == 212 and files(tmp_path / 'src') == before and _nothing_known(tmp_path) and b'Remove skipped' in (tmp_path / 'log').read_bytes()


def test_dev_null_as_a_stored_list_deletes_nothing(tmp_path):
	before = _tree_and_list(tmp_path, b'')
	rc, shown = in_a_terminal(hpcp_command(['src', '-fl', '/dev/null', '-rme']), tmp_path, b'y')
	assert rc == 212 and files(tmp_path / 'src') == before and _nothing_known(tmp_path)


@pytest.mark.parametrize('made, used', [('src/', 'src'), ('src', 'src/')], ids=['slash-then-bare', 'bare-then-slash'])
def test_a_list_stored_under_the_other_spelling_matches_nothing_and_deletes_nothing(tmp_path, made, used):
	write(tmp_path / 'src/a', '1')
	write(tmp_path / 'src/b', '2')
	assert run([made, '-fl', 'S'], tmp_path)[0] == 0
	write(tmp_path / 'src/stale', 'x')
	rc, shown = in_a_terminal(hpcp_command([used, '-fl', 'S', '-rme']), tmp_path, b'y')
	assert rc == 212 and files(tmp_path / 'src') == {b'a': b'1', b'b': b'2', b'stale': b'x'}
	assert _nothing_known(tmp_path)


def test_a_stored_list_with_one_matching_name_still_deletes_the_extras(tmp_path):
	write(tmp_path / 'src/one', '1')
	write(tmp_path / 'src/stale', 'x')
	(tmp_path / 'S').write_bytes(b'src/one\n')
	rc, shown = stored_rme(tmp_path)
	assert rc == 0 and files(tmp_path / 'src') == {b'one': b'1'}


def test_a_stored_list_and_an_empty_tree_report_no_such_note(tmp_path):
	os.makedirs(str(tmp_path / 'src'))
	(tmp_path / 'S').write_bytes(b'')
	rc, shown = stored_rme(tmp_path)
	assert rc == 0 and _nothing_known(tmp_path, False)


# ---- the third review of 9.81 ----

# -rme with only named files or links checks nothing, as a list of files does.

def _rme_in_a_terminal(tmp_path, args):
	"""hpcp ARGS -rme, answering yes: (rc, the whole output). A prompt would be answered, so a kept file proves none ran."""
	rc, _ = in_a_terminal(hpcp_command(args + ['-rme']), tmp_path, b'y')
	return rc, (tmp_path / 'log').read_bytes()


def _but_log(tree):
	return {name: content for name, content in tree.items() if os.path.basename(name) != b'log'}


def _a_merge_out_of_the_destination(tmp_path):
	write(tmp_path / 'out/sub/x', 'S')
	write(tmp_path / 'b/x', 'B')
	return ['out/sub/', 'b/', '-d', 'out/', '-rf']


def test_safe_remove_skipped_says_every_source_stays(tmp_path):
	rc, output = run(_a_merge_out_of_the_destination(tmp_path), tmp_path)
	assert rc == 212 and b'every source stays' in output, output[-2000:]
	assert files(tmp_path) == {b'out/sub/x': b'S', b'b/x': b'B', b'out/x': b'B'}


def test_remove_skipped_does_not_claim_what_rme_took(tmp_path):
	# -rme runs first and takes a source inside the destination for extra (docs/limitations.md).
	rc, output = _rme_in_a_terminal(tmp_path, _a_merge_out_of_the_destination(tmp_path))
	assert rc == 212 and b'unless -rme removed it above' in output and b'every source stays' not in output, output[-2000:]


def test_rme_with_a_named_file_into_its_own_directory_checks_nothing(tmp_path):
	write(tmp_path / 'a/x', 'X')
	write(tmp_path / 'a/other', 'O')
	rc, output = _rme_in_a_terminal(tmp_path, ['a/x', '-d', 'a'])
	assert rc == 0, output[-2000:]
	assert _but_log(files(tmp_path)) == {b'a/x': b'X', b'a/other': b'O'} and b'-rme checked nothing' in output and b'(y/n)' not in output


def test_rme_with_a_link_into_the_working_directory_deletes_nothing(tmp_path):
	write(tmp_path / 'w/target', 'T')
	write(tmp_path / 'w/precious', 'P')
	write(tmp_path / 'w/sub/q', 'Q')
	os.symlink('target', str(tmp_path / 'w/lf'))
	before = files(tmp_path)
	rc, output = _rme_in_a_terminal(tmp_path / 'w', ['lf', '-d', '.'])
	assert rc == 0, output[-2000:]
	assert _but_log(files(tmp_path)) == before and b'-rme checked nothing' in output


@pytest.mark.parametrize('dest', ['d', 'd/'])
def test_rme_with_a_named_file_copies_it_and_leaves_the_rest_of_the_destination(tmp_path, dest):
	write(tmp_path / 'a/x', 'X')
	write(tmp_path / 'd/keep', 'K')
	write(tmp_path / 'd/sub/s', 'S')
	rc, output = _rme_in_a_terminal(tmp_path, ['a/x', '-d', dest])
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'd') == {b'keep': b'K', b'sub/s': b'S', b'x': b'X'} and b'-rme checked nothing' in output


def test_rme_with_two_named_files_and_a_dangling_link_deletes_nothing(tmp_path):
	write(tmp_path / 'a/x', 'X')
	write(tmp_path / 'a/y', 'Y')
	os.symlink('nowhere', str(tmp_path / 'a/dg'))
	write(tmp_path / 'd/keep', 'K')
	rc, output = _rme_in_a_terminal(tmp_path, ['a/x', 'a/y', 'a/dg', '-d', 'd'])
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'd') == {b'keep': b'K', b'x': b'X', b'y': b'Y', b'dg': b'-> nowhere'} and b'-rme checked nothing' in output


def test_rme_with_a_link_to_a_directory_deletes_nothing(tmp_path):
	write(tmp_path / 'a/x', 'X')
	os.symlink('a', str(tmp_path / 'ld'))
	write(tmp_path / 'd/keep', 'K')
	rc, output = _rme_in_a_terminal(tmp_path, ['ld', '-d', 'd'])
	assert files(tmp_path / 'd').get(b'keep') == b'K' and b'(y/n)' not in output
	assert files(tmp_path / 'a') == {b'x': b'X'}


def test_rme_with_a_file_and_a_directory_still_cleans(tmp_path):
	write(tmp_path / 'a/x', 'X')
	write(tmp_path / 'e/z', 'Z')
	write(tmp_path / 'd/keep', 'K')
	rc, output = _rme_in_a_terminal(tmp_path, ['a/x', 'e/', '-d', 'd'])
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'd') == {b'x': b'X', b'z': b'Z'} and b'-rme checked nothing' not in output


def test_rme_with_a_directory_alone_still_cleans(tmp_path):
	write(tmp_path / 'e/z', 'Z')
	write(tmp_path / 'd/keep', 'K')
	rc, output = _rme_in_a_terminal(tmp_path, ['e/', '-d', 'd'])
	assert rc == 0 and files(tmp_path / 'd') == {b'z': b'Z'} and b'-rme checked nothing' not in output


# R2, the tighter rule: a relative entry is copied only when it and the working directory part strictly inside
# every destination that holds it, and its landing is neither the entry nor below it.

def _nest(tmp_path):
	write(tmp_path / 't/a/f', '1')
	write(tmp_path / 't/a/b/g', '2')
	write(tmp_path / 't/a/c/h', '3')
	os.symlink('..', str(tmp_path / 't/a/b/up'))
	os.symlink('a', str(tmp_path / 't/viaa'))
	return tmp_path


@pytest.mark.parametrize('entry', ['../../a', '../../a/', './../../a/.', '../../../t/a', '../../a/c', '../c', '../f', 'up/f'],
	ids=['name', 'slash', 'dot', 'through-t', 'sibling-dir', 'sibling', 'file-above', 'link-up'])
@pytest.mark.parametrize('dest', ['..', 'abs', 'link'])
def test_an_entry_that_is_or_is_below_the_destination_stays_and_a_rerun_agrees(tmp_path, entry, dest):
	_nest(tmp_path)
	cwd = tmp_path / 't/a/b'
	target = {'..': '..', 'abs': str(tmp_path / 't/a'), 'link': str(tmp_path / 't/viaa')}[dest]
	before = files(tmp_path)
	for _ in range(2):
		rc, output = run(['-d', target], cwd, (entry + '\n').encode())
		assert rc == 0, output[-2000:]
		assert files(tmp_path) == before, (entry, dest, output[-2000:])


def test_find_of_the_destination_run_twice_adds_nothing(tmp_path):
	_nest(tmp_path)
	cwd = tmp_path / 't/a/b'
	before = files(tmp_path)
	after = []
	for _ in range(2):
		listing = subprocess.run(['find', '../../a', '-type', 'f'], cwd=str(cwd), stdout=subprocess.PIPE).stdout
		rc, output = run(['-d', '..'], cwd, listing)
		assert rc == 0, output[-2000:]
		after.append(files(tmp_path))
	# The file below the working directory (a/b/g) is copied once, as an entry that parts from the destination at b; the
	# files directly in the destination stay, and the second run finds nothing new.
	assert after[0] == after[1] and set(after[0]) - set(before) == {b't/a/a/b/g'}


def test_a_repeated_name_run_twice_does_not_nest(tmp_path):
	write(tmp_path / 't/b/b/f', '1')
	before = files(tmp_path)
	for _ in range(2):
		rc, output = run(['-d', '..'], tmp_path / 't/b/b', b'../../b/b\n')
		assert rc == 0 and files(tmp_path) == before, output[-2000:]


def test_two_destinations_one_holding_the_entry_keeps_it_and_copies_the_other(tmp_path):
	write(tmp_path / 't/a/o/log', 'L')
	write(tmp_path / 't/a/new', 'N')
	rc, output = run(['-d', 'o', '-d', '..', '-rf'], tmp_path / 't/a', b'o/log\nnew\n')
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 't') == {b'a/o/log': b'L', b'a/o/new': b'N'}


def test_entries_below_the_working_directory_still_copy_to_its_parent(tmp_path):
	overlay = _overlay_dir(tmp_path)
	listing = subprocess.run(['find', 'etc', '-type', 'f'], cwd=str(overlay), stdout=subprocess.PIPE).stdout
	for _ in range(2):
		rc, output = run(['-d', '..'], overlay, listing)
		assert rc == 0, output[-2000:]
	assert files(tmp_path / 't') == {b'overlay/etc/conf': b'C', b'overlay/etc/sub/more': b'M', b'etc/conf': b'C', b'etc/sub/more': b'M'}


def test_the_landing_rule_survives_mixed_drives(hpcp_mod, monkeypatch):
	spellings = ('/t/a/b/g', '/t/a/b/g')
	assert hpcp_mod._lands_elsewhere(spellings, 'a/b/g', {'/t/a'}, '/t/a/b')
	def other_drive(paths):
		raise ValueError("Paths don't have the same drive")
	monkeypatch.setattr(os.path, 'commonpath', other_drive)
	assert not hpcp_mod._lands_elsewhere(spellings, 'a/b/g', {'/t/a'}, '/t/a/b')


# An entry inside a destination by its given path, outside it by its real path (through a link), still parts from the
# destination at the working directory, or it stays.

@pytest.mark.parametrize('flags', [[], ['-rf']], ids=['copy', 'move'])
def test_an_entry_through_a_link_that_leaves_the_destination_stays_when_it_does_not_part(tmp_path, flags):
	write(tmp_path / 'data/x', 'D')
	os.makedirs(str(tmp_path / 'home/u'))
	os.symlink('../../data', str(tmp_path / 'home/u/scratch'))
	before = files(tmp_path)
	for _ in range(2):
		rc, output = run(['-d', 'home'] + flags, tmp_path, b'home/u/scratch/x\n')
		assert rc == 0, output[-2000:]
		assert files(tmp_path) == before and not os.path.exists(str(tmp_path / 'home/home'))


def test_find_dash_l_through_a_link_out_of_the_destination_adds_no_nested_copy(tmp_path):
	write(tmp_path / 'ext/x', 'E')
	write(tmp_path / 'out/own', 'O')
	os.symlink('../ext', str(tmp_path / 'out/lk'))
	listing = subprocess.run(['find', '-L', '.', '-type', 'f'], cwd=str(tmp_path), stdout=subprocess.PIPE).stdout
	rc, output = run(['-d', 'out'], tmp_path, listing)
	assert rc == 0, output[-2000:]
	assert not os.path.exists(str(tmp_path / 'out/out')) and files(tmp_path / 'ext') == {b'x': b'E'}


@pytest.mark.parametrize('entry', [b'a/l/x', b'a/l/sub', b'a/l'], ids=['file', 'directory-below-link', 'link'])
def test_an_entry_below_a_link_in_the_destination_stays(tmp_path, entry):
	write(tmp_path / 'o/x', 'X')
	write(tmp_path / 'o/sub/y', 'Y')
	os.makedirs(str(tmp_path / 't/a'))
	os.symlink('../../o', str(tmp_path / 't/a/l'))
	before = files(tmp_path)
	rc, output = run(['-d', 'a'], tmp_path / 't', entry + b'\n')
	assert rc == 0, output[-2000:]
	assert files(tmp_path) == before and not os.path.exists(str(tmp_path / 't/a/a'))


def test_find_dash_l_into_a_directory_with_a_link_out_of_it_run_with_a_move_keeps_everything(tmp_path):
	write(tmp_path / 'o/x', 'X')
	os.makedirs(str(tmp_path / 't/a'))
	os.symlink('../../o', str(tmp_path / 't/a/l'))
	before = files(tmp_path)
	listing = subprocess.run(['find', '-L', 'a', '-type', 'f'], cwd=str(tmp_path / 't'), stdout=subprocess.PIPE).stdout
	rc, output = run(['-d', 'a/', '-rf'], tmp_path / 't', listing)
	assert rc == 0, output[-2000:]
	assert files(tmp_path) == before


@pytest.mark.parametrize('flags', [[], ['-rf']], ids=['copy', 'move'])
def test_a_link_out_of_the_overlay_still_copies_where_the_path_parts_from_the_destination(tmp_path, flags):
	overlay = _overlay_dir(tmp_path)
	write(tmp_path / 'o2/x', 'L')
	os.symlink('../../../o2', str(overlay / 'etc/l'))
	listing = subprocess.run(['find', '-L', 'etc', '-type', 'f'], cwd=str(overlay), stdout=subprocess.PIPE).stdout
	rc, output = run(['-d', '..'] + flags, overlay, listing)
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 't').get(b'etc/l/x') == b'L' and files(tmp_path / 't').get(b'etc/conf') == b'C'
	assert os.path.exists(str(tmp_path / 'o2/x')) == (not flags)


# ---- a source's landing is another source's path (refused before anything changes, exit 213) ----

def _race_tree(tmp_path):
	"""w/c/r = OUT and w/D/c/r = IN; commands run in w/D/c: '../../c/r' lands at w/D/c/r, which is the other entry."""
	write(tmp_path / 'w/c/r', 'OUT')
	write(tmp_path / 'w/D/c/r', 'IN')
	return tmp_path / 'w/D/c'


@pytest.mark.parametrize('flags', [['-rf'], [], ['-s'], ['-nb'], ['-rf', '-nb'], ['-rf', '-s']], ids=lambda flags: ' '.join(flags) or 'copy')
def test_a_list_entry_that_lands_on_another_entry_is_refused_before_anything_changes(tmp_path, flags):
	# 9.81 left a 0-byte w/D/r in about 1 run in 20 (exit 0, sources removed); with -s or -nb IN was always lost.
	cwd = _race_tree(tmp_path)
	rc, output = run(['-d', '..'] + flags, cwd, b'../../c/r\nr\n')
	assert rc == 213, output[-2000:]
	assert files(tmp_path / 'w') == {b'c/r': b'OUT', b'D/c/r': b'IN'}
	assert str(tmp_path / 'w/c/r').encode() in output and str(tmp_path / 'w/D/c/r').encode() in output
	assert b'nothing is copied or removed' in output


@pytest.mark.parametrize('flags', [['-rf'], []], ids=['move', 'copy'])
def test_a_directory_that_lands_on_another_source_is_refused_before_anything_changes(tmp_path, flags):
	# d1/x=A and d2/d1/x=B: the second source's contents land in d2/, the first's directory on the second's path.
	write(tmp_path / 'd1/x', 'A')
	write(tmp_path / 'd2/d1/x', 'B')
	rc, output = run(['d1', 'd2/d1/', '-d', 'd2/'] + flags, tmp_path)
	assert rc == 213, output[-2000:]
	assert files(tmp_path) == {b'd1/x': b'A', b'd2/d1/x': b'B'}
	assert str(tmp_path / 'd1').encode() in output and str(tmp_path / 'd2/d1').encode() in output


@pytest.mark.parametrize('flags', [['-rf'], []], ids=['move', 'copy'])
def test_a_directory_that_holds_another_source_and_has_its_file_is_refused(tmp_path, flags):
	# a lands at out/a, which holds the source out/a/x; a/x would overwrite it.
	write(tmp_path / 'a/x', 'A')
	write(tmp_path / 'out/a/x', 'B')
	rc, output = run(['a', 'out/a/x', '-d', 'out/'] + flags, tmp_path)
	assert rc == 213, output[-2000:]
	assert files(tmp_path) == {b'a/x': b'A', b'out/a/x': b'B'}
	assert b'is the source file' in output


def test_a_landing_inside_a_source_directory_is_refused(tmp_path):
	write(tmp_path / 'out/p/q', 'B')
	write(tmp_path / 'src/p/q', 'A')
	rc, output = run(['-sfl', '-', str(tmp_path / 'out/p') + '/', '-d', str(tmp_path / 'out')], tmp_path / 'src', b'p/q\n')
	assert rc == 213 and b'is the source file' in output
	assert str(tmp_path / 'out/p/q').encode() in output and str(tmp_path / 'src/p/q').encode() in output
	# Nothing there yet: a file added to the other source's tree loses nothing.
	os.remove(tmp_path / 'out/p/q')
	rc, output = run(['-sfl', '-', str(tmp_path / 'out/p') + '/', '-d', str(tmp_path / 'out')], tmp_path / 'src', b'p/q\n')
	assert rc == 0 and (tmp_path / 'out/p/q').read_text() == 'A'


def test_an_excluded_source_is_no_clash(tmp_path):
	cwd = _race_tree(tmp_path)
	rc, output = run(['-d', '..', '-e', '*/D/c/r'], cwd, b'../../c/r\nr\n')
	assert b'would land on' not in output


# The forms around it that stay as they are.

def test_find_dot_into_a_directory_below_it_still_runs(tmp_path):
	write(tmp_path / 'a/x', 'A')
	write(tmp_path / 'b', 'B')
	rc, output = run(['-d', 'out'], tmp_path, b'.\n./a\n./a/x\n./b\n')
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'out') == {b'a/x': b'A', b'b': b'B'}


def test_a_source_holding_the_destination_with_another_source_still_runs(tmp_path):
	write(tmp_path / 's/a', 'A')
	write(tmp_path / 'x', 'X')
	rc, output = run(['.', 'x', '-d', 'out'], tmp_path)
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'out')[b'x'] == b'X' and files(tmp_path / 'out')[b's/a'] == b'A'


def test_a_landing_on_a_path_that_is_not_a_source_overwrites_it_as_before(tmp_path):
	cwd = _race_tree(tmp_path)
	write(tmp_path / 'w/D/other', 'O')
	rc, output = run(['-d', '..'], cwd, b'../../c/r\n../other\n')
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'w/D')[b'c/r'] == b'OUT'


def test_a_source_that_lands_on_itself_with_another_source_is_no_clash(tmp_path):
	write(tmp_path / 'out/x', 'X')
	write(tmp_path / 'y', 'Y')
	rc, output = run(['out/x', 'y', '-d', 'out'], tmp_path)
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'out') == {b'x': b'X', b'y': b'Y'}


def test_a_directory_that_lands_on_itself_and_holds_another_source_still_runs(tmp_path):
	# a copies nothing (it is its own landing), so it overwrites nothing of a/x, which lands at ./x.
	write(tmp_path / 'a/x', 'A')
	rc, output = run(['a/x', 'a', '-d', '.'], tmp_path)
	assert rc == 0, output[-2000:]
	assert files(tmp_path) == {b'a/x': b'A', b'x': b'A'}


def test_a_named_file_that_is_the_destination_is_not_landed_on(tmp_path):
	write(tmp_path / 'a/x', 'A')
	os.symlink('a', tmp_path / 'ldir')
	rc, output = run(['a/x', 'ldir/', '-d', 'a/x'], tmp_path)
	assert rc == 213 and b'Destination clash' in output
	assert files(tmp_path) == {b'a/x': b'A', b'ldir': b'-> a'}


def test_a_directory_that_holds_another_source_without_its_file_still_runs(tmp_path):
	write(tmp_path / 'a/f', 'F')
	write(tmp_path / 'out/a/x', 'B')
	rc, output = run(['a', 'out/a/x', '-d', 'out/'], tmp_path)
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'out')[b'a/f'] == b'F' and files(tmp_path / 'out')[b'x'] == b'B'


# ---- the one rule: a file landing on the place of a file the run still takes from there ----
# Tests named test_safe_ pin forms that must stay safe, and also pass before the rule; every other test here fails there.

MOVE_AND_COPY = pytest.mark.parametrize('flags', [['-rf'], [], ['-s'], ['-nb'], ['-pfl'], ['-rf', '-s'], ['-rf', '-nb']], ids=lambda flags: ' '.join(flags) or 'copy')


def _own_tree(tmp_path):
	"""p/q/x = OUTER and p/q/q/x = INNER: p/q/ flattened into p/ lands the inner file on the outer one."""
	write(tmp_path / 'p/q/x', 'OUTER')
	write(tmp_path / 'p/q/q/x', 'INNER')
	return {b'p/q/x': b'OUTER', b'p/q/q/x': b'INNER'}


@MOVE_AND_COPY
def test_a_source_landing_over_its_own_tree_is_refused(tmp_path, flags):
	tree = _own_tree(tmp_path)
	rc, output = run(['p/q/', '-d', 'p/'] + flags, tmp_path)
	assert rc == 213 and files(tmp_path) == tree, output[-2000:]
	assert str(tmp_path / 'p/q/q/x').encode() in output and str(tmp_path / 'p/q/x').encode() in output


@pytest.mark.parametrize('listing', [b'.\n./x\n./q\n./q/x\n', b'.\n'], ids=['find', 'dot'])
def test_a_list_landing_over_its_own_tree_is_refused(tmp_path, listing):
	tree = _own_tree(tmp_path)
	rc, output = run(['-d', '..', '-rf'], tmp_path / 'p/q', listing)
	assert rc == 213 and files(tmp_path) == tree, output[-2000:]


def test_safe_file_entries_landing_over_each_other_stay_refused(tmp_path):
	tree = _own_tree(tmp_path)
	rc, output = run(['-d', '..', '-rf'], tmp_path / 'p/q', b'./x\n./q/x\n')
	assert rc == 213 and files(tmp_path) == tree, output[-2000:]


def test_safe_a_flatten_that_lands_on_no_file_keeps_working(tmp_path):
	write(tmp_path / 'p/q/q/only', 'I')
	write(tmp_path / 'p/q/top', 'T')
	rc, output = run(['p/q/', '-d', 'p/', '-rf'], tmp_path)
	assert rc == 0, output[-2000:]
	assert files(tmp_path) == {b'p/top': b'T', b'p/q/only': b'I'}


def _linked_pair(tmp_path):
	write(tmp_path / 'd1/x', 'A')
	write(tmp_path / 'd2/d1/x', 'B')
	os.symlink('d2', tmp_path / 'l2')
	return {b'd1/x': b'A', b'd2/d1/x': b'B', b'l2': b'-> d2'}


@pytest.mark.parametrize('flags', [['-rf'], []], ids=['move', 'copy'])
@pytest.mark.parametrize('args', [['d1', 'l2/d1/', '-d', 'd2/'], ['d1', 'd2/d1/', '-d', 'l2/']], ids=['source-link', 'dest-link'])
def test_a_link_in_the_path_hides_no_clash(tmp_path, args, flags):
	tree = _linked_pair(tmp_path)
	rc, output = run(args + flags, tmp_path)
	assert rc == 213 and files(tmp_path) == tree, output[-2000:]


@pytest.mark.parametrize('listing, dest', [(b'../../c/r\n../../lnk/c/r\n', '..'), (b'../../c/r\nr\n', '../../lnk')], ids=['entry-link', 'dest-link'])
def test_a_link_in_a_list_path_hides_no_clash(tmp_path, listing, dest):
	cwd = _race_tree(tmp_path)
	os.symlink('D', tmp_path / 'w/lnk')
	rc, output = run(['-d', dest, '-rf'], cwd, listing)
	assert rc == 213 and files(tmp_path / 'w') == {b'c/r': b'OUT', b'D/c/r': b'IN', b'lnk': b'-> D'}, output[-2000:]


def test_a_link_in_the_destination_hides_no_listed_directory(tmp_path):
	write(tmp_path / 'p/x', 'A')
	write(tmp_path / 'src/p/x', 'B')
	os.makedirs(tmp_path / 'out')
	os.symlink('../src/p', tmp_path / 'out/p')
	rc, output = run(['-d', 'out', '-rf'], tmp_path, b'p/x\nsrc/p\n')
	assert rc == 213 and files(tmp_path) == {b'p/x': b'A', b'src/p/x': b'B', b'out/p': b'-> ../src/p'}, output[-2000:]


def test_a_directory_link_in_the_destination_hides_no_clash(tmp_path):
	# src/e/x would go through out/e onto src/d/x: a directory source meets a link below the destination.
	write(tmp_path / 'src/d/x', 'D')
	write(tmp_path / 'src/e/x', 'E')
	os.makedirs(tmp_path / 'out')
	os.symlink('../src/d', tmp_path / 'out/e')
	rc, output = run(['src/', '-d', 'out/', '-rf'], tmp_path)
	assert rc == 213 and files(tmp_path) == {b'src/d/x': b'D', b'src/e/x': b'E', b'out/e': b'-> ../src/d'}, output[-2000:]


def test_a_move_through_a_file_link_onto_its_own_file_is_refused(tmp_path):
	# cp writes through out/x, onto a/x itself: the copy skips it, and the removal would leave a dangling link.
	write(tmp_path / 'a/x', 'A')
	os.makedirs(tmp_path / 'out')
	os.symlink('../a/x', tmp_path / 'out/x')
	rc, output = run(['a/', '-d', 'out/', '-rf'], tmp_path)
	assert rc == 213 and files(tmp_path) == {b'a/x': b'A', b'out/x': b'-> ../a/x'}, output[-2000:]


def test_a_file_link_in_the_destination_hides_no_clash(tmp_path):
	write(tmp_path / 'a/y', 'A')
	write(tmp_path / 'b/x', 'B')
	os.makedirs(tmp_path / 'out')
	os.symlink('../b/x', tmp_path / 'out/y')
	rc, output = run(['a/', 'b/', '-d', 'out/', '-rf'], tmp_path)
	assert rc == 213 and files(tmp_path) == {b'a/y': b'A', b'b/x': b'B', b'out/y': b'-> ../b/x'}, output[-2000:]


def test_safe_a_link_source_replaces_a_link_at_its_landing(tmp_path):
	# A link is copied as a link: it replaces what is there, and writes through nothing.
	write(tmp_path / 'b/x', 'B')
	os.makedirs(tmp_path / 'a')
	os.makedirs(tmp_path / 'out')
	os.symlink('elsewhere', tmp_path / 'a/y')
	os.symlink('../b/x', tmp_path / 'out/y')
	rc, output = run(['a/', 'b/', '-d', 'out/'], tmp_path)
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'out') == {b'y': b'-> elsewhere', b'x': b'B'} and (tmp_path / 'b/x').read_text() == 'B'


def test_two_spellings_of_one_landing_are_one_file(tmp_path):
	# out/l -> real: both entries land on out/real/x. The first stays and nothing is removed.
	write(tmp_path / 'l/x', 'A')
	write(tmp_path / 'real/x', 'B')
	os.makedirs(tmp_path / 'out/real')
	os.symlink('real', tmp_path / 'out/l')
	rc, output = run(['-d', 'out', '-rf'], tmp_path, b'l/x\nreal/x\n')
	assert rc == 213, output[-2000:]
	assert (tmp_path / 'l/x').read_text() == 'A' and (tmp_path / 'real/x').read_text() == 'B' and (tmp_path / 'out/real/x').read_text() == 'A'


def test_safe_a_landing_in_a_later_destination_counts(tmp_path):
	# A file goes to one destination of several, whichever has room: each landing has to be safe.
	write(tmp_path / 'a/x', 'A')
	write(tmp_path / 'd2/a/x', 'B')
	rc, output = run(['a', 'd2/a/x', '-d', 'd1/', '-d', 'd2/', '-rf'], tmp_path)
	assert rc == 213 and (tmp_path / 'a/x').read_text() == 'A' and (tmp_path / 'd2/a/x').read_text() == 'B', output[-2000:]


# '..' lands where '.' does: a landing never leaves its destination.

@pytest.mark.parametrize('flags', [[], ['-s']], ids=['parallel', 'serial'])
def test_dot_dot_lands_in_the_destination_as_dot_does(tmp_path, flags):
	write(tmp_path / 'w/t', 'T')
	write(tmp_path / 'w/sub/s', 'S')
	rc, output = run(['..', '-d', 'out/'] + flags, tmp_path / 'w/sub')
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'w/sub/out') == {b't': b'T', b'sub/s': b'S'}
	assert files(tmp_path / 'w') == {b't': b'T', b'sub/s': b'S', b'sub/out/t': b'T', b'sub/out/sub/s': b'S'}


def test_dot_dot_and_dot_give_the_same_landing(hpcp_mod):
	land = hpcp_mod._source_dest
	assert land('..', '/o/out/') == land('.', '/o/out/') == land('../', '/o/out/') == land('a/..', '/o/out/') == land('a/../..', '/o/out/') == '/o/out'
	assert land('sub', '/o/out/') == '/o/out/sub' and land('sub', '/o/out') == '/o/out'


def test_a_move_of_dot_dot_onto_itself_keeps_both_files(tmp_path):
	# w/x = X, w/c/x = C. '..' used to land in w, swapping the two and losing both: now it lands on itself, as '.' would.
	write(tmp_path / 'w/x', 'X')
	write(tmp_path / 'w/c/x', 'C')
	os.makedirs(tmp_path / 'w/c/s')
	rc, output = run(['../../x', '..', '-d', '../', '-rf'], tmp_path / 'w/c/s')
	assert rc == 241 and files(tmp_path / 'w') == {b'x': b'X', b'c/x': b'C'}, output[-2000:]


# A directory that lands on itself copies nothing, so it neither reads nor writes.

@pytest.mark.parametrize('flags', MODES, ids=MODE_IDS)
def test_a_directory_landing_on_itself_takes_the_update(tmp_path, flags):
	# A glob rerun: hpcp */conf -d merged/.
	write(tmp_path / 'a/conf/x', 'A')
	write(tmp_path / 'merged/conf/x', 'old')
	write(tmp_path / 'merged/conf/keep', 'K')
	rc, output = run(['a/conf', 'merged/conf', '-d', 'merged/'] + flags, tmp_path)
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'merged') == {b'conf/x': b'A', b'conf/keep': b'K'} and b'Merged directories' not in output


@pytest.mark.parametrize('args, source', [(['d1', 'd2/d1', '-d', 'd2/'], 'd1/x'), (['a/', 'd2/d1', '-d', 'd2/'], 'a/d1/x')], ids=['named', 'contents'])
def test_another_source_landing_on_a_directory_that_lands_on_itself_runs(tmp_path, args, source):
	write(tmp_path / source, 'A')
	write(tmp_path / 'd2/d1/x', 'B')
	write(tmp_path / 'd2/d1/k', 'K')
	rc, output = run(args, tmp_path)
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'd2') == {b'd1/x': b'A', b'd1/k': b'K'}


def test_a_directory_at_a_sources_place_is_no_clash(tmp_path):
	# Only a file at another source's place is: out/a/s has no z, so a/s/z lands on nothing there.
	write(tmp_path / 'a/y', 'Y')
	write(tmp_path / 'a/s/z', 'Z')
	write(tmp_path / 'out/a/s/o', 'O')
	rc, output = run(['a', 'out/a/s', '-d', 'out/'], tmp_path)
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'out') == {b'a/y': b'Y', b'a/s/z': b'Z', b'a/s/o': b'O', b's/o': b'O'}


def test_safe_a_move_onto_itself_stays_refused(tmp_path):
	write(tmp_path / 'merged/conf/x', 'M')
	write(tmp_path / 'a/conf/x', 'A')
	rc, output = run(['a/conf', 'merged/conf', '-d', 'merged/', '-rf'], tmp_path)
	assert rc == 241 and files(tmp_path) == {b'merged/conf/x': b'M', b'a/conf/x': b'A'}, output[-2000:]


def test_a_source_holding_the_destination_leaves_it_out(tmp_path):
	# As a list does. Its copy would read out/x while ./x overwrites it, into a nested copy nobody asked for.
	write(tmp_path / 'x', 'X')
	write(tmp_path / 's/a', 'A')
	for _ in range(2):
		rc, output = run(['.', '-d', 'out/'], tmp_path)
		assert rc == 0, output[-2000:]
		assert files(tmp_path / 'out') == {b'x': b'X', b's/a': b'A'}


@pytest.mark.parametrize('source', [['.'], ['-sfl', 'names']], ids=['argument', 'list'])
def test_rme_takes_nothing_from_a_destination_a_source_holds(tmp_path, source):
	# d/keep was left out of the copy of '.', so it looks extra there, and it is a source file.
	write(tmp_path / 'x', 'X')
	write(tmp_path / 'd/keep', 'K')
	write(tmp_path / 'names', '.\n')
	rc, output = _rme_in_a_terminal(tmp_path, source + ['-d', 'd/'])
	assert rc == 0 and b'-rme checked nothing: a source is or holds the destination' in output and b'(y/n)' not in output, output[-2000:]
	assert (tmp_path / 'd/keep').read_text() == 'K' and (tmp_path / 'd/x').read_text() == 'X'


def test_dot_dot_holding_the_destination_runs_twice_the_same(tmp_path):
	write(tmp_path / 'w/t', 'T')
	write(tmp_path / 'w/sub/s', 'S')
	for _ in range(2):
		rc, output = run(['..', '-d', 'out/'], tmp_path / 'w/sub')
		assert rc == 0, output[-2000:]
		assert files(tmp_path / 'w/sub/out') == {b't': b'T', b'sub/s': b'S'}


def test_safe_a_move_of_a_source_holding_the_destination_stays_refused(tmp_path):
	write(tmp_path / 'x', 'X')
	rc, output = run(['.', '-d', 'out/', '-rf'], tmp_path)
	assert rc == 241 and (tmp_path / 'x').read_text() == 'X', output[-2000:]


def test_safe_a_named_file_that_lands_on_itself_is_not_overwritten(tmp_path):
	# Only a directory that lands on itself gives way: a file named directly was asked for as it is.
	write(tmp_path / 'x', 'mine')
	write(tmp_path / 'y/x', 'other')
	for args in (['x', 'y/x', '-d', './'], ['y/x', 'x', '-d', './']):
		rc, output = run(args, tmp_path)
		assert rc == 213 and files(tmp_path) == {b'x': b'mine', b'y/x': b'other'}, output[-2000:]


@pytest.mark.parametrize('spelling', ['merged/conf', 'lm/conf'], ids=['plain', 'through-a-link'])
def test_a_directory_landing_on_itself_runs_no_copy(hpcp_mod, reset_hpcp_globals, tmp_path, monkeypatch, spelling):
	# Another source may be writing its files right now: reading one (a hash, a size) fails, and cp would fail on the same file.
	write(tmp_path / 'a/conf/x', 'A')
	write(tmp_path / 'merged/conf/x', 'old')
	write(tmp_path / 'merged/conf/keep', 'K')
	os.symlink('merged', tmp_path / 'lm')
	monkeypatch.chdir(tmp_path)
	copied = []
	real_copy = hpcp_mod.copy_file
	monkeypatch.setattr(hpcp_mod, 'copy_file', lambda src, *args, **kwargs: copied.append(src) or real_copy(src, *args, **kwargs))
	assert hpcp_mod.hpcp(['a/conf', spelling], dest_paths=['merged/'], single_thread=True, batch=True) == 0
	assert copied == [str(tmp_path / 'a/conf/x')] and (tmp_path / 'merged/conf/x').read_text() == 'A'


def test_rme_knows_the_files_of_a_directory_that_lands_on_itself(tmp_path):
	write(tmp_path / 'a/sub/x', 'A')
	write(tmp_path / 'out/sub/k', 'K')
	os.symlink('k', tmp_path / 'out/sub/lk')
	os.makedirs(tmp_path / 'out/sub/empty')
	write(tmp_path / 'out/stale', 'gone')
	rc, output = _rme_in_a_terminal(tmp_path, ['a/', 'out/sub', '-d', 'out/'])
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'out') == {b'sub/x': b'A', b'sub/k': b'K', b'sub/lk': b'-> k'} and (tmp_path / 'out/sub/empty').is_dir()


# The rule judges what is really copied: the listing's files, not what happens to exist.

def test_an_excluded_entry_lands_nowhere(tmp_path):
	# a/b is left out per entry, so nothing lands on out/b, the other source.
	write(tmp_path / 'a/b', 'A')
	write(tmp_path / 'a/y', 'Y')
	write(tmp_path / 'out/b/x', 'B')
	rc, output = run(['a/', 'out/b/', '-d', 'out/', '-e', '*/a/b'], tmp_path)
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'out') == {b'y': b'Y', b'x': b'B', b'b/x': b'B'}


def test_safe_an_excluded_file_is_not_read_so_landing_there_is_no_clash(tmp_path):
	# out/d/x is left out of its source: it is an ordinary destination file for a/d/x.
	write(tmp_path / 'a/d/x', 'A')
	write(tmp_path / 'out/d/x', 'old')
	write(tmp_path / 'out/d/y', 'Y')
	rc, output = run(['a', 'out/d/', '-d', 'out/', '-e', '*/out/d/x'], tmp_path)
	assert b'would land on' not in output and rc == 0, output[-2000:]


@pytest.mark.parametrize('kind', ['fifo', 'socket'])
def test_a_special_file_lands_nowhere(tmp_path, monkeypatch, kind):
	# The listing skips it, so a/b lands nowhere, whatever lies at out/b.
	import socket
	write(tmp_path / 'a/y', 'Y')
	write(tmp_path / 'out/b/x', 'B')
	monkeypatch.chdir(tmp_path)
	if kind == 'fifo':
		os.mkfifo('a/b')
	else:
		sock = socket.socket(socket.AF_UNIX)
		sock.bind('a/b')
		sock.close()
	rc, output = run(['a/', 'out/b/', '-d', 'out/'], tmp_path)
	assert rc == 0 and b'would land on' not in output, output[-2000:]
	assert files(tmp_path / 'out') == {b'y': b'Y', b'x': b'B', b'b/x': b'B'}


@pytest.mark.parametrize('flags', [[], ['-rf']], ids=['copy', 'move'])
def test_safe_no_working_directory_is_needed(tmp_path, flags):
	# Absolute entries with the working directory gone: nothing here may ask for it.
	write(tmp_path / 'a', 'A')
	write(tmp_path / 'out/keep', 'K')
	os.mkdir(tmp_path / 'gone')
	code = ('import os, subprocess, sys\n'
		'os.chdir(sys.argv[1]); os.rmdir(sys.argv[1])\n'
		'task = subprocess.run([sys.executable] + sys.argv[2:], input=sys.stdin.buffer.read(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)\n'
		'sys.stdout.buffer.write(task.stdout); sys.exit(task.returncode)\n')
	listing = (str(tmp_path / 'a') + '\n' + str(tmp_path / 'out/keep') + '\n').encode()
	rc, output = run_python(code, [str(tmp_path / 'gone'), HPCP, '-d', str(tmp_path / 'out') + '/'] + flags, listing)
	assert rc == 0 and b'Traceback' not in output, output[-2000:]
	assert (tmp_path / 'out/keep').read_text() == 'K' and os.path.lexists(tmp_path / 'a') == (not flags)
	assert (tmp_path / 'out' / str(tmp_path / 'a').lstrip(os.sep)).read_text() == 'A'


def test_safe_a_relative_destination_without_a_working_directory_lands_nowhere(tmp_path):
	# Leaving a removed working directory (Python 3.14's workers ask for one) must not turn this into a path under /.
	write(tmp_path / 'a', 'A')
	os.mkdir(tmp_path / 'gone')
	code = ('import os, subprocess, sys\n'
		'os.chdir(sys.argv[1]); os.rmdir(sys.argv[1])\n'
		'sys.exit(subprocess.run([sys.executable] + sys.argv[2:], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode)\n')
	rc, _ = run_python(code, [str(tmp_path / 'gone'), HPCP, str(tmp_path / 'a'), '-d', 'hpcp_relative_landing/'], b'')
	assert rc != 0 and not os.path.lexists('/hpcp_relative_landing')


def run_python(code, args, stdin):
	task = subprocess.run([sys.executable, '-c', code] + args, input=stdin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600, start_new_session=True)
	return task.returncode, task.stdout


# Cost: a directory is looked at once, and no file gets a stat.

def test_real_places_look_at_each_directory_once(hpcp_mod, tmp_path, monkeypatch):
	for name in range(50):
		write(tmp_path / 'out/d' / f'f{name}', 'x')
	os.symlink('d', tmp_path / 'out/l')
	scans, stats = [], []
	real_scandir, real_islink = os.scandir, os.path.islink
	monkeypatch.setattr(hpcp_mod.os, 'scandir', lambda path: scans.append(path) or real_scandir(path))
	monkeypatch.setattr(hpcp_mod.os.path, 'islink', lambda path: stats.append(path) or real_islink(path))
	places = hpcp_mod._RealPlaces()
	root = os.path.realpath(tmp_path)
	for name in range(50):
		assert places.landing(root + f'/out/l/f{name}', root + f'/s/f{name}') == root + f'/out/d/f{name}'
		assert places.place(root + f'/out/l/f{name}') == root + f'/out/d/f{name}'
	assert scans == [root + '/out/d'] and not [path for path in stats if os.path.basename(path).startswith('f')]


def test_real_places_agree_with_realpath(hpcp_mod, tmp_path):
	write(tmp_path / 'r/d/f', 'x')
	os.symlink('d', tmp_path / 'r/one')
	os.symlink('../r/one', tmp_path / 'r/two')
	os.symlink('nowhere', tmp_path / 'r/dangling')
	places = hpcp_mod._RealPlaces()
	root = os.path.realpath(tmp_path)
	for path in ('r/two', 'r/two/f', 'r/one/missing/deeper', 'r/dangling', 'r/dangling/x', 'r/d/f', 'nothing/here'):
		assert places.directory(root + '/' + path) == os.path.realpath(root + '/' + path)


def test_safe_a_copy_into_a_new_destination_lists_nothing_early(hpcp_mod, tmp_path, monkeypatch):
	write(tmp_path / 'src/d/x', 'x')
	monkeypatch.setattr(hpcp_mod, 'get_file_list', lambda *args, **kwargs: pytest.fail('listed'))
	assert hpcp_mod._settle_destinations([str(tmp_path / 'src/')], [str(tmp_path / 'new') + '/'])[:2] == ({}, 0)


def test_a_copy_onto_an_existing_tree_reads_each_directory_once_and_stats_no_file(hpcp_mod, tmp_path, monkeypatch):
	for name in range(30):
		write(tmp_path / 'src/d' / f'f{name}', 'x')
		write(tmp_path / 'out/d' / f'f{name}', 'x')
	scans = []
	real_scandir, real_lstat = os.scandir, os.lstat
	monkeypatch.chdir(tmp_path)
	hpcp_mod.get_file_list(str(tmp_path / 'src') + '/', parallel_file_listing=False, max_workers=1)
	monkeypatch.setattr(hpcp_mod.os, 'scandir', lambda path: scans.append(path) or real_scandir(path))
	lstats = []
	monkeypatch.setattr(hpcp_mod.os, 'lstat', lambda path, **kwargs: lstats.append(path) or real_lstat(path, **kwargs))
	try:
		assert hpcp_mod._settle_destinations(['src/'], ['out/'], max_workers=1)[:2] == ({}, 0)
	finally:
		hpcp_mod._get_file_list_cache.clear()
	root = str(tmp_path)
	assert sorted(scans) == [root + '/out', root + '/out/d']
	assert not [path for path in lstats if os.path.basename(str(path)).startswith('f')]


# ---- a link entry whose directory nobody made, and a link to a directory in the list ----

def _linked_tree(root):
	write(root / 'f0', 'X')
	os.makedirs(root / 'real')
	os.symlink('../f0', root / 'real/l')
	os.symlink('real', root / 'lnk')


@pytest.mark.parametrize('args, stdin, expected', [
	(['-d', 'out'], b'lnk/l\n', {b'lnk/l': b'-> ../f0'}),
	(['-d', 'out', '-nds'], b'real/l\n', {b'real/l': b'-> ../f0'}),
	(['-d', 'out', '-rf'], b'lnk/l\n', {b'lnk/l': b'-> ../f0'}),
	(['-d', 'out'], b'lnk/l\nlnk\n', {b'lnk/l': b'-> ../f0'}),
], ids=['through a link', '-nds', 'move', 'the link listed too'])
def test_a_link_entry_gets_its_directory(tmp_path, args, stdin, expected):
	_linked_tree(tmp_path)
	rc, output = run(args, tmp_path, stdin)
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'out') == expected


def test_find_with_links_followed_copies_every_link_entry(tmp_path):
	_linked_tree(tmp_path)
	listing = subprocess.run(['find', '-L', '.', '-type', 'f'], cwd=str(tmp_path), stdout=subprocess.PIPE).stdout
	rc, output = run(['-d', 'out'], tmp_path, listing)
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'out') == {b'f0': b'X', b'lnk/l': b'-> ../f0', b'real/l': b'-> ../f0'}


def test_a_link_in_a_subdirectory_gets_its_directory_without_directory_sync(tmp_path):
	# -nds makes no directory, and a directory holding only links gets none from a file's copy either: 9.80 exited 162.
	write(tmp_path / 'f0', 'X')
	os.makedirs(tmp_path / 'real/sub')
	os.symlink('../../f0', tmp_path / 'real/sub/m')
	rc, output = run(['real', '-d', 'out/', '-nds'], tmp_path)
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'out') == {b'real/sub/m': b'-> ../../f0'}


def test_a_directory_is_not_made_for_a_link_when_directories_are_not_to_be(tmp_path):
	_linked_tree(tmp_path)
	rc, output = run(['-d', 'out', '-ncd'], tmp_path, b'lnk/l\n')
	assert rc != 0 and not os.path.exists(tmp_path / 'out/lnk')


def test_a_directory_reached_through_a_link_keeps_the_mode_of_the_real_one(tmp_path):
	write(tmp_path / 'real/f', 'F')
	os.chmod(tmp_path / 'real', 0o700)
	os.symlink('real', tmp_path / 'lnk')
	for name in ('lnk', 'real'):
		rc, output = run(['-d', 'out_' + name], tmp_path, name.encode() + b'/f\n')
		assert rc == 0, output[-2000:]
		assert stat.S_IMODE(os.stat(tmp_path / ('out_' + name) / name).st_mode) == 0o700


@pytest.mark.parametrize('flags', [[], ['-rf']], ids=['copy', 'move'])
def test_find_following_a_link_to_a_directory_copies_its_files(tmp_path, flags):
	write(tmp_path / 'd/a', 'A')
	os.symlink('d', tmp_path / 'lnk')
	rc, output = run(['-d', 'out'] + flags, tmp_path, b'lnk\nlnk/a\n')
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'out') == {b'lnk/a': b'A'}
	assert os.path.isdir(tmp_path / 'out/lnk') and not os.path.islink(tmp_path / 'out/lnk')
	if flags:
		assert not os.path.exists(tmp_path / 'd/a')


def test_a_link_to_a_directory_listed_alone_is_still_copied_as_a_link(tmp_path):
	write(tmp_path / 'd/a', 'A')
	os.symlink('d', tmp_path / 'lnk')
	rc, output = run(['-d', 'out'], tmp_path, b'lnk\n')
	assert rc == 0, output[-2000:]
	assert files(tmp_path / 'out') == {b'lnk': b'-> d'}


# ---- modes: -di (needs root and loop devices: VM 901) ----

@pytest.mark.root
@pytest.mark.loop
@pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() != 0, reason='needs root and loop devices')
def test_dest_image_keeps_absolute_paths_and_a_list_only_cleans_its_directories(work, hpcp_mod, reset_hpcp_globals, at_a_tty, monkeypatch):
	monkeypatch.chdir(work)
	image = str(work / 'dest.img')
	mount = str(work / 'mnt')
	os.mkdir(mount)
	try:
		assert hpcp_list(hpcp_mod, work, b'src/a\nsrc/b/x.txt\n', dest=None, dest_image=image, dest_image_size=80 * 1024 * 1024) == 0
		subprocess.run(['mount', '-o', 'loop', image, mount], check=True)
		inside = mount + str(work)
		assert (open(inside + '/src/a/deep/y.log').read(), open(inside + '/src/b/x.txt').read()) == ('three', 'two')
		write(mount + '/elsewhere/keep', 'stays')
		write(inside + '/src/a/stale', 'gone')
		subprocess.run(['umount', mount], check=True)
		assert hpcp_list(hpcp_mod, work, b'src/a\nsrc/b/x.txt\n', dest=None, dest_image=image, remove_extra=True) == 0
		subprocess.run(['mount', '-o', 'loop', image, mount], check=True)
		assert os.path.exists(mount + '/elsewhere/keep') and not os.path.exists(inside + '/src/a/stale')
	finally:
		subprocess.run(['umount', mount], stderr=subprocess.DEVNULL)


@pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() != 0, reason='needs root and loop devices')
def test_an_image_mounted_inside_its_source_takes_a_rerun(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# hpcp / -di in small: the source reads what the last run put in the image, and that is no clash.
	write(tmp_path / 'src/x', 'X')
	os.mkdir(tmp_path / 'src/tmp')
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path / 'src/tmp'))
	for _ in range(2):
		assert hpcp_mod.hpcp(['src/'], dest_image=str(tmp_path / 'dest.img'), dest_image_size=80 * 1024 * 1024, batch=True) == 0


def test_a_source_that_holds_the_image_is_not_listed_before_the_copy(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# hpcp / -di in small: an image is a tree of its own, so the run pays nothing to look for clashes with it.
	write(tmp_path / 'src/x', 'X')
	write(tmp_path / 'src/tmp/x', 'earlier run')
	def listed(*args, **kwargs):
		raise AssertionError('listed before the copy')
	monkeypatch.setattr(hpcp_mod, 'get_file_list', listed)
	got = hpcp_mod._settle_destinations([str(tmp_path / 'src')], [str(tmp_path / 'src/tmp')], dest_image=str(tmp_path / 'dest.img'))
	assert got == ({}, 0, set())


# ---- a directory job judged as a whole; a hard link at the landing; a source that is the destination ----


def test_a_link_in_the_destination_onto_a_source_directory_protects_its_files(tmp_path):
	# src/d/x lands on itself through out/d, yet src/ as a whole does not land on itself, so it still reads src/d/x.
	write(tmp_path / 'src/d/x', 'S')
	write(tmp_path / 'b/d/x', 'B')
	os.makedirs(tmp_path / 'out')
	os.symlink('../src/d', tmp_path / 'out/d')
	tree = files(tmp_path)
	rc, output = run(['src/', 'b/', '-d', 'out/'], tmp_path)
	assert rc == 213 and files(tmp_path) == tree, output[-2000:]


def test_a_link_in_the_destination_onto_a_source_file_protects_it(tmp_path):
	write(tmp_path / 'a/x', 'A')
	write(tmp_path / 'b/x', 'B')
	os.makedirs(tmp_path / 'out')
	os.symlink('../a/x', tmp_path / 'out/x')
	tree = files(tmp_path)
	rc, output = run(['a/', 'b/', '-d', 'out/'], tmp_path)
	assert rc == 213 and files(tmp_path) == tree, output[-2000:]


def test_safe_one_source_over_a_link_to_itself_copies_nothing_and_runs(tmp_path):
	write(tmp_path / 'src/d/x', 'S')
	os.makedirs(tmp_path / 'out')
	os.symlink('../src/d', tmp_path / 'out/d')
	rc, output = run(['src/', '-d', 'out/'], tmp_path)
	assert rc == 0 and (tmp_path / 'src/d/x').read_text() == 'S', output[-2000:]


def test_content_only_writes_no_source_through_a_hard_link(tmp_path):
	# cp -f rewrites the inode, and b/y is that inode too.
	write(tmp_path / 'a/x', 'A' * 5000)
	write(tmp_path / 'b/y', 'B' * 3000)
	os.makedirs(tmp_path / 'out')
	os.link(tmp_path / 'b/y', tmp_path / 'out/x')
	rc, output = run(['a/x', '-d', 'out/', '-co'], tmp_path)
	assert rc == 0, output[-2000:]
	assert (tmp_path / 'b/y').read_text() == 'B' * 3000 and (tmp_path / 'out/x').read_text() == 'A' * 5000


def test_content_only_writes_no_source_through_a_symlink_and_a_hard_link(tmp_path):
	write(tmp_path / 'a/x', 'A' * 5000)
	write(tmp_path / 'b/y', 'B' * 3000)
	os.makedirs(tmp_path / 'out')
	os.link(tmp_path / 'b/y', tmp_path / 't')
	os.symlink('../t', tmp_path / 'out/x')
	rc, output = run(['a/x', '-d', 'out/', '-co'], tmp_path)
	assert rc == 0, output[-2000:]
	assert (tmp_path / 'b/y').read_text() == 'B' * 3000 and (tmp_path / 'out/x').read_text() == 'A' * 5000


@pytest.mark.parametrize('sources, link, landing, source', [
	(['a/'], 'a/y', 'out/x', 'out/y'),
	(['a/', 'b/'], 'b/x', 'out/y', 'out/x'),
], ids=['own-file', 'another-source'])
def test_a_content_only_move_keeps_every_file_through_a_hard_link(tmp_path, sources, link, landing, source):
	for attempt in range(4):
		root = tmp_path / str(attempt)
		write(root / 'a/x', 'X' * 200000)
		write(root / 'a/y', 'Y' * 200000)
		write(root / 'b/x', 'B' * 200000)
		os.makedirs(root / 'out')
		os.link(root / link, root / landing)
		rc, output = run(sources + ['-d', 'out/', '-co', '-rf'] + (['-s'] if attempt else []), root)
		wanted = {b'X' * 200000, b'Y' * 200000} | ({b'B' * 200000} if len(sources) > 1 else set())
		assert wanted <= set(files(root).values()), (rc, output[-2000:])


def test_safe_content_only_keeps_a_plain_destination_and_makes_a_new_one(tmp_path):
	write(tmp_path / 'a/x', 'new')
	write(tmp_path / 'a/y', 'made')
	write(tmp_path / 'out/x', 'old')
	os.chmod(tmp_path / 'out/x', 0o640)
	inode = os.stat(tmp_path / 'out/x').st_ino
	rc, output = run(['a/', '-d', 'out/', '-co'], tmp_path)
	assert rc == 0, output[-2000:]
	assert (tmp_path / 'out/x').read_text() == 'new' and (tmp_path / 'out/y').read_text() == 'made'
	assert os.stat(tmp_path / 'out/x').st_ino == inode and stat.S_IMODE(os.stat(tmp_path / 'out/x').st_mode) == 0o640


@pytest.mark.parametrize('sources, dest, nested', [
	(['a', 'b', 'backup'], 'backup/', 'backup/backup'),
	(['a'], 'a/', 'a/a'),
	(['a', 'b'], 'b/', 'b/b'),
], ids=['glob', 'alone', 'last'])
def test_a_source_that_is_the_destination_is_left_out_on_every_run(tmp_path, sources, dest, nested):
	write(tmp_path / 'a/x', 'A')
	write(tmp_path / 'b/y', 'B')
	os.makedirs(tmp_path / 'backup')
	for _ in range(3):
		rc, output = run(sources + ['-d', dest], tmp_path)
		assert rc == 0 and b'list entries' not in output, output[-2000:]
		assert not os.path.lexists(tmp_path / nested)


def test_a_move_leaves_out_the_source_that_is_the_destination(tmp_path):
	write(tmp_path / 'a/x', 'A')
	write(tmp_path / 'b/y', 'B')
	rc, output = run(['a', 'b', '-d', 'b/', '-rf'], tmp_path)
	assert rc == 0, output[-2000:]
	assert files(tmp_path) == {b'b/a/x': b'A', b'b/y': b'B'}


def test_a_move_of_the_destination_alone_moves_nothing(tmp_path):
	write(tmp_path / 'a/x', 'A')
	rc, output = run(['a', '-d', 'a/', '-rf'], tmp_path)
	assert rc == 0 and b'Nothing to copy' in output, output[-2000:]
	assert files(tmp_path) == {b'a/x': b'A'}


def test_rme_takes_nothing_from_a_source_left_out_as_the_destination(tmp_path):
	write(tmp_path / 'a/x', 'A')
	write(tmp_path / 'b/y', 'B')
	rc, output = _rme_in_a_terminal(tmp_path, ['a', 'b', '-d', 'b/'])
	assert rc == 0 and b'-rme checked nothing' in output and b'(y/n)' not in output, output[-2000:]
	assert (tmp_path / 'b/y').read_text() == 'B' and (tmp_path / 'b/a/x').read_text() == 'A'


@pytest.mark.parametrize('dests', [['a/'], ['a/', 'out/']], ids=['one', 'two'])
@pytest.mark.parametrize('sources', [['a/', 'b/'], ['a/', 'a', 'b/']], ids=['alone', 'beside-its-named-twin'])
def test_safe_a_source_that_lands_on_itself_is_not_left_out(tmp_path, dests, sources):
	# 'a/' has no name to land under: it is the landing, not something to move into it, and b/x would land on a/x.
	write(tmp_path / 'a/x', 'A')
	write(tmp_path / 'b/x', 'B')
	os.makedirs(tmp_path / 'out')
	tree = files(tmp_path)
	rc, output = run(sources + [arg for dest in dests for arg in ('-d', dest)] + ['-rf'], tmp_path)
	assert rc == 241 and files(tmp_path) == tree, output[-2000:]


# ---- what a landing directory can and cannot answer ----


no_permissions_as_root = pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() == 0, reason='root reads every directory')


def _link_at_a_landing(tmp_path, mode):
	"""a/y would write through out/y onto b/x, in an out of that mode. (out's mode comes back when the test ends.)"""
	write(tmp_path / 'a/x', 'A')
	write(tmp_path / 'a/y', 'C')
	write(tmp_path / 'b/x', 'B')
	os.makedirs(tmp_path / 'out')
	os.symlink('../b/x', tmp_path / 'out/y')
	tree = files(tmp_path)
	os.chmod(tmp_path / 'out', mode)
	return tree


@pytest.fixture
def out_back(tmp_path):
	yield
	if os.path.isdir(tmp_path / 'out'):
		os.chmod(tmp_path / 'out', 0o700)


@no_permissions_as_root
def test_a_directory_that_cannot_be_listed_is_asked_by_name(tmp_path, out_back):
	tree = _link_at_a_landing(tmp_path, 0o300)
	rc, output = run(['a/', 'b/', '-d', 'out/', '-rf'], tmp_path)
	os.chmod(tmp_path / 'out', 0o700)
	assert rc == 213 and files(tmp_path) == tree, output[-2000:]


@no_permissions_as_root
def test_safe_a_directory_that_cannot_be_listed_still_takes_a_safe_copy(tmp_path, out_back):
	write(tmp_path / 'a/x', 'A')
	os.makedirs(tmp_path / 'out')
	os.chmod(tmp_path / 'out', 0o300)
	rc, output = run(['a/', '-d', 'out/', '-rf'], tmp_path)
	os.chmod(tmp_path / 'out', 0o700)
	assert rc == 0, output[-2000:]
	assert files(tmp_path) == {b'out/x': b'A'}


@no_permissions_as_root
@pytest.mark.parametrize('mode', [0o000, 0o600], ids=['nothing', 'read-only'])
def test_a_directory_that_cannot_be_searched_is_refused_by_name(tmp_path, out_back, mode):
	tree = _link_at_a_landing(tmp_path, mode)
	rc, output = run(['a/', 'b/', '-d', 'out/', '-rf'], tmp_path)
	os.chmod(tmp_path / 'out', 0o700)
	assert rc == 213 and files(tmp_path) == tree, output[-2000:]
	assert b'chmod u+rx' in output and str(tmp_path / 'out').encode() in output


@no_permissions_as_root
def test_a_run_into_a_directory_that_cannot_be_searched_is_refused_without_a_link(tmp_path, out_back):
	write(tmp_path / 'a/x', 'A')
	os.makedirs(tmp_path / 'out')
	os.chmod(tmp_path / 'out', 0o600)
	rc, output = run(['a/', '-d', 'out/', '-rf'], tmp_path)
	os.chmod(tmp_path / 'out', 0o700)
	assert rc == 213 and files(tmp_path) == {b'a/x': b'A'}, output[-2000:]


def _a_big_directory(root, entries):
	os.makedirs(root)
	for number in range(entries):
		os.close(os.open(os.path.join(str(root), f'e{number}'), os.O_CREAT | os.O_WRONLY))


def test_safe_a_link_past_the_listing_limit_is_still_found(tmp_path):
	_a_big_directory(tmp_path / 'out', 10100)
	write(tmp_path / 'a/y', 'A')
	write(tmp_path / 'b/x', 'B')
	os.symlink('../b/x', tmp_path / 'out/y')
	rc, output = run(['a/', 'b/', '-d', 'out/', '-rf'], tmp_path)
	assert rc == 213 and (tmp_path / 'a/y').read_text() == 'A' and (tmp_path / 'b/x').read_text() == 'B', output[-2000:]


def test_safe_a_run_past_the_listing_limit_copies(tmp_path):
	_a_big_directory(tmp_path / 'out', 10100)
	write(tmp_path / 'a/y', 'A')
	write(tmp_path / 'b/x', 'B')
	rc, output = run(['a/', 'b/', '-d', 'out/', '-rf'], tmp_path)
	assert rc == 0, output[-2000:]
	assert (tmp_path / 'out/y').read_text() == 'A' and (tmp_path / 'out/x').read_text() == 'B'


def test_one_file_into_a_big_directory_reads_only_the_limit(hpcp_mod, reset_hpcp_globals, tmp_path, monkeypatch):
	_a_big_directory(tmp_path / 'out', 20000)
	write(tmp_path / 'f', 'F')
	read = []
	real_scandir = os.scandir

	class Counted:
		def __init__(self, path):
			self.entries = real_scandir(path)
		def __enter__(self):
			return self
		def __exit__(self, *args):
			self.entries.close()
		def __iter__(self):
			for entry in self.entries:
				read.append(entry.name)
				yield entry

	monkeypatch.setattr(os, 'scandir', Counted)
	monkeypatch.chdir(tmp_path)
	places = hpcp_mod._RealPlaces()
	assert places.landing(str(tmp_path / 'out/f'), str(tmp_path / 'f')) == str(tmp_path / 'out/f')
	assert len(read) <= 10001


def test_the_listing_reads_as_far_as_the_names_asked_about(hpcp_mod, tmp_path):
	_a_big_directory(tmp_path / 'out', 10100)
	os.symlink('e0', tmp_path / 'out/l')
	assert hpcp_mod._RealPlaces().links(str(tmp_path / 'out')) is None
	assert hpcp_mod._RealPlaces().links(str(tmp_path / 'out'), 20000) == {'l'}
	assert hpcp_mod._RealPlaces(20000).links(str(tmp_path / 'out')) == {'l'}


def test_no_directory_answers_no_links_and_a_failed_scan_answers_nothing(hpcp_mod, tmp_path, monkeypatch):
	os.makedirs(tmp_path / 'out')
	assert hpcp_mod._RealPlaces().links(str(tmp_path / 'gone')) == set()

	def refused(path):
		raise PermissionError(13, 'Permission denied', path)

	monkeypatch.setattr(os, 'scandir', refused)
	assert hpcp_mod._RealPlaces().links(str(tmp_path / 'out')) is None


def test_two_paths_of_one_directory_are_one_place(hpcp_mod, tmp_path, monkeypatch):
	# What a bind mount or a casefolded name shows: the same directory by another name.
	os.makedirs(tmp_path / 'one')
	os.makedirs(tmp_path / 'two')
	real_lstat = os.lstat
	seen = real_lstat(tmp_path / 'one')

	def lstat(path, **kwargs):
		found = real_lstat(path, **kwargs)
		if os.path.basename(str(path)) != 'two':
			return found
		return os.stat_result(found[:1] + (seen.st_ino, seen.st_dev) + found[3:])

	monkeypatch.setattr(os, 'lstat', lstat)
	places = hpcp_mod._RealPlaces()
	assert places.directory(str(tmp_path / 'one')) == places.directory(str(tmp_path / 'two'))
	assert places.place(str(tmp_path / 'two/x')) == str(tmp_path / 'one/x')


# ---- a failed copy takes only what it changed ----


@no_permissions_as_root
@pytest.mark.parametrize('flags', [[], ['-co']], ids=['plain', 'content-only'])
@pytest.mark.parametrize('source', ['a/', 'a/x'], ids=['directory', 'file'])
def test_a_source_that_cannot_be_read_leaves_the_destination_alone(tmp_path, source, flags):
	write(tmp_path / 'a/x', 'new and longer')
	write(tmp_path / 'out/x', 'good')
	os.chmod(tmp_path / 'a/x', 0o000)
	try:
		rc, output = run([source, '-d', 'out/'] + flags, tmp_path)
	finally:
		os.chmod(tmp_path / 'a/x', 0o600)
	assert rc != 0 and (tmp_path / 'out/x').read_text() == 'good', output[-2000:]


@no_permissions_as_root
@pytest.mark.parametrize('flags', [[], ['-co']], ids=['plain', 'content-only'])
def test_a_source_that_cannot_be_read_leaves_a_hard_linked_destination_alone(tmp_path, flags):
	write(tmp_path / 'a/x', 'new and longer')
	write(tmp_path / 'out/x', 'good')
	os.link(tmp_path / 'out/x', tmp_path / 'other')
	os.chmod(tmp_path / 'a/x', 0o000)
	try:
		rc, output = run(['a/', '-d', 'out/'] + flags, tmp_path)
	finally:
		os.chmod(tmp_path / 'a/x', 0o600)
	assert rc != 0 and (tmp_path / 'out/x').read_text() == 'good', output[-2000:]


def _cp_writing_a_part(hpcp_mod, monkeypatch, fails):
	"""A cp that dies part way through its write to each path in fails, and works elsewhere."""
	real = hpcp_mod.run_command_in_multicmd_with_path_check
	def cp(command, *args, **kwargs):
		if command[-1] in fails:
			with open(command[-1], 'w') as target:
				target.write('par')
			raise RuntimeError('cp: write error: No space left on device')
		return real(command, *args, **kwargs)
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', cp)


def test_a_cp_that_wrote_nothing_leaves_the_destination_alone(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	write(tmp_path / 'a/x', 'new and longer')
	write(tmp_path / 'out/x', 'good')
	def cp(*args, **kwargs):
		raise RuntimeError('cp: Too many open files')
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', cp)
	hpcp_mod.copy_file(str(tmp_path / 'a/x'), [str(tmp_path / 'out/x')])
	assert (tmp_path / 'out/x').read_text() == 'good'


@pytest.mark.parametrize('there', [None, 'old content'], ids=['new', 'over-a-file'])
def test_a_part_of_a_copy_goes_before_the_next_destination(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, there):
	write(tmp_path / 'a/x', 'the whole file')
	first, second = str(tmp_path / 'one/x'), str(tmp_path / 'two/x')
	os.makedirs(tmp_path / 'two')
	if there:
		write(tmp_path / 'one/x', there)
	_cp_writing_a_part(hpcp_mod, monkeypatch, [first])
	hpcp_mod.copy_file(str(tmp_path / 'a/x'), [first, second])
	assert not os.path.exists(first) and (tmp_path / 'two/x').read_text() == 'the whole file'


def test_a_part_of_a_copy_goes_when_no_destination_takes_it(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	write(tmp_path / 'a/x', 'the whole file')
	first, second = str(tmp_path / 'one/x'), str(tmp_path / 'two/x')
	_cp_writing_a_part(hpcp_mod, monkeypatch, [first, second])
	hpcp_mod.copy_file(str(tmp_path / 'a/x'), [first, second])
	assert not os.path.exists(first) and not os.path.exists(second)


def test_content_only_takes_a_link_loop_at_a_landing_without_a_crash(tmp_path):
	write(tmp_path / 'a/x', 'A')
	os.makedirs(tmp_path / 'out')
	os.symlink('x', tmp_path / 'out/x')
	rc, output = run(['a/', '-d', 'out/', '-co'], tmp_path)
	assert b'Traceback' not in output, output[-2000:]
