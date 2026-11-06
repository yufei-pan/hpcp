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
import time

import pytest

HPCP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hpcp.py')


def run(args, cwd, stdin=b'', env=None):
	"""The CLI, stdin given as bytes (never a tty): (return code, stdout + stderr as bytes)."""
	full_env = dict(os.environ, **(env or {}))
	# Its own session: no controlling terminal, wherever the tests run, so a prompt cannot reach a person's.
	task = subprocess.run([sys.executable, HPCP] + args, input=stdin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
		cwd=str(cwd), env=full_env, timeout=120, start_new_session=True)
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
	# A later ancestor covers an earlier entry; a repeat goes whatever its spelling; a symlink covers nothing.
	assert kept == ['d/e', 'g/h', 'ln', 'ln/e']


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
	assert hpcp_mod.hpcp(['src/a/x.txt'], dest_paths=[str(work / 'out') + os.sep], single_thread=True, remove_extra=True) == 0
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
	start = time.monotonic()
	got = hpcp_mod._expected_dest_paths(sources, sources, ['/o/'])
	assert time.monotonic() - start < 3
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
	assert rc == 213 and b'Destination clash' in output and b'is a source' in output
	assert (tmp_path / 'out/x').read_text() == 'mine' and (tmp_path / 'out/sub/x').read_text() == 'mine'


def test_a_move_that_would_overwrite_a_source_is_refused_before_any_copy(tmp_path):
	write(tmp_path / 'a/sub/x', 'from a')
	write(tmp_path / 'a/y', 'y')
	write(tmp_path / 'out/sub/x', 'mine')
	rc, output = run(['a/', 'out/sub/x', '-d', 'out/', '-rm', '-rf'], tmp_path)
	assert rc == 241 and not (tmp_path / 'out/y').exists()
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
	assert hpcp_mod._settle_destinations(['/data/source_a', '/data/source_b'], ['/o/out/'], None, False, None, 4, False) == ({}, 0)


def test_settling_lists_when_destinations_overlap(hpcp_mod, monkeypatch):
	listings = {'/data/source_a/': (frozenset(['/data/source_a/f']), frozenset(), 0, frozenset(['/data/source_a/'])),
		'/data/source_b/': (frozenset(['/data/source_b/f']), frozenset(), 0, frozenset(['/data/source_b/']))}
	monkeypatch.setattr(hpcp_mod, 'get_file_list', lambda path, **kwargs: listings[path])
	monkeypatch.setattr(hpcp_mod.os.path, 'isdir', lambda path: path.startswith('/data/source'))
	dropped, replaced = hpcp_mod._settle_destinations(['/data/source_a', '/data/source_b'], ['/o/out'], None, False, None, 4, False)
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
