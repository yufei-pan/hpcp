import multiprocessing
import os
import pytest


def test_get_file_list_serial_lists_files(tmp_tree, hpcp_mod, reset_hpcp_globals):
	tmp_tree.add_file('a.txt', 'A')
	tmp_tree.add_file('sub/b.txt', 'B')
	# Returns (file_list, links, size, folders)
	files, links, size, folders = hpcp_mod.get_file_list(
		str(tmp_tree.src),
		parallel_file_listing=False,
		drop_cache=True,
	)
	joined = ' '.join(map(str, files))
	assert 'a.txt' in joined
	assert 'b.txt' in joined
	assert size >= 2
	assert isinstance(links, (set, frozenset))
	assert isinstance(folders, (set, frozenset))


def test_get_file_list_respects_exclude(tmp_tree, hpcp_mod, reset_hpcp_globals):
	tmp_tree.add_file('keep.txt', 'k')
	tmp_tree.add_file('skip.dat', 's')
	exclude = hpcp_mod.format_exclude(['*.dat'])
	files, links, size, folders = hpcp_mod.get_file_list(
		str(tmp_tree.src),
		exclude=exclude,
		parallel_file_listing=False,
		drop_cache=True,
	)
	joined = ' '.join(map(str, files))
	assert 'keep.txt' in joined
	assert 'skip.dat' not in joined


def test_get_file_list_parallel_matches_serial(tmp_tree, hpcp_mod, reset_hpcp_globals):
	tmp_tree.add_file('a.txt', 'A')
	tmp_tree.add_file('sub/b.txt', 'B')
	serial, _, _, _ = hpcp_mod.get_file_list(
		str(tmp_tree.src), parallel_file_listing=False, drop_cache=True,
	)
	parallel, _, _, _ = hpcp_mod.get_file_list(
		str(tmp_tree.src), parallel_file_listing=True, drop_cache=True, max_workers=2,
	)
	assert set(serial) == set(parallel)


def test_get_file_list_puts_symlinks_in_links(tmp_tree, hpcp_mod, reset_hpcp_globals):
	tmp_tree.add_file('target.txt', 't')
	tmp_tree.add_symlink('link', 'target.txt')
	files, links, _, _ = hpcp_mod.get_file_list(
		str(tmp_tree.src), parallel_file_listing=False, drop_cache=True,
	)
	joined_files = ' '.join(map(str, files))
	joined_links = ' '.join(map(str, links))
	assert 'target.txt' in joined_files
	assert 'link' in joined_links
	assert 'link' not in joined_files


def test_format_exclude_reads_exclude_file(tmp_tree, hpcp_mod, reset_hpcp_globals):
	ex = tmp_tree.root / 'exclude.txt'
	ex.write_text('*.tmp\nbuild\n')
	got = hpcp_mod.format_exclude(exclude_file=str(ex))
	assert any(p.endswith('*.tmp') or p.endswith('/.tmp') or '*.tmp' in p for p in got)
	assert any('build' in p for p in got)


def test_format_exclude_missing_file_is_skipped(hpcp_mod, reset_hpcp_globals, tmp_path):
	got = hpcp_mod.format_exclude(['cache'], exclude_file=str(tmp_path / 'nope'))
	assert '*/cache' in got


def test_get_file_list_append_hash(tmp_tree, hpcp_mod, reset_hpcp_globals):
	tmp_tree.add_file('a.txt', 'payload')
	files, _, _, _ = hpcp_mod.get_file_list(
		str(tmp_tree.src),
		append_hash=True,
		parallel_file_listing=False,
		drop_cache=True,
	)
	assert any(':' in str(f) for f in files)


def test_trim_paths_relative(hpcp_mod):
	base = '/home/user/project/main.py'
	got = hpcp_mod.trim_paths({'/home/user/project/file1.py', '/home/user/project/file2.py'}, base)
	assert got == {'file1.py', 'file2.py'}


# ---- 9.73: an unreadable entry is recorded, the rest is still listed (review C3) ----

def _deny_scandir(monkeypatch, hpcp_mod, denied):
	real = os.scandir
	def scandir(path='.'):
		if os.path.abspath(path) == os.path.abspath(denied):
			raise PermissionError(13, 'Permission denied', path)
		return real(path)
	monkeypatch.setattr(hpcp_mod.os, 'scandir', scandir)


@pytest.mark.parametrize('parallel', [False, True])
def test_unreadable_directory_is_recorded_and_the_rest_listed(tmp_tree, hpcp_mod, reset_hpcp_globals, monkeypatch, parallel):
	if parallel and multiprocessing.get_start_method() != 'fork':
		pytest.skip('the denied scandir is patched in this process: workers that do not fork (forkserver, Python 3.14) list with the real one')
	tmp_tree.add_file('a.txt', 'A')
	tmp_tree.add_file('ok/b.txt', 'B')
	tmp_tree.add_file('secret/c.txt', 'C')
	_deny_scandir(monkeypatch, hpcp_mod, tmp_tree.src / 'secret')
	files, _, _, _ = hpcp_mod.get_file_list(str(tmp_tree.src), parallel_file_listing=parallel, max_workers=2, drop_cache=True)
	names = sorted(os.path.relpath(path, tmp_tree.src) for path in files)
	assert names == ['a.txt', os.path.join('ok', 'b.txt')]
	assert [error.partition(':')[0] for error in hpcp_mod.ERRORS] == ['Listing error']


def test_unreadable_file_hash_is_recorded_and_the_rest_listed(tmp_tree, hpcp_mod, reset_hpcp_globals, monkeypatch):
	tmp_tree.add_file('a.txt', 'A')
	denied = tmp_tree.add_file('b.txt', 'B')
	real = hpcp_mod.hash_file
	def hash_file(path, *args, **kwargs):
		if path == denied:
			raise PermissionError(13, 'Permission denied', path)
		return real(path, *args, **kwargs)
	monkeypatch.setattr(hpcp_mod, 'hash_file', hash_file)
	files, _, _, _ = hpcp_mod.get_file_list(str(tmp_tree.src), append_hash=True, parallel_file_listing=False, drop_cache=True)
	assert [os.path.basename(entry.rpartition(':')[0]) for entry in files] == ['a.txt']
	assert [error.partition(':')[0] for error in hpcp_mod.ERRORS] == ['Listing error']


def test_copy_with_an_unreadable_directory_copies_the_rest(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, copy_args, monkeypatch):
	tmp_tree.add_file('a.txt', 'A')
	tmp_tree.add_file('secret/c.txt', 'C')
	_deny_scandir(monkeypatch, hpcp_mod, tmp_tree.src / 'secret')
	srcs, opts = copy_args()
	assert hpcp_mod.hpcp(srcs, **opts) == 209
	assert (tmp_tree.dst / 'a.txt').read_text() == 'A'


def test_move_after_a_cached_incomplete_listing_keeps_the_sources(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch):
	"""The remove gate sees the copy's listing errors even when that listing was cached before the copy started."""
	tmp_tree.add_file('a.txt', 'A')
	tmp_tree.add_file('secret/c.txt', 'C')
	_deny_scandir(monkeypatch, hpcp_mod, tmp_tree.src / 'secret')
	src = str(tmp_tree.src) + os.sep
	hpcp_mod.get_file_list(src, exclude=hpcp_mod.format_exclude(None), parallel_file_listing=False)
	hpcp_mod.ERRORS.clear()
	rc = hpcp_mod.hpcp([src], dest_paths=[str(tmp_tree.dst) + os.sep], remove_force=True, single_thread=True,
		batch=True, do_not_remove_files_while_listing=True)
	assert (tmp_tree.src / 'secret' / 'c.txt').exists()
	assert rc == 212
