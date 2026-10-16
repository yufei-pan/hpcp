import os
import sys
import pytest


def test_delete_file_bulk_removes_files(tmp_tree, hpcp_mod, reset_hpcp_globals):
	p1 = tmp_tree.add_file('a.txt', 'a', under='dst')
	p2 = tmp_tree.add_file('b.txt', 'b', under='dst')
	hpcp_mod.delete_file_bulk([p1, p2])
	assert not os.path.exists(p1)
	assert not os.path.exists(p2)


def test_remove_extra_files_deletes_only_extras(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch):
	# Mirror process_copy mapping: src path WITHOUT trailing sep => dest/src_basename/...
	# Use trailing sep on both so files land directly under dst (see _src_to_dest_map).
	src_keep = tmp_tree.add_file('keep.txt', 'k', under='src')
	dst_keep = tmp_tree.add_file('keep.txt', 'k', under='dst')
	dst_extra = tmp_tree.add_file('extra.txt', 'e', under='dst')
	src_paths = [str(tmp_tree.src) + os.sep]
	dests = [str(tmp_tree.dst) + os.sep]
	# remove_extra_files requires a TTY + interactive 'y' before deleting (safety gate).
	monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
	monkeypatch.setattr('builtins.input', lambda: 'y')
	hpcp_mod.remove_extra_files(
		[src_keep],
		dests,
		max_workers=2,
		verbose=False,
		files_per_job=1,
		single_thread=True,
		exclude=None,
		parallel_file_listing=False,
		src_paths=src_paths,
		sym_links=None,
	)
	assert os.path.exists(dst_keep)
	assert not os.path.exists(dst_extra)


def test_process_remove_force_deletes_tree(tmp_tree, hpcp_mod, reset_hpcp_globals):
	gone = tmp_tree.add_file('a.txt', 'x')
	hpcp_mod.process_remove([str(tmp_tree.src)], single_thread=True, remove_force=True)
	assert not os.path.exists(gone)
	assert not tmp_tree.src.exists()


def test_process_remove_without_force_requires_tty(tmp_tree, hpcp_mod, reset_hpcp_globals, monkeypatch):
	tmp_tree.add_file('a.txt', 'x')
	monkeypatch.setattr(sys.stdin, 'isatty', lambda: False)
	with pytest.raises(RuntimeError, match='not a tty'):
		hpcp_mod.process_remove([str(tmp_tree.src)], single_thread=True, remove_force=False)


def test_remove_files_while_listing_deletes_entries(tmp_tree, hpcp_mod, reset_hpcp_globals):
	p = tmp_tree.add_file('gone.txt', 'x')
	hpcp_mod.get_file_list(
		str(tmp_tree.src), parallel_file_listing=False, drop_cache=True, remove_files_while_listing=True,
	)
	assert not os.path.exists(p)


def test_rwloff_listing_leaves_files(tmp_tree, hpcp_mod, reset_hpcp_globals):
	p = tmp_tree.add_file('stay.txt', 'x')
	hpcp_mod.get_file_list(
		str(tmp_tree.src), parallel_file_listing=False, drop_cache=True, remove_files_while_listing=False,
	)
	assert os.path.exists(p)


def test_remove_extra_dirs_drops_empty_extra(tmp_tree, hpcp_mod, reset_hpcp_globals):
	extra = tmp_tree.dst / 'extra_dir'
	extra.mkdir()
	hpcp_mod.remove_extra_dirs([str(tmp_tree.src) + os.sep], [str(tmp_tree.dst) + os.sep])
	assert not extra.exists()


def test_remove_force_implies_remove(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, copy_args, monkeypatch):
	called = []
	monkeypatch.setattr(hpcp_mod, 'process_remove', lambda *a, **k: called.append(True))
	tmp_tree.add_file('a.txt', 'x')
	srcs, opts = copy_args(remove=False, remove_force=True)
	hpcp_mod.hpcp(srcs, **opts)
	assert called


def test_delete_files_parallel_counts_files_removed_while_listing(
	tmp_tree, hpcp_mod, reset_hpcp_globals, capsys,
):
	"""Files unlinked during the scan must still be counted in the summary."""
	hpcp_mod.REMOVE_FILES_WHILE_LISTING = True
	for name in ('a.txt', 'b.txt', 'sub/c.txt'):
		tmp_tree.add_file(name, 'x')
	tmp_tree.add_symlink('l.lnk', 'a.txt')
	count, _ = hpcp_mod.delete_files_parallel(
		str(tmp_tree.src), max_workers=2, parallel_file_listing=False,
	)
	assert 'Number of files: 4' in capsys.readouterr().out
	# 4 entries unlinked during the scan, +1 for the directory structure itself.
	assert count == 5
	assert not tmp_tree.src.exists()


def test_delete_files_parallel_init_size_spans_all_paths(
	tmp_tree, hpcp_mod, reset_hpcp_globals, monkeypatch,
):
	"""The progress-bar baseline must be the total over every path, not the last one."""
	hpcp_mod.REMOVE_FILES_WHILE_LISTING = False
	a = tmp_tree.add_file('one/a.txt', b'a' * 100)
	b = tmp_tree.add_file('two/b.txt', b'b' * 5)
	expected = hpcp_mod.get_file_size(a) + hpcp_mod.get_file_size(b)
	seen = {}

	def fake_delete(file_list, max_workers, *a, **k):
		seen['init_size'] = k.get('init_size')
		return len(file_list), 0

	monkeypatch.setattr(hpcp_mod, 'delete_file_list_parallel', fake_delete)
	hpcp_mod.delete_files_parallel(
		[str(tmp_tree.src / 'one'), str(tmp_tree.src / 'two')],
		max_workers=2, batch=True, parallel_file_listing=False,
	)
	assert seen['init_size'] == expected


def test_cached_listing_still_removes_files(tmp_tree, hpcp_mod, reset_hpcp_globals):
	"""A warm cache entry must not let remove-while-listing skip the actual unlink."""
	p = tmp_tree.add_file('gone.txt', 'x')
	hpcp_mod.get_file_list(str(tmp_tree.src), parallel_file_listing=False)
	assert os.path.exists(p)
	hpcp_mod.get_file_list(
		str(tmp_tree.src), parallel_file_listing=False, remove_files_while_listing=True,
	)
	assert not os.path.exists(p)


@pytest.mark.parametrize('single_thread,parallel_listing', [(True, False), (False, False), (False, True)])
@pytest.mark.parametrize('remove_while_listing', [True, False])
@pytest.mark.parametrize('batch', [True, False])
def test_remove_preserves_excluded_entries(
	tmp_tree, hpcp_mod, reset_hpcp_globals, single_thread, parallel_listing,
	remove_while_listing, batch,
):
	keep = tmp_tree.add_file('parent/keep.txt', 'keep')
	protected = tmp_tree.add_file('protected/inside.txt', 'protected')
	empty = tmp_tree.add_dir('protected-empty')
	link = tmp_tree.add_symlink('keep-link', 'parent/keep.txt')
	removed = tmp_tree.add_file('gone/sub/file.txt', 'remove')
	tmp_tree.add_symlink('gone/dangling', 'missing')
	other = tmp_tree.root / 'other'
	other.mkdir()
	(other / 'file.txt').write_text('remove')
	rc = hpcp_mod.hpcp(
		[str(tmp_tree.src), str(other)], remove=True, remove_force=True,
		exclude=['keep.txt', 'protected', 'protected-empty', 'keep-link'],
		single_thread=single_thread, max_workers=2, batch=batch,
		parallel_file_listing=parallel_listing,
		do_not_remove_files_while_listing=not remove_while_listing,
	)
	assert rc == 0
	assert open(keep).read() == 'keep'
	assert open(protected).read() == 'protected'
	assert os.path.isdir(empty)
	assert os.path.islink(link)
	assert not os.path.exists(removed)
	assert not (tmp_tree.src / 'gone').exists()
	assert not other.exists()


def test_remove_extra_nested_directories_deepest_first(tmp_tree, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# Deliberately expose parent-first set iteration: cleanup must impose its own order.
	class ParentFirstSet(set):
		def __iter__(self):
			return iter(sorted(set.__iter__(self), key=lambda p: p.count(os.sep)))
	monkeypatch.setattr(hpcp_mod, 'set', ParentFirstSet, raising=False)
	(tmp_tree.dst / 'extra' / 'child' / 'leaf').mkdir(parents=True)
	hpcp_mod.remove_extra_dirs([str(tmp_tree.src) + os.sep], [str(tmp_tree.dst) + os.sep])
	assert not (tmp_tree.dst / 'extra').exists()


def test_remove_force_alone_removes_without_copying(tmp_tree, hpcp_mod, reset_hpcp_globals, monkeypatch):
	src = tmp_tree.add_file('remove-me.txt', 'remove')
	monkeypatch.chdir(tmp_tree.dst)
	rc = hpcp_mod.hpcp([src], remove_force=True, single_thread=True)
	assert rc == 0
	assert not os.path.lexists(src)
	assert not (tmp_tree.dst / 'remove-me.txt').exists()


# ---- 9.73: no removal after a copy that reported errors ----

def _copy_with_one_failure(tmp_tree):
	tmp_tree.add_file('ok/a.txt', 'a')
	tmp_tree.add_file('missing/lost.txt', 'lost')
	(tmp_tree.dst / 'ok').mkdir()


@pytest.mark.parametrize('single_thread', [True, False])
def test_move_keeps_the_sources_when_the_copy_failed(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, copy_args, single_thread):
	"""-rm with a destination is a move: a file that was not copied must not be deleted from the source."""
	_copy_with_one_failure(tmp_tree)
	srcs, opts = copy_args(single_thread=single_thread, max_workers=1 if single_thread else 2,
		no_create_dir=True, no_directory_sync=True, remove_force=True)
	rc = hpcp_mod.hpcp(srcs, **opts)
	assert (tmp_tree.src / 'missing' / 'lost.txt').read_text() == 'lost'
	assert (tmp_tree.src / 'ok' / 'a.txt').exists()
	assert rc == 212


def test_move_still_removes_the_sources_after_a_clean_copy(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, copy_args):
	tmp_tree.add_file('ok/a.txt', 'a')
	srcs, opts = copy_args(remove_force=True)
	assert hpcp_mod.hpcp(srcs, **opts) == 0
	assert (tmp_tree.dst / 'ok' / 'a.txt').read_text() == 'a'
	assert not tmp_tree.src.exists()


def test_remove_extra_is_skipped_when_the_copy_failed(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, copy_args, monkeypatch):
	_copy_with_one_failure(tmp_tree)
	extra = tmp_tree.add_file('ok/extra.txt', 'keep me', under='dst')
	monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
	monkeypatch.setattr('builtins.input', lambda: 'y')
	srcs, opts = copy_args(no_create_dir=True, no_directory_sync=True, remove_extra=True)
	assert hpcp_mod.hpcp(srcs, **opts) == 212
	assert os.path.exists(extra)


# ---- 9.73: -rme with -di (review C1) ----

def test_src_to_dest_map_mirrors_dest_image_layout(hpcp_mod, tmp_path):
	"""process_copy puts each source at dest + its absolute path in -di mode, trailing separator or not."""
	dest = str(tmp_path / 'mnt') + os.sep
	for src in ('/home/u/data', '/home/u/data/'):
		assert hpcp_mod._src_to_dest_map([src], [dest], dest_image=True) == [(os.path.abspath(src), [os.path.join(str(tmp_path / 'mnt'), 'home/u/data')])]


def test_remove_extra_after_dest_image_layout_copy_keeps_the_copies(tmp_path, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch):
	"""A -di copy (here into a plain directory standing in for the mounted image) then -rme removes only the real extra."""
	src = tmp_path / 'live' / 'data'
	(src / 'sub').mkdir(parents=True)
	(src / 'keep.txt').write_text('keep')
	(src / 'sub' / 'deep.txt').write_text('deep')
	dest = str(tmp_path / 'mnt') + os.sep
	os.makedirs(dest)
	files, links = hpcp_mod.process_copy([str(src)], [dest], single_thread=True, max_workers=1, dest_image='stand-in.img', batch=True)
	mirrored = tmp_path / 'mnt' / os.path.relpath(str(src), '/')
	assert (mirrored / 'sub' / 'deep.txt').read_text() == 'deep'
	(mirrored / 'old.txt').write_text('extra')
	(mirrored / 'olddir').mkdir()
	monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
	monkeypatch.setattr('builtins.input', lambda: 'y')
	hpcp_mod.remove_extra_files(files, [dest], 1, False, 1, single_thread=True, src_paths=[str(src)], sym_links=links, dest_image=True)
	hpcp_mod.remove_extra_dirs([str(src)], [dest], dest_image=True)
	assert (mirrored / 'keep.txt').read_text() == 'keep'
	assert (mirrored / 'sub' / 'deep.txt').read_text() == 'deep'
	assert not (mirrored / 'old.txt').exists()
	assert not (mirrored / 'olddir').exists()


# ---- 9.73: remove_extra_dirs and excludes (review C10) ----

def test_remove_extra_dirs_keeps_a_directory_holding_excluded_files(tmp_tree, hpcp_mod, reset_hpcp_globals):
	kept = tmp_tree.add_file('old/a.log', 'log', under='dst')
	empty = tmp_tree.add_dir('gone', under='dst')
	exclude = hpcp_mod.format_exclude(['*.log'])
	hpcp_mod.remove_extra_dirs([str(tmp_tree.src) + os.sep], [str(tmp_tree.dst) + os.sep], exclude=exclude)
	assert os.path.exists(kept)
	assert not os.path.exists(empty)
	assert hpcp_mod.ERRORS == []


def test_remove_extra_dirs_matches_dir_excludes_and_leaves_excluded_trees_alone(tmp_tree, hpcp_mod, reset_hpcp_globals):
	cache_sub = tmp_tree.add_dir('x/cache/sub', under='dst')
	empty = tmp_tree.add_dir('y', under='dst')
	exclude = hpcp_mod.format_exclude(['cache'])
	hpcp_mod.remove_extra_dirs([str(tmp_tree.src) + os.sep], [str(tmp_tree.dst) + os.sep], exclude=exclude)
	assert os.path.isdir(cache_sub)
	assert not os.path.exists(empty)
	assert hpcp_mod.ERRORS == []


def test_remove_extra_dirs_skips_symlinks_to_directories(tmp_tree, hpcp_mod, reset_hpcp_globals):
	target = tmp_tree.add_dir('real', under='src')
	link = tmp_tree.add_symlink('alias', target, under='dst')
	hpcp_mod.remove_extra_dirs([str(tmp_tree.src) + os.sep], [str(tmp_tree.dst) + os.sep])
	assert os.path.islink(link)


def test_remove_extra_dirs_leaves_the_callers_source_list_alone(tmp_tree, hpcp_mod, reset_hpcp_globals):
	src_paths = [str(tmp_tree.src / 'a.tmp'), str(tmp_tree.src / 'b.tmp'), str(tmp_tree.src) + os.sep]
	hpcp_mod.remove_extra_dirs(src_paths, [str(tmp_tree.dst) + os.sep], exclude=hpcp_mod.format_exclude(['*.tmp']))
	assert src_paths == [str(tmp_tree.src / 'a.tmp'), str(tmp_tree.src / 'b.tmp'), str(tmp_tree.src) + os.sep]


_IMAGE_TOOLS = ('losetup', 'mkfs.ext4', 'mount', 'umount', 'lsblk')


@pytest.mark.skipif(os.name == 'nt' or os.geteuid() != 0 or any(__import__('shutil').which(tool) is None for tool in _IMAGE_TOOLS),
	reason='needs root and losetup/mkfs.ext4/mount/umount/lsblk')
@pytest.mark.root
@pytest.mark.loop
def test_remove_extra_into_a_dest_image_removes_only_the_extra(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Review C1 end to end: hpcp SRC -di IMG -rme on an existing image keeps every copied file."""
	src = tmp_path / 'live' / 'data'
	(src / 'sub').mkdir(parents=True)
	(src / 'keep.txt').write_text('keep')
	(src / 'sub' / 'deep.txt').write_text('deep')
	(src / 'gone.txt').write_text('gone')
	(src / 'gonedir').mkdir()
	image = str(tmp_path / 'backup.img')
	opts = dict(single_thread=True, max_workers=1, batch=True, do_not_remove_files_while_listing=True)
	assert hpcp_mod.hpcp([str(src)], dest_image=image, **opts) == 0
	(src / 'gone.txt').unlink()
	(src / 'gonedir').rmdir()
	monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
	monkeypatch.setattr('builtins.input', lambda: 'y')
	assert hpcp_mod.hpcp([str(src)], dest_image=image, remove_extra=True, **opts) == 0
	mount_points, loop_devices = [], []
	try:
		root = hpcp_mod._mount_single_fs_image(image, True, mount_points, loop_devices)
		mirrored = os.path.join(root, os.path.relpath(str(src), '/'))
		assert open(os.path.join(mirrored, 'keep.txt')).read() == 'keep'
		assert open(os.path.join(mirrored, 'sub', 'deep.txt')).read() == 'deep'
		assert not os.path.exists(os.path.join(mirrored, 'gone.txt'))
		assert not os.path.exists(os.path.join(mirrored, 'gonedir'))
	finally:
		hpcp_mod.clean_up(mount_points, loop_devices)


def test_move_to_a_destination_glob_that_matches_nothing_removes_nothing(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	"""Requested destinations that all fall away must stop the run, not turn -rm into a plain delete."""
	tmp_tree.add_file('a.txt', 'a')
	rc = hpcp_mod.hpcp([str(tmp_tree.src) + os.sep], dest_paths=[str(tmp_tree.root / 'node*') + os.sep], remove_force=True,
		single_thread=True, batch=True)
	assert (tmp_tree.src / 'a.txt').read_text() == 'a'
	assert rc == 202


@pytest.mark.skipif(os.name == 'nt' or os.geteuid() != 0 or any(__import__('shutil').which(tool) is None for tool in _IMAGE_TOOLS),
	reason='needs root and losetup/mkfs.ext4/mount/umount/lsblk')
@pytest.mark.root
@pytest.mark.loop
def test_move_into_a_new_dest_image_creates_it_before_removing_the_sources(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""-rm -di NEW.img: the image is created and filled first; before 9.73 the sources were removed and no image made."""
	src = tmp_path / 'live' / 'data'
	src.mkdir(parents=True)
	(src / 'keep.txt').write_text('keep')
	image = str(tmp_path / 'moved.img')
	assert hpcp_mod.hpcp([str(src)], dest_image=image, remove_force=True, single_thread=True, max_workers=1, batch=True) == 0
	assert os.path.isfile(image) and not src.exists()
	mount_points, loop_devices = [], []
	try:
		root = hpcp_mod._mount_single_fs_image(image, True, mount_points, loop_devices)
		assert open(os.path.join(root, os.path.relpath(str(src), '/'), 'keep.txt')).read() == 'keep'
	finally:
		hpcp_mod.clean_up(mount_points, loop_devices)


def test_remove_extra_dirs_never_enters_a_mount_point(tmp_tree, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""A filesystem mounted below the destination is skipped whole, not just its root directory."""
	inner = tmp_tree.add_dir('scratch/a', under='dst')
	mount_point = str(tmp_tree.dst / 'scratch')
	real_ismount = os.path.ismount
	monkeypatch.setattr(hpcp_mod.os.path, 'ismount', lambda path: os.path.normpath(path) == mount_point or real_ismount(path))
	hpcp_mod.remove_extra_dirs([str(tmp_tree.src) + os.sep], [str(tmp_tree.dst) + os.sep])
	assert os.path.isdir(inner)


# ---- 9.73 re-review: -e names source paths, also for -rme ----

@pytest.mark.parametrize('pattern', ['absolute', 'relative'])
def test_remove_extra_keeps_the_copy_of_an_excluded_source_path(tmp_path, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch, pattern):
	"""-e /data/src/cache leaves the source's cache out; -rme must not delete the destination's earlier copy of it."""
	src = tmp_path / 'data' / 'src'
	(src / 'cache').mkdir(parents=True)
	(src / 'a.txt').write_text('a')
	(src / 'cache' / 'c.txt').write_text('cached')
	backup = str(tmp_path / 'backup') + os.sep
	os.makedirs(backup)
	opts = dict(single_thread=True, batch=True, do_not_remove_files_while_listing=True)
	assert hpcp_mod.hpcp([str(src)], dest_paths=[backup], **opts) == 0
	monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
	monkeypatch.setattr('builtins.input', lambda: 'y')
	exclude = [str(src / 'cache')] if pattern == 'absolute' else ['cache']
	assert hpcp_mod.hpcp([str(src)], dest_paths=[backup], remove_extra=True, exclude=exclude, **opts) == 0
	assert (tmp_path / 'backup' / 'src' / 'cache' / 'c.txt').read_text() == 'cached'
	assert (tmp_path / 'backup' / 'src' / 'a.txt').read_text() == 'a'
