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


@pytest.mark.parametrize('track_on_rme', [True, False])
def test_remove_extra_keeps_link_targets_that_tracking_placed(tmp_path, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch, track_on_rme):
	"""A tracked target belongs to the copy: -rme -lt must not delete it (it would come back on every run). Without -lt
	it is not part of the copy, so -rme removes it like any other extra."""
	src = tmp_path / 's' / 'src'
	src.mkdir(parents=True)
	(tmp_path / 's' / 'other').mkdir()
	(tmp_path / 's' / 'other' / 'x').write_text('x')
	os.symlink('../other/x', str(src / 'lnk'))
	dst = str(tmp_path / 'dst') + os.sep
	os.makedirs(dst)
	opts = dict(single_thread=True, batch=True, do_not_remove_files_while_listing=True)
	assert hpcp_mod.hpcp([str(src)], dest_paths=[dst], link_tracking=True, **opts) == 0
	assert (tmp_path / 'dst' / 'src' / 'lnk').read_text() == 'x'
	(tmp_path / 'dst' / 'stale.txt').write_text('extra')
	monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
	monkeypatch.setattr('builtins.input', lambda: 'y')
	assert hpcp_mod.hpcp([str(src)], dest_paths=[dst], remove_extra=True, link_tracking=track_on_rme, **opts) == 0
	assert (tmp_path / 'dst' / 'other' / 'x').exists() is track_on_rme
	assert os.readlink(tmp_path / 'dst' / 'src' / 'lnk') == '../other/x'
	assert not (tmp_path / 'dst' / 'stale.txt').exists()


def test_remove_extra_dirs_with_dest_image_follows_this_runs_sources(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""-rme -di mirrors the image to this run's sources: a directory left from another source goes, even though that
	path exists on this host; the sources' own directories and their parents stay."""
	live = tmp_path / 'live'
	(live / 'a' / 'kept').mkdir(parents=True)
	(live / 'b').mkdir()
	image_root = tmp_path / 'mnt'
	mirrored_a = image_root / os.path.relpath(str(live / 'a'), '/')
	mirrored_b = image_root / os.path.relpath(str(live / 'b'), '/')
	(mirrored_a / 'kept').mkdir(parents=True)
	(mirrored_a / 'gone').mkdir()
	(mirrored_b / 'old').mkdir(parents=True)
	hpcp_mod.remove_extra_dirs([str(live / 'a')], [str(image_root) + os.sep], dest_image=True)
	assert (mirrored_a / 'kept').is_dir()
	assert not (mirrored_a / 'gone').exists()
	assert not mirrored_b.exists()


def test_move_with_a_dangling_link_whose_path_has_a_colon_goes_on(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, copy_args):
	"""A dangling relative link is copied as is; its note is not an error, even when the path holds ':' (rc 128)."""
	tmp_tree.add_file('a.txt', 'a')
	tmp_tree.add_symlink('at:12:00', 'missing:target')
	srcs, opts = copy_args(remove_force=True, link_tracking=True)
	assert hpcp_mod.hpcp(srcs, **opts) == 0
	assert os.readlink(tmp_tree.dst / 'at:12:00') == 'missing:target'
	assert not tmp_tree.src.exists()


def test_excluded_destination_is_a_note_not_an_error(tmp_tree, hpcp_mod, reset_hpcp_globals):
	tmp_tree.add_dir('extra', under='dst')
	hpcp_mod.remove_extra_dirs([str(tmp_tree.src) + os.sep], [str(tmp_tree.dst)], exclude=hpcp_mod.format_exclude(['dst']))
	assert hpcp_mod.ERRORS == []
	assert (tmp_tree.dst / 'extra').is_dir()


# ---- 9.73 third review: what -rme keeps ----

def _rme(hpcp_mod, monkeypatch, srcs, dests, **kwargs):
	monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
	monkeypatch.setattr('builtins.input', lambda: 'y')
	return hpcp_mod.hpcp(srcs, dest_paths=dests, remove_extra=True, single_thread=True, batch=True,
		do_not_remove_files_while_listing=True, **kwargs)


@pytest.mark.parametrize('target', ['.', '..', 'releases/v3'])
def test_in_tree_directory_link_does_not_shield_its_subtree_from_remove_extra(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch, target):
	"""usr/bin/X11 -> . (or current -> releases/v3): the target is part of the copy anyway, not a tracked extra."""
	tmp_tree.add_file('sub/releases/v3/keep.txt', 'keep')
	tmp_tree.add_symlink('sub/link', target)
	stale = tmp_tree.add_file('sub/releases/v3/stale.txt', 'stale', under='dst')
	assert _rme(hpcp_mod, monkeypatch, [str(tmp_tree.src) + os.sep], [str(tmp_tree.dst) + os.sep], link_tracking=True) == 0
	assert not os.path.exists(stale)
	assert (tmp_tree.dst / 'sub' / 'releases' / 'v3' / 'keep.txt').exists()


def test_dangling_link_does_not_shield_a_stale_file_from_remove_extra(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch):
	tmp_tree.add_symlink('dangle', 'gone.txt')
	stale = tmp_tree.add_file('gone.txt', 'deleted at the source', under='dst')
	assert _rme(hpcp_mod, monkeypatch, [str(tmp_tree.src) + os.sep], [str(tmp_tree.dst) + os.sep], link_tracking=True) == 0
	assert not os.path.exists(stale)


def test_exclude_matching_the_source_root_name_does_not_shield_the_whole_destination(tmp_path, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch):
	"""hpcp /home/u/cache/ /backup/ -rme -e cache: the copy runs (the root is listed as 'cache/'); so must -rme."""
	src = tmp_path / 'u' / 'cache'
	src.mkdir(parents=True)
	(src / 'a.txt').write_text('a')
	dst = tmp_path / 'backup'
	dst.mkdir()
	(dst / 'stale.txt').write_text('stale')
	assert _rme(hpcp_mod, monkeypatch, [str(src) + os.sep], [str(dst) + os.sep], exclude=['cache']) == 0
	assert (dst / 'a.txt').read_text() == 'a'
	assert not (dst / 'stale.txt').exists()


def test_remove_extra_dirs_keeps_tracked_link_target_directories(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""-di: an empty directory that link tracking placed outside this run's sources is kept like tracked files."""
	live = tmp_path / 'live'
	(live / 'src').mkdir(parents=True)
	image_root = tmp_path / 'mnt'
	spool = image_root / os.path.relpath(str(live / 'shared' / 'spool'), '/')
	spool.mkdir(parents=True)
	(image_root / os.path.relpath(str(live / 'src'), '/')).mkdir(parents=True)
	hpcp_mod.remove_extra_dirs([str(live / 'src')], [str(image_root) + os.sep], dest_image=True, keep={(str(spool.parent), str(live / 'shared'), str(image_root / os.path.relpath(str(live / 'src' / 'l'), '/')))})
	assert spool.is_dir()


# ---- 9.73 fourth review ----

def test_exclude_that_drops_the_whole_source_root_shields_its_copy(tmp_path, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch):
	"""-e 'cache/' matches the root as the listing names it ('…/cache/'): nothing is copied, and -rme removes nothing."""
	src = tmp_path / 'u' / 'cache'
	src.mkdir(parents=True)
	(src / 'a.txt').write_text('a')
	dst = tmp_path / 'backup'
	dst.mkdir()
	(dst / 'a.txt').write_text('earlier copy')
	assert _rme(hpcp_mod, monkeypatch, [str(src) + os.sep], [str(dst) + os.sep], exclude=['cache/']) == 0
	assert (dst / 'a.txt').read_text() == 'earlier copy'


@pytest.mark.parametrize('target', ['..', '../..'])
def test_tracked_target_that_contains_the_copy_shields_nothing(tmp_path, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch, target):
	"""proj/up -> .. resolves to the destination root (or above): keeping it would keep everything."""
	src = tmp_path / 'home' / 'proj'
	(src / 'sub').mkdir(parents=True)
	(src / 'a.txt').write_text('a')
	os.symlink(target, str(src / 'sub' / 'up' if target == '../..' else src / 'up'))
	dst = tmp_path / 'backup'
	dst.mkdir()
	(dst / 'stale.txt').write_text('stale')
	assert _rme(hpcp_mod, monkeypatch, [str(src)], [str(dst) + os.sep], link_tracking=True) == 0
	assert (dst / 'proj' / 'a.txt').read_text() == 'a'
	assert not (dst / 'stale.txt').exists()


def test_remove_extra_keeps_the_copy_of_an_excluded_link_target(tmp_path, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch):
	home = tmp_path / 'home'
	(home / 'proj').mkdir(parents=True)
	(home / 'shared').mkdir()
	(home / 'shared' / 'lib').write_text('lib')
	os.symlink('../shared/lib', str(home / 'proj' / 'lib'))
	dst = tmp_path / 'backup'
	(dst / 'shared').mkdir(parents=True)
	(dst / 'shared' / 'lib').write_text('earlier copy')
	assert _rme(hpcp_mod, monkeypatch, [str(home / 'proj')], [str(dst) + os.sep], exclude=[str(home / 'shared' / 'lib')], link_tracking=True) == 0
	assert (dst / 'shared' / 'lib').read_text() == 'earlier copy'


def test_tracked_target_inside_another_sources_tree_is_kept(tmp_path, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch):
	"""Sources /a/x and /b/y both go to /backup/: x/l -> ../y/f brings /a/y/f to /backup/y/f, inside /b/y's copy."""
	a, b = tmp_path / 'a', tmp_path / 'b'
	(a / 'x').mkdir(parents=True)
	(a / 'y').mkdir()
	(a / 'y' / 'f').write_text('tracked from a')
	(b / 'y').mkdir(parents=True)
	(b / 'y' / 'g').write_text('g')
	os.symlink('../y/f', str(a / 'x' / 'l'))
	dst = tmp_path / 'backup'
	dst.mkdir()
	opts = dict(single_thread=True, batch=True, do_not_remove_files_while_listing=True)
	assert hpcp_mod.hpcp([str(a / 'x'), str(b / 'y')], dest_paths=[str(dst) + os.sep], link_tracking=True, **opts) == 0
	assert (dst / 'x' / 'l').read_text() == 'tracked from a'
	assert _rme(hpcp_mod, monkeypatch, [str(a / 'x'), str(b / 'y')], [str(dst) + os.sep], link_tracking=True) == 0
	assert (dst / 'x' / 'l').read_text() == 'tracked from a'


# ---- 9.73 fifth review ----

def test_tracked_directory_holding_another_sources_copy_keeps_its_own_contents(tmp_path, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch):
	"""-di, sources s/a and s/c/sub, link a/l -> ../c: img/s/c is tracked and holds sub's copy. Its other contents are
	kept; sub's copy is still judged as sub's (a stale file there goes)."""
	s = tmp_path / 's'
	(s / 'a').mkdir(parents=True)
	(s / 'c' / 'sub').mkdir(parents=True)
	(s / 'c' / 'other').write_text('tracked content')
	(s / 'c' / 'sub' / 'f').write_text('f')
	os.symlink('../c', str(s / 'a' / 'l'))
	mnt = str(tmp_path / 'mnt') + os.sep
	os.makedirs(mnt)
	srcs = [str(s / 'a'), str(s / 'c' / 'sub')]
	files, links = hpcp_mod.process_copy(srcs, [mnt], single_thread=True, max_workers=1, dest_image='stand-in.img', batch=True)
	keep = hpcp_mod.create_sym_links(links, link_tracking=True)
	image_c = tmp_path / 'mnt' / os.path.relpath(str(s / 'c'), '/')
	# img/s/c exists (sub's parent), so tracking found its target there; other is from an earlier run.
	(image_c / 'other').write_text('tracked content')
	(image_c / 'sub' / 'stale').write_text('stale')
	monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
	monkeypatch.setattr('builtins.input', lambda: 'y')
	hpcp_mod.remove_extra_files(files, [mnt], 1, False, 1, single_thread=True, src_paths=srcs, sym_links=links, dest_image=True, keep=keep)
	assert (image_c / 'other').read_text() == 'tracked content'
	assert not (image_c / 'sub' / 'stale').exists()


def test_exclude_above_the_source_root_does_not_stop_link_tracking(tmp_path, hpcp_mod, reset_hpcp_globals, require_linux):
	"""Source /srv/tmp/proj with -e tmp: the copy runs (the listing never tests above its root), and so does tracking."""
	srv = tmp_path / 'srv' / 'tmp'
	(srv / 'proj').mkdir(parents=True)
	(srv / 'shared').mkdir()
	(srv / 'shared' / 'lib').write_text('lib')
	os.symlink('../shared/lib', str(srv / 'proj' / 'l'))
	dst = tmp_path / 'd'
	dst.mkdir()
	assert hpcp_mod.hpcp([str(srv / 'proj')], dest_paths=[str(dst) + os.sep], exclude=['tmp'], single_thread=True,
		link_tracking=True, batch=True, do_not_remove_files_while_listing=True) == 0
	assert (dst / 'proj' / 'l').read_text() == 'lib'


def test_symlink_source_root_is_judged_as_the_link_it_is_copied_as(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""A source root that is a symlink is recreated as a link (no 'path/' listing): '*/cache/' does not exclude it."""
	real = tmp_path / 'real'
	real.mkdir()
	root = tmp_path / 'cache'
	os.symlink(str(real), str(root))
	mapping = hpcp_mod._src_to_dest_map([str(root)], [str(tmp_path / 'd') + os.sep])
	assert not hpcp_mod._excluded_at_source(str(tmp_path / 'd' / 'cache'), mapping, hpcp_mod.format_exclude(['cache/']))


# ---- 9.73 ultrareview of the fix branch ----

def _as_mount_root(hpcp_mod, monkeypatch, root):
	real = os.path.ismount
	monkeypatch.setattr(hpcp_mod.os.path, 'ismount', lambda path: os.path.abspath(path) == str(root) or real(path))


def test_remove_extra_keeps_lost_and_found_at_a_filesystem_root(tmp_path, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch):
	"""Ultrareview: -rme into an ext4 root removed lost+found (and would delete the orphans fsck put there)."""
	src = tmp_path / 'src'
	src.mkdir()
	(src / 'a.txt').write_text('a')
	dst = tmp_path / 'disk'
	(dst / 'lost+found').mkdir(parents=True)
	(dst / 'lost+found' / '#1234').write_text('recovered')
	(dst / 'lost+found' / 'empty-orphan-dir').mkdir()
	(dst / 'stale.txt').write_text('stale')
	(dst / 'sub' / 'lost+found').mkdir(parents=True)
	_as_mount_root(hpcp_mod, monkeypatch, dst)
	assert _rme(hpcp_mod, monkeypatch, [str(src) + os.sep], [str(dst) + os.sep]) == 0
	assert (dst / 'lost+found' / '#1234').read_text() == 'recovered'
	assert (dst / 'lost+found' / 'empty-orphan-dir').is_dir()
	assert not (dst / 'stale.txt').exists()
	# Not at a filesystem root: an ordinary directory, judged like any other.
	assert not (dst / 'sub').exists()


def test_remove_extra_dirs_with_dest_image_keeps_the_image_lost_and_found(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	live = tmp_path / 'live'
	(live / 'a').mkdir(parents=True)
	image_root = tmp_path / 'mnt'
	(image_root / 'lost+found').mkdir(parents=True)
	(image_root / os.path.relpath(str(live / 'a'), '/')).mkdir(parents=True)
	_as_mount_root(hpcp_mod, monkeypatch, image_root)
	hpcp_mod.remove_extra_dirs([str(live / 'a')], [str(image_root) + os.sep], dest_image=True)
	assert (image_root / 'lost+found').is_dir()


def test_lost_and_found_check_stops_at_the_root(hpcp_mod):
	assert hpcp_mod._in_root_lost_and_found('/no/such/dir/file', '/') is False
	assert hpcp_mod._in_root_lost_and_found('/lost+found/#12', '/') is os.path.ismount('/')
