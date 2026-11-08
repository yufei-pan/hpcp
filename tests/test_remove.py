import errno
import os
import shutil
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


def test_rwloff_listing_leaves_files(tmp_tree, hpcp_mod, reset_hpcp_globals):
	p = tmp_tree.add_file('stay.txt', 'x')
	hpcp_mod.get_file_list(
		str(tmp_tree.src), parallel_file_listing=False, drop_cache=True,
	)
	assert os.path.exists(p)


def test_remove_extra_dirs_drops_empty_extra(tmp_tree, hpcp_mod, reset_hpcp_globals):
	extra = tmp_tree.dst / 'extra_dir'
	extra.mkdir()
	hpcp_mod.remove_extra_dirs([str(tmp_tree.src) + os.sep], [str(tmp_tree.dst) + os.sep])
	assert not extra.exists()


def test_remove_force_implies_remove(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, copy_args, monkeypatch):
	called = []
	monkeypatch.setattr(hpcp_mod, 'remove_moved_sources', lambda *a, **k: called.append(True))
	tmp_tree.add_file('a.txt', 'x')
	srcs, opts = copy_args(remove=False, remove_force=True)
	hpcp_mod.hpcp(srcs, **opts)
	assert called


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
	# A mount point at dst/scratch, as _filesystem_boundary would find it (tests/test_rme_mounts.py mounts for real).
	def boundary(listed_root, real_root, visible=None):
		return (frozenset([os.path.join(listed_root, 'scratch')]), None)
	monkeypatch.setattr(hpcp_mod, '_filesystem_boundary', boundary)
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
	"""A tracked target belongs to the copy: -rme -lt must not delete it (it would come back on every run). It lands at
	dst/other/x, outside dst/src where src lands, which -rme does not judge since 9.84: it stays without -lt too."""
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
	(tmp_path / 'dst' / 'src' / 'stale.txt').write_text('extra')
	monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
	monkeypatch.setattr('builtins.input', lambda: 'y')
	assert hpcp_mod.hpcp([str(src)], dest_paths=[dst], remove_extra=True, link_tracking=track_on_rme, **opts) == 0
	assert (tmp_path / 'dst' / 'other' / 'x').exists()
	assert os.readlink(tmp_path / 'dst' / 'src' / 'lnk') == '../other/x'
	assert not (tmp_path / 'dst' / 'src' / 'stale.txt').exists()


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


@pytest.mark.parametrize('link_tracking', [True, False])
def test_move_with_a_dangling_link_whose_path_has_a_colon_goes_on(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, copy_args, link_tracking):
	"""A dangling relative link is copied as is; its note is not an error, even when the path holds ':' (rc 128)."""
	tmp_tree.add_file('a.txt', 'a')
	tmp_tree.add_symlink('at:12:00', 'missing:target')
	srcs, opts = copy_args(remove_force=True, link_tracking=link_tracking)
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


@pytest.mark.parametrize('link_tracking', [True, False])
@pytest.mark.parametrize('target', ['.', '..', 'releases/v3'])
def test_in_tree_directory_link_does_not_shield_its_subtree_from_remove_extra(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch, target, link_tracking):
	"""usr/bin/X11 -> . (or current -> releases/v3): the target is part of the copy anyway, not a tracked extra."""
	tmp_tree.add_file('sub/releases/v3/keep.txt', 'keep')
	tmp_tree.add_symlink('sub/link', target)
	stale = tmp_tree.add_file('sub/releases/v3/stale.txt', 'stale', under='dst')
	assert _rme(hpcp_mod, monkeypatch, [str(tmp_tree.src) + os.sep], [str(tmp_tree.dst) + os.sep], link_tracking=link_tracking) == 0
	assert not os.path.exists(stale)
	assert (tmp_tree.dst / 'sub' / 'releases' / 'v3' / 'keep.txt').exists()


@pytest.mark.parametrize('link_tracking', [True, False])
def test_dangling_link_does_not_shield_a_stale_file_from_remove_extra(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch, link_tracking):
	tmp_tree.add_symlink('dangle', 'gone.txt')
	stale = tmp_tree.add_file('gone.txt', 'deleted at the source', under='dst')
	assert _rme(hpcp_mod, monkeypatch, [str(tmp_tree.src) + os.sep], [str(tmp_tree.dst) + os.sep], link_tracking=link_tracking) == 0
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
	(dst / 'proj').mkdir(parents=True)
	(dst / 'proj' / 'stale.txt').write_text('stale')
	assert _rme(hpcp_mod, monkeypatch, [str(src)], [str(dst) + os.sep], link_tracking=True) == 0
	assert (dst / 'proj' / 'a.txt').read_text() == 'a'
	assert not (dst / 'proj' / 'stale.txt').exists()


@pytest.mark.parametrize('link_tracking', [True, False])
def test_remove_extra_keeps_the_copy_of_an_excluded_link_target(tmp_path, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch, link_tracking):
	"""An excluded path's destination copy is never removed, tracking or not (review of eaee294: without -lt it went)."""
	home = tmp_path / 'home'
	(home / 'proj').mkdir(parents=True)
	(home / 'shared').mkdir()
	(home / 'shared' / 'lib').write_text('lib')
	os.symlink('../shared/lib', str(home / 'proj' / 'lib'))
	dst = tmp_path / 'backup'
	(dst / 'shared').mkdir(parents=True)
	(dst / 'shared' / 'lib').write_text('earlier copy')
	assert _rme(hpcp_mod, monkeypatch, [str(home / 'proj')], [str(dst) + os.sep], exclude=[str(home / 'shared' / 'lib')], link_tracking=link_tracking) == 0
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


@pytest.mark.parametrize('single_thread, parallel_file_listing', [(True, False), (False, False), (False, True)])
def test_move_leaves_a_special_file_it_cannot_copy(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, copy_args, single_thread, parallel_file_listing):
	"""Ultrareview: the copy skips a FIFO (or socket, device node) without an error, and the move's rmtree then
	deleted the only copy. A move removes only what the copy listed: the FIFO and its directory stay."""
	import stat
	tmp_tree.add_file('a.txt', 'a')
	tmp_tree.add_file('sub/b.txt', 'b')
	os.mkfifo(str(tmp_tree.src / 'sub' / 'pipe') if parallel_file_listing else str(tmp_tree.src / 'pipe'))
	srcs, opts = copy_args(remove_force=True, single_thread=single_thread, max_workers=1 if single_thread else 2,
		parallel_file_listing=parallel_file_listing)
	assert hpcp_mod.hpcp(srcs, **opts) == 0
	assert (tmp_tree.dst / 'a.txt').read_text() == 'a'
	assert (tmp_tree.dst / 'sub' / 'b.txt').read_text() == 'b'
	assert not (tmp_tree.src / 'a.txt').exists() and not (tmp_tree.src / 'sub' / 'b.txt').exists()
	assert stat.S_ISFIFO(os.lstat(tmp_tree.src / 'sub' / 'pipe' if parallel_file_listing else tmp_tree.src / 'pipe').st_mode)


def test_copy_with_a_special_file_still_succeeds_and_says_so(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, copy_args, capsys):
	tmp_tree.add_file('a.txt', 'a')
	os.mkfifo(str(tmp_tree.src / 'pipe'))
	srcs, opts = copy_args()
	assert hpcp_mod.hpcp(srcs, **opts) == 0
	assert not os.path.lexists(tmp_tree.dst / 'pipe')
	assert 'Special file skipped' in capsys.readouterr().err


@pytest.mark.parametrize('lister', ['get_file_list_serial', 'get_file_list_parallel'])
def test_overlong_path_is_a_listing_error(hpcp_mod, reset_hpcp_globals, lister):
	"""A path over 4096 characters is not listed, so not copied: an error (a move must not delete it)."""
	result = getattr(hpcp_mod, lister)('/' + 'x' * 4100)
	assert result[:2] == (frozenset(), frozenset())
	assert [error.partition(':')[0] for error in hpcp_mod.ERRORS] == ['Listing error']


def test_entry_that_went_away_during_the_listing_is_no_special_file(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Without d_type (xfs ftype=0, some NFS) is_file() stats the entry: one deleted meanwhile reads as neither file
	nor directory nor link. It is gone, as in 9.72: no note and no error."""
	class Gone:
		path = str(tmp_path / 'gone.txt')
		def is_symlink(self):
			return False
		def is_file(self, follow_symlinks=True):
			return False
		def is_dir(self, follow_symlinks=True):
			return False
		def stat(self, follow_symlinks=True):
			raise FileNotFoundError(2, 'No such file or directory', self.path)
	assert hpcp_mod._special_file_kind(Gone()) is None


def test_special_file_kind_names_a_fifo(tmp_path, hpcp_mod):
	os.mkfifo(str(tmp_path / 'p'))
	(tmp_path / 'f').write_text('f')
	kinds = {entry.name: hpcp_mod._special_file_kind(entry) for entry in os.scandir(tmp_path)}
	assert kinds == {'p': 'FIFO', 'f': None}


# ---- 9.73: a move removes only what the copy listed ----

@pytest.mark.parametrize('single_thread', [True, False])
def test_move_keeps_a_file_created_in_the_source_after_the_copy_listed_it(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, copy_args, monkeypatch, single_thread):
	"""Ultrareview: the removal re-listed the source and removed whole trees, so a file written during the move went
	with no copy. Only the copied files and links go, then directories that are empty."""
	tmp_tree.add_file('a.txt', 'a')
	tmp_tree.add_file('sub/b.txt', 'b')
	tmp_tree.add_file('sub/deeper/c.txt', 'c')
	tmp_tree.add_symlink('ln', 'a.txt')
	real = hpcp_mod.create_sym_links
	def create_sym_links(*args, **kwargs):
		(tmp_tree.src / 'sub' / 'new.txt').write_text('written during the move')
		return real(*args, **kwargs)
	monkeypatch.setattr(hpcp_mod, 'create_sym_links', create_sym_links)
	srcs, opts = copy_args(remove_force=True, single_thread=single_thread, max_workers=1 if single_thread else 2)
	assert hpcp_mod.hpcp(srcs, **opts) == 0
	assert (tmp_tree.src / 'sub' / 'new.txt').read_text() == 'written during the move'
	assert sorted(os.listdir(tmp_tree.src)) == ['sub'] and os.listdir(tmp_tree.src / 'sub') == ['new.txt']
	assert (tmp_tree.dst / 'sub' / 'deeper' / 'c.txt').read_text() == 'c'
	assert os.readlink(tmp_tree.dst / 'ln') == 'a.txt'


def test_move_reports_what_it_left(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, copy_args, monkeypatch, capsys):
	tmp_tree.add_file('a.txt', 'a')
	real = hpcp_mod.create_sym_links
	def create_sym_links(*args, **kwargs):
		(tmp_tree.src / 'late.txt').write_text('late')
		return real(*args, **kwargs)
	monkeypatch.setattr(hpcp_mod, 'create_sym_links', create_sym_links)
	srcs, opts = copy_args(remove_force=True)
	assert hpcp_mod.hpcp(srcs, **opts) == 0
	assert 'late.txt' in capsys.readouterr().err


def test_move_with_an_exclude_leaves_the_excluded_paths(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, copy_args, capsys):
	tmp_tree.add_file('a.txt', 'a')
	tmp_tree.add_file('cache/c.txt', 'cached')
	srcs, opts = copy_args(remove_force=True, exclude=['cache'])
	assert hpcp_mod.hpcp(srcs, **opts) == 0
	assert (tmp_tree.src / 'cache' / 'c.txt').read_text() == 'cached'
	assert not (tmp_tree.src / 'a.txt').exists()
	assert not (tmp_tree.dst / 'cache').exists()
	# Left as asked: not reported as left behind.
	assert 'c.txt' not in capsys.readouterr().err


def test_move_of_directories_only_removes_no_file(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, copy_args):
	"""-do copies the directory tree without files: a move then has no file to remove (before, it removed them all)."""
	tmp_tree.add_file('sub/a.txt', 'a')
	srcs, opts = copy_args(remove_force=True, directory_only=True)
	hpcp_mod.hpcp(srcs, **opts)
	assert (tmp_tree.src / 'sub' / 'a.txt').read_text() == 'a'


def test_listed_file_that_became_a_directory_is_left(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""The move's remover only unlinks: a listed path that is now a directory keeps what it holds."""
	(tmp_path / 'was-a-file').mkdir()
	(tmp_path / 'was-a-file' / 'inside.txt').write_text('new')
	(tmp_path / 'f.txt').write_text('f')
	root_id = hpcp_mod._path_identity(str(tmp_path))
	hpcp_mod._remove_listed_files([(str(tmp_path), root_id, rel) for rel in ('was-a-file', 'f.txt', 'gone.txt')])
	assert (tmp_path / 'was-a-file' / 'inside.txt').read_text() == 'new'
	assert not (tmp_path / 'f.txt').exists()
	assert hpcp_mod.ERRORS == []



def _swap_for_a_symlink(directory, target):
	"""Replace directory (and what it holds) with a symlink to target, as another user writing the tree could."""
	import shutil
	shutil.rmtree(str(directory))
	os.symlink(str(target), str(directory))


def test_move_never_removes_through_a_directory_swapped_for_a_symlink(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Security review: src/sub/passwd was listed; sub is then swapped for a symlink to /etc (here: victim). The unlink
	goes by descriptors with O_NOFOLLOW, so it stops at sub instead of deleting victim/passwd."""
	src, victim = tmp_path / 'src', tmp_path / 'victim'
	(src / 'sub').mkdir(parents=True)
	(src / 'sub' / 'passwd').write_text('listed')
	victim.mkdir()
	(victim / 'passwd').write_text('not yours')
	(victim / 'empty').mkdir()
	root_id = hpcp_mod._path_identity(str(src))
	_swap_for_a_symlink(src / 'sub', victim)
	hpcp_mod._remove_listed_files([(str(src), root_id, os.path.join('sub', 'passwd'))])
	hpcp_mod._rmdir_below(str(src), root_id, os.path.join('sub', 'empty'))
	assert (victim / 'passwd').read_text() == 'not yours'
	assert (victim / 'empty').is_dir()
	assert os.path.islink(src / 'sub')


def test_move_whose_source_directory_is_swapped_during_the_copy_removes_nothing_outside(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, copy_args, monkeypatch):
	victim = tmp_tree.root / 'victim'
	victim.mkdir()
	(victim / 'f.txt').write_text('not yours')
	(victim / 'empty').mkdir()
	tmp_tree.add_file('sub/f.txt', 'listed')
	tmp_tree.add_dir('sub/empty')
	real = hpcp_mod.create_sym_links
	def create_sym_links(*args, **kwargs):
		_swap_for_a_symlink(tmp_tree.src / 'sub', victim)
		return real(*args, **kwargs)
	monkeypatch.setattr(hpcp_mod, 'create_sym_links', create_sym_links)
	srcs, opts = copy_args(remove_force=True)
	hpcp_mod.hpcp(srcs, **opts)
	assert (victim / 'f.txt').read_text() == 'not yours'
	assert (victim / 'empty').is_dir()



@pytest.mark.parametrize('swap', ['symlink', 'directory'])
def test_move_never_removes_through_a_source_root_replaced_since_the_copy(tmp_path, hpcp_mod, reset_hpcp_globals, swap):
	"""Security review: the root itself was opened by path for every removal. A user who can write its parent renames it
	away and leaves a symlink (or another directory) there: nothing below the replaced root is removed."""
	src, victim = tmp_path / 'src', tmp_path / 'victim'
	src.mkdir()
	(src / 'f.txt').write_text('listed')
	(src / 'empty').mkdir()
	victim.mkdir()
	(victim / 'f.txt').write_text('not yours')
	(victim / 'empty').mkdir()
	root_id = hpcp_mod._path_identity(str(src))
	os.rename(str(src), str(tmp_path / 'moved-away'))
	if swap == 'symlink':
		os.symlink(str(victim), str(src))
	else:
		os.rename(str(victim), str(src))
		victim = src
	hpcp_mod._remove_listed_files([(str(src), root_id, 'f.txt')])
	hpcp_mod._rmdir_below(str(src), root_id, 'empty')
	assert (victim / 'f.txt').read_text() == 'not yours'
	assert (victim / 'empty').is_dir()
	assert (tmp_path / 'moved-away' / 'f.txt').read_text() == 'listed'


def test_move_of_nested_sources_never_removes_through_the_inner_one_swapped(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch):
	"""Sources src and src/sub: sub swapped for a symlink during the copy is stopped like any directory below src."""
	victim = tmp_tree.root / 'victim'
	victim.mkdir()
	(victim / 'f.txt').write_text('not yours')
	tmp_tree.add_file('a.txt', 'a')
	tmp_tree.add_file('sub/f.txt', 'listed')
	real = hpcp_mod.create_sym_links
	def create_sym_links(*args, **kwargs):
		_swap_for_a_symlink(tmp_tree.src / 'sub', victim)
		return real(*args, **kwargs)
	monkeypatch.setattr(hpcp_mod, 'create_sym_links', create_sym_links)
	hpcp_mod.hpcp([str(tmp_tree.src), str(tmp_tree.src / 'sub')], dest_paths=[str(tmp_tree.dst) + os.sep], remove_force=True,
		single_thread=True, batch=True, do_not_remove_files_while_listing=True)
	assert (victim / 'f.txt').read_text() == 'not yours'


def test_move_of_a_source_root_replaced_during_the_copy_removes_nothing_from_the_new_one(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, copy_args, monkeypatch):
	tmp_tree.add_file('f.txt', 'listed')
	victim = tmp_tree.root / 'victim'
	victim.mkdir()
	(victim / 'f.txt').write_text('not yours')
	real = hpcp_mod.create_sym_links
	def create_sym_links(*args, **kwargs):
		os.rename(str(tmp_tree.src), str(tmp_tree.root / 'moved-away'))
		os.symlink(str(victim), str(tmp_tree.src))
		return real(*args, **kwargs)
	monkeypatch.setattr(hpcp_mod, 'create_sym_links', create_sym_links)
	srcs, opts = copy_args(remove_force=True)
	hpcp_mod.hpcp(srcs, **opts)
	assert (victim / 'f.txt').read_text() == 'not yours'


def test_move_removal_opens_each_directory_about_once(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""9.72 removed by path; the safe removal must not reopen every directory above each file (Lustre: a server round
	trip per open). Sorted entries share their directories' descriptors."""
	root = tmp_path / 'src'
	rels = []
	for top in range(4):
		for depth in range(4):
			directory = os.path.join(*(['d%d' % top] + ['s%d' % level for level in range(depth)]))
			os.makedirs(root / directory, exist_ok=True)
			for index in range(6):
				(root / directory / ('f%d' % index)).write_text('x')
				rels.append(os.path.join(directory, 'f%d' % index))
	directories = {os.path.dirname(rel) for rel in rels}
	opens = []
	real_open = os.open
	monkeypatch.setattr(hpcp_mod.os, 'open', lambda *args, **kwargs: opens.append(args[0]) or real_open(*args, **kwargs))
	root_id = hpcp_mod._path_identity(str(root))
	hpcp_mod._remove_listed_files([(str(root), root_id, rel) for rel in sorted(rels)])
	assert not any((root / rel).exists() for rel in rels)
	# The root's parent and the root, then each directory once.
	assert len(opens) <= len(directories) + 2


def test_a_directory_the_move_cannot_enter_does_not_stop_its_siblings(tmp_path, hpcp_mod, reset_hpcp_globals):
	root, victim = tmp_path / 'src', tmp_path / 'victim'
	for name in ('a', 'b', 'c'):
		(root / name).mkdir(parents=True)
		(root / name / 'f').write_text(name)
	victim.mkdir()
	(victim / 'f').write_text('not yours')
	root_id = hpcp_mod._path_identity(str(root))
	_swap_for_a_symlink(root / 'b', victim)
	hpcp_mod._remove_listed_files([(str(root), root_id, os.path.join(name, 'f')) for name in ('a', 'b', 'c')])
	assert not (root / 'a' / 'f').exists() and not (root / 'c' / 'f').exists()
	assert (victim / 'f').read_text() == 'not yours'
	assert [error.partition(':')[0] for error in hpcp_mod.ERRORS] == ['Remove process exception']


# ---- 9.73: review of eaee294..HEAD ----

@pytest.mark.parametrize('layout', ['dest-is-source', 'dest-inside-source', 'file-onto-itself'])
def test_move_onto_itself_is_refused_before_anything_changes(tmp_path, hpcp_mod, reset_hpcp_globals, require_linux, layout):
	"""`hpcp -rf /data/w/x -d /data/w/` resolves the destination to the source: every file is identical to itself, the
	copy skips it, and the move then removed the source (9.72 too). Refused, nothing copied or removed."""
	w = tmp_path / 'w'
	(w / 'x' / 'sub').mkdir(parents=True)
	(w / 'x' / 'a.txt').write_text('a')
	(w / 'x' / 'sub' / 'b.txt').write_text('b')
	if layout == 'dest-is-source':
		src, dest = str(w / 'x'), str(w) + os.sep
	elif layout == 'dest-inside-source':
		src, dest = str(w / 'x'), str(w / 'x' / 'sub') + os.sep
	else:
		src, dest = str(w / 'x' / 'a.txt'), str(w / 'x') + os.sep
	rc = hpcp_mod.hpcp([src], dest_paths=[dest], remove_force=True, single_thread=True, batch=True)
	assert rc == 241
	assert (w / 'x' / 'a.txt').read_text() == 'a' and (w / 'x' / 'sub' / 'b.txt').read_text() == 'b'
	assert not (w / 'x' / 'sub' / 'x').exists()


def test_move_with_many_source_roots_matches_each_entry_in_linear_time(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""-sfl gives one root per file: matching every listed entry against every root was quadratic."""
	import time
	src = tmp_path / 'src'
	src.mkdir()
	roots = []
	for index in range(30000):
		path = src / ('f%05d' % index)
		path.write_text('x')
		roots.append(str(path))
	ids = {root: hpcp_mod._path_identity(root) for root in roots}
	start = time.process_time()
	hpcp_mod.remove_moved_sources(roots, roots, {}, single_thread=True, remove_force=True, root_ids=ids)
	assert time.process_time() - start < 3
	assert os.listdir(src) == []


def test_move_of_nested_sources_names_what_it_left_once(tmp_path, hpcp_mod, reset_hpcp_globals):
	a = tmp_path / 'a'
	(a / 'b').mkdir(parents=True)
	(a / 'b' / 'new.txt').write_text('written during the move')
	roots = [str(a), str(a / 'b')]
	ids = {root: hpcp_mod._path_identity(root) for root in roots}
	hpcp_mod.remove_moved_sources(roots, [], {}, single_thread=True, remove_force=True, root_ids=ids)
	notes = [error for error in hpcp_mod.ERRORS if error.startswith('Move left entries')]
	assert len(notes) == 1 and notes[0].count('new.txt') == 1 and 'Move left entries: 1 ' in notes[0]


def test_source_identity_of_a_symlink_given_with_a_slash_is_the_links(tmp_path, hpcp_mod):
	"""'ln/' is copied (and moved) as the link ln: its pinned identity must be the link's, not its target's."""
	(tmp_path / 'real').mkdir()
	os.symlink(str(tmp_path / 'real'), str(tmp_path / 'ln'))
	ids = hpcp_mod._source_identities([str(tmp_path / 'ln') + os.sep])
	assert ids == {str(tmp_path / 'ln'): hpcp_mod._path_identity(str(tmp_path / 'ln'))}
	assert ids[str(tmp_path / 'ln')] != hpcp_mod._path_identity(str(tmp_path / 'real'))


def test_plain_remove_without_remove_while_listing_reports_no_copy_notes(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux, monkeypatch):
	"""-rm -rwloff (no destination): rmtree removes everything, special files included; the listing's copy notes are
	not the run's errors."""
	tmp_tree.add_file('a.txt', 'a')
	os.mkfifo(str(tmp_tree.src / 'pipe'))
	monkeypatch.setattr(hpcp_mod, 'REMOVE_FILES_WHILE_LISTING', False)
	hpcp_mod.process_remove([str(tmp_tree.src)], single_thread=True, remove_force=True)
	assert not tmp_tree.src.exists()
	assert [error for error in hpcp_mod.ERRORS if error.startswith(('Special file skipped', 'Listing error'))] == []


# The move backstop (_not_held): before a move removes a source, a landing must hold it. A destination that takes two
# names for one (an 8.3 alias, a legacy charset's other case) is stood in for here by two landings that are one file.

def _write(path, content, mtime=None):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'w') as f:
		f.write(content)
	if mtime is not None:
		os.utime(path, (mtime, mtime))
	return path


def _not_held(hpcp_mod, tmp_path, entries, dests=('out/',), sources=('a/',)):
	return hpcp_mod._not_held([str(tmp_path / entry) for entry in entries], [str(tmp_path / source) for source in sources],
		[str(tmp_path / dest) for dest in dests])


def test_a_landing_of_the_source_s_size_and_time_holds_it(tmp_path, hpcp_mod, reset_hpcp_globals):
	_write(str(tmp_path / 'a/f'), 'data', 1_600_000_000)
	_write(str(tmp_path / 'out/f'), 'data', 1_600_000_000)
	assert _not_held(hpcp_mod, tmp_path, ['a/f']) == {}


@pytest.mark.parametrize('landing, why', [
	(None, 'it is at none of its landings'),
	('other content', 'holds other content (13 bytes, not 4)'),
	('dirt', 'holds other content (modified at another time)'),
])
def test_a_landing_that_is_not_the_source_s_holds_nothing(tmp_path, hpcp_mod, reset_hpcp_globals, landing, why):
	_write(str(tmp_path / 'a/f'), 'data', 1_600_000_000)
	if landing is not None:
		_write(str(tmp_path / 'out/f'), landing, 1_600_000_100)
	assert why in _not_held(hpcp_mod, tmp_path, ['a/f'])[str(tmp_path / 'a/f')]


def test_a_content_only_move_does_not_ask_for_the_time(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# -co sets no times.
	monkeypatch.setattr(hpcp_mod, 'CONTENT_ONLY', True)
	_write(str(tmp_path / 'a/f'), 'data', 1_600_000_000)
	_write(str(tmp_path / 'out/f'), 'dirt', 1_600_000_100)
	assert _not_held(hpcp_mod, tmp_path, ['a/f']) == {}


def test_a_directory_at_the_landing_holds_no_file(tmp_path, hpcp_mod, reset_hpcp_globals):
	_write(str(tmp_path / 'a/f'), 'data')
	os.makedirs(tmp_path / 'out/f')
	assert 'another kind of entry' in _not_held(hpcp_mod, tmp_path, ['a/f'])[str(tmp_path / 'a/f')]


@pytest.mark.skipif(not hasattr(os, 'link'), reason='hard links')
def test_a_landing_that_is_the_source_holds_it_only_through_another_name(tmp_path, hpcp_mod, reset_hpcp_globals):
	# out leads into a: the landing out/f is a/f itself, and removing a/f removes it. A source with another name (a hard link)
	# keeps its content there.
	_write(str(tmp_path / 'a/f'), 'data')
	os.symlink('a', str(tmp_path / 'out'))
	assert 'is the source itself' in _not_held(hpcp_mod, tmp_path, ['a/f'])[str(tmp_path / 'a/f')]
	os.unlink(str(tmp_path / 'out'))
	os.makedirs(tmp_path / 'out')
	os.link(str(tmp_path / 'a/f'), str(tmp_path / 'out/f'))
	assert _not_held(hpcp_mod, tmp_path, ['a/f']) == {}


@pytest.mark.skipif(not hasattr(os, 'link') or os.name == 'nt', reason='hard links and symlinks')
def test_two_sources_that_one_landing_file_holds_are_both_kept(tmp_path, hpcp_mod, reset_hpcp_globals):
	# Of one size and one time, each landing would pass for its source: that they are one file says they were taken for one.
	# out/q leads to out/p, one file of one name, as a folding destination has it.
	_write(str(tmp_path / 'a/p'), 'AAAA', 1_600_000_000)
	_write(str(tmp_path / 'b/q'), 'BBBB', 1_600_000_000)
	_write(str(tmp_path / 'out/p'), 'BBBB', 1_600_000_000)
	os.symlink('p', str(tmp_path / 'out/q'))
	not_held = _not_held(hpcp_mod, tmp_path, ['a/p', 'b/q'], sources=('a/', 'b/'))
	assert sorted(not_held) == [str(tmp_path / 'a/p'), str(tmp_path / 'b/q')]
	assert 'the destination took the names for one' in not_held[str(tmp_path / 'a/p')] and str(tmp_path / 'b/q') in not_held[str(tmp_path / 'a/p')]
	# One source given twice is no such pair, and neither are two names of one source file (a hard link): one content.
	assert _not_held(hpcp_mod, tmp_path, ['b/q'], sources=('b/', 'b/')) == {}
	os.unlink(str(tmp_path / 'a/p'))
	os.link(str(tmp_path / 'b/q'), str(tmp_path / 'a/p'))
	assert _not_held(hpcp_mod, tmp_path, ['a/p', 'b/q'], sources=('a/', 'b/')) == {}


@pytest.mark.parametrize('one_name', [False, True])
def test_a_rename_in_one_directory_asks_the_directory_whether_the_names_are_one(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, one_name):
	# exFAT through FUSE gives each spelling of one file a number of its own: only the directory can say.
	_write(str(tmp_path / 'd/p'), 'data', 1_600_000_000)
	_write(str(tmp_path / 'd/q'), 'data', 1_600_000_000)
	asked = []
	def ask(directory, names):
		asked.append((directory, sorted(names)))
		return ({name: names[0] for name in names} if one_name else {name: name for name in names}), {}
	monkeypatch.setattr(hpcp_mod, '_ask_names', ask)
	not_held = _not_held(hpcp_mod, tmp_path, ['d/p'], dests=('d/q',), sources=('d/p',))
	assert asked == [(str(tmp_path / 'd'), ['p', 'q'])]
	assert ('is the source itself, under another spelling' in not_held.get(str(tmp_path / 'd/p'), '')) == one_name


def test_a_rename_the_directory_cannot_be_asked_about_keeps_the_source(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	_write(str(tmp_path / 'd/p'), 'data', 1_600_000_000)
	_write(str(tmp_path / 'd/q'), 'data', 1_600_000_000)
	monkeypatch.setattr(hpcp_mod, '_ask_names', lambda directory, names: ({}, {name: 'the reason given' for name in names}))
	assert 'cannot be asked: the reason given' in _not_held(hpcp_mod, tmp_path, ['d/p'], dests=('d/q',), sources=('d/p',))[str(tmp_path / 'd/p')]


def test_a_move_keeps_the_sources_whose_landings_another_source_overwrote(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, capsys):
	# What the net misses, on a destination that takes two names for one: out/q leads to out/p, and nothing checks before the
	# copy, so b/q is written onto a/p's landing. Without the backstop the move exited 0 with A gone.
	monkeypatch.chdir(tmp_path)
	_write('a/p', 'A')
	_write('b/q', 'BB')
	_write('out/p', 'old')
	os.symlink('p', 'out/q')
	monkeypatch.setattr(hpcp_mod, '_settle_destinations', lambda *args, **kwargs: ({}, 0, set()))
	assert hpcp_mod.hpcp(['a/', 'b/'], ['out/'], single_thread=True, max_workers=1, batch=True, remove_force=True) == 213
	error = capsys.readouterr().err
	# Both: the landing that holds b/q is one file with a/p's (9.84 kept a/p only, as its landing's size differs).
	for path, landing in (('a/p', 'out/p'), ('b/q', 'out/q')):
		assert f'Destination clash: not removing {tmp_path / path}: {tmp_path / landing} is one file with ' in error
	assert 'Move left entries' not in error
	assert open('a/p').read() == 'A' and open('b/q').read() == 'BB' and open('out/p').read() == 'BB'


@pytest.mark.skipif(os.name == 'nt', reason='symlinks')
def test_a_move_keeps_both_sources_that_landed_in_one_file_whatever_it_holds(tmp_path, hpcp_mod, reset_hpcp_globals):
	# Of other sizes, only a/p's landing fails its own checks; b/q's holds it, yet a/p's content is gone from the
	# destination too: both stay (9.84 removed b/q).
	_write(str(tmp_path / 'a/p'), 'A', 1_600_000_000)
	_write(str(tmp_path / 'b/q'), 'BB', 1_600_000_000)
	_write(str(tmp_path / 'out/p'), 'BB', 1_600_000_000)
	os.symlink('p', str(tmp_path / 'out/q'))
	not_held = _not_held(hpcp_mod, tmp_path, ['a/p', 'b/q'], sources=('a/', 'b/'))
	assert sorted(not_held) == [str(tmp_path / 'a/p'), str(tmp_path / 'b/q')]
	assert all('the destination took the names for one' in why for why in not_held.values())


# The same backstop at the end of a copy (move=False): only two sources in one landing file, the clash itself.

def _copy_not_held(hpcp_mod, tmp_path, entries, dests=('out/',), sources=('a/',)):
	return hpcp_mod._not_held([str(tmp_path / entry) for entry in entries], [str(tmp_path / source) for source in sources],
		[str(tmp_path / dest) for dest in dests], move=False)


@pytest.mark.skipif(os.name == 'nt', reason='symlinks')
def test_a_copy_reports_two_sources_that_landed_in_one_file(tmp_path, hpcp_mod, reset_hpcp_globals):
	_write(str(tmp_path / 'a/p'), 'A', 1_600_000_000)
	_write(str(tmp_path / 'b/q'), 'BB', 1_600_000_000)
	_write(str(tmp_path / 'out/p'), 'BB', 1_600_000_000)
	os.symlink('p', str(tmp_path / 'out/q'))
	not_held = _copy_not_held(hpcp_mod, tmp_path, ['a/p', 'b/q'], sources=('a/', 'b/'))
	assert sorted(not_held) == [str(tmp_path / 'a/p'), str(tmp_path / 'b/q')]
	assert not_held[str(tmp_path / 'a/p')] == (f'{tmp_path / "out/p"} is one file with {tmp_path / "out/q"} (from {tmp_path / "b/q"}): '
		'the destination took the names for one')


def test_a_copy_reports_a_source_at_none_of_its_landings(tmp_path, hpcp_mod, reset_hpcp_globals):
	_write(str(tmp_path / 'a/f'), 'data', 1_600_000_000)
	assert 'it is at none of its landings' in _copy_not_held(hpcp_mod, tmp_path, ['a/f'])[str(tmp_path / 'a/f')]
	# A source gone since has nothing to say.
	assert _copy_not_held(hpcp_mod, tmp_path, ['a/gone']) == {}


@pytest.mark.parametrize('landing', ['other content', 'dirt', 'self'])
def test_a_copy_that_is_no_clash_reports_nothing_and_reads_nothing(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, landing):
	# A source changed during the copy, or a directory that lands on itself, is what cp did, not a clash.
	monkeypatch.setattr(hpcp_mod, '_same_contents', lambda *args: pytest.fail('a copy reads no file'))
	_write(str(tmp_path / 'a/f'), 'data', 1_600_000_000)
	if landing == 'self':
		assert _copy_not_held(hpcp_mod, tmp_path, ['a/f'], dests=('a/',)) == {}
		return
	if landing is not None:
		_write(str(tmp_path / 'out/f'), landing, 1_600_000_100)
	assert _copy_not_held(hpcp_mod, tmp_path, ['a/f']) == {}


def test_a_directory_that_lands_on_itself_and_another_source_written_into_it_is_no_clash(tmp_path, hpcp_mod, reset_hpcp_globals):
	# hpcp out/ a/ -d out/: out/x is its own landing, and a/x's too, as merged directories give (the later source wins).
	_write(str(tmp_path / 'a/x'), 'ax', 1_600_000_000)
	_write(str(tmp_path / 'out/x'), 'ax', 1_600_000_000)
	assert _copy_not_held(hpcp_mod, tmp_path, ['out/x', 'a/x'], sources=('out/', 'a/')) == {}


@pytest.mark.skipif(not hasattr(os, 'link'), reason='hard links')
def test_a_copy_onto_a_destination_deduplicated_by_hard_links_is_no_clash(tmp_path, hpcp_mod, reset_hpcp_globals):
	# Two names of one file at the destination were there before: the copy never writes such a file (cp -a replaces it).
	_write(str(tmp_path / 'a/p'), 'same', 1_600_000_000)
	_write(str(tmp_path / 'a/q'), 'same', 1_600_000_000)
	_write(str(tmp_path / 'out/p'), 'same', 1_600_000_000)
	os.link(str(tmp_path / 'out/p'), str(tmp_path / 'out/q'))
	assert _copy_not_held(hpcp_mod, tmp_path, ['a/p', 'a/q']) == {}


@pytest.mark.skipif(os.name == 'nt', reason='symlinks')
@pytest.mark.parametrize('single_thread', [True, False])
def test_a_copy_whose_sources_landed_in_one_file_exits_213_naming_both(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, capsys, single_thread):
	# What the net misses, on a destination that takes two names for one (an 8.3 alias on vfat): out/q leads to out/p, so b/q
	# is written onto a/p's landing. 9.84 exited 0 with A in neither landing; both sources stay, and so does the landing.
	monkeypatch.chdir(tmp_path)
	_write('a/p', 'A')
	_write('b/q', 'BB')
	_write('out/p', 'old')
	os.symlink('p', 'out/q')
	monkeypatch.setattr(hpcp_mod, '_settle_destinations', lambda *args, **kwargs: ({}, 0, set()))
	assert hpcp_mod.hpcp(['a/', 'b/'], ['out/'], single_thread=single_thread, max_workers=1 if single_thread else 4, batch=True) == 213
	error = capsys.readouterr().err
	for path, landing, other, other_landing in (('a/p', 'out/p', 'b/q', 'out/q'), ('b/q', 'out/q', 'a/p', 'out/p')):
		assert (f'Destination clash: copied {tmp_path / path}, but {tmp_path / landing} is one file with {tmp_path / other_landing} '
			f'(from {tmp_path / other}): the destination took the names for one') in error
	# The landing holds whichever was written last (in parallel, either), whole.
	assert open('a/p').read() == 'A' and open('b/q').read() == 'BB' and open('out/p').read() in ('A', 'BB')


@pytest.mark.parametrize('copy_failed', [False, True])
def test_a_copy_whose_landing_is_gone_exits_213_unless_its_copy_said_why(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, capsys, copy_failed):
	# On vfat beside an 8.3 alias a source can end at no landing, and 9.85 at f7d03d8 exited 0. Stood in for here by a
	# landing removed once the copy is done. A copy that failed has its own error, and no clash line is added to it.
	monkeypatch.chdir(tmp_path)
	_write('a/p', 'A')
	_write('a/q', 'Q')
	real = hpcp_mod.create_sym_links
	def then_lost(*args, **kwargs):
		os.unlink('out/p')
		if copy_failed:
			hpcp_mod.eprint(f'Copy failed: {tmp_path / "a/p"}: stood in for')
		return real(*args, **kwargs)
	monkeypatch.setattr(hpcp_mod, 'create_sym_links', then_lost)
	rc = hpcp_mod.hpcp(['a/'], ['out/'], single_thread=True, max_workers=1, batch=True)
	error = capsys.readouterr().err
	said = f'Destination clash: copied {tmp_path / "a/p"}, but it is at none of its landings ({tmp_path / "out/p"})'
	assert (rc, said in error) == ((190, False) if copy_failed else (213, True))
	assert 'a/q' not in error


def test_a_copy_onto_a_tree_with_other_contents_exits_0(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, capsys):
	# The common case: a rerun onto a tree, merged directories (the later source wins), nothing shared.
	monkeypatch.chdir(tmp_path)
	_write('a/p', 'A')
	_write('a/s/x', 'ax')
	_write('b/s/x', 'bx')
	_write('out/p', 'old')
	assert hpcp_mod.hpcp(['a/', 'b/'], ['out/'], single_thread=True, max_workers=1, batch=True) == 0
	assert 'Destination clash' not in capsys.readouterr().err
	assert open('out/p').read() == 'A' and open('out/s/x').read() == 'bx'


def test_a_file_that_went_to_a_later_destination_is_held_there(tmp_path, hpcp_mod, reset_hpcp_globals):
	# Files spill from one -d to the next: any landing that holds it will do.
	_write(str(tmp_path / 'a/f'), 'data', 1_600_000_000)
	_write(str(tmp_path / 'two/f'), 'data', 1_600_000_000)
	assert _not_held(hpcp_mod, tmp_path, ['a/f'], dests=('one/', 'two/')) == {}
	_write(str(tmp_path / 'one/f'), 'other', 1_600_000_000)
	assert _not_held(hpcp_mod, tmp_path, ['a/f'], dests=('one/', 'two/')) == {}


# Times as each filesystem keeps them: a copy's time lies within one step of the coarser time (_mtime_step_ns).

_SOURCE_NS = 1_700_000_000_123_456_789


@pytest.mark.parametrize('step', [100, 1000, 10**6, 10**7, 2 * 10**9, None], ids=['100ns-ntfs', '1us-udf', '1ms', '10ms-exfat', '2s-vfat', 'float-seconds'])
def test_a_landing_whose_filesystem_keeps_a_coarser_time_holds_the_source(tmp_path, hpcp_mod, reset_hpcp_globals, step):
	# 9.84 at a1d61e8 compared float seconds and guessed no 1 us or 1 ms step: ntfs-3g and udf moves exited 213.
	_write(str(tmp_path / 'a/f'), 'data')
	os.utime(str(tmp_path / 'a/f'), ns=(_SOURCE_NS, _SOURCE_NS))
	_write(str(tmp_path / 'out/f'), 'data')
	if step is None:
		# Set through float seconds (os.utime with st_mtime): within half a float step, 119 ns.
		os.utime(str(tmp_path / 'out/f'), (os.stat(str(tmp_path / 'a/f')).st_mtime,) * 2)
		assert _not_held(hpcp_mod, tmp_path, ['a/f']) == {} and os.stat(str(tmp_path / 'out/f')).st_mtime_ns != _SOURCE_NS
		assert hpcp_mod.is_file_identical(str(tmp_path / 'a/f'), str(tmp_path / 'out/f'), 4) is True
		return
	for landing_ns in (_SOURCE_NS // step * step, -(-_SOURCE_NS // step) * step):
		os.utime(str(tmp_path / 'out/f'), ns=(landing_ns, landing_ns))
		assert _not_held(hpcp_mod, tmp_path, ['a/f']) == {}, landing_ns
		# The skip check: such a copy is not copied again on every run.
		assert hpcp_mod.is_file_identical(str(tmp_path / 'a/f'), str(tmp_path / 'out/f'), 4) is True, landing_ns
	# One step more (at least 1 us, past a float step) is another time, and the contents say whose.
	_write(str(tmp_path / 'out/f'), 'dirt')
	other_ns = _SOURCE_NS // step * step - max(step, 1000)
	os.utime(str(tmp_path / 'out/f'), ns=(other_ns, other_ns))
	assert 'holds other content (modified at another time)' in _not_held(hpcp_mod, tmp_path, ['a/f'])[str(tmp_path / 'a/f')]
	assert hpcp_mod.is_file_identical(str(tmp_path / 'a/f'), str(tmp_path / 'out/f'), 4) is False


_1973_NS = 100_000_000_123_456_789


@pytest.mark.parametrize('first, second, agree', [
	# One float step at 1.7e9 s is 238.4 ns: a time set through float seconds lands within half of it.
	(_SOURCE_NS, _SOURCE_NS + 238, True),
	(_SOURCE_NS, _SOURCE_NS - 238, True),
	(_SOURCE_NS, _SOURCE_NS + 239, False),
	(_SOURCE_NS, _SOURCE_NS // 1000 * 1000, True),
	(_SOURCE_NS, _SOURCE_NS // 1000 * 1000 - 1000, False),
	(_SOURCE_NS, _SOURCE_NS // 10**6 * 10**6, True),
	# In 1973 a float step is 14.9 ns, and NTFS's 100 ns step decides.
	(_1973_NS, _1973_NS // 100 * 100, True),
	(_1973_NS, _1973_NS // 100 * 100 - 1, False),
	(_1973_NS, _1973_NS + 14, True),
	(_1973_NS, _1973_NS + 16, False),
	(1_700_000_000 * 10**9, 1_700_000_002 * 10**9, True),
	(1_700_000_000 * 10**9, 1_700_000_002 * 10**9 + 1, False),
	(1_700_000_001 * 10**9, 1_700_000_002 * 10**9, True),
	(1_700_000_001 * 10**9, 1_700_000_003 * 10**9, False),
	# A 1 s filesystem's time against a finer one within that second.
	(1_700_000_001 * 10**9, 1_700_000_001_500_000_000, True),
	# Across 2**31 s (2038) the float step doubles: the later time's, 476.8 ns, decides.
	(2**31 * 10**9 - 200, 2**31 * 10**9 + 250, True),
	(2**31 * 10**9 - 200, 2**31 * 10**9 + 276, True),
	(2**31 * 10**9 - 200, 2**31 * 10**9 + 277, False),
	(1_700_000_000 * 10**9 + 10**7, 1_700_000_000 * 10**9 + 2 * 10**7, True),
	(1_700_000_000 * 10**9 + 10**7, 1_700_000_000 * 10**9 + 3 * 10**7, False),
])
def test_two_times_agree_within_one_step_of_the_coarser(hpcp_mod, first, second, agree):
	assert hpcp_mod._mtimes_agree(first, second) is agree and hpcp_mod._mtimes_agree(second, first) is agree
	state = 'f:4:{}:644:0:0:h:-'
	assert hpcp_mod.states_equal(state.format(first), state.format(second)) is agree


def test_a_rerun_keeps_the_times_to_the_nanosecond(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# A file found identical gets its times again: 9.84 at a1d61e8 set them through float seconds, up to 119 ns off.
	monkeypatch.chdir(tmp_path)
	_write('a/d/f', 'data')
	os.utime('a/d/f', ns=(_SOURCE_NS, _SOURCE_NS))
	for _ in range(2):
		assert hpcp_mod.hpcp(['a/'], ['out/'], single_thread=True, max_workers=1, batch=True) == 0
		assert os.stat('out/d/f').st_mtime_ns == _SOURCE_NS


def test_a_landing_whose_time_vfat_could_not_keep_holds_the_source_of_its_contents(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# vfat keeps no time before 1980: it sets 315532800. 9.84 at a1d61e8 kept every such source, with 213.
	_write(str(tmp_path / 'a/f'), 'data', 0)
	_write(str(tmp_path / 'out/f'), 'data', 315532800)
	assert _not_held(hpcp_mod, tmp_path, ['a/f']) == {}
	_write(str(tmp_path / 'out/f'), 'dirt', 315532800)
	assert 'holds other content (modified at another time)' in _not_held(hpcp_mod, tmp_path, ['a/f'])[str(tmp_path / 'a/f')]
	# A landing that cannot be read is not known to hold it.
	_write(str(tmp_path / 'out/f'), 'data', 315532800)
	def refused(first, second):
		raise PermissionError(errno.EACCES, 'Permission denied', second)
	monkeypatch.setattr(hpcp_mod, '_same_contents', refused)
	assert 'could not be compared with it (Permission denied)' in _not_held(hpcp_mod, tmp_path, ['a/f'])[str(tmp_path / 'a/f')]


@pytest.mark.parametrize('landing, landing_s, identical', [
	('data', 315532800, True),
	# vfat keeps the floor in local time: any time at or after it is one it could not have kept.
	('data', 315532800 + 3600, True),
	# Of one size, differing only where the tail hash does not look: read in full.
	('DAta', 315532800, False),
	# A landing that kept a time before 1980 kept the source's: one that differs is another time.
	('data', 5, False),
])
def test_the_skip_check_reads_a_file_from_before_1980_against_its_vfat_landing(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch,
		landing, landing_s, identical):
	# 9.84 copied every such file again on every rerun onto vfat.
	monkeypatch.setattr(hpcp_mod, 'HASH_SIZE', 2)
	_write(str(tmp_path / 'a/f'), 'data', 0)
	_write(str(tmp_path / 'out/f'), landing, landing_s)
	assert hpcp_mod.is_file_identical(str(tmp_path / 'a/f'), str(tmp_path / 'out/f'), 4) is identical


def test_the_skip_check_reads_nothing_more_for_a_time_after_1980(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	monkeypatch.setattr(hpcp_mod, '_same_contents', lambda *args: pytest.fail('read in full'))
	_write(str(tmp_path / 'a/f'), 'data', 1_600_000_000)
	_write(str(tmp_path / 'out/f'), 'data', 1_600_000_100)
	assert hpcp_mod.is_file_identical(str(tmp_path / 'a/f'), str(tmp_path / 'out/f'), 4) is False
	_write(str(tmp_path / 'out/f'), 'data', 1_600_000_000)
	assert hpcp_mod.is_file_identical(str(tmp_path / 'a/f'), str(tmp_path / 'out/f'), 4) is True


def test_the_contents_are_read_only_when_the_time_disagrees(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	read = []
	real = hpcp_mod._same_contents
	monkeypatch.setattr(hpcp_mod, '_same_contents', lambda first, second: read.append(first) or real(first, second))
	_write(str(tmp_path / 'a/same'), 'data', 1_600_000_000)
	_write(str(tmp_path / 'out/same'), 'data', 1_600_000_000)
	_write(str(tmp_path / 'a/longer'), 'data', 1_600_000_000)
	_write(str(tmp_path / 'out/longer'), 'datadata', 1_600_000_100)
	_write(str(tmp_path / 'a/later'), 'data', 1_600_000_000)
	_write(str(tmp_path / 'out/later'), 'data', 1_600_000_100)
	# A time its landing's filesystem kept to the microsecond is the same time: nothing to read.
	_write(str(tmp_path / 'a/coarse'), 'data')
	os.utime(str(tmp_path / 'a/coarse'), ns=(_SOURCE_NS, _SOURCE_NS))
	_write(str(tmp_path / 'out/coarse'), 'data')
	os.utime(str(tmp_path / 'out/coarse'), ns=(_SOURCE_NS // 1000 * 1000,) * 2)
	assert sorted(_not_held(hpcp_mod, tmp_path, ['a/same', 'a/longer', 'a/later', 'a/coarse'])) == [str(tmp_path / 'a/longer')]
	assert read == [str(tmp_path / 'a/later')]


def test_contents_compare_to_the_end(tmp_path, hpcp_mod):
	big = b'x' * (3 << 20)
	(tmp_path / 'one').write_bytes(big)
	(tmp_path / 'two').write_bytes(big)
	(tmp_path / 'three').write_bytes(big[:-1] + b'y')
	assert hpcp_mod._same_contents(str(tmp_path / 'one'), str(tmp_path / 'two'))
	assert not hpcp_mod._same_contents(str(tmp_path / 'one'), str(tmp_path / 'three'))


@pytest.mark.skipif(not hasattr(os, 'symlink') or os.name == 'nt', reason='symlinks')
def test_a_landing_that_is_a_link_to_a_file_holds_what_the_copy_wrote_through_it(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# cp writes through a link at the landing: out/x -> ../t/f holds a/x in t/f. 9.84 at a1d61e8 said 'another kind of entry', 213.
	monkeypatch.chdir(tmp_path)
	_write('a/x', 'A')
	_write('t/f', 'old')
	os.makedirs('out')
	os.symlink(os.path.join('..', 't', 'f'), 'out/x')
	assert hpcp_mod.hpcp(['a/'], ['out/'], single_thread=True, max_workers=1, batch=True, remove_force=True) == 0
	assert not os.path.exists('a/x') and open('t/f').read() == 'A' and os.path.islink('out/x')


@pytest.mark.skipif(not hasattr(os, 'symlink') or not hasattr(os, 'link') or os.name == 'nt', reason='symlinks and hard links')
def test_a_landing_that_is_a_link_to_the_source_is_the_source_itself(tmp_path, hpcp_mod, reset_hpcp_globals):
	# However many names the source has, removing it takes what the link shows. A link that leads nowhere holds nothing.
	_write(str(tmp_path / 'a/x'), 'A')
	os.makedirs(tmp_path / 'out')
	os.symlink(os.path.join('..', 'a', 'x'), str(tmp_path / 'out/x'))
	assert 'is the source itself' in _not_held(hpcp_mod, tmp_path, ['a/x'])[str(tmp_path / 'a/x')]
	os.link(str(tmp_path / 'a/x'), str(tmp_path / 'a/other-name'))
	assert 'is the source itself' in _not_held(hpcp_mod, tmp_path, ['a/x'])[str(tmp_path / 'a/x')]
	os.unlink(str(tmp_path / 'out/x'))
	os.symlink('nowhere', str(tmp_path / 'out/x'))
	assert 'another kind of entry' in _not_held(hpcp_mod, tmp_path, ['a/x'])[str(tmp_path / 'a/x')]


@pytest.mark.skipif(not hasattr(os, 'link'), reason='hard links')
def test_a_move_onto_a_destination_deduplicated_by_hard_links_removes_the_sources(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# out/p and out/q are one file, as a deduplicating tool leaves them, and hold a/p and b/q: the copy skips both and writes
	# nothing. 9.84 at a1d61e8 took the shared file for two names the destination took for one, 213.
	monkeypatch.chdir(tmp_path)
	_write('a/p', 'same', 1_600_000_000)
	_write('b/q', 'same', 1_600_000_000)
	_write('out/p', 'same', 1_600_000_000)
	os.link('out/p', 'out/q')
	assert hpcp_mod.hpcp(['a/', 'b/'], ['out/'], single_thread=True, max_workers=1, batch=True, remove_force=True) == 0
	assert not os.path.exists('a/p') and not os.path.exists('b/q') and open('out/q').read() == 'same'


@pytest.mark.skipif(not hasattr(os, 'link'), reason='hard links')
def test_a_copy_leaves_a_landing_with_other_names_before_writing_it(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# What the backstop's hard-link rule rests on: a landing the copy writes has one name after it, -co too.
	monkeypatch.chdir(tmp_path)
	for content_only in (False, True):
		monkeypatch.setattr(hpcp_mod, 'CONTENT_ONLY', content_only)
		shutil.rmtree('out', ignore_errors=True)
		_write('a/p', 'new %s' % content_only)
		_write('out/p', 'old')
		os.link('out/p', 'out/elsewhere')
		assert hpcp_mod.hpcp(['a/'], ['out/'], single_thread=True, max_workers=1, batch=True) == 0
		assert os.stat('out/p').st_nlink == 1 and open('out/p').read() == 'new %s' % content_only and open('out/elsewhere').read() == 'old'


# -rme judges only below where a directory source lands, and never removes a source, or what lies in or holds one.

def _rme_answered(hpcp_mod, monkeypatch, sources, dest='out/'):
	monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
	monkeypatch.setattr('builtins.input', lambda *args: 'y')
	return hpcp_mod.hpcp(sources, [dest], single_thread=True, max_workers=1, batch=True, remove_extra=True)


def _tree(root):
	return sorted(os.path.relpath(os.path.join(path, name), root) for path, dirs, names in os.walk(root) for name in names + [d for d in dirs if os.path.islink(os.path.join(path, d))])


@pytest.mark.parametrize('sources', [['b'], ['a/x', 'b']], ids=['directory', 'file-and-directory'])
def test_rme_with_a_source_that_lands_in_a_directory_of_its_name_judges_only_that_directory(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, sources):
	# hpcp b -d out/ -rme lands in out/b: out/keep is not its to judge. 9.83 deleted it.
	monkeypatch.chdir(tmp_path)
	_write('a/x', 'x')
	_write('b/f', 'f')
	_write('out/keep', 'k')
	_write('out/b/extra', 'e')
	os.makedirs('out/kept-dir')
	os.makedirs('out/b/extra-dir')
	assert _rme_answered(hpcp_mod, monkeypatch, sources) == 0
	assert 'keep' in _tree('out') and 'b/extra' not in _tree('out') and 'b/f' in _tree('out')
	assert os.path.isdir('out/kept-dir') and not os.path.exists('out/b/extra-dir')


def test_rme_with_a_source_that_mirrors_into_the_destination_still_judges_all_of_it(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	monkeypatch.chdir(tmp_path)
	_write('b/f', 'f')
	_write('out/extra', 'e')
	_write('out/sub/extra', 'e')
	assert _rme_answered(hpcp_mod, monkeypatch, ['b/']) == 0
	assert _tree('out') == ['f']


def test_rme_never_removes_a_source_inside_the_destination(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# out/sub/x is a source: 9.83 took it for extra and deleted it, its only copy when b/x replaced it at out/x.
	monkeypatch.chdir(tmp_path)
	_write('out/sub/x', 'S')
	_write('out/deep/sub2/y', 'Y')
	_write('b/x', 'B')
	_write('out/extra', 'e')
	# A directory of a source is a source too, an empty one included, and so is an empty source.
	os.makedirs('out/sub/empty')
	os.makedirs('out/empty-source')
	assert _rme_answered(hpcp_mod, monkeypatch, ['out/sub/', 'out/deep/sub2/', 'out/empty-source/', 'b/']) == 0
	assert open('out/sub/x').read() == 'S' and open('out/deep/sub2/y').read() == 'Y' and not os.path.exists('out/extra')
	assert os.path.isdir('out/sub/empty') and os.path.isdir('out/empty-source')


@pytest.mark.skipif(not hasattr(os, 'symlink') or os.name == 'nt', reason='symlinks')
def test_rme_removes_a_destination_link_that_leads_into_a_source_as_an_extra(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, capsys):
	# Where a link leads makes it no source. Kept, it stayed on every run and src/d was never mirrored (9.83 removed it).
	# 1. out/d -> ../src/d at src/d's landing: the first run removes it, the second mirrors d.
	# 2. a link left by a source that is gone (an absolute one into the tree) is an extra, not kept for its target's sake.
	monkeypatch.chdir(tmp_path)
	_write('src/d/f', 'F')
	_write('src/rel/v1', 'V')
	_write('elsewhere/z', 'Z')
	os.makedirs('out')
	os.symlink(os.path.join('..', 'src', 'd'), 'out/d')
	os.symlink(os.path.join('..', 'elsewhere'), 'out/l')
	os.symlink(os.path.abspath('src/rel/v1'), 'out/old')
	assert _rme_answered(hpcp_mod, monkeypatch, ['src/']) == 0
	assert 'Not removing' not in capsys.readouterr().out
	assert not any(os.path.lexists('out/' + name) for name in ('old', 'l')) and not os.path.islink('out/d')
	assert open('src/d/f').read() == 'F' and open('src/rel/v1').read() == 'V' and open('elsewhere/z').read() == 'Z'
	assert _rme_answered(hpcp_mod, monkeypatch, ['src/']) == 0
	assert open('out/d/f').read() == 'F' and not os.path.islink('out/d') and open('src/d/f').read() == 'F'


@pytest.mark.skipif(not hasattr(os, 'symlink') or os.name == 'nt', reason='symlinks')
@pytest.mark.parametrize('flags', [{'remove': True}, {'remove': True, 'remove_extra': True}], ids=['move', 'move-rme'])
def test_a_move_onto_a_link_that_leads_to_its_source_is_still_refused_before_any_copy(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, flags):
	# The case the removed rule was written for: out/d -> ../src/d with src/ moved. Refused as landing on itself, nothing touched.
	monkeypatch.chdir(tmp_path)
	_write('src/d/f', 'F')
	_write('src/g', 'G')
	os.makedirs('out')
	os.symlink(os.path.join('..', 'src', 'd'), 'out/d')
	monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
	monkeypatch.setattr('builtins.input', lambda *args: 'y')
	rc = hpcp_mod.hpcp(['src/'], ['out/'], single_thread=True, max_workers=1, batch=True, **flags)
	assert rc == 213
	assert open('src/d/f').read() == 'F' and open('src/g').read() == 'G' and os.path.islink('out/d') and not os.path.exists('out/g')


@pytest.mark.skipif(not hasattr(os, 'symlink') or os.name == 'nt', reason='symlinks')
def test_rme_never_removes_what_lies_deep_in_a_source_or_a_link_a_source_holds_or_is(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	# out/sub/ mirrors into out/, so all of out/ is judged: its own files and links are sources, however deep, wherever a link
	# leads, and a named link is a source though it leads nowhere. 9.84 at a1d61e8 followed each link first, and removed both.
	monkeypatch.chdir(tmp_path)
	_write('out/sub/inner/z', 'Z')
	_write('elsewhere/e', 'E')
	os.symlink(os.path.join('..', '..', 'elsewhere'), 'out/sub/l')
	os.makedirs('out/x')
	os.symlink('nowhere', 'out/x/dangling')
	_write('b/f', 'F')
	_write('out/extra', 'e')
	assert _rme_answered(hpcp_mod, monkeypatch, ['out/sub/', 'out/x/dangling', 'b/']) == 0
	assert open('out/sub/inner/z').read() == 'Z' and os.path.islink('out/sub/l') and os.path.islink('out/x/dangling')
	assert not os.path.exists('out/extra') and open('out/f').read() == 'F'


@pytest.mark.parametrize('extra', ['file', 'link'])
def test_rme_asks_about_sources_once_a_directory_not_once_an_extra(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, extra):
	# On NFS with actimeo=0 each stat is a round trip: 9.84 at a1d61e8 took a realpath (an lstat of each part) and a stat for
	# every extra, 52 s for 10,000 extras where the walk took 0.35 s. 9.84 still lstat'ed each extra link.
	if extra == 'link' and os.name == 'nt':
		pytest.skip('symlinks')
	monkeypatch.chdir(tmp_path)
	_write('b/f', 'f')
	_write('out/sub/x', 'S')
	os.makedirs('out/b/many')
	for i in range(500):
		if extra == 'link':
			os.symlink('nowhere', 'out/b/many/%d' % i)
		else:
			_write('out/b/many/%d' % i, 'e')
	calls, inside = [0], [False]
	for name in ('stat', 'lstat'):
		real = getattr(os, name)
		def counted(*args, _real=real, **kwargs):
			if inside[0]:
				calls[0] += 1
			return _real(*args, **kwargs)
		monkeypatch.setattr(os, name, counted)
	make = hpcp_mod._source_guard
	def counting_guard(*args, **kwargs):
		guarded = make(*args, **kwargs)
		def wrapped(*args, **kwargs):
			inside[0] = True
			try:
				return guarded(*args, **kwargs)
			finally:
				inside[0] = False
		return wrapped
	monkeypatch.setattr(hpcp_mod, '_source_guard', counting_guard)
	assert _rme_answered(hpcp_mod, monkeypatch, ['b', 'out/sub/']) == 0
	assert not os.path.exists('out/b/many') and open('out/sub/x').read() == 'S'
	assert calls[0] < 50, calls[0]
