import os
import io
import pytest
import time


def _h(hpcp_mod, char):
	"""A digest-shaped string for the current hash algorithm."""
	return char * (16 if hpcp_mod.xxhash_available else 128)


def test_load_file_list_reads_lines(tmp_path, hpcp_mod, reset_hpcp_globals):
	p = tmp_path / 'list.txt'
	p.write_text('a.txt\n\nb.txt\n')
	got = hpcp_mod.load_file_list(str(p))
	assert got == frozenset(['a.txt', 'b.txt'])


def test_load_file_list_missing_returns_empty(hpcp_mod, reset_hpcp_globals, tmp_path):
	assert hpcp_mod.load_file_list(str(tmp_path / 'nope.txt')) == frozenset()


def test_load_file_list_from_stdin(hpcp_mod, reset_hpcp_globals, monkeypatch):
	monkeypatch.setattr('sys.stdin', io.StringIO('one\n two \n'))
	assert hpcp_mod.load_file_list('-') == frozenset(['one', 'two'])


def test_store_file_list_writes_relative_paths(tmp_tree, hpcp_mod, reset_hpcp_globals):
	tmp_tree.add_file('a.txt', 'A')
	tmp_tree.add_file('sub/b.txt', 'B')
	out = str(tmp_tree.root / 'files.txt')
	hpcp_mod.store_file_list(
		out,
		[str(tmp_tree.src)],
		parallel_file_listing=False,
		append_hash=False,
	)
	text = open(out).read()
	assert 'a.txt' in text
	assert 'b.txt' in text
	assert ':' not in text.replace(os.sep, '/')


def test_store_file_list_with_hash_appends_digest(tmp_tree, hpcp_mod, reset_hpcp_globals):
	tmp_tree.add_file('a.txt', 'A')
	out = str(tmp_tree.root / 'files.txt')
	hpcp_mod.store_file_list(
		out,
		[str(tmp_tree.src)],
		parallel_file_listing=False,
		append_hash=True,
	)
	assert any(':' in line for line in open(out).read().splitlines())


def test_hpcp_writes_target_file_list(tmp_tree, hpcp_mod, reset_hpcp_globals, copy_args, require_linux):
	tmp_tree.add_file('a.txt', 'A')
	out = str(tmp_tree.root / 'out_list.txt')
	srcs, opts = copy_args(target_file_list=out, append_hash_to_file_list=False)
	hpcp_mod.hpcp(srcs, **opts)
	assert os.path.isfile(out)
	assert 'a.txt' in open(out).read()


def test_compare_file_list_writes_diff(tmp_path, hpcp_mod, reset_hpcp_globals):
	diff = tmp_path / 'diff.txt'
	a = _h(hpcp_mod, 'a')
	src = {f'a.txt:{a}', f'only_src.txt:{a}'}
	dst = {f'a.txt:{a}', f'only_dst.txt:{a}'}
	hpcp_mod.compare_file_list(src, dst, diff_file_list=str(diff), tar_diff_file_list=False)
	rows = hpcp_mod.read_changes(str(diff))
	assert {(r.op, r.path) for r in rows} == {('A', 'only_src.txt'), ('D', 'only_dst.txt')}


def test_tar_diff_file_list_is_new_or_updated_only(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Help text: tar-compatible diff lists update/new files only (in src not dest)."""
	diff = tmp_path / 'tar.txt'
	a, b, c = _h(hpcp_mod, 'a'), _h(hpcp_mod, 'b'), _h(hpcp_mod, 'c')
	src = {f'new.txt:{a}', f'updated.txt:{b}', f'same.txt:{a}'}
	dst = {f'same.txt:{a}', f'updated.txt:{c}', f'extra.txt:{a}'}
	hpcp_mod.compare_file_list(src, dst, diff_file_list=str(diff), tar_diff_file_list=True)
	assert set(diff.read_text().splitlines()) == {'new.txt', 'updated.txt'}


def test_tar_diff_file_list_skips_directory_metadata_rows(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""F11: an M row for a directory would make `tar -T` archive its whole subtree; added directories stay."""
	R = hpcp_mod.ChangeRow
	d, f = 'd:40755:0:0:-', 'f:1:1:100644:0:0:aa:-'
	rows = [R('A', 'newdir/', '-', d), R('A', 'newdir/f.txt', '-', f), R('M', 'olddir/', 'd:40700:0:0:-', d),
		R('M', 'mod.txt', f, 'f:2:1:100644:0:0:bb:-'), R('M', './', 'd:40700:0:0:-', d), R('D', 'gone.txt', f, '-')]
	out = tmp_path / 'tar.txt'
	hpcp_mod.report_changes(rows, str(out), tar_diff_file_list=True)
	assert set(out.read_text().splitlines()) == {'newdir/', 'newdir/f.txt', 'mod.txt'}


@pytest.mark.parametrize('tar', [False, True])
def test_report_changes_survives_non_utf8_names(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, tar):
	"""F8: the list file is written before printing, and printing a non-UTF-8 name never raises."""
	name = os.fsdecode(b'bad\xffname')
	rows = [hpcp_mod.ChangeRow('A', name, '-', 'l:x')]
	out = tmp_path / 'diff.tsv'
	strict = io.TextIOWrapper(io.BytesIO(), encoding='utf-8', errors='strict')
	monkeypatch.setattr('sys.stdout', strict)
	hpcp_mod.report_changes(rows, str(out), tar_diff_file_list=tar)
	strict.flush()
	printed = strict.buffer.getvalue()
	monkeypatch.undo()
	if tar:
		assert out.read_bytes() == b'bad\xffname\n'
	else:
		assert hpcp_mod.read_changes(str(out)) == rows
	assert b'bad\\xffname' in printed


def test_hpcp_compare_file_list_with_dest(tmp_tree, hpcp_mod, reset_hpcp_globals, copy_args, require_linux):
	tmp_tree.add_file('keep.txt', 'k', under='src')
	tmp_tree.add_file('keep.txt', 'k', under='dst')
	tmp_tree.add_file('only_src.txt', 's', under='src')
	tmp_tree.add_file('only_dst.txt', 'd', under='dst')
	diff = str(tmp_tree.root / 'cmp.txt')
	srcs, opts = copy_args(compare_file_list=True, diff_file_list=diff, append_hash_to_file_list=False)
	hpcp_mod.hpcp(srcs, **opts)
	text = open(diff).read()
	assert 'only_src.txt' in text or 'only_dst.txt' in text


def test_source_file_list_copies_listed_file(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	src_file = tmp_tree.add_file('listed.txt', 'from-list')
	lst = tmp_tree.root / 'sfl.txt'
	lst.write_text(src_file + '\n')
	hpcp_mod.hpcp(
		[],
		dest_paths=[str(tmp_tree.dst) + os.sep],
		source_file_list=str(lst),
		single_thread=True,
		max_workers=1,
		batch=True,
		do_not_remove_files_while_listing=True,
	)
	copied = list(tmp_tree.dst.rglob('listed.txt'))
	assert copied
	assert copied[0].read_text() == 'from-list'


@pytest.mark.parametrize('compare_requested', [True, False])
def test_compare_stored_list_preserves_baseline_and_writes_diff(
	tmp_tree, hpcp_mod, reset_hpcp_globals, compare_requested,
):
	tmp_tree.add_file('same.txt', 'same')
	tmp_tree.add_file('new.txt', 'new')
	baseline = tmp_tree.root / 'baseline.txt'
	baseline.write_text('./\nsame.txt\n')
	diff = tmp_tree.root / 'diff.txt'
	rc = hpcp_mod.hpcp(
		[str(tmp_tree.src) + os.sep], target_file_list=str(baseline),
		compare_file_list=compare_requested, diff_file_list=str(diff),
		append_hash_to_file_list=False, single_thread=True,
	)
	assert rc == 0
	assert baseline.read_text() == './\nsame.txt\n'
	rows = hpcp_mod.read_changes(str(diff))
	assert [(r.op, r.path) for r in rows] == [('A', 'new.txt')]
	assert rows[0].new.startswith('f:')


def test_diff_file_list_keeps_names_with_colons(tmp_tree, hpcp_mod, reset_hpcp_globals):
	"""Regression: -dfl cut every entry at its last ':' (directory 'a:b/' became 'a')."""
	tmp_tree.add_file('a:b/file:1.txt', 'x')
	tmp_tree.add_file('plain.txt', 'p')
	baseline = tmp_tree.root / 'baseline.txt'
	baseline.write_text('./\nplain.txt\nold:dir/\n')
	diff = tmp_tree.root / 'diff.tsv'
	rc = hpcp_mod.hpcp(
		[str(tmp_tree.src) + os.sep], target_file_list=str(baseline), diff_file_list=str(diff),
		append_hash_to_file_list=False, single_thread=True,
	)
	assert rc == 0
	ops = {(r.op, r.path) for r in hpcp_mod.read_changes(str(diff))}
	assert ops == {('A', 'a:b/'), ('A', 'a:b/file:1.txt'), ('D', 'old:dir/')}


def test_hashed_list_symlink_lines_parse_to_their_path(hpcp_mod, reset_hpcp_globals):
	a = _h(hpcp_mod, 'a')
	lines = [f'f.txt:{a}', 'ln:target:with:colons', 'dir/']
	entries = hpcp_mod._parse_file_list_lines(lines)
	assert entries == {'f.txt': a, 'ln': 'target:with:colons', 'dir/': None}
	# a live key that itself contains ':' wins over the first-colon split
	entries = hpcp_mod._parse_file_list_lines([f'f.txt:{a}', 'my:link:dest'], live_keys={'my:link'})
	assert entries['my:link'] == 'dest'


def test_hpcp_diff_file_list_against_dest_reports_changes(tmp_tree, hpcp_mod, reset_hpcp_globals, copy_args, require_linux, capsys):
	t = time.time() - 1000
	for path in (tmp_tree.add_file('keep.txt', 'k', under='src'), tmp_tree.add_file('keep.txt', 'k', under='dst')):
		os.utime(path, (t, t))
	tmp_tree.add_file('changed.txt', 'new!', under='src')
	tmp_tree.add_file('changed.txt', 'old', under='dst')
	tmp_tree.add_file('only_src.txt', 's', under='src')
	tmp_tree.add_file('only_dst.txt', 'd', under='dst')
	diff = str(tmp_tree.root / 'cmp.tsv')
	srcs, opts = copy_args(diff_file_list=diff)
	assert hpcp_mod.hpcp(srcs, **opts) == 0
	ops = {(r.op, r.path) for r in hpcp_mod.read_changes(diff)}
	assert ops == {('A', 'only_src.txt'), ('M', 'changed.txt'), ('D', 'only_dst.txt')}
	assert 'Modified:' in capsys.readouterr().out


def test_diff_file_list_against_hashed_stored_list(tmp_tree, hpcp_mod, reset_hpcp_globals):
	"""A live tree compared with a stored hashed list: only real changes produce rows."""
	hpcp_mod.HASH_SIZE = 65536
	tmp_tree.add_file('keep.txt', 'keep')
	tmp_tree.add_file('changed.txt', 'before')
	tmp_tree.add_symlink('ln', 'keep.txt')
	out = str(tmp_tree.root / 'stored.txt')
	src = str(tmp_tree.src) + os.sep
	hpcp_mod.store_file_list(out, [src], parallel_file_listing=False, append_hash=True)
	tmp_tree.add_file('changed.txt', 'after, longer')
	os.remove(str(tmp_tree.src / 'ln'))
	tmp_tree.add_symlink('ln', 'changed.txt')
	tmp_tree.add_file('new.txt', 'n')
	diff = str(tmp_tree.root / 'diff.tsv')
	rc = hpcp_mod.hpcp([src], target_file_list=out, diff_file_list=diff, single_thread=True)
	assert rc == 0
	ops = {(r.op, r.path) for r in hpcp_mod.read_changes(diff)}
	assert ops == {('M', 'changed.txt'), ('M', 'ln'), ('A', 'new.txt')}


def _populate(root, files):
	for rel, content in files.items():
		path = os.path.join(str(root), rel)
		os.makedirs(os.path.dirname(path), exist_ok=True)
		with open(path, 'w') as f:
			f.write(content)


def test_process_compare_store_layout_limits_to_src_subtree(tmp_path, hpcp_mod, reset_hpcp_globals):
	src = tmp_path / 'live'
	store = tmp_path / 'store'
	src.mkdir()
	store.mkdir()
	rel = os.path.relpath(str(src), os.sep)
	_populate(src, {'same.txt': 's', 'changed.txt': 'new!!', 'only_src.txt': 'a'})
	_populate(store / rel, {'changed.txt': 'old', 'only_store.txt': 'd', 'same.txt': 's'})
	t = os.stat(str(src / 'same.txt')).st_mtime
	os.utime(str(store / rel / 'same.txt'), (t, t))
	_populate(store / 'unrelated', {'x.txt': 'x'})
	diff = str(tmp_path / 'diff.tsv')
	hpcp_mod.process_compare_file_list([str(src)], [str(store) + os.sep], max_workers=1, dest_image='store.img', diff_file_list=diff)
	got = {(r.op, r.path) for r in hpcp_mod.read_changes(diff)}
	prefix = rel.replace(os.sep, '/') + '/'
	assert ('A', prefix + 'only_src.txt') in got
	assert ('M', prefix + 'changed.txt') in got
	assert ('D', prefix + 'only_store.txt') in got
	assert not any(p.startswith('unrelated') for _, p in got)
	assert all(p.startswith(prefix) or prefix.startswith(p) for _, p in got)
	assert (('M', prefix + 'same.txt') not in got) and (('A', prefix + 'same.txt') not in got)


def test_process_compare_src_image_root_layout_uses_relative_keys(tmp_path, hpcp_mod, reset_hpcp_globals):
	src_root = tmp_path / 'srcimg'
	store = tmp_path / 'store'
	src_root.mkdir()
	store.mkdir()
	_populate(src_root, {'same.txt': 's', 'sub/changed.txt': 'new!!', 'only_src.txt': 'a'})
	_populate(store, {'same.txt': 's', 'sub/changed.txt': 'old', 'only_store.txt': 'd'})
	t = os.stat(str(src_root / 'same.txt')).st_mtime
	os.utime(str(store / 'same.txt'), (t, t))
	diff = str(tmp_path / 'diff.tsv')
	root = str(src_root)
	hpcp_mod.process_compare_file_list([root], [str(store) + os.sep], max_workers=1, dest_image='store.img',
		diff_file_list=diff, src_image_roots=[root])
	got = {(r.op, r.path) for r in hpcp_mod.read_changes(diff)}
	assert got == {('A', 'only_src.txt'), ('M', 'sub/changed.txt'), ('D', 'only_store.txt')}


# ---- 9.73: lists from -hs 0 or another hash algorithm (review D2) ----

XXH = 'a' * 16
BLAKE = 'b' * 128


def test_unhashed_list_lines_are_files_with_unknown_content(hpcp_mod):
	"""-hs 0 writes 'path:' (an empty digest) for files and 'path:target' for links."""
	entries = hpcp_mod._parse_file_list_lines(['src/a.txt:', 'src/ln:target', 'src/sub/'], live_keys={'src/a.txt', 'src/ln'})
	assert entries == {'src/a.txt': None, 'src/ln': 'target', 'src/sub/': None}


@pytest.mark.parametrize('local_xxhash, foreign, own', [(False, XXH, BLAKE), (True, BLAKE, XXH)])
def test_digest_of_the_other_algorithm_is_a_file_with_unknown_content(hpcp_mod, monkeypatch, local_xxhash, foreign, own):
	monkeypatch.setattr(hpcp_mod, 'xxhash_available', local_xxhash)
	entries = hpcp_mod._parse_file_list_lines([f'src/a.txt:{foreign}', f'src/b.txt:{own}', 'src/ln:target'], live_keys={'src/ln'})
	assert entries == {'src/a.txt': None, 'src/b.txt': own, 'src/ln': 'target'}


def test_compare_with_an_unhashed_list_reports_no_changes(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	tmp_tree.add_file('a.txt', 'A')
	tmp_tree.add_file('sub/b.txt', 'B')
	tmp_tree.add_symlink('ln', 'a.txt')
	listing = str(tmp_tree.root / 'list.txt')
	diff = str(tmp_tree.root / 'diff.tsv')
	src = [str(tmp_tree.src)]
	assert hpcp_mod.hpcp(src, target_file_list=listing, hash_size=0, single_thread=True) == 0
	assert hpcp_mod.hpcp(src, target_file_list=listing, diff_file_list=diff, hash_size=0, single_thread=True) == 0
	assert hpcp_mod.read_changes(diff) == []


def test_unhashed_list_with_a_name_ending_in_a_colon_stays_unhashed(hpcp_mod):
	"""Only a list whose every file and link line has a ':' (as -fl with hashes writes it) can be a -hs 0 list."""
	lines = ['notes:', 'logs/12:30:00.txt', 'plain.txt', 'logs/']
	assert hpcp_mod._parse_file_list_lines(lines) == {line: None for line in lines}


@pytest.mark.parametrize('lines', [
	['snap:2026-10-16/', 'snap:2026-10-16/a.txt', 'snap:2026-10-16/logs/', 'snap:2026-10-16/logs/10:00.log'],
	['logs/2026-10-16T10:00.log'],
])
def test_unhashed_list_whose_every_name_has_a_colon_stays_unhashed(hpcp_mod, lines):
	"""Ultrareview: a source root named 'snap:…' (or one timestamped file) puts ':' in every line of a plain -nhfl
	list. Without a bare 'path:' line (what -hs 0 writes for a file) nothing marks the list as hashed."""
	assert hpcp_mod._parse_file_list_lines(lines) == {line: None for line in lines}
