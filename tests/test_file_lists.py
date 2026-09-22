import os
import io
import pytest


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
	src = {'a.txt:h', 'only_src.txt:h'}
	dst = {'a.txt:h', 'only_dst.txt:h'}
	hpcp_mod.compare_file_list(src, dst, diff_file_list=str(diff), tar_diff_file_list=False)
	text = diff.read_text()
	assert 'only_dst.txt' in text
	assert 'only_src.txt' in text


def test_tar_diff_file_list_is_new_or_updated_only(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Help text: tar-compatible diff lists update/new files only (in src not dest)."""
	diff = tmp_path / 'tar.txt'
	src = {'new.txt:h', 'updated.txt:new', 'same.txt:h'}
	dst = {'same.txt:h', 'updated.txt:old', 'extra.txt:h'}
	hpcp_mod.compare_file_list(src, dst, diff_file_list=str(diff), tar_diff_file_list=True)
	text = diff.read_text()
	assert set(text.splitlines()) == {'new.txt', 'updated.txt'}


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
	assert diff.read_bytes() == b'-\x00new.txt\n'
