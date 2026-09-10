import sys
import pytest


def _parse(hpcp_mod, argv):
	# get_args prints; that is fine
	sys.argv = argv
	return hpcp_mod.get_args()


def test_full_hash_and_hash_size(hpcp_mod, restore_argv, reset_hpcp_globals):
	args = _parse(hpcp_mod, ['hpcp', '-fh', '-hs', '0', '-d', '/tmp', '/tmp/src'])
	assert args.full_hash is True
	assert args.hash_size == 0


def test_remove_force_flag(hpcp_mod, restore_argv, reset_hpcp_globals):
	# argparse stores remove_force independently; it does not auto-set remove=
	args = _parse(hpcp_mod, ['hpcp', '-rf', '/tmp/src'])
	assert args.remove_force is True
	assert args.remove is False


def test_content_only_and_no_create_dir(hpcp_mod, restore_argv, reset_hpcp_globals):
	args = _parse(hpcp_mod, ['hpcp', '-co', '-ncd', '-d', '/tmp', '/tmp/src'])
	assert args.content_only is True
	assert args.no_create_dir is True


def test_batch_default_true(hpcp_mod, restore_argv, reset_hpcp_globals):
	args = _parse(hpcp_mod, ['hpcp', '-d', '/tmp', '/tmp/src'])
	assert args.batch is True


def test_no_batch(hpcp_mod, restore_argv, reset_hpcp_globals):
	args = _parse(hpcp_mod, ['hpcp', '-nb', '-d', '/tmp', '/tmp/src'])
	assert args.batch is False


def test_exclude_append(hpcp_mod, restore_argv, reset_hpcp_globals):
	args = _parse(hpcp_mod, ['hpcp', '-e', '*.o', '-e', 'tmp*', '-d', '/tmp', '/tmp/src'])
	assert '*.o' in args.exclude
	assert 'tmp*' in args.exclude


def test_dest_image_size_flag(hpcp_mod, restore_argv, reset_hpcp_globals):
	args = _parse(hpcp_mod, ['hpcp', '-di', '/tmp/d.img', '-dis', '1G', '/tmp/src'])
	assert args.dest_image == '/tmp/d.img'
	assert args.dest_image_size == '1G'


def test_remaining_copy_mode_flags(hpcp_mod, restore_argv, reset_hpcp_globals):
	args = _parse(hpcp_mod, [
		'hpcp', '-s', '-v', '-do', '-nds', '-pfl', '-nlt', '-rds',
		'-fpj', '8', '-j', '2', '-d', '/tmp', '/tmp/src',
	])
	assert args.single_thread is True
	assert args.verbose is True
	assert args.directory_only is True
	assert args.no_directory_sync is True
	assert args.parallel_file_listing is True
	assert args.no_link_tracking is True
	assert args.random_dest_selection is True
	assert args.files_per_job == 8
	assert args.max_workers == 2


def test_file_list_cli_flags(hpcp_mod, restore_argv, reset_hpcp_globals):
	args = _parse(hpcp_mod, [
		'hpcp', '-sfl', 'src.txt', '-fl', 'out.txt', '-cfl', '-dfl', 'diff.txt',
		'-tdfl', '-nhfl', '-d', '/tmp', '/tmp/src',
	])
	assert args.source_file_list == 'src.txt'
	assert args.target_file_list == 'out.txt'
	assert args.compare_file_list is True
	assert args.diff_file_list == 'diff.txt'
	assert args.tar_diff_file_list is True
	assert args.no_hash_file_list is True


def test_diff_file_list_does_not_set_compare_flag(hpcp_mod, restore_argv, reset_hpcp_globals):
	# argparse does not set compare_file_list; hpcp() treats a set -dfl as compare.
	args = _parse(hpcp_mod, ['hpcp', '-dfl', 'diff.txt', '-d', '/tmp', '/tmp/src'])
	assert args.diff_file_list == 'diff.txt'
	assert args.compare_file_list is False


def test_remove_and_rate_limit_flags(hpcp_mod, restore_argv, reset_hpcp_globals):
	args = _parse(hpcp_mod, [
		'hpcp', '-rm', '-rme', '-rwloff', '-L', '10M', '-F', '100',
		'-tfs', 'ext4', '-enes', '-ctl', '30', '-x', '/tmp/ex.txt',
		'-d', '/tmp', '/tmp/src',
	])
	assert args.remove is True
	assert args.remove_extra is True
	assert args.do_not_remove_files_while_listing is True
	assert args.rate_limit == '10M'
	assert args.file_rate_limit == '100'
	assert args.target_file_system == 'ext4'
	assert args.exit_not_enough_space is True
	assert args.command_timeout_limit == 30
	assert args.exclude_file == '/tmp/ex.txt'


def test_image_and_unimplemented_flags_parse(hpcp_mod, restore_argv, reset_hpcp_globals):
	args = _parse(hpcp_mod, [
		'hpcp', '-si', '/tmp/src.img', '-siff', 'diff.img', '-diff',
		'-dd', '-ddr', '10G', '-d', '/tmp/out.img', '/tmp/src.img',
	])
	assert args.src_image == ['/tmp/src.img']
	assert args.load_diff_image == ['diff.img']
	assert args.get_diff_image is True
	assert args.disk_dump is True
	assert args.dd_resize == ['10G']


def test_main_does_not_pass_unimplemented_image_flags(hpcp_mod):
	import inspect
	src = inspect.getsource(hpcp_mod.main)
	assert 'load_diff_image' not in src
	assert 'get_diff_image' not in src
