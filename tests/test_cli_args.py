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
