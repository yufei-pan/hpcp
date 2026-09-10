import pytest


def test_format_bytes_parses_rate_strings(hpcp_mod):
	n = hpcp_mod.format_bytes('10M', to_int=True)
	assert n in (10 * 1000 * 1000, 10 * 1024 * 1024)
	assert hpcp_mod.format_bytes('1Ki', to_int=True) == 1024


def test_hpcp_applies_rate_and_timeout_globals(tmp_tree, hpcp_mod, reset_hpcp_globals, copy_args, require_linux):
	tmp_tree.add_file('a.txt', 'x')
	srcs, opts = copy_args(
		bytes_rate_limit='10M',
		files_rate_limit='1K',
		command_timeout_limit=12,
	)
	hpcp_mod.hpcp(srcs, **opts)
	assert hpcp_mod.BYTES_RATE_LIMIT == hpcp_mod.format_bytes('10M', to_int=True)
	assert hpcp_mod.FILES_RATE_LIMIT == hpcp_mod.format_bytes('1K', to_int=True)
	assert hpcp_mod.COMMAND_TIMEOUT == 12


def test_progress_bar_under_rate_limit(hpcp_mod):
	apb = hpcp_mod.Adaptive_Progress_Bar(
		total_count=100, total_size=1000, suppress_all_output=True,
		bytes_rate_limit=1, files_rate_limit=0,
	)
	apb.size_counter = 10**9
	assert apb.under_rate_limit() is False
	apb.size_counter = 0
	assert apb.under_rate_limit() is True


def test_max_workers_zero_is_half_cpu(tmp_tree, hpcp_mod, reset_hpcp_globals, copy_args, require_linux, monkeypatch):
	seen = {}
	Orig = hpcp_mod.copy_scheduler

	class Spy(Orig):
		def __init__(self, *args, **kwargs):
			seen['max_workers'] = kwargs.get('max_workers', args[0] if args else None)
			super().__init__(*args, **kwargs)

	monkeypatch.setattr(hpcp_mod, 'copy_scheduler', Spy)
	monkeypatch.setattr(hpcp_mod.multiprocessing, 'cpu_count', lambda: 8)
	tmp_tree.add_file('a.txt', 'x')
	srcs, opts = copy_args(max_workers=0, single_thread=False)
	hpcp_mod.hpcp(srcs, **opts)
	assert seen['max_workers'] == 4


def test_max_workers_negative_is_cpu_multiple(tmp_tree, hpcp_mod, reset_hpcp_globals, copy_args, require_linux, monkeypatch):
	seen = {}
	Orig = hpcp_mod.copy_scheduler

	class Spy(Orig):
		def __init__(self, *args, **kwargs):
			seen['max_workers'] = kwargs.get('max_workers', args[0] if args else None)
			super().__init__(*args, **kwargs)

	monkeypatch.setattr(hpcp_mod, 'copy_scheduler', Spy)
	monkeypatch.setattr(hpcp_mod.multiprocessing, 'cpu_count', lambda: 8)
	tmp_tree.add_file('a.txt', 'x')
	srcs, opts = copy_args(max_workers=-2, single_thread=False)
	hpcp_mod.hpcp(srcs, **opts)
	assert seen['max_workers'] == 16


def test_target_fs_mismatch_skips_dest(tmp_tree, hpcp_mod, reset_hpcp_globals, copy_args, require_linux):
	tmp_tree.add_file('a.txt', 'x')
	real = hpcp_mod.get_fs_type_from_path(str(tmp_tree.dst))
	if not real:
		pytest.skip('could not detect dest filesystem type')
	srcs, opts = copy_args(target_file_system='no_such_fs')
	rc = hpcp_mod.hpcp(srcs, **opts)
	assert not (tmp_tree.dst / 'a.txt').exists()
	assert rc != 0 or not (tmp_tree.dst / 'a.txt').exists()


def test_exit_not_enough_space_skips_copy(tmp_tree, hpcp_mod, reset_hpcp_globals, copy_args, require_linux, monkeypatch):
	tmp_tree.add_file('a.txt', 'payload-bytes')
	monkeypatch.setattr(hpcp_mod, 'get_free_space_bytes', lambda path: 0)
	srcs, opts = copy_args(exit_not_enough_space=True)
	hpcp_mod.hpcp(srcs, **opts)
	assert not (tmp_tree.dst / 'a.txt').exists()


def test_check_path_finds_cp(hpcp_mod, reset_hpcp_globals, require_linux):
	assert hpcp_mod.check_path('cp') is True


def test_get_fs_type_from_path_nonzero(tmp_path, hpcp_mod, reset_hpcp_globals, require_linux):
	fs = hpcp_mod.get_fs_type_from_path(str(tmp_path))
	assert isinstance(fs, str)
	assert len(fs) > 0
