import os
import pytest


def test_get_last_existing_parent_finds_root(tmp_path, hpcp_mod, reset_hpcp_globals):
	missing = tmp_path / 'a' / 'b' / 'c'
	parent = hpcp_mod.get_last_existing_parent(str(missing))
	assert os.path.isdir(parent)
	assert os.path.samefile(parent, tmp_path)


def test_get_file_size_missing_returns_zero(hpcp_mod, reset_hpcp_globals, tmp_path):
	missing = str(tmp_path / 'nope')
	assert hpcp_mod.get_file_size(missing) == 0


def test_natural_sort_orders_numeric_suffixes(hpcp_mod):
	assert hpcp_mod.natural_sort(['file10.txt', 'file2.txt', 'file1.txt']) == [
		'file1.txt', 'file2.txt', 'file10.txt',
	]


def test_format_time_examples(hpcp_mod):
	assert hpcp_mod.format_time(0) == '0s'
	assert hpcp_mod.format_time(65) == '1m5s'
	assert hpcp_mod.format_time(3661) == '1h1m1s'


def test_is_excluded_globs(hpcp_mod):
	assert hpcp_mod.is_excluded('/data/tmp/cache', ['*/cache']) is True
	assert hpcp_mod.is_excluded('/data/tmp/cache', ['*/logs']) is False
	assert hpcp_mod.is_excluded('/data/tmp/cache', None) is False


def test_get_file_size_reports_allocated_size_of_sparse_file(hpcp_mod, tmp_path):
	path = tmp_path / 'sparse.img'
	with open(path, 'wb') as f:
		f.truncate(64 << 20)
		f.write(b'x')
	st = os.stat(path)
	if not hasattr(st, 'st_blocks') or st.st_blocks * 512 >= st.st_size:
		pytest.skip('filesystem does not report sparse allocation')
	assert hpcp_mod.get_file_size(str(path)) == st.st_blocks * 512
	assert hpcp_mod.get_file_size(str(path)) < st.st_size


@pytest.mark.parametrize('messages,expected', [
	(['Unregistered failure: something failed'], 128),
	(['FS param warning: using defaults'], 0),
	(['FS param warning: using defaults', 'Unregistered failure: failed'], 128),
	([], 0),
])
def test_error_status_preserves_unknown_failures(hpcp_mod, reset_hpcp_globals, messages, expected):
	for message in messages:
		hpcp_mod.eprint(message)
	assert hpcp_mod.get_rc_from_error() == expected
	assert hpcp_mod.get_rc_from_error() == 0


def test_worker_reports_only_current_task_errors(hpcp_mod, reset_hpcp_globals, monkeypatch):
	monkeypatch.setattr(hpcp_mod, '_worker_initialized', True)
	hpcp_mod.ERRORS.append('Copy failed: parent error')
	settings = {name: getattr(hpcp_mod, name) for name in hpcp_mod._TASK_SETTINGS}
	assert hpcp_mod._run_task(settings, hpcp_mod.eprint, 'FS param warning: defaults') == (None, ['FS param warning: defaults'], None)
	assert hpcp_mod._run_task(settings, hpcp_mod.eprint, 'Copy failed: worker error') == (None, ['Copy failed: worker error'], None)
	assert hpcp_mod._run_task(settings, int, '3') == (3, [], None)
	result, errors, crash = hpcp_mod._run_task(settings, int, 'x')
	assert result is None and errors == [] and crash.startswith('ValueError')
	assert hpcp_mod.ERRORS == ['Copy failed: parent error']


def test_call_raising_recorded_failures_raises_only_failures(hpcp_mod, reset_hpcp_globals):
	hpcp_mod.ERRORS.append('Copy failed: parent error')
	hpcp_mod._call_raising_recorded_failures(hpcp_mod.eprint, 'FS param warning: defaults')
	with pytest.raises(RuntimeError, match='Copy failed: worker error'):
		hpcp_mod._call_raising_recorded_failures(hpcp_mod.eprint, 'Copy failed: worker error')
	assert hpcp_mod.ERRORS == ['Copy failed: parent error']


def test_task_result_records_a_task_whose_worker_died(hpcp_mod, reset_hpcp_globals):
	"""A worker killed mid-task (OOM) fails its future with BrokenProcessPool: recorded like a crash, not raised."""
	from concurrent.futures.process import BrokenProcessPool

	class DeadFuture:
		def result(self):
			raise BrokenProcessPool('A process in the process pool was terminated abruptly')

	assert hpcp_mod._task_result(DeadFuture(), (0, 0), 'Remove process exception') == (0, 0)
	assert [error.partition(':')[0] for error in hpcp_mod.ERRORS] == ['Remove process exception']
