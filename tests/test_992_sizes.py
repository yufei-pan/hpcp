'''hpcp 9.92: a size hpcp cannot read ends the run as a usage error (241) before anything is touched. Each refusal test fails on 9.91
(9348355), which took a word for 0 and an unknown unit for None (multiCMD 1.64).'''
import os
import subprocess
import sys

import pytest

import hpcp

G = 1024 ** 3
HPCP = os.path.abspath(hpcp.__file__)
WORDS = [
	('-ddr', ['one']), ('-ddr', ['5foo']), ('-ddr', ['100G', '5foo']), ('-L', 'fast'), ('-L', '5foo'), ('-F', '5foo'), ('-dis', '5foo'),
]


def _library_kwargs(option, value):
	return {'-ddr': 'dd_resize', '-L': 'bytes_rate_limit', '-F': 'files_rate_limit', '-dis': 'dest_image_size'}[option], value


def _dd_recorders(monkeypatch):
	'''-dd's outside-world steps (the source's partition table, the image resize, the partition table write) as recorders.'''
	steps = []
	infos = {'src': {'size': 300 * G, 'disk_identifier': 'x', 'sector_size': 512}, '1': {'size': 100 * G, 'sector_size': 512},
		'2': {'size': 50 * G, 'sector_size': 512}}
	monkeypatch.setattr(hpcp.os, 'geteuid', lambda: 0)   # as root: no five-second warning
	monkeypatch.setattr(hpcp, 'validate_dd_source_path', lambda *a, **k: steps.append('validate_dd_source_path') or 'src')
	def get_partition_infos(device):
		steps.append('get_partition_infos')
		return {k: dict(v) for k, v in infos.items()}
	get_partition_infos.cache_clear = lambda: None   # hpcp's own is cached, and a run's end clears it
	monkeypatch.setattr(hpcp, 'get_partition_infos', get_partition_infos)
	monkeypatch.setattr(hpcp, 'resize_image', lambda name, size: steps.append(('resize_image', size)))
	monkeypatch.setattr(hpcp, 'create_partition_table', lambda *a, **k: steps.append('create_partition_table') or ([], None))
	return steps


@pytest.mark.parametrize('option, value', WORDS)
def test_a_size_that_is_none_ends_the_library_call_with_241_and_touches_nothing(tmp_path, reset_hpcp_globals, monkeypatch, capsys, option, value):
	# 9.91: -ddr one resized the destination image and wrote its partition table (sgdisk --clear) with a 0 size; -ddr 5foo
	# crashed in sum(); -L fast copied with no limit; -L 5foo ended in a TypeError at the first progress bar; -F 5foo and -dis 5foo
	# copied with none
	steps = _dd_recorders(monkeypatch)
	src = tmp_path / 'src'
	src.mkdir()
	(src / 'f').write_text('data')
	dst = tmp_path / 'dst'
	before = (hpcp.BYTES_RATE_LIMIT, hpcp.FILES_RATE_LIMIT)
	name, given = _library_kwargs(option, value)
	kwargs = {name: given}
	if option == '-ddr':
		(tmp_path / 'img').write_text('keep')
		rc = hpcp.hpcp(['src'], dest_paths=[str(tmp_path / 'img')], dd=True, **kwargs)
	else:
		rc = hpcp.hpcp([str(src) + '/'], dest_paths=[str(dst)], single_thread=True, **kwargs)
	said = capsys.readouterr().err
	assert rc == 241, (rc, said[-1500:])
	assert 'Invalid arguments: ' + option + ' ' in said and 'is not a size' in said and repr(value if isinstance(value, str) else value[-1]) in said
	assert steps == [] and not dst.exists()
	assert (hpcp.BYTES_RATE_LIMIT, hpcp.FILES_RATE_LIMIT) == before
	if option == '-ddr':
		assert (tmp_path / 'img').read_text() == 'keep'


@pytest.mark.parametrize('option, value', WORDS)
def test_a_size_that_is_none_ends_the_command_line_with_241(tmp_path, option, value):
	# the same through the command line: -dd as nobody would wait five seconds for its warning first
	src = tmp_path / 'src'
	src.mkdir()
	(src / 'f').write_text('data')
	dst = tmp_path / 'dst'
	dst.write_text('keep')
	argv = [sys.executable, HPCP, str(src) + '/', '-d', str(dst), '-s']
	for one in value if isinstance(value, list) else [value]:
		argv += [option, one]
	if option == '-ddr':
		argv = [sys.executable, HPCP, '-dd', str(src), str(dst)] + argv[argv.index('-s') + 1:]
	task = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, cwd=str(tmp_path))
	assert task.returncode == 241, (task.returncode, task.stdout[-800:], task.stderr[-800:])
	assert b'Invalid arguments: ' + option.encode() + b' ' in task.stderr
	assert dst.read_text() == 'keep' and sorted(os.listdir(str(tmp_path))) == ['dst', 'src']


@pytest.mark.parametrize('text', ['0', '512', '100', '10K', '1Ki', '10M', '1G', '1Gi', '200G', '100GiB', '10TiB', '1.5 GiB', '1.5GiB', '5kb', '5B',
	'5 b', '2PiB', '1EB', '3ZiB', '4yb', ' 10M ', '10m', '10gIb'])
def test_every_size_the_help_and_readme_show_and_every_one_format_bytes_reads_is_taken(text):
	clean = hpcp._clean_size(text)
	assert clean == text.strip()
	assert isinstance(hpcp.format_bytes(clean, to_int=True), int)


@pytest.mark.parametrize('text', ['one', '5foo', '', ' ', 'fast', 'G', '-5', '+5', '5 G B', '1e3', '.5G', '5GG', '5Gx', '5Gib!', '5ib', '1,5G', '5 GiB extra',
	'0x10', '٥', None, True, -1, float('nan'), float('inf'), [], {}])
def test_what_is_no_size_is_none(text):
	assert hpcp._clean_size(text) is None


def test_the_unit_letters_are_the_ones_multicmd_reads():
	# format_bytes' unit_labels are local to it: what it reads as a size, a letter at a time, is what _clean_size takes (but for
	# an i with no letter before it, which it reads as nothing and hpcp does not take)
	import string
	for letter in string.ascii_lowercase.replace('i', ''):
		for text in ('1' + letter, '1' + letter.upper() + 'iB'):
			assert (hpcp._clean_size(text) is not None) == multiCMD_reads(text), text


def multiCMD_reads(text):
	import io
	import contextlib
	with contextlib.redirect_stderr(io.StringIO()):
		try:
			result = hpcp.multiCMD.format_bytes(text, to_int=True)
		except Exception:
			return False
	return isinstance(result, int) and result > 0


def test_a_lone_size_for_ddr_is_a_list_of_one_and_a_valid_run_goes_on(tmp_path, reset_hpcp_globals, monkeypatch):
	steps = _dd_recorders(monkeypatch)
	monkeypatch.setattr(hpcp, 'validate_dd_source_path', lambda *a, **k: 'src')
	hpcp.hpcp(['src'], dest_paths=[str(tmp_path / 'img')], dd=True, dd_resize='100G')
	assert ('resize_image', 100 * 1000 ** 3 + 50 * G + 2 * 1024 ** 2 + hpcp._GPT_STRUCTURE_RESERVE) in steps


def test_the_valid_rate_limits_and_image_size_still_copy(tmp_path, reset_hpcp_globals):
	src = tmp_path / 'src'
	src.mkdir()
	(src / 'f').write_text('data')
	for text in ('0', '10M', '1Gi'):
		dst = tmp_path / ('dst' + text)
		assert hpcp.hpcp([str(src) + '/'], dest_paths=[str(dst)], single_thread=True, bytes_rate_limit=text, files_rate_limit='1K', dest_image_size='0') == 0
		assert (dst / 'f').read_text() == 'data'


def test_a_zero_partition_size_is_refused_as_no_size(tmp_path, reset_hpcp_globals, monkeypatch, capsys):
	steps = _dd_recorders(monkeypatch)
	assert hpcp.hpcp(['src'], dest_paths=[str(tmp_path / 'img')], dd=True, dd_resize=['100G', '0G']) == 241
	assert "-ddr '0G' is no partition size" in capsys.readouterr().err and steps == []
