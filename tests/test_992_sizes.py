'''hpcp 9.92: the sizes of -ddr, -L, -F and -dis are read once, by hpcp's own rule, before anything is touched. A text that holds
a digit and is no size ends the run as a usage error (241), and so does everything -ddr cannot take as a partition size; for -L,
-F and -dis a text with no digit, a negative number and a size below 1 run as before, with no limit and no split, and now say so.
9.91 (9348355) handed each text to multiCMD's format_bytes, which answers a word with 0 and an unknown unit with None (multiCMD
1.61 to 1.64).'''
import os
import subprocess
import sys

import pytest

import hpcp

G = 1024 ** 3
HPCP = os.path.abspath(hpcp.__file__)
# refused: (option, value); a list for -ddr, which the command line gives as one
REFUSED = [
	('-ddr', ['one']), ('-ddr', ['5foo']), ('-ddr', ['100G', '5foo']), ('-ddr', ['']), ('-ddr', ['-1']), ('-ddr', ['10G/s']),
	('-L', '5foo'), ('-L', '1e3'), ('-L', '1,5M'), ('-L', '10M per second'), ('-L', '{10TiB}'), ('-F', '5foo'), ('-dis', '5foo'),
	('-dis', '10G/s'), ('-dis', '{10TiB}'),
]
# run with no limit or no split, and said: (option, value, the note)
WITHOUT = [
	('-L', 'none', "Note: -L 'none' has no number: no rate limit"), ('-L', '', "Note: -L '' has no number: no rate limit"),
	('-L', '-1', "Note: -L '-1' is below 0: no rate limit"), ('-L', 'fast', "Note: -L 'fast' has no number: no rate limit"),
	('-F', '', "Note: -F '' has no number: no rate limit"), ('-F', '0.5', "Note: -F '0.5' is below 1: no rate limit"),
	('-F', '-1', "Note: -F '-1' is below 0: no rate limit"), ('-dis', '-5G', "Note: -dis '-5G' is below 0: no split"),
	('-dis', '', "Note: -dis '' has no number: no split"), ('-dis', 'none', "Note: -dis 'none' has no number: no split"),
]
KEYWORD = {'-ddr': 'dd_resize', '-L': 'bytes_rate_limit', '-F': 'files_rate_limit', '-dis': 'dest_image_size'}


def _dd_recorders(monkeypatch):
	'''-dd's outside-world steps (the source's partition table, the image resize, the partition table write, a loop device) as
	recorders: no test here reaches losetup, also when the refusal is undone.'''
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
	monkeypatch.setattr(hpcp, 'create_loop_device', lambda *a, **k: steps.append('create_loop_device') or '/dev/null')
	return steps


def _tree(tmp_path):
	src = tmp_path / 'src'
	src.mkdir()
	(src / 'f').write_text('data')
	return src, tmp_path / 'dst'


@pytest.mark.parametrize('option, value', REFUSED)
def test_a_size_that_cannot_be_read_ends_the_library_call_with_241_and_touches_nothing(tmp_path, reset_hpcp_globals, monkeypatch, capsys, option, value):
	# 9.91: -ddr one resized the destination image and wrote its partition table (sgdisk --clear) with a 0 size; -ddr 5foo
	# crashed in sum(); -L 5foo ended in a TypeError at the first progress bar; -L 1e3 limited to 1 EB, -L 1,5M to 1 byte a second
	steps = _dd_recorders(monkeypatch)
	src, dst = _tree(tmp_path)
	before = (hpcp.BYTES_RATE_LIMIT, hpcp.FILES_RATE_LIMIT)
	kwargs = {KEYWORD[option]: value}
	if option == '-ddr':
		(tmp_path / 'img').write_text('keep')
		rc = hpcp.hpcp(['src'], dest_paths=[str(tmp_path / 'img')], dd=True, **kwargs)
	else:
		rc = hpcp.hpcp([str(src) + '/'], dest_paths=[str(dst)], single_thread=True, **kwargs)
	said = capsys.readouterr().err
	assert rc == 241, (rc, said[-1500:])
	assert 'Invalid arguments: ' + option + ' ' in said and (' is not a size' in said or ' is no partition size' in said) and repr(value if isinstance(value, str) else value[-1]) in said
	assert steps == [] and not dst.exists()
	assert (hpcp.BYTES_RATE_LIMIT, hpcp.FILES_RATE_LIMIT) == before
	if option == '-ddr':
		assert (tmp_path / 'img').read_text() == 'keep'


@pytest.mark.parametrize('option, value', REFUSED)
def test_a_size_that_cannot_be_read_ends_the_command_line_with_241(tmp_path, option, value):
	# the same through the command line: -dd as nobody would wait five seconds for its warning first
	src, dst = _tree(tmp_path)
	dst.write_text('keep')
	argv = [sys.executable, HPCP, str(src) + '/', '-d', str(dst), '-s']
	for one in value if isinstance(value, list) else [value]:
		argv += [option + '=' + one] if one.startswith('-') else [option, one]
	if option == '-ddr':
		argv = [sys.executable, HPCP, '-dd', str(src), str(dst)] + argv[argv.index('-s') + 1:]
	task = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, cwd=str(tmp_path))
	assert task.returncode == 241, (task.returncode, task.stdout[-800:], task.stderr[-800:])
	assert b'Invalid arguments: ' + option.encode() + b' ' in task.stderr
	assert dst.read_text() == 'keep' and sorted(os.listdir(str(tmp_path))) == ['dst', 'src']


@pytest.mark.parametrize('option, value, note', WITHOUT)
def test_a_value_with_no_number_runs_without_a_limit_and_says_so(tmp_path, reset_hpcp_globals, capsys, option, value, note):
	# 9.91 ran these with no limit and no split, saying nothing of it on stderr; 9.92's first build refused them (241)
	src, dst = _tree(tmp_path)
	rc = hpcp.hpcp([str(src) + '/'], dest_paths=[str(dst)], single_thread=True, **{KEYWORD[option]: value})
	said = capsys.readouterr().err
	assert rc == 0, (rc, said[-1500:])
	assert note in said.splitlines()
	assert (dst / 'f').read_text() == 'data'
	assert (hpcp.BYTES_RATE_LIMIT, hpcp.FILES_RATE_LIMIT) == (0, 0)


def test_a_value_with_no_number_runs_through_the_command_line_too(tmp_path):
	src, dst = _tree(tmp_path)
	task = subprocess.run([sys.executable, HPCP, str(src) + '/', '-d', str(dst) + '/', '-s', '-L', '', '-F=-1', '-dis', 'none'],
		stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, cwd=str(tmp_path))
	assert task.returncode == 0, (task.returncode, task.stdout[-800:], task.stderr[-800:])
	said = task.stderr.decode().splitlines()
	assert "Note: -L '' has no number: no rate limit" in said and "Note: -F '-1' is below 0: no rate limit" in said
	assert "Note: -dis 'none' has no number: no split" in said
	assert (dst / 'f').read_text() == 'data'


@pytest.mark.parametrize('value, limit', [('10MB/s', 10 ** 7), ('10M/s', 10 ** 7), ('10 MiB / S', 10 * 1024 ** 2), (' 512 ', 512), ('0', 0), ('.5G', 5 * 10 ** 8), ('+5G', 5 * 10 ** 9),
	('+.5K/s', 500)])
def test_a_rate_written_as_a_rate_is_read(tmp_path, reset_hpcp_globals, capsys, value, limit):
	# 9.91 read the front of the text, and a .5G or a +5G as no number; 9.92's first build refused the /s (241)
	src, dst = _tree(tmp_path)
	assert hpcp.hpcp([str(src) + '/'], dest_paths=[str(dst)], single_thread=True, bytes_rate_limit=value, files_rate_limit='100/s') == 0
	assert (hpcp.BYTES_RATE_LIMIT, hpcp.FILES_RATE_LIMIT) == (limit, 100)
	assert 'Note:' not in capsys.readouterr().err and (dst / 'f').read_text() == 'data'


def test_a_rate_written_as_a_rate_is_read_through_the_command_line(tmp_path):
	src, dst = _tree(tmp_path)
	task = subprocess.run([sys.executable, HPCP, str(src) + '/', '-d', str(dst) + '/', '-s', '-L', '10MB/s', '-F', '100/s'],
		stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, cwd=str(tmp_path))
	assert task.returncode == 0, (task.returncode, task.stdout[-800:], task.stderr[-800:])
	assert (dst / 'f').read_text() == 'data'


@pytest.mark.parametrize('text', ['0', '512', '100', '10K', '1Ki', '10M', '1G', '1Gi', '200G', '100GiB', '10TiB', '1.5 GiB', '1.5GiB', '5kb', '5B',
	'5 b', '2PiB', '1EB', '3ZiB', '4yb', ' 10M ', '10m', '10gIb', '999Y'])
def test_every_size_the_help_shows_is_taken_with_the_value_multicmd_reads(text):
	for option in ('-L', '-F', '-dis') + (('-ddr',) if text != '0' else ()):
		size, note, wrong = hpcp._read_size(option, text)
		assert (note, wrong) == (None, None), (option, note, wrong)
		assert size == hpcp.multiCMD.format_bytes(text.strip(), to_int=True) and isinstance(size, int)


@pytest.mark.parametrize('text', ['5foo', '5 G B', '1e3', '5GG', '5Gx', '5Gib!', '5ib', '1,5G', '5 GiB extra', '0x10', '٥', '１０',
	'5.', '10M10', '{10TiB}', '-5foo', '+5foo', '.5.5G', '- 5', '+ 5G', '1' * 28, '9' * 400, '1000Y', '2000Y', True, float('nan'), float('inf'), [], {}, b'10'])
def test_what_holds_a_digit_and_is_no_size_is_refused_for_every_option(text):
	# 9.91: '9' * 400 was an OverflowError's traceback and -L 2000Y a KeyError at the first progress bar (exit 70)
	for option in ('-L', '-F', '-dis', '-ddr'):
		size, note, wrong = hpcp._read_size(option, text)
		assert (size, note) == (None, None) and wrong.startswith(option + ' ' + repr(text) + ' is '), (option, size, note, wrong)


@pytest.mark.parametrize('value', ['one', '', ' ', 'fast', 'G', '-5', '-5G', '-0', '/s', -1, -0.5, 0.5, '0.5', '.5', '0.0004K', None, False, 0, '0', '0G'])
def test_ddr_takes_a_size_above_0_and_nothing_else(value):
	size, note, wrong = hpcp._read_size('-ddr', value)
	assert (size, note) == (None, None) and wrong.startswith('-ddr ' + repr(value) + ' is no'), wrong   # 'is not a size', 'is no partition size'


@pytest.mark.parametrize('value, why', [('one', 'has no number'), ('none', 'has no number'), ('', 'has no number'), (' ', 'has no number'), ('inf', 'has no number'), ('/s', 'has no number'),
	('-5', 'is below 0'), ('-5G', 'is below 0'), ('-.5', 'is below 0'), (-1, 'is below 0'), (-0.5, 'is below 0'), (0.5, 'is below 1'),
	('0.5', 'is below 1'), ('.5', 'is below 1'), ('+.5', 'is below 1'), ('0.0004K', 'is below 1')])
def test_what_holds_no_size_means_no_limit_and_no_split_with_its_note(value, why):
	for option, without in (('-L', 'no rate limit'), ('-F', 'no rate limit'), ('-dis', 'no split')):
		assert hpcp._read_size(option, value) == (0, 'Note: %s %r %s: %s' % (option, value, why, without), None)


@pytest.mark.parametrize('value, size', [(None, None), (False, None), (0, 0), ('0', 0), ('0.0', 0), ('0K', 0), ('-0', 0), (512, 512), (1.5e3, 1500),
	(10 ** 26, 10 ** 26)])
def test_unset_zero_and_numbers_are_taken_without_a_note(value, size):
	for option in ('-L', '-F', '-dis'):
		assert hpcp._read_size(option, value) == (size, None, None)


@pytest.mark.parametrize('text, size', [('.5G', 5 * 10 ** 8), ('+5G', 5 * 10 ** 9), ('+.5K', 500), ('+5', 5), ('.5Ki', 512)])
def test_a_leading_plus_or_point_is_number_syntax(text, size):
	# 9.91 found no number in these and ran with no limit; -ddr read .5G as a 0-byte partition
	for option in ('-L', '-F', '-dis', '-ddr'):
		assert hpcp._read_size(option, text) == (size, None, None), option


def test_a_number_too_large_for_a_float_is_refused_not_a_traceback():
	from fractions import Fraction
	for value in (10 ** 400, Fraction(10 ** 400)):
		for option in ('-L', '-ddr'):
			size, note, wrong = hpcp._read_size(option, value)
			assert (size, note) == (None, None) and ' is too large ' in wrong


def test_a_number_that_is_no_int_or_float_is_taken_as_its_number(tmp_path, reset_hpcp_globals):
	# a numpy integer is such a number (a numbers.Integral that is no int), a Fraction another: 9.91 took them, and False as
	# unset; 9.92's first build refused all three (241)
	from fractions import Fraction
	assert hpcp._read_size('-L', Fraction(10 ** 7)) == (10 ** 7, None, None) and type(hpcp._read_size('-L', Fraction(10 ** 7))[0]) is int
	src, dst = _tree(tmp_path)
	assert hpcp.hpcp([str(src) + '/'], dest_paths=[str(dst)], single_thread=True, bytes_rate_limit=Fraction(10 ** 7), files_rate_limit=False, dest_image_size=False) == 0
	assert (hpcp.BYTES_RATE_LIMIT, hpcp.FILES_RATE_LIMIT) == (10 ** 7, 0) and (dst / 'f').read_text() == 'data'


def test_the_unit_letters_are_the_ones_multicmd_reads():
	# format_bytes' unit_labels are local to it: what it reads as a size, a letter at a time, is what hpcp takes (but for
	# an i with no letter before it, which it reads as nothing and hpcp does not take)
	import string
	for letter in string.ascii_lowercase.replace('i', ''):
		for text in ('1' + letter, '1' + letter.upper() + 'iB'):
			assert (hpcp._read_size('-ddr', text)[2] is None) == multiCMD_reads(text), text


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


def test_a_rate_limit_of_0_takes_an_earlier_calls_limit_away(tmp_path, reset_hpcp_globals):
	# a library caller's second run: '0' has always set the limit back to none
	src, dst = _tree(tmp_path)
	assert hpcp.hpcp([str(src) + '/'], dest_paths=[str(dst)], single_thread=True, bytes_rate_limit='10M', files_rate_limit='1K') == 0
	assert (hpcp.BYTES_RATE_LIMIT, hpcp.FILES_RATE_LIMIT) == (10 ** 7, 1000)
	assert hpcp.hpcp([str(src) + '/'], dest_paths=[str(dst)], single_thread=True, bytes_rate_limit='0', files_rate_limit='0') == 0
	assert (hpcp.BYTES_RATE_LIMIT, hpcp.FILES_RATE_LIMIT) == (0, 0)


def test_a_zero_partition_size_is_refused_as_no_size(tmp_path, reset_hpcp_globals, monkeypatch, capsys):
	steps = _dd_recorders(monkeypatch)
	assert hpcp.hpcp(['src'], dest_paths=[str(tmp_path / 'img')], dd=True, dd_resize=['100G', '0G']) == 241
	assert "-ddr '0G' is no partition size" in capsys.readouterr().err and steps == []


def test_a_size_too_large_for_the_units_is_refused_by_the_command_line(tmp_path):
	# 9.91: -L 2000Y ended in a KeyError at the first progress bar (70), 400 digits in an OverflowError's traceback (1)
	src, dst = _tree(tmp_path)
	for option, value in (('-L', '2000Y'), ('-F', '9' * 30), ('-ddr', '9' * 400), ('-dis', '9' * 400)):
		task = subprocess.run([sys.executable, HPCP, str(src) + '/', '-d', str(dst) + '/', '-s', option, value],
			stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, cwd=str(tmp_path))
		assert task.returncode == 241 and b'Traceback' not in task.stderr, (option, task.returncode, task.stderr[-600:])
		assert b'Invalid arguments: ' + option.encode() + b' ' in task.stderr and not dst.exists()
