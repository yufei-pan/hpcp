import os
import subprocess
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _run_with_embedded_multicmd(tmp_path, code):
	(tmp_path / 'multiCMD.py').write_text("version = '1.0'\n")  # too old: the embedded copy is used
	env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), _ROOT]))
	return subprocess.run([sys.executable, '-c', code], stdout=subprocess.PIPE, cwd=str(tmp_path), env=env,
		universal_newlines=True, check=True).stdout


@pytest.mark.parametrize('installed, uses_installed', [
	("'1.70'", True),
	("'1.100'", True),   # float('1.100') == 1.1 would wrongly reject it
	("'2.0'", True),
	("'1.69'", False),   # older than the embedded copy, whose bar keeps its percent beside hpcp's count from 110 columns (1.70)
	("'1.68'", False),   # older still: a bar of fewer than 10 cells, a line left open at its total (1.69)
	("'1.67'", False),   # older still: a frame wider than the terminal, which wraps (1.68)
	("'1.66'", False),   # older than the embedded copy, whose format_bytes counts '4.1G' as 4100000000 bytes (1.66 gives 4099999999)
	("'1.65'", False),   # older still: a countdown at a terminal that waited for a line to end past its timeout (1.66)
	("'1.64'", False),   # older still: an ended stdin at the countdown raised, where 1.65 takes it for no answer
	("'1.63'", False),   # older still: shown calls that went on when stdout's reader quit (1.64)
	("'1.61'", False),   # older still: 9.91's copy, with the exec wrapper's SIGINT mask
	("'1.60'", False),   # no SIGINT mask in the exec wrapper (1.61)
	("'1.59'", False),   # older still: no thread-start fix (1.60)
	("'1.58'", False),   # older still: no pass_fds
	("'1.56'", False),   # the floor pip accepts, older than the embedded copy
	("'1.9'", False),    # minor 9 < 70, though float('1.9') > 1.70
	("'garbage'", False),
	(None, False),       # a module without a version
])
def test_an_installed_multicmd_is_used_only_when_not_older_than_the_embedded_one(tmp_path, installed, uses_installed):
	(tmp_path / 'multiCMD.py').write_text('marker = 1\n' + ('version = %s\n' % installed if installed else ''))
	env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), _ROOT]))
	out = subprocess.run([sys.executable, '-c', 'import hpcp; print(hasattr(hpcp.multiCMD, "marker"), hpcp.multiCMD.version)'], stdout=subprocess.PIPE,
		cwd=str(tmp_path), env=env, universal_newlines=True, check=True).stdout.split()
	assert out[0] == str(uses_installed)
	if not uses_installed:
		assert out[1] == '1.70'


def test_embedded_multicmd_is_used_when_the_installed_one_is_older(tmp_path):
	assert _run_with_embedded_multicmd(tmp_path, 'import hpcp; print(hpcp.multiCMD.version)').strip() == '1.70'


@pytest.mark.skipif(os.name != 'posix', reason='pass_fds is POSIX')
def test_pass_fds_reaches_the_command_through_the_embedded_copy(tmp_path):
	# hpcp 9.87 hands cp the held file's descriptor (pass_fds); the embedded copy (1.59 on) passes it on, through 3.6's exec
	# wrapper as through 3.7 and later's process group
	code = """import os, hpcp
r, w = os.pipe()
task = hpcp.multiCMD.run_commands([['sh', '-c', 'echo through >&%d' % w]], quiet=True, return_object=True, pass_fds=(w,))[0]
os.close(w)
print(hpcp.multiCMD.version, task.returncode, os.read(r, 100).decode().strip())"""
	assert _run_with_embedded_multicmd(tmp_path, code).split() == ['1.70', '0', 'through']


def test_the_import_gate_tracks_the_embedded_copy_and_no_installer_is_asked_for_multicmd():
	# hpcp runs its embedded copy wherever an installed multiCMD is older than it, missing or does not import, so neither
	# setup.py nor the script header (PEP 723: uv run, pipx run) asks for one: a requirement guaranteed nothing, and stock pip
	# 9.0.3 on EL7 and EL8 upgrading an old multiCMD for it stopped `pip3 install -U hpcp` (9.89, docs/lessons.md 2026-11-11).
	import re
	with open(os.path.join(_ROOT, 'hpcp.py'), encoding='utf-8') as f:
		source = f.read()
	# the embed sits one tab in, inside the import's signal mask (9.88)
	gate = re.search(r"\) < \((\d+), (\d+)\):\n\t\t\traise ImportError\('multiCMD too old'\)", source)
	assert '%s.%s' % gate.groups() == '1.70'
	header = source[:source.index('# ///\n', source.index('# /// script\n') + 1)]
	assert 'xxhash' in header and 'multicmd' not in header.lower()
	with open(os.path.join(_ROOT, 'setup.py'), encoding='utf-8') as f:
		requires = f.read().split('install_requires=[', 1)[1].split(']', 1)[0]
	assert 'xxhash' in requires and not [line for line in requires.splitlines()
		if 'multicmd' in line.lower() and not line.strip().startswith('#')]


def test_xxhash_is_not_required_where_stock_pip_cannot_install_it():
	# pip 9.0.3 on EL7 and EL8 (Python 3.6) takes no xxhash wheel and building one needs a compiler, so a plain requirement
	# fails the whole install; hpcp hashes with blake2b without it
	with open(os.path.join(_ROOT, 'setup.py'), encoding='utf-8') as f:
		assert "\"xxhash; python_version >= '3.7'\"" in f.read()


@pytest.mark.skipif(os.name != 'posix', reason='a pseudo-terminal')
@pytest.mark.parametrize('typed, answer', [(b'n\n', "'n'"), (b'n', 'None'), (b'n\x04', 'None')])
def test_the_embedded_countdown_takes_only_a_line_ended_before_it_runs_out(tmp_path, typed, answer):
	# hpcp's prompts (-dd's "All data ... will be lost", the destination prompt, a mount that fails) ask through the embedded
	# multiCMD's countdown, and take no answer for their default. From 1.66 (hpcp 9.97 carries 1.70): at a terminal a line ended by Enter is the
	# answer, and text with no Enter is none when the countdown runs out, also typed and then one Ctrl-D, which 1.64 (hpcp
	# 9.93) waited on past the countdown for the line to end.
	import pty
	master, slave = pty.openpty()
	try:
		os.write(master, typed)
		env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), _ROOT]))
		(tmp_path / 'multiCMD.py').write_text("version = '1.0'\n")  # too old: the embedded copy is used
		code = 'import hpcp; print(repr(hpcp.multiCMD.input_with_timeout_and_countdown(1, "Q?")))'
		out = subprocess.run([sys.executable, '-c', code], stdin=slave, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
			cwd=str(tmp_path), env=env, universal_newlines=True, timeout=20, start_new_session=True).stdout
	finally:
		os.close(slave)
		os.close(master)
	assert out.strip().splitlines()[-1] == answer, out
