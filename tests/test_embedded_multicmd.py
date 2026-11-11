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
	("'1.60'", True),
	("'1.100'", True),   # float('1.100') == 1.1 would wrongly reject it
	("'2.0'", True),
	("'1.59'", False),   # older than the embedded copy, which brings the thread-start fix (1.60)
	("'1.58'", False),   # older still: no pass_fds
	("'1.56'", False),   # the floor pip accepts, older than the embedded copy
	("'1.9'", False),    # minor 9 < 60, though float('1.9') > 1.60
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
		assert out[1] == '1.60'


def test_embedded_multicmd_is_used_when_the_installed_one_is_older(tmp_path):
	assert _run_with_embedded_multicmd(tmp_path, 'import hpcp; print(hpcp.multiCMD.version)').strip() == '1.60'


@pytest.mark.skipif(os.name != 'posix', reason='pass_fds is POSIX')
def test_pass_fds_reaches_the_command_through_the_embedded_copy(tmp_path):
	# hpcp 9.87 hands cp the held file's descriptor (pass_fds); the embedded copy (1.59 on) passes it on, through 3.6's exec
	# wrapper as through 3.7 and later's process group
	code = """import os, hpcp
r, w = os.pipe()
task = hpcp.multiCMD.run_commands([['sh', '-c', 'echo through >&%d' % w]], quiet=True, return_object=True, pass_fds=(w,))[0]
os.close(w)
print(hpcp.multiCMD.version, task.returncode, os.read(r, 100).decode().strip())"""
	assert _run_with_embedded_multicmd(tmp_path, code).split() == ['1.60', '0', 'through']


def test_the_import_gate_tracks_the_embedded_copy_and_the_floor_is_one_number():
	# The package floor stays below the gate: stock pip 9.0.3 on EL7 and EL8 cannot upgrade multiCMD, so a raised floor fails
	# `pip3 install -U hpcp`, and an older installed multiCMD costs nothing, because the embedded copy runs instead.
	import re
	with open(os.path.join(_ROOT, 'hpcp.py'), encoding='utf-8') as f:
		source = f.read()
	# the embed sits one tab in, inside the import's signal mask (9.88)
	gate = re.search(r"\) < \((\d+), (\d+)\):\n\t\t\traise ImportError\('multiCMD too old'\)", source)
	assert '%s.%s' % gate.groups() == '1.60'
	with open(os.path.join(_ROOT, 'setup.py'), encoding='utf-8') as f:
		assert "'multiCMD>=1.56'" in f.read()
	assert '"multiCMD>=1.56"' in source


def test_xxhash_is_not_required_where_stock_pip_cannot_install_it():
	# pip 9.0.3 on EL7 and EL8 (Python 3.6) takes no xxhash wheel and building one needs a compiler, so a plain requirement
	# fails the whole install; hpcp hashes with blake2b without it
	with open(os.path.join(_ROOT, 'setup.py'), encoding='utf-8') as f:
		assert "\"xxhash; python_version >= '3.7'\"" in f.read()
