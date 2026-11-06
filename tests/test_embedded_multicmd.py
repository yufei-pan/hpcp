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
	("'1.57'", True),
	("'1.100'", True),   # float('1.100') == 1.1 would wrongly reject it
	("'2.0'", True),
	("'1.56'", False),   # the floor pip accepts, older than the embedded copy
	("'1.9'", False),    # minor 9 < 57, though float('1.9') > 1.57
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
		assert out[1] == '1.57'


def test_embedded_multicmd_is_used_when_the_installed_one_is_older(tmp_path):
	assert _run_with_embedded_multicmd(tmp_path, 'import hpcp; print(hpcp.multiCMD.version)').strip() == '1.57'


def test_the_import_gate_tracks_the_embedded_copy_and_the_floor_is_one_number():
	# The package floor stays below the gate: stock pip 9.0.3 on EL7 and EL8 cannot upgrade multiCMD, so a raised floor fails
	# `pip3 install -U hpcp`, and an older installed multiCMD costs nothing, because the embedded copy runs instead.
	import re
	with open(os.path.join(_ROOT, 'hpcp.py'), encoding='utf-8') as f:
		source = f.read()
	gate = re.search(r"\) < \((\d+), (\d+)\):\n\t\traise ImportError\('multiCMD too old'\)", source)
	assert '%s.%s' % gate.groups() == '1.57'
	with open(os.path.join(_ROOT, 'setup.py'), encoding='utf-8') as f:
		assert "'multiCMD>=1.56'" in f.read()
	assert '"multiCMD>=1.56"' in source
