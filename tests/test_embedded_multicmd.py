import os
import subprocess
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _run_with_embedded_multicmd(tmp_path, code):
	(tmp_path / 'multiCMD.py').write_text("version = '1.0'\n")  # too old: the embedded copy is used
	env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), _ROOT]))
	return subprocess.run([sys.executable, '-c', code], stdout=subprocess.PIPE, cwd=str(tmp_path), env=env,
		universal_newlines=True, check=True).stdout


def test_embedded_multicmd_is_used_when_the_installed_one_is_older(tmp_path):
	assert _run_with_embedded_multicmd(tmp_path, 'import hpcp; print(hpcp.multiCMD.version)').strip() == '1.56'


def test_declared_multicmd_floor_is_the_one_the_import_gate_enforces():
	# below the gate hpcp ignores an installed multiCMD for its embedded copy, so a lower floor in setup.py
	# (or in the script header) would install a multiCMD that is never used
	import re
	with open(os.path.join(_ROOT, 'hpcp.py'), encoding='utf-8') as f:
		source = f.read()
	gate = re.search(r"\) < \((\d+), (\d+)\):\n\t\traise ImportError\('multiCMD too old'\)", source)
	floor = '%s.%s' % gate.groups()
	with open(os.path.join(_ROOT, 'setup.py'), encoding='utf-8') as f:
		assert "'multiCMD>=%s'" % floor in f.read()
	assert '"multiCMD>=%s"' % floor in source
