'''9.87 group 3: small fixes (A1 is in test_tool_path_overrides.py). Subprocess runs use the CLI as a person does.'''
import os
import subprocess
import sys

import pytest

HPCP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hpcp.py')
posix_only = pytest.mark.skipif(os.name != 'posix', reason='POSIX paths and links')


def run(args, cwd, env=None, timeout=120, stdin=subprocess.DEVNULL):
	'''The CLI in a session of its own: (return code, output as text)'''
	task = subprocess.run([sys.executable, HPCP] + args, cwd=str(cwd), stdin=stdin, stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT, env=dict(os.environ, **(env or {})), timeout=timeout, start_new_session=True)
	return task.returncode, task.stdout.decode(errors='replace')


# ---- A2: -d folded once (entry 19)

@posix_only
@pytest.mark.parametrize('dest', ['l/../out/', 'l/../out'], ids=['into', 'as'])
def test_a_destination_through_a_link_and_dotdot_lands_where_it_says(tmp_path, dest):
	# 9.86 made deep/out (the mkdir followed l before '..') while the files went to out/
	(tmp_path / 'a').mkdir()
	(tmp_path / 'a' / 'f').write_text('f')
	(tmp_path / 'deep' / 'inner').mkdir(parents=True)
	os.symlink('deep/inner', str(tmp_path / 'l'))
	rc, output = run(['a/', '-d', dest], tmp_path)
	assert rc == 0, output[-2000:]
	assert (tmp_path / 'out' / 'f').read_text() == 'f'
	assert sorted(os.listdir(str(tmp_path / 'deep'))) == ['inner'] and os.listdir(str(tmp_path / 'deep' / 'inner')) == []
