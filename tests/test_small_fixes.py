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


# ---- A3: -siff reads its lists as bytes (entry 9)

_PATCH = '''
import os, sys
sys.path.insert(0, %(root)r)
import hpcp
if __name__ == '__main__':  # a pool's forkserver or spawn worker imports this file
	hpcp.HASH_SIZE = 65536
	step, new, base, vol, target = sys.argv[1:6]
	if step == 'make':
		rows = hpcp.compute_changes(*hpcp.build_root_maps(new, base, max_workers=1), max_workers=1)
		hpcp.build_patch_tree(rows, new, [vol], hpcp.make_patch_meta(rows, 'n', 'b', 'root'), max_workers=1)
	else:
		print('failed', hpcp.apply_patch_tree([vol], target, max_workers=1)['failed'])
'''

# Python's C locale with neither its UTF-8 mode nor locale coercion (3.7+): names are ASCII, as under Python 3.6 with LANG unset
_C = {'LC_ALL': 'C', 'LANG': 'C', 'PYTHONUTF8': '0', 'PYTHONCOERCECLOCALE': '0'}


@posix_only
@pytest.mark.parametrize('made', ['C', 'C.UTF-8'])
def test_a_patch_with_a_non_ascii_name_applies_under_the_c_locale(tmp_path, made):
	# 9.86 decoded the lists as UTF-8, and under the C locale the name could not be encoded back: UnicodeEncodeError
	name = b'new\xc3\xa9'
	new, base, vol, target = (tmp_path / part for part in ('new', 'base', 'vol', 'target'))
	for directory in (new, base, vol, target):
		directory.mkdir()
	with open(os.path.join(os.fsencode(str(new)), name), 'wb') as f:
		f.write(b'content')
	script = tmp_path / 'patch.py'
	script.write_text(_PATCH % {'root': os.path.dirname(HPCP)})
	def step(what, env):
		done = subprocess.run([sys.executable, str(script), what, str(new), str(base), str(vol), str(target)],
			stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=dict(os.environ, **env), timeout=120)
		assert done.returncode == 0, done.stdout.decode(errors='replace')[-3000:]
		return done.stdout.decode(errors='replace')
	step('make', _C if made == 'C' else dict(_C, LC_ALL='C.UTF-8', LANG='C.UTF-8'))
	assert 'failed 0' in step('apply', _C)
	with open(os.path.join(os.fsencode(str(target)), name), 'rb') as f:
		assert f.read() == b'content'
