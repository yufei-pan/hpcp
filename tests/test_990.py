'''hpcp 9.90: the deferred entries of 9.88 and 9.89 (docs/limitations.md), each test failing on 9.89 (ae258b0) unless it
says it guards a case 9.89 already gets right.'''
import contextlib
import errno
import io
import os
import stat
import subprocess
import sys

import pytest

import hpcp
from test_held import _RUNS, _w1_run, fresh, read, write  # noqa: F401 (fresh is a fixture)

HPCP = hpcp.__file__
posix_only = pytest.mark.skipif(os.name != 'posix', reason='POSIX')
linux_only = pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux')
not_root = pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() == 0, reason='root may do it')



def _run_named(monkeypatch, *named):
	'''One run's settings: named (-d each) and their pin, a run of its own.'''
	monkeypatch.setattr(hpcp, 'NAMED_DESTS', frozenset(os.path.abspath(path) for path in named))
	monkeypatch.setattr(hpcp, 'RUN_ID', '990-%d' % next(_RUNS))
	monkeypatch.setattr(hpcp, 'NAMED_IDS', hpcp._named_ids(hpcp.NAMED_DESTS))


def _copy(src, dests):
	with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
		return hpcp.copy_file(str(src), [str(dest) for dest in dests])


def _files(root):
	return {os.path.relpath(os.path.join(d, name), str(root)): read(os.path.join(d, name))
		for d, _, names in os.walk(str(root)) for name in names}


# -- A rerun to several destinations: the door's answers are duplicated only where two landings can share a cursor --------

@posix_only
def test_990_two_destinations_side_by_side_lend_the_doors_answers(tmp_path, fresh, monkeypatch):
	# 9.89 duplicated (and closed) each landing's directory descriptor whenever a file had several landings; two -d side by
	# side never share a door cursor, so 9.90 lends both (no dup), as with one destination.
	write(tmp_path / 'src' / 'a' / 'f', 'F')
	for out in ('out', 'out2'):
		os.makedirs(str(tmp_path / out / 'a'))
	_run_named(monkeypatch, tmp_path / 'out', tmp_path / 'out2')
	dups, real_dup = [], os.dup
	def dup(fd):
		dups.append(fd)
		return real_dup(fd)
	monkeypatch.setattr(os, 'dup', dup)
	try:
		_copy(tmp_path / 'src' / 'a' / 'f', [tmp_path / 'out' / 'a' / 'f', tmp_path / 'out2' / 'a' / 'f'])
	finally:
		monkeypatch.setattr(os, 'dup', real_dup)
		hpcp._close_cursors()
	assert read(tmp_path / 'out' / 'a' / 'f') == 'F' and not hpcp.ERRORS, hpcp.ERRORS
	assert dups == [], dups


@posix_only
def test_990_nested_destinations_still_duplicate(tmp_path, fresh, monkeypatch):
	# Guards what 9.89 gets right: one -d inside another shares a cursor (the 9.89 two-landings test), so the answers are dups.
	write(tmp_path / 'src' / 'x' / 'a' / 'b' / 'f', 'F')
	out = str(tmp_path / 'out')
	for directory in ('x/a/b', 'x/x/a/b'):
		os.makedirs(os.path.join(out, directory))
	_run_named(monkeypatch, out, os.path.join(out, 'x'))
	assert getattr(hpcp, '_nested_named', lambda: True)()
	try:
		_copy(tmp_path / 'src' / 'x' / 'a' / 'b' / 'f', [os.path.join(out, 'x/a/b/f'), os.path.join(out, 'x/x/a/b/f')])
	finally:
		hpcp._close_cursors()
	assert _files(out) == {'x/a/b/f': 'F'} and not hpcp.ERRORS, hpcp.ERRORS


@posix_only
@pytest.mark.parametrize('rerun', [False, True])
def test_990_a_named_link_landing_below_another_destination_keeps_its_directory(tmp_path, fresh, monkeypatch, rerun):
	# The re-review's case: -d out/l -d other/, l -> ../other/a/b/real. Neither -d lies below the other, but the link's landing
	# (named, followed at the copy) lies below other/, whose cursor the second landing walks: lent, the answer for out/l's
	# directory was closed by the second door call, and the file went to other/f (exit 0, no message). Named: dup'd.
	write(tmp_path / 'src' / 'f', 'NEW')
	os.makedirs(str(tmp_path / 'other' / 'x' / 'y'))
	write(tmp_path / 'other' / 'a' / 'b' / 'real', 'NEW' if rerun else 'old')
	if rerun:
		st = os.stat(str(tmp_path / 'src' / 'f'))
		os.utime(str(tmp_path / 'other' / 'a' / 'b' / 'real'), ns=(st.st_atime_ns, st.st_mtime_ns))
	os.makedirs(str(tmp_path / 'out'))
	os.symlink('../other/a/b/real', str(tmp_path / 'out' / 'l'))
	link, other = str(tmp_path / 'out' / 'l'), str(tmp_path / 'other')
	_run_named(monkeypatch, link, other)
	assert not getattr(hpcp, '_nested_named', lambda: False)()
	try:
		hpcp._dest_directory(os.path.join(other, 'x', 'y'), make=False)  # the cursor elsewhere below other/ first
		_copy(tmp_path / 'src' / 'f', [link, os.path.join(other, 'f')])
	finally:
		hpcp._close_cursors()
	assert _files(other) == {'a/b/real': 'NEW'} and not hpcp.ERRORS, (_files(other), hpcp.ERRORS)


# -- A -d link retargeted between a failed first attempt and its retry -----------------------------------------------------

@posix_only
def test_990_a_retry_goes_where_the_first_attempt_went(tmp_path, fresh, monkeypatch):
	# Probe C of the 9.89 final review's leads.py: -d out/l, l -> ../other/real; the first attempt fails and l is pointed at
	# ../third/name2 before the retry. 9.89 worked the landing out again for the retry and wrote other/name2 (the new name in
	# the old directory). The landing is worked out once a file: the retry writes other/real.
	write(tmp_path / 'src' / 'f', 'NEW')
	for directory, name in (('other', 'real'), ('other', 'name2'), ('third', 'name2')):
		write(tmp_path / directory / name, 'old %s/%s' % (directory, name))
	os.makedirs(str(tmp_path / 'out'))
	link = str(tmp_path / 'out' / 'l')
	os.symlink('../other/real', link)
	_run_named(monkeypatch, link)
	real, calls = hpcp._cp_once, []
	def once(*args, **kwargs):
		calls.append(1)
		if len(calls) == 1:
			os.unlink(link)
			os.symlink('../third/name2', link)
			raise OSError(errno.ENOSPC, 'No space left on device', args[1])
		return real(*args, **kwargs)
	monkeypatch.setattr(hpcp, '_cp_once', once)
	try:
		_copy(tmp_path / 'src' / 'f', [link])
	finally:
		hpcp._close_cursors()
	assert len(calls) == 2
	assert read(tmp_path / 'other' / 'real') == 'NEW', hpcp.ERRORS
	assert read(tmp_path / 'other' / 'name2') == 'old other/name2' and read(tmp_path / 'third' / 'name2') == 'old third/name2'


# -- A device node copied by a non-root user onto a -d link ------------------------------------------------------------------

@posix_only
@not_root
def test_990_a_device_node_onto_a_named_link_says_it_cannot_make_the_node(tmp_path):
	# 9.88 and 9.89 looked the landing up by the link's own name in the target's directory, found nothing, did not ask, and
	# left cp to fail on the copy beside ("the new copy beside it failed"); 9.87 said it could not make the node.
	if not os.path.exists('/dev/null') or not stat.S_ISCHR(os.stat('/dev/null').st_mode):
		pytest.skip('no /dev/null device here')
	write(tmp_path / 'other' / 'real', 'keep')
	os.makedirs(str(tmp_path / 'out'))
	os.symlink('../other/real', str(tmp_path / 'out' / 'l'))
	p = subprocess.run([sys.executable, HPCP, '/dev/null', '-d', str(tmp_path / 'out' / 'l')], stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT, universal_newlines=True, cwd=str(tmp_path))
	assert p.returncode == 190, p.stdout
	assert 'cannot make a device node like /dev/null here' in p.stdout, p.stdout
	assert 'beside' not in p.stdout, p.stdout
	assert read(tmp_path / 'other' / 'real') == 'keep' and os.readlink(str(tmp_path / 'out' / 'l')) == '../other/real'
	assert sorted(os.listdir(str(tmp_path / 'other'))) == ['real']


# -- -lt: a link whose target directory holds it, and whose copy fails ---------------------------------------------------------

@posix_only
def test_990_link_tracking_tries_a_failing_target_once(tmp_path):
	# src/k/a/a -> ../a, and out/k/a a file, so the tracked target's copy fails every time: 9.89 found the link nested in the
	# target again after each attempt and tracked it again, until Python's recursion limit (992 'Copy failed' lines here,
	# about 20,000 lines in all). The target is tried once.
	os.makedirs(str(tmp_path / 'src' / 'k' / 'a'))
	os.symlink('../a', str(tmp_path / 'src' / 'k' / 'a' / 'a'))
	write(tmp_path / 'src' / 'k' / 'a' / 'g', 'x')
	write(tmp_path / 'out' / 'k' / 'a', 'file')
	p = subprocess.run([sys.executable, HPCP, 'src/k', '-d', 'out/', '-lt'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
		universal_newlines=True, cwd=str(tmp_path))
	assert p.returncode == 213, p.stdout[-2000:]
	failed = [line for line in p.stdout.splitlines() if line.startswith('Copy failed')]
	assert 1 <= len(failed) <= 3, (len(failed), p.stdout[-2000:])
	assert read(tmp_path / 'out' / 'k' / 'a') == 'file'


# -- A serial run's progress bar ------------------------------------------------------------------------------------------------

@posix_only
def test_990_a_serial_copy_draws_a_frame_at_most_each_refresh_and_the_last_at_the_total(tmp_path, fresh, monkeypatch):
	# 9.89 drew a frame after every file of a serial run (rate_limit), a third of a serial rerun's instructions a file; one
	# every 0.1 s is what a terminal shows anyway. The frame at the total is always drawn: it ends the bar's line.
	count = 300
	for index in range(count):
		write(tmp_path / 'src' / ('f%03d' % index), 'x')
	frames, real = [], hpcp.multiCMD.print_progress_bar
	def frame(*args, **kwargs):
		frames.append(args[:2])
	monkeypatch.setattr(hpcp.multiCMD, 'print_progress_bar', frame)
	with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
		hpcp.copy_files_serial(str(tmp_path / 'src'), [str(tmp_path / 'out')])
	assert len(os.listdir(str(tmp_path / 'out'))) == count
	assert frames and frames[-1] == (count, count), frames[-3:]
	assert len(frames) < count // 4, len(frames)
