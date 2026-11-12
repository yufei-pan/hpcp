"""The helpers behind conftest's _a_test_closes_what_it_opens: they see what a test leaves open, and nothing else."""
import gc
import os
import warnings

import pytest

from conftest import (_can_list_descriptors, close_descriptors, descriptors_left_since, descriptors_opened_since, open_descriptors,
	warn_if_descriptors_cannot_be_listed)

can_look = pytest.mark.skipif(not _can_list_descriptors(), reason='no way to list the descriptors here (/proc/self/fd, /dev/fd)')


@can_look
def test_a_file_and_a_directory_left_open_are_seen(tmp_path):
	before = open_descriptors()
	(tmp_path / 'f').write_text('x')
	opened = [os.open(str(tmp_path / 'f'), os.O_RDONLY), os.open(str(tmp_path), os.O_RDONLY)]
	try:
		found = descriptors_opened_since(before)
		assert found == {opened[0]: os.path.realpath(str(tmp_path / 'f')), opened[1]: os.path.realpath(str(tmp_path))}
	finally:
		for fd in opened:
			os.close(fd)
	assert descriptors_opened_since(before) == {}


@can_look
def test_a_pipe_has_no_path_and_is_not_one_of_them():
	before = open_descriptors()
	read, write = os.pipe()
	try:
		assert descriptors_opened_since(before) == {}
	finally:
		os.close(read)
		os.close(write)


@can_look
def test_multiprocessing_s_arena_is_not_one_of_them(tmp_path):
	# the allocator makes it when a process first shares memory with a pool, and keeps it for the process's life
	before = open_descriptors()
	fd = os.open(str(tmp_path / 'pym-1-abc'), os.O_RDWR | os.O_CREAT, 0o600)
	try:
		os.unlink(str(tmp_path / 'pym-1-abc'))
		assert descriptors_opened_since(before) == {}
	finally:
		os.close(fd)


@can_look
def test_close_descriptors_closes_what_was_found_and_not_a_number_something_else_took(tmp_path):
	(tmp_path / 'a').write_text('a')
	(tmp_path / 'b').write_text('b')
	before = open_descriptors()
	fd = os.open(str(tmp_path / 'a'), os.O_RDONLY)
	found = descriptors_opened_since(before)
	assert list(found) == [fd]
	close_descriptors(found)
	with pytest.raises(OSError):
		os.fstat(fd)
	# found again, now closed by someone else, and the number taken by another file: that one stays
	found = {fd: found[fd]}
	taken = os.open(str(tmp_path / 'b'), os.O_RDONLY)
	try:
		assert taken == fd
		close_descriptors(found)
		assert os.fstat(taken).st_size == 1
	finally:
		os.close(taken)


@can_look
def test_a_file_a_reference_cycle_holds_is_no_leak_once_it_is_collected(tmp_path):
	# a failed test's traceback or a cycle can keep a file object alive until the next collection: closed by it, not a leak
	(tmp_path / 'a').write_text('a')
	before = open_descriptors()
	gc.disable()  # the collection is the one descriptors_left_since makes
	try:
		cycle = []
		cycle.append(cycle)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore', ResourceWarning)
			cycle.append(open(str(tmp_path / 'a')))
			del cycle
			assert len(descriptors_opened_since(before)) == 1
			assert descriptors_left_since(before) == {}
	finally:
		gc.enable()
	assert descriptors_opened_since(before) == {}


@can_look
def test_a_file_something_still_holds_is_a_leak_and_is_not_closed_for_it(tmp_path):
	# the guard reports, and does not close: the number would be the next test's, and the holder would close it when it let go
	(tmp_path / 'a').write_text('a')
	before = open_descriptors()
	held = open(str(tmp_path / 'a'))
	try:
		assert descriptors_left_since(before) == {held.fileno(): os.path.realpath(str(tmp_path / 'a'))}
		assert not held.closed and os.fstat(held.fileno()).st_size == 1
	finally:
		held.close()


def test_a_run_that_cannot_list_descriptors_says_so(monkeypatch):
	monkeypatch.setattr(os.path, 'isdir', lambda path: False)
	assert not _can_list_descriptors() and open_descriptors() == {}
	with pytest.warns(RuntimeWarning, match='cannot list'):
		warn_if_descriptors_cannot_be_listed()


@can_look
def test_a_run_that_can_list_descriptors_says_nothing():
	with warnings.catch_warnings(record=True) as caught:
		warnings.simplefilter('always')
		warn_if_descriptors_cannot_be_listed()
	assert caught == []
