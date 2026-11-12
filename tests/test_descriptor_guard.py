"""The helpers behind conftest's _a_test_closes_what_it_opens: they see what a test leaves open, and nothing else."""
import os

import pytest

from conftest import close_descriptors, descriptors_opened_since, open_descriptors

pytestmark = pytest.mark.skipif(os.name == 'nt', reason='no descriptor listing (/proc/self/fd, /dev/fd) here')


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


def test_a_pipe_has_no_path_and_is_not_one_of_them():
	before = open_descriptors()
	read, write = os.pipe()
	try:
		assert descriptors_opened_since(before) == {}
	finally:
		os.close(read)
		os.close(write)


def test_multiprocessing_s_arena_is_not_one_of_them(tmp_path):
	# the allocator makes it when a process first shares memory with a pool, and keeps it for the process's life
	before = open_descriptors()
	fd = os.open(str(tmp_path / 'pym-1-abc'), os.O_RDWR | os.O_CREAT, 0o600)
	try:
		os.unlink(str(tmp_path / 'pym-1-abc'))
		assert descriptors_opened_since(before) == {}
	finally:
		os.close(fd)


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
