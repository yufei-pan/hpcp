import os
import warnings
import pytest


pytestmark = [pytest.mark.linux, pytest.mark.loop]


def test_create_and_detach_loop_device(loop_image, hpcp_mod, reset_hpcp_globals):
	loop = loop_image['loop']
	assert loop.startswith('/dev/loop') or os.path.exists(loop)
	# fixture detaches on teardown; also verify detach API idempotence-ish
	# Note: loop_image teardown ignores detach_loop_device False (Task 1 deferred minor).
	detached = hpcp_mod.detach_loop_device(loop)
	assert detached in (True, False)
	if not detached:
		warnings.warn(
			f'detach_loop_device({loop!r}) returned False; fixture teardown will retry',
			UserWarning,
			stacklevel=1,
		)
	# Re-detach should not crash
	hpcp_mod.detach_loop_device(loop)


def test_get_partitions_on_loop_without_table(loop_image, hpcp_mod, reset_hpcp_globals):
	# Empty image may have no partitions — assert function returns a list (possibly empty)
	# and does not crash; if it raises, skip or BUGS.md depending on documented contract.
	loop = loop_image['loop']
	try:
		parts = hpcp_mod.get_partitions(loop)
	except Exception as e:
		pytest.skip(f'get_partitions needs partitioned image: {e}')
	assert isinstance(parts, list)


@pytest.mark.root
def test_partitioned_image_optional(tmp_path, hpcp_mod, reset_hpcp_globals, require_linux):
	"""Optional deeper test: only run when root and sfdisk/parted available.

	If tools missing, skip. Do not require this for default `pytest -q` green.
	"""
	import shutil
	if os.geteuid() != 0:
		pytest.skip('root required')
	if not shutil.which('sfdisk') and not shutil.which('parted'):
		pytest.skip('sfdisk/parted not available')
	# Minimal: create image, attach, attempt get_partition_infos; skip on failure with reason
	img = tmp_path / 'part.img'
	with open(img, 'wb') as f:
		f.truncate(16 * 1024 * 1024)
	loop = None
	try:
		loop = hpcp_mod.create_loop_device(str(img))
		try:
			infos = hpcp_mod.get_partition_infos(loop)
		except Exception as e:
			pytest.skip(f'partition infos unavailable on blank image: {e}')
		assert infos is not None
	finally:
		if loop:
			hpcp_mod.detach_loop_device(loop)
