import os
import inspect
import pytest


def test_hpcp_gui_is_callable(hpcp_mod):
	assert callable(hpcp_mod.hpcp_gui)


def test_main_mentions_windows_gui_branch(hpcp_mod):
	src = inspect.getsource(hpcp_mod.main)
	assert 'hpcp_gui' in src
	assert "os.name == 'nt'" in src


@pytest.mark.windows
@pytest.mark.gui
def test_gui_launch_skipped_or_runs_on_windows(hpcp_mod, require_windows):
	# Do not start mainloop in CI: only verify Tk import works on Windows
	import tkinter
	root = tkinter.Tk()
	root.withdraw()
	root.destroy()
