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


def _fake_tkinter(entry_values):
	"""A stand-in tkinter: Entry widgets return entry_values in creation order; mainloop presses 'Start Copy'."""
	import types
	values = iter(entry_values)
	buttons = {}

	class Widget:
		def __init__(self, *args, **kwargs):
			self.kwargs = kwargs
		def grid(self, *args, **kwargs):
			return None

	class Entry(Widget):
		def __init__(self, *args, **kwargs):
			super().__init__(*args, **kwargs)
			self.value = '' if 'textvariable' in kwargs else next(values, '')
		def get(self):
			return self.value
		def insert(self, index, text):
			self.value = text + self.value

	class Button(Widget):
		def __init__(self, *args, **kwargs):
			super().__init__(*args, **kwargs)
			buttons[kwargs.get('text')] = kwargs.get('command')

	class Var:
		def __init__(self, value=None):
			self.value = value
		def get(self):
			return self.value
		def set(self, value):
			self.value = value

	class Tk:
		def title(self, text):
			return None
		def withdraw(self):
			return None
		def deiconify(self):
			return None
		def mainloop(self):
			buttons['Start Copy']()

	tk = types.ModuleType('tkinter')
	tk.Tk, tk.Entry, tk.Label, tk.Button, tk.Checkbutton = Tk, Entry, Widget, Button, Widget
	tk.IntVar = lambda value=0: Var(value)
	tk.BooleanVar = lambda value=False: Var(value)
	tk.StringVar = lambda value='': Var(value)
	tk.filedialog = types.SimpleNamespace(askdirectory=lambda: '', askopenfilename=lambda **kwargs: '')
	tk.messagebox = types.SimpleNamespace(showerror=lambda *args: (_ for _ in ()).throw(AssertionError(args)))
	return tk


@pytest.mark.parametrize('dest, expected', [('D:/backup', ['D:/backup']), ('', [])])
def test_gui_passes_the_destination_as_one_path(hpcp_mod, monkeypatch, dest, expected):
	"""Review C7: the destination field is one path, not a list of its characters."""
	import sys
	tk = _fake_tkinter(['C:/data', dest])
	monkeypatch.setitem(sys.modules, 'tkinter', tk)
	monkeypatch.setitem(sys.modules, 'tkinter.filedialog', tk.filedialog)
	monkeypatch.setitem(sys.modules, 'tkinter.messagebox', tk.messagebox)
	calls = []
	monkeypatch.setattr(hpcp_mod, 'hpcp', lambda *args, **kwargs: calls.append((args, kwargs)) or 0)
	hpcp_mod.hpcp_gui()
	assert calls[0][0] == (['C:/data'], expected)
