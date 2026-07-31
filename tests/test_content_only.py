import os
import sys

import pytest

# Import sibling module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hpcp


def test_content_only_flag_parses():
	old = sys.argv
	try:
		sys.argv = ['hpcp', '-co', '-d', '/tmp', '/tmp/src']
		args = hpcp.get_args()
		assert args.content_only is True
	finally:
		sys.argv = old


def test_content_only_default_off():
	old = sys.argv
	try:
		sys.argv = ['hpcp', '-d', '/tmp', '/tmp/src']
		args = hpcp.get_args()
		assert args.content_only is False
	finally:
		sys.argv = old
