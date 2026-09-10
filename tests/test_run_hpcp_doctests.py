import doctest
import hpcp


def test_hpcp_module_doctests():
	failures, _ = doctest.testmod(hpcp, optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
	assert failures == 0
