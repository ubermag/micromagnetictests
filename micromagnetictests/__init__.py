"""Test computational magnetism tools."""
import pkg_resources
import pytest

import micromagnetictests.calculatortests

from .get_tests import get_tests

__version__ = pkg_resources.get_distribution(__name__).version


def test():
    """Run all package tests.

    Examples
    --------
    1. Run all tests.

    >>> import micromagnetictests as mt
    ...
    >>> # mt.test()

    """
    return pytest.main(['-v', '--pyargs',
                        'micromagnetictests', '-l'])  # pragma: no cover
