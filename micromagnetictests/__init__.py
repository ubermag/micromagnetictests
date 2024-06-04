"""Test computational magnetism tools."""

import importlib.metadata

import pytest

from .get_tests import get_tests as get_tests
from micromagnetictests import calculatortests as calculatortests

__version__ = importlib.metadata.version(__package__)


def test():
    """Run all package tests.

    Examples
    --------
    1. Run all tests.

    >>> import micromagnetictests as mt
    ...
    >>> # mt.test()

    """
    return pytest.main(
        ["-v", "--pyargs", "micromagnetictests", "-l"]
    )  # pragma: no cover
