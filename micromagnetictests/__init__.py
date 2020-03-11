import pytest
import pkg_resources
import micromagnetictests.calculatortests
from .get_tests import get_tests


def test():
    return pytest.main(['-v', '--pyargs',
                        'micromagnetictests'])  # pragma: no cover


__version__ = pkg_resources.get_distribution(__name__).version
__dependencies__ = pkg_resources.require(__name__)
