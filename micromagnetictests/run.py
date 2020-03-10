import pytest


def run(calc):
    @pytest.fixture(scope='module')
    def calculator():
        return calc

    return pytest.main(['-v', 'calculatortests'])
