import pytest


def run(calculator):
    @pytest.fixture(scope='module')
    def calculator():
        return oc

    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for f in files:
        if f != '__init__' and f != 'run.py':
        return pytest.main(['-v', '--pyargs',
                            'micromagnetictests'])  # pragma: no cover
