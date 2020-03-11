import inspect
import micromagnetictests as mt


def test_micromagnetictests():
    for name, object in inspect.getmembers(mt):
        assert isinstance(name, str)
        if inspect.isclass(object):
            starting_strings = ['__', 'test_', 'Test']
            assert any([name.startswith(s) for s in starting_strings])
