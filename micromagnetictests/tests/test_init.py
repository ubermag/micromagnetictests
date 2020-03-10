import micromagnetictests as mt


def test_version():
    assert isinstance(mt.__version__, str)
    assert '.' in mt.__version__


def test_dependencies():
    assert isinstance(mt.__dependencies__, list)
    assert len(mt.__dependencies__) > 0
