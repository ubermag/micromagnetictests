import micromagnetictests as mt


def test_version():
    assert isinstance(mt.__version__, str)
    assert "." in mt.__version__
