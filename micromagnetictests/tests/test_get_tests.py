import micromagnetictests as mt


def test_get_tests():
    tests = list(mt.get_tests())
    assert len(tests) == 29
