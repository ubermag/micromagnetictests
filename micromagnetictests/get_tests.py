import inspect

import micromagnetictests as mt


def get_tests():
    """Generator yielding all available test names.

    Returns
    -------
    Generator

        Tests that can be imported from ``micromagnetictests``. The list
        consists of tuples, where the first element is the name of the tests,
        whereas the second element is the test function.

    Examples
    --------
    1. Getting the list of available tests.

    >>> import micromagnetictests as mt
    ...
    >>> list(mt.get_tests())
    [...]

    """
    for name, object in inspect.getmembers(mt.calculatortests):
        if inspect.isfunction(object) and name.startswith("test_"):
            yield (name, object)
