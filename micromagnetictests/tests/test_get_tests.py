import importlib
import inspect

import micromagnetictests as mt


def _module_test_functions():
    test_functions = {}
    for module_name in mt.calculatortests._MODULES:
        module = importlib.import_module(
            f"{mt.calculatortests.__name__}.{module_name}"
        )
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("test_"):
                assert name not in test_functions
                test_functions[name] = obj
    return test_functions


def test_get_tests():
    tests = list(mt.get_tests())
    test_names = [name for name, _ in tests]

    assert test_names
    assert len(test_names) == len(set(test_names))
    assert all(name.startswith("test_") for name in test_names)
    assert all(inspect.isfunction(obj) for _, obj in tests)
    assert dict(tests) == _module_test_functions()
