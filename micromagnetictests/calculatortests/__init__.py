"""Calculator tests."""

from importlib import import_module

_MODULES = (
    "compute",
    "cubicanisotropy",
    "damping",
    "demag",
    "dirname",
    "dmi",
    "dynamics",
    "energy",
    "exchange",
    "fixedsubregions",
    "hysteresisdriver",
    "info_file",
    "mesh",
    "mindriver",
    "multiple_drives",
    "outputformat",
    "outputstep",
    "precession",
    "relaxdriver",
    "rkky",
    "schedule",
    "skyrmion",
    "slonczewski",
    "stdprob3",
    "stdprob4",
    "stdprob5",
    "threads",
    "timedriver",
    "uniaxialanisotropy",
    "zeeman",
    "zhangli",
)


def _is_test_or_fixture(name, obj):
    return name.startswith("test_") or hasattr(obj, "_fixture_function_marker")


for _module_name in _MODULES:
    _module = import_module(f"{__name__}.{_module_name}")
    globals().update(
        {
            name: obj
            for name, obj in vars(_module).items()
            if _is_test_or_fixture(name, obj)
        }
    )
    globals().pop(_module_name, None)

del import_module, _is_test_or_fixture, _module, _module_name
