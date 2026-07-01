from types import SimpleNamespace

import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import pytest


@pytest.fixture
def fixed_subregions_case(calculator):
    case = SimpleNamespace()
    case.calculator = calculator
    p1 = (-10e-9, -5e-9, -3e-9)
    p2 = (10e-9, 5e-9, 3e-9)
    case.region = df.Region(p1=p1, p2=p2)
    case.cell = (1e-9, 1e-9, 1e-9)
    case.subregions = {
        "r1": df.Region(p1=(-10e-9, -5e-9, -3e-9), p2=(10e-9, 0, 3e-9)),
        "r2": df.Region(p1=(-10e-9, 0, -3e-9), p2=(10e-9, 5e-9, 3e-9)),
    }

    return case


def test_fixed_subregions_fixed_subregions(fixed_subregions_case):
    self = fixed_subregions_case
    name = "fixed_subregions"

    H = (0, 0, 1e5)
    Ms = 1e6

    system = mm.System(name=name)
    system.energy = mm.Zeeman(H=H)

    mesh = df.Mesh(region=self.region, cell=self.cell, subregions=self.subregions)
    system.m = df.Field(mesh, nvdim=3, value=(1, 0, 0), norm=Ms)

    md = self.calculator.MinDriver()
    md.drive(system, fixed_subregions=["r1"])

    assert np.linalg.norm(np.subtract(system.m["r1"].mean(), (Ms, 0, 0))) < 1

    assert np.linalg.norm(np.subtract(system.m["r2"].mean(), (0, 0, Ms))) < 1

    self.calculator.delete(system)
