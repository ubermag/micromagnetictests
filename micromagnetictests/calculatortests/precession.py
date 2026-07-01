from types import SimpleNamespace

import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import pytest


@pytest.fixture
def precession_case(calculator):
    case = SimpleNamespace()
    case.calculator = calculator
    p1 = (-5e-9, -5e-9, -3e-9)
    p2 = (5e-9, 5e-9, 3e-9)
    case.region = df.Region(p1=p1, p2=p2)
    case.n = (10, 10, 10)
    case.subregions = {
        "r1": df.Region(p1=(-5e-9, -5e-9, -3e-9), p2=(5e-9, 0, 3e-9)),
        "r2": df.Region(p1=(-5e-9, 0, -3e-9), p2=(5e-9, 5e-9, 3e-9)),
    }

    return case


def test_precession_scalar(precession_case):
    self = precession_case
    name = "precession_scalar"

    H = (0, 0, 1e6)
    gamma0 = 0
    Ms = 1e6

    mesh = df.Mesh(region=self.region, n=self.n)

    system = mm.System(name=name)
    system.energy = mm.Zeeman(H=H)
    system.dynamics = mm.Precession(gamma0=gamma0)
    system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

    td = self.calculator.TimeDriver()
    td.drive(system, t=0.2e-9, n=50)

    # Gamma is zero, nothing should change.
    value = system.m(mesh.region.center)
    assert np.linalg.norm(np.cross(value, (0, 0.1 * Ms, Ms))) < 1e-3

    self.calculator.delete(system)


def test_precession_dict(precession_case):
    self = precession_case
    name = "precession_dict"

    H = (0, 0, 1e6)
    gamma0 = {"r1": 0, "r2": 2.211e5}
    Ms = 1e6

    mesh = df.Mesh(region=self.region, n=self.n, subregions=self.subregions)

    system = mm.System(name=name)
    system.energy = mm.Zeeman(H=H)
    system.dynamics = mm.Precession(gamma0=gamma0)
    system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

    td = self.calculator.TimeDriver()
    td.drive(system, t=0.2e-9, n=50)

    # gamma=0 region
    value = system.m((1e-9, -4e-9, 3e-9))
    assert np.linalg.norm(np.cross(value, (0, 0.1 * Ms, Ms))) < 1e-3

    # gamma!=0 region
    value = system.m((1e-9, 4e-9, 3e-9))
    assert np.linalg.norm(np.cross(value, (0, 0.1 * Ms, Ms))) > 1

    self.calculator.delete(system)


def test_precession_field(precession_case):
    self = precession_case
    name = "precession_field"

    mesh = df.Mesh(region=self.region, n=self.n)

    def value_fun(pos):
        x, y, z = pos
        if y <= 0:
            return 0
        else:
            return 2.211e5

    H = (0, 0, 1e6)
    gamma0 = df.Field(mesh, nvdim=1, value=value_fun)
    Ms = 1e6

    system = mm.System(name=name)
    system.energy = mm.Zeeman(H=H)
    system.dynamics = mm.Precession(gamma0=gamma0)
    system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

    td = self.calculator.TimeDriver()
    td.drive(system, t=0.2e-9, n=50)

    # gamma=0 region
    value = system.m((1e-9, -4e-9, 3e-9))
    assert np.linalg.norm(np.cross(value, (0, 0.1 * Ms, Ms))) < 1e-3

    # gamma!=0 region
    value = system.m((1e-9, 4e-9, 3e-9))
    assert np.linalg.norm(np.cross(value, (0, 0.1 * Ms, Ms))) > 1

    self.calculator.delete(system)
