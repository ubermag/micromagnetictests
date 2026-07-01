from types import SimpleNamespace

import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import pytest


@pytest.fixture
def rkky_case(calculator):
    case = SimpleNamespace()
    case.calculator = calculator
    p1 = (-5e-9, -5e-9, -3e-9)
    p2 = (5e-9, 5e-9, 3e-9)
    case.region = df.Region(p1=p1, p2=p2)
    case.n = (10, 10, 6)
    case.subregions = {
        "r1": df.Region(p1=(-5e-9, -5e-9, -3e-9), p2=(5e-9, 0, 3e-9)),
        "r2": df.Region(p1=(-5e-9, 0, -3e-9), p2=(5e-9, 1e-9, 3e-9)),
        "r3": df.Region(p1=(-5e-9, 1e-9, -3e-9), p2=(5e-9, 5e-9, 3e-9)),
    }

    return case


def m_init(pos):
    x, y, z = pos
    if y <= 0:
        return (0, 0.2, 1)
    else:
        return (0, -0.5, -1)


def test_rkky_scalar(rkky_case):
    self = rkky_case
    name = "rkky_scalar"

    sigma = -1e4
    sigma2 = 0
    Ms = 1e6

    system = mm.System(name=name)
    system.energy = mm.RKKY(sigma=sigma, sigma2=sigma2, subregions=["r1", "r3"])

    mesh = df.Mesh(region=self.region, n=self.n, subregions=self.subregions)
    system.m = df.Field(mesh, nvdim=3, value=m_init, norm=Ms)

    md = self.calculator.MinDriver()
    md.drive(system)

    # AFM
    m1 = system.m.orientation((0, -0.5e-9, 0))
    m2 = system.m.orientation((0, 1.5e-9, 0))
    assert abs(np.dot(m1, m2) - (-1)) < 1e-3

    system.energy.rkky.sigma = 1e4
    system.energy.rkky.sigma2 = 0

    system.m = df.Field(mesh, nvdim=3, value=m_init, norm=Ms)

    md.drive(system)

    # FM
    m1 = system.m.orientation((0, -0.5e-9, 0))
    m2 = system.m.orientation((0, 1.5e-9, 0))
    assert abs(np.dot(m1, m2) - 1) < 1e-3

    self.calculator.delete(system)
