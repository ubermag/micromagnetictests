from types import SimpleNamespace

import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import pytest


@pytest.fixture
def exchange_case(calculator):
    case = SimpleNamespace()
    case.calculator = calculator
    p1 = (-5e-9, -5e-9, -3e-9)
    p2 = (5e-9, 5e-9, 3e-9)
    case.region = df.Region(p1=p1, p2=p2)
    case.n = (10, 10, 6)
    case.subregions = {
        "r1": df.Region(p1=(-5e-9, -5e-9, -3e-9), p2=(5e-9, 0, 3e-9)),
        "r2": df.Region(p1=(-5e-9, 0, -3e-9), p2=(5e-9, 5e-9, 3e-9)),
    }

    return case


def m_init(pos):
    x, y, z = pos
    if y <= 0:
        return (0, 0.2, 1)
    else:
        return (0, -0.7, -0.4)


def test_exchange_scalar(exchange_case):
    self = exchange_case
    name = "exchange_scalar"

    A = 1e-12
    Ms = 1e6

    system = mm.System(name=name)
    system.energy = mm.Exchange(A=A)

    mesh = df.Mesh(region=self.region, n=self.n)
    system.m = df.Field(mesh, nvdim=3, value=m_init, norm=Ms)

    md = self.calculator.MinDriver()
    md.drive(system)

    assert abs(np.linalg.norm(system.m.mean()) - Ms) < 1

    self.calculator.delete(system)


def test_exchange_dict(exchange_case):
    self = exchange_case
    name = "exchange_dict"

    A = {"r1": 3e-12, "r2": 1e-12, "r1:r2": -1e-12}
    Ms = 1e6

    system = mm.System(name=name)
    system.energy = mm.Exchange(A=A)

    mesh = df.Mesh(region=self.region, n=self.n, subregions=self.subregions)
    system.m = df.Field(mesh, nvdim=3, value=m_init, norm=Ms)

    md = self.calculator.MinDriver()
    md.drive(system)

    # r1
    assert abs(np.linalg.norm(system.m["r1"].mean()) - Ms) < 1
    # r2
    assert abs(np.linalg.norm(system.m["r2"].mean()) - Ms) < 1

    assert (
        abs(
            np.dot(
                system.m["r1"].orientation.mean(),
                system.m["r2"].orientation.mean(),
            )
            - (-1)
        )
        < 1e-3
    )

    self.calculator.delete(system)


def test_exchange_field(exchange_case):
    self = exchange_case
    name = "exchange_field"

    def A_fun(pos):
        x, y, z = pos
        if x <= 0:
            return 1e-10  # for 0, OOMMF gives nan
        else:
            return 1e-12

    mesh = df.Mesh(region=self.region, n=self.n)
    A = df.Field(mesh, nvdim=1, value=A_fun)
    Ms = 1e6

    system = mm.System(name=name)
    system.energy = mm.Exchange(A=A)

    system.m = df.Field(mesh, nvdim=3, value=m_init, norm=Ms)

    md = self.calculator.MinDriver()
    md.drive(system)

    assert abs(np.linalg.norm(system.m.mean()) - Ms) < 1

    self.calculator.delete(system)
