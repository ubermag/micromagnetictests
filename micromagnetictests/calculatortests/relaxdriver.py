import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import pytest


def test_relaxdriver(calculator):
    p1 = (0, 0, 0)
    p2 = (5e-9, 5e-9, 5e-9)
    n = (5, 5, 5)
    Ms = 1e6
    A = 1e-12
    H = (0, 0, 1e6)
    region = df.Region(p1=p1, p2=p2)
    mesh = df.Mesh(region=region, n=n)

    system = mm.System(name="relaxdriver")
    system.energy = mm.Exchange(A=A) + mm.Zeeman(H=H)
    system.m = df.Field(mesh, nvdim=3, value=(0, 1, 0), norm=Ms)

    md = calculator.RelaxDriver()
    md.drive(system)

    value = system.m(mesh.region.center)
    assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) < 1e-2

    assert system.table.x == md._x

    calculator.delete(system)


def test_relax_check_for_energy(calculator):
    system = mm.examples.macrospin()
    system.energy = 0
    md = calculator.RelaxDriver()

    with pytest.raises(RuntimeError, match="System's energy is not defined"):
        md.drive(system)
