from types import SimpleNamespace

import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import pytest


@pytest.fixture
def threads_case(calculator):
    case = SimpleNamespace()
    case.calculator = calculator
    p1 = (0, 0, 0)
    p2 = (5e-9, 5e-9, 5e-9)
    n = (2, 2, 2)
    case.Ms = 1e6
    A = 1e-12
    H = (0, 0, 1e6)
    region = df.Region(p1=p1, p2=p2)
    case.mesh = df.Mesh(region=region, n=n)
    case.energy = mm.Exchange(A=A) + mm.Zeeman(H=H)
    case.precession = mm.Precession(gamma0=mm.consts.gamma0)
    case.damping = mm.Damping(alpha=1)
    case.m = df.Field(case.mesh, nvdim=3, value=(0, 0.1, 1), norm=case.Ms)

    return case


def test_threads_threads(threads_case):
    name = "timedriver_noevolver_nodriver"

    system = mm.System(name=name)
    system.energy = threads_case.energy
    system.dynamics = threads_case.precession + threads_case.damping
    system.m = threads_case.m

    # One thread
    td = threads_case.calculator.TimeDriver()
    td.drive(system, t=0.2e-9, n=50, n_threads=1)

    value = system.m(threads_case.mesh.region.center)
    assert np.linalg.norm(np.subtract(value, (0, 0, threads_case.Ms))) < 1

    # Two threads
    td.drive(system, t=0.2e-9, n=50, n_threads=2)

    value = system.m(threads_case.mesh.region.center)
    assert np.linalg.norm(np.subtract(value, (0, 0, threads_case.Ms))) < 1

    threads_case.calculator.delete(system)
