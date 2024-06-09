import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import pytest


@pytest.fixture
def Ms():
    return 1e6


@pytest.fixture
def system(Ms):
    system = mm.System(name="hysteresisdriver_noevolver_nodriver")

    A = 1e-12
    H = (0, 0, 1e6)
    system.energy = mm.Exchange(A=A) + mm.Zeeman(H=H)

    p1 = (0, 0, 0)
    p2 = (5e-9, 5e-9, 5e-9)
    n = (5, 5, 5)
    region = df.Region(p1=p1, p2=p2)
    mesh = df.Mesh(region=region, n=n)
    system.m = df.Field(mesh, nvdim=3, value=(0, 1, 0), norm=Ms)
    return system


def test_simple_hysteresis_loop(calculator, system, Ms):
    """Simple hysteresis loop between Hmin and Hmax with symmetric number of steps."""
    hd = calculator.HysteresisDriver()
    hd.drive(system, Hmin=(0, 0, -1e6), Hmax=(0, 0, 1e6), n=3)

    value = system.m(system.m.mesh.region.center)
    assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) < 1e-3

    assert len(system.table.data.index) == 5

    assert system.table.x == "B_hysteresis"

    calculator.delete(system)


def test_stepped_hysteresis_loop(calculator, system, Ms):
    """Simple hysteresis loop with uneven steps using `Hsteps` as keyword argument."""
    hd = calculator.HysteresisDriver()
    hd.drive(
        system,
        Hsteps=[
            [(0, 0, -1e6), (0, 0, 1e6), 3],
            [(0, 0, 1e6), (0, 0, -1e6), 5],
        ],
    )

    value = system.m(system.m.mesh.region.center)
    assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) < 1e-3

    assert len(system.table.data.index) == 7

    assert system.table.x == "B_hysteresis"

    calculator.delete(system)


def test_hysteresis_check_for_energy(calculator):
    system = mm.examples.macrospin()
    system.energy = 0
    hd = calculator.HysteresisDriver()

    with pytest.raises(RuntimeError, match="System's energy is not defined"):
        hd.drive(system, Hmin=(0, 0, -1e6), Hmax=(0, 0, 1e6), n=3)
