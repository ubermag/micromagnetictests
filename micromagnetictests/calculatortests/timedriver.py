import glob
import os
from types import SimpleNamespace

import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import pytest


@pytest.fixture
def time_driver_case(calculator):
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


def test_time_driver_noevolver_nodriver(time_driver_case):
    name = "timedriver_noevolver_nodriver"

    system = mm.System(name=name)
    system.energy = time_driver_case.energy
    system.dynamics = time_driver_case.precession + time_driver_case.damping
    system.m = time_driver_case.m

    td = time_driver_case.calculator.TimeDriver()
    td.drive(system, t=0.2e-9, n=50)

    value = system.m(time_driver_case.mesh.region.center)
    assert np.linalg.norm(np.subtract(value, (0, 0, time_driver_case.Ms))) < 10

    assert system.table.x == "t"

    time_driver_case.calculator.delete(system)


def test_time_driver_rungekutta_evolver_nodriver(time_driver_case):
    name = "timedriver_rungekutta_evolver_nodriver"

    system = mm.System(name=name)
    system.energy = time_driver_case.energy
    system.dynamics = time_driver_case.precession + time_driver_case.damping
    system.m = time_driver_case.m

    evolver = time_driver_case.calculator.RungeKuttaEvolver(method="rkf54s")
    td = time_driver_case.calculator.TimeDriver(evolver=evolver)
    td.drive(system, t=0.2e-9, n=50)

    value = system.m(time_driver_case.mesh.region.center)
    assert np.linalg.norm(np.subtract(value, (0, 0, time_driver_case.Ms))) < 1

    time_driver_case.calculator.delete(system)


def test_time_driver_euler_evolver_nodriver(time_driver_case):
    name = "timedriver_euler_evolver_nodriver"

    system = mm.System(name=name)
    system.energy = time_driver_case.energy
    system.dynamics = time_driver_case.precession + time_driver_case.damping
    system.m = time_driver_case.m

    evolver = time_driver_case.calculator.EulerEvolver(start_dm=0.02)
    td = time_driver_case.calculator.TimeDriver(evolver=evolver)
    td.drive(system, t=0.2e-9, n=50)

    value = system.m(time_driver_case.mesh.region.center)
    assert np.linalg.norm(np.subtract(value, (0, 0, time_driver_case.Ms))) < 1

    time_driver_case.calculator.delete(system)


def test_time_driver_theta_evolver_nodriver(time_driver_case):
    name = "timedriver_theta_evolver_nodriver"

    system = mm.System(name=name)
    system.energy = time_driver_case.energy
    system.dynamics = time_driver_case.precession + time_driver_case.damping
    system.m = time_driver_case.m
    system.T = 10

    evolver = time_driver_case.calculator.UHH_ThetaEvolver(fixed_timestep=2e-13)
    td = time_driver_case.calculator.TimeDriver(evolver=evolver)
    td.drive(system, t=0.2e-9, n=50)

    # Check if it runs.

    time_driver_case.calculator.delete(system)


def test_time_driver_therm_heun_evolver_nodriver(time_driver_case):
    name = "timedriver_therm_heun_evolver_nodriver"

    system = mm.System(name=name)
    system.energy = time_driver_case.energy
    system.dynamics = time_driver_case.precession + time_driver_case.damping
    system.m = time_driver_case.m
    system.T = 10

    evolver = time_driver_case.calculator.Xf_ThermHeunEvolver()
    td = time_driver_case.calculator.TimeDriver(evolver=evolver)
    td.drive(system, t=1e-11, n=1)

    # Check if it runs.

    time_driver_case.calculator.delete(system)


def test_time_driver_noevolver_nodriver_finite_temperature(time_driver_case):
    name = "timedriver_therm_heun_evolver_nodriver"

    system = mm.System(name=name)
    system.energy = time_driver_case.energy
    system.dynamics = time_driver_case.precession + time_driver_case.damping
    system.m = time_driver_case.m
    system.T = 10

    td = time_driver_case.calculator.TimeDriver()
    with pytest.raises(RuntimeError):
        td.drive(system, t=0.2e-9, n=50)


def test_time_driver_noevolver_driver(time_driver_case):
    name = "timedriver_noevolver_driver"

    system = mm.System(name=name)
    system.energy = time_driver_case.energy
    system.dynamics = time_driver_case.precession + time_driver_case.damping
    system.m = time_driver_case.m

    td = time_driver_case.calculator.TimeDriver(stopping_dm_dt=0.01)
    td.drive(system, t=0.3e-9, n=50)

    value = system.m(time_driver_case.mesh.region.center)
    assert np.linalg.norm(np.subtract(value, (0, 0, time_driver_case.Ms))) < 1

    time_driver_case.calculator.delete(system)


def test_time_driver_noprecession(time_driver_case):
    name = "timedriver_noprecession"

    system = mm.System(name=name)
    system.energy = time_driver_case.energy
    system.dynamics = time_driver_case.damping
    system.m = time_driver_case.m

    td = time_driver_case.calculator.TimeDriver()
    td.drive(system, t=0.2e-9, n=50)

    value = system.m(time_driver_case.mesh.region.center)
    assert np.linalg.norm(np.subtract(value, (0, 0, time_driver_case.Ms))) < 10

    time_driver_case.calculator.delete(system)


def test_time_driver_nodamping(time_driver_case):
    name = "timedriver_nodamping"

    system = mm.System(name=name)
    system.energy = time_driver_case.energy
    system.dynamics = time_driver_case.precession
    system.m = time_driver_case.m

    td = time_driver_case.calculator.TimeDriver()
    td.drive(system, t=0.2e-9, n=50)

    value = system.m(time_driver_case.mesh.region.center)
    assert np.linalg.norm(np.subtract(value, (0, 0, time_driver_case.Ms))) > 1e3

    time_driver_case.calculator.delete(system)


def test_time_driver_output_files(time_driver_case):
    name = "timedriver_output_files"

    system = mm.System(name=name)
    system.energy = time_driver_case.energy
    system.dynamics = time_driver_case.precession + time_driver_case.damping
    system.m = time_driver_case.m

    td = time_driver_case.calculator.TimeDriver()
    td.drive(system, t=0.2e-9, n=50, save=True, overwrite=True)

    dirname = os.path.join(f"{name}", f"drive-{system.drive_number - 1}")
    assert os.path.exists(dirname)
    if os.path.exists(os.path.join(dirname, f"{name}.out")):
        mumax3_path = os.path.join(dirname, f"{name}.out")
        mx3filename = os.path.join(dirname, f"{name}.mx3")
        assert os.path.isfile(mx3filename)
        omffilename = os.path.join(dirname, "m0.omf")
        assert os.path.isfile(omffilename)
        omf_files = list(glob.iglob(os.path.join(mumax3_path, "*.ovf")))
        assert len(omf_files) == 50
    else:
        miffilename = os.path.join(dirname, f"{name}.mif")
        assert os.path.isfile(miffilename)
        omf_files = list(glob.iglob(os.path.join(dirname, "*.omf")))
        assert len(omf_files) == 51
        odt_files = list(glob.iglob(os.path.join(dirname, "*.odt")))
        assert len(odt_files) == 1
        omffilename = os.path.join(dirname, "m0.omf")
        assert omffilename in omf_files

    time_driver_case.calculator.delete(system)


def test_time_driver_drive_exception(time_driver_case):
    name = "timedriver_exception"

    system = mm.System(name=name)
    system.energy = time_driver_case.energy
    system.dynamics = time_driver_case.precession + time_driver_case.damping
    system.m = time_driver_case.m

    td = time_driver_case.calculator.TimeDriver()
    with pytest.raises(ValueError):
        td.drive(system, t=-0.1e-9, n=10)
    with pytest.raises(ValueError):
        td.drive(system, t=0.1e-9, n=-10)


def test_time_driver_wrong_evolver(time_driver_case):
    system = mm.examples.macrospin()
    evolver = time_driver_case.calculator.CGEvolver()
    td = time_driver_case.calculator.TimeDriver(evolver=evolver)

    with pytest.raises(TypeError):
        td.drive(system, t=1e-12, n=1)

    time_driver_case.calculator.delete(system)


def test_time_driver_check_for_energy_and_dynamics(time_driver_case):
    system = mm.examples.macrospin()
    system.energy = 0
    td = time_driver_case.calculator.TimeDriver()

    with pytest.raises(RuntimeError, match="System's energy is not defined"):
        td.drive(system, t=1e-12, n=1)

    system.dynamics = 0

    with pytest.raises(RuntimeError, match="System's dynamics is not defined"):
        td.drive(system, t=1e-12, n=1)
