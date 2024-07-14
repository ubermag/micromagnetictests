import glob
import os

import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import pytest


class TestTimeDriver:
    @pytest.fixture(autouse=True)
    def _setup_calculator(self, calculator):
        self.calculator = calculator

    def setup_method(self):
        p1 = (0, 0, 0)
        p2 = (5e-9, 5e-9, 5e-9)
        n = (2, 2, 2)
        self.Ms = 1e6
        A = 1e-12
        H = (0, 0, 1e6)
        region = df.Region(p1=p1, p2=p2)
        self.mesh = df.Mesh(region=region, n=n)
        self.energy = mm.Exchange(A=A) + mm.Zeeman(H=H)
        self.precession = mm.Precession(gamma0=mm.consts.gamma0)
        self.damping = mm.Damping(alpha=1)
        self.m = df.Field(self.mesh, nvdim=3, value=(0, 0.1, 1), norm=self.Ms)

    def test_noevolver_nodriver(self):
        name = "timedriver_noevolver_nodriver"

        system = mm.System(name=name)
        system.energy = self.energy
        system.dynamics = self.precession + self.damping
        system.m = self.m

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.2e-9, n=50)

        value = system.m(self.mesh.region.center)
        assert np.linalg.norm(np.subtract(value, (0, 0, self.Ms))) < 10

        assert system.table.x == "t"

        self.calculator.delete(system)

    def test_rungekutta_evolver_nodriver(self):
        name = "timedriver_rungekutta_evolver_nodriver"

        system = mm.System(name=name)
        system.energy = self.energy
        system.dynamics = self.precession + self.damping
        system.m = self.m

        evolver = self.calculator.RungeKuttaEvolver(method="rkf54s")
        td = self.calculator.TimeDriver(evolver=evolver)
        td.drive(system, t=0.2e-9, n=50)

        value = system.m(self.mesh.region.center)
        assert np.linalg.norm(np.subtract(value, (0, 0, self.Ms))) < 1

        self.calculator.delete(system)

    def test_euler_evolver_nodriver(self):
        name = "timedriver_euler_evolver_nodriver"

        system = mm.System(name=name)
        system.energy = self.energy
        system.dynamics = self.precession + self.damping
        system.m = self.m

        evolver = self.calculator.EulerEvolver(start_dm=0.02)
        td = self.calculator.TimeDriver(evolver=evolver)
        td.drive(system, t=0.2e-9, n=50)

        value = system.m(self.mesh.region.center)
        assert np.linalg.norm(np.subtract(value, (0, 0, self.Ms))) < 1

        self.calculator.delete(system)

    def test_theta_evolver_nodriver(self):
        name = "timedriver_theta_evolver_nodriver"

        system = mm.System(name=name)
        system.energy = self.energy
        system.dynamics = self.precession + self.damping
        system.m = self.m
        system.T = 10

        evolver = self.calculator.UHH_ThetaEvolver(fixed_timestep=2e-13)
        td = self.calculator.TimeDriver(evolver=evolver)
        td.drive(system, t=0.2e-9, n=50)

        # Check if it runs.

        self.calculator.delete(system)

    def test_therm_heun_evolver_nodriver(self):
        name = "timedriver_therm_heun_evolver_nodriver"

        system = mm.System(name=name)
        system.energy = self.energy
        system.dynamics = self.precession + self.damping
        system.m = self.m
        system.T = 10

        evolver = self.calculator.Xf_ThermHeunEvolver()
        td = self.calculator.TimeDriver(evolver=evolver)
        td.drive(system, t=1e-11, n=1)

        # Check if it runs.

        self.calculator.delete(system)

    def test_noevolver_nodriver_finite_temperature(self):
        name = "timedriver_therm_heun_evolver_nodriver"

        system = mm.System(name=name)
        system.energy = self.energy
        system.dynamics = self.precession + self.damping
        system.m = self.m
        system.T = 10

        td = self.calculator.TimeDriver()
        with pytest.raises(RuntimeError):
            td.drive(system, t=0.2e-9, n=50)

    def test_noevolver_driver(self):
        name = "timedriver_noevolver_driver"

        system = mm.System(name=name)
        system.energy = self.energy
        system.dynamics = self.precession + self.damping
        system.m = self.m

        td = self.calculator.TimeDriver(stopping_dm_dt=0.01)
        td.drive(system, t=0.3e-9, n=50)

        value = system.m(self.mesh.region.center)
        assert np.linalg.norm(np.subtract(value, (0, 0, self.Ms))) < 1

        self.calculator.delete(system)

    def test_noprecession(self):
        name = "timedriver_noprecession"

        system = mm.System(name=name)
        system.energy = self.energy
        system.dynamics = self.damping
        system.m = self.m

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.2e-9, n=50)

        value = system.m(self.mesh.region.center)
        assert np.linalg.norm(np.subtract(value, (0, 0, self.Ms))) < 10

        self.calculator.delete(system)

    def test_nodamping(self):
        name = "timedriver_nodamping"

        system = mm.System(name=name)
        system.energy = self.energy
        system.dynamics = self.precession
        system.m = self.m

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.2e-9, n=50)

        value = system.m(self.mesh.region.center)
        assert np.linalg.norm(np.subtract(value, (0, 0, self.Ms))) > 1e3

        self.calculator.delete(system)

    def test_output_files(self):
        name = "timedriver_output_files"

        system = mm.System(name=name)
        system.energy = self.energy
        system.dynamics = self.precession + self.damping
        system.m = self.m

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.2e-9, n=50, save=True, overwrite=True)

        dirname = os.path.join(f"{name}", f"drive-{system.drive_number-1}")
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

        self.calculator.delete(system)

    def test_drive_exception(self):
        name = "timedriver_exception"

        system = mm.System(name=name)
        system.energy = self.energy
        system.dynamics = self.precession + self.damping
        system.m = self.m

        td = self.calculator.TimeDriver()
        with pytest.raises(ValueError):
            td.drive(system, t=-0.1e-9, n=10)
        with pytest.raises(ValueError):
            td.drive(system, t=0.1e-9, n=-10)

    def test_wrong_evolver(self):
        system = mm.examples.macrospin()
        evolver = self.calculator.CGEvolver()
        td = self.calculator.TimeDriver(evolver=evolver)

        with pytest.raises(TypeError):
            td.drive(system, t=1e-12, n=1)

        self.calculator.delete(system)

    def test_check_for_energy_and_dynamics(self):
        system = mm.examples.macrospin()
        system.energy = 0
        td = self.calculator.TimeDriver()

        with pytest.raises(RuntimeError, match="System's energy is not defined"):
            td.drive(system, t=1e-12, n=1)

        system.dynamics = 0

        with pytest.raises(RuntimeError, match="System's dynamics is not defined"):
            td.drive(system, t=1e-12, n=1)
