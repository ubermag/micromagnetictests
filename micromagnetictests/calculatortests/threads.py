import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import pytest


class TestThreads:
    @pytest.fixture(autouse=True)
    def _setup_calculator(self, calculator):
        self.calculator = calculator

    def setup(self):
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
        self.m = df.Field(self.mesh, dim=3, value=(0, 0.1, 1), norm=self.Ms)

    def test_threads(self):
        name = "timedriver_noevolver_nodriver"

        system = mm.System(name=name)
        system.energy = self.energy
        system.dynamics = self.precession + self.damping
        system.m = self.m

        # One thread
        td = self.calculator.TimeDriver()
        td.drive(system, t=0.2e-9, n=50, n_threads=1)

        value = system.m(self.mesh.region.random_point())
        assert np.linalg.norm(np.subtract(value, (0, 0, self.Ms))) < 1

        # Two threads
        td.drive(system, t=0.2e-9, n=50, n_threads=2)

        value = system.m(self.mesh.region.random_point())
        assert np.linalg.norm(np.subtract(value, (0, 0, self.Ms))) < 1

        self.calculator.delete(system)
