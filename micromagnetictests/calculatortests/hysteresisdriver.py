import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import pytest


class TestHysteresisDriver:
    @pytest.fixture(autouse=True)
    def _setup_calculator(self, calculator):
        self.calculator = calculator

    def setup(self):
        p1 = (0, 0, 0)
        p2 = (5e-9, 5e-9, 5e-9)
        n = (5, 5, 5)
        self.Ms = 1e6
        A = 1e-12
        H = (0, 0, 1e6)
        region = df.Region(p1=p1, p2=p2)
        self.mesh = df.Mesh(region=region, n=n)
        self.energy = mm.Exchange(A=A) + mm.Zeeman(H=H)
        self.m = df.Field(self.mesh, dim=3, value=(0, 1, 0), norm=self.Ms)

    def test_noevolver_nodriver(self):
        name = "hysteresisdriver_noevolver_nodriver"

        system = mm.System(name=name)
        system.energy = self.energy
        system.m = self.m

        hd = self.calculator.HysteresisDriver()
        hd.drive(system, Hmin=(0, 0, -1e6), Hmax=(0, 0, 1e6), n=3)

        value = system.m(self.mesh.region.random_point())
        assert np.linalg.norm(np.subtract(value, (0, 0, self.Ms))) < 1e-3

        assert len(system.table.data.index) == 5

        assert system.table.x == "B_hysteresis"

        self.calculator.delete(system)
