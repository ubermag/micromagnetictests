import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import pytest


class TestDamping:
    @pytest.fixture(autouse=True)
    def _setup_calculator(self, calculator):
        self.calculator = calculator

    def setup_method(self):
        p1 = (-5e-9, -5e-9, -3e-9)
        p2 = (5e-9, 5e-9, 3e-9)
        self.region = df.Region(p1=p1, p2=p2)
        self.n = (10, 10, 10)
        self.subregions = {
            "r1": df.Region(p1=(-5e-9, -5e-9, -3e-9), p2=(5e-9, 0, 3e-9)),
            "r2": df.Region(p1=(-5e-9, 0, -3e-9), p2=(5e-9, 5e-9, 3e-9)),
        }

    def test_scalar(self):
        name = "damping_scalar"

        H = (0, 0, 1e6)
        alpha = 0
        Ms = 1e6

        mesh = df.Mesh(region=self.region, n=self.n)

        system = mm.System(name=name)
        system.energy = mm.Zeeman(H=H)
        system.dynamics = mm.Damping(alpha=alpha)
        system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.2e-9, n=50)

        # Alpha is zero, nothing should change.
        value = system.m(mesh.region.center)
        assert np.linalg.norm(np.cross(value, (0, 0.1 * Ms, Ms))) < 1e-3

        self.calculator.delete(system)

    def test_dict(self):
        name = "damping_dict"

        H = (0, 0, 1e6)
        alpha = {"r1": 0, "r2": 1}
        Ms = 1e6

        mesh = df.Mesh(region=self.region, n=self.n, subregions=self.subregions)

        system = mm.System(name=name)
        system.energy = mm.Zeeman(H=H)
        system.dynamics = mm.Damping(alpha=alpha)
        system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.2e-9, n=50)

        # alpha=0 region
        value = system.m((1e-9, -4e-9, 3e-9))
        assert np.linalg.norm(np.cross(value, (0, 0.1 * Ms, Ms))) < 1e-3

        # alpha!=0 region
        value = system.m((1e-9, 4e-9, 3e-9))
        assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) < 1e-3

        self.calculator.delete(system)

    def test_field(self):
        name = "damping_field"

        mesh = df.Mesh(region=self.region, n=self.n)

        def value_fun(pos):
            x, y, z = pos
            if y <= 0:
                return 0
            else:
                return 1

        H = (0, 0, 1e6)
        alpha = df.Field(mesh, nvdim=1, value=value_fun)
        Ms = 1e6

        system = mm.System(name=name)
        system.energy = mm.Zeeman(H=H)
        system.dynamics = mm.Damping(alpha=alpha)
        system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.2e-9, n=50)

        # alpha=0 region
        value = system.m((1e-9, -4e-9, 3e-9))
        assert np.linalg.norm(np.cross(value, (0, 0.1 * Ms, Ms))) < 1e-3

        # alpha!=0 region
        value = system.m((1e-9, 4e-9, 3e-9))
        assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) < 1e-3

        self.calculator.delete(system)
