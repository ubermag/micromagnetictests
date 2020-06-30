import pytest
import numpy as np
import discretisedfield as df
import micromagneticmodel as mm


class TestZeeman:
    @pytest.fixture(autouse=True)
    def _setup_calculator(self, calculator):
        self.calculator = calculator

    def setup(self):
        p1 = (-10e-9, -5e-9, -3e-9)
        p2 = (10e-9, 5e-9, 3e-9)
        self.region = df.Region(p1=p1, p2=p2)
        self.cell = (1e-9, 1e-9, 1e-9)
        self.subregions = {'r1': df.Region(p1=(-10e-9, -5e-9, -3e-9),
                                           p2=(10e-9, 0, 3e-9)),
                           'r2': df.Region(p1=(-10e-9, 0, -3e-9),
                                           p2=(10e-9, 5e-9, 3e-9))}

    def test_vector(self):
        name = 'zeeman_vector'

        H = (0, 0, 1e6)
        Ms = 1e6

        system = mm.System(name=name)

        # time-independent
        system.energy = mm.Zeeman(H=H)

        mesh = df.Mesh(region=self.region, cell=self.cell)
        system.m = df.Field(mesh, dim=3, value=(1, 1, 1), norm=Ms)

        md = self.calculator.MinDriver()
        md.drive(system)

        value = system.m(mesh.region.random_point())
        assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) < 1e-3

        # time-dependent - sine
        system.energy = mm.Zeeman(H=H, wave='sin', f=1e9, t0=1e-12)

        mesh = df.Mesh(region=self.region, cell=self.cell)
        system.m = df.Field(mesh, dim=3, value=(1, 1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.1e-9, n=20)

        # time-dependent - sinc
        system.energy = mm.Zeeman(H=H, wave='sinc', f=1e9, t0=0)

        mesh = df.Mesh(region=self.region, cell=self.cell)
        system.m = df.Field(mesh, dim=3, value=(1, 1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.1e-9, n=20)

        self.calculator.delete(system)

    def test_dict(self):
        name = 'zeeman_dict'

        H = {'r1': (1e5, 0, 0), 'r2': (0, 0, 1e5)}
        Ms = 1e6

        system = mm.System(name=name)
        system.energy = mm.Zeeman(H=H)

        mesh = df.Mesh(region=self.region, cell=self.cell,
                       subregions=self.subregions)
        system.m = df.Field(mesh, dim=3, value=(1, 1, 1), norm=Ms)

        md = self.calculator.MinDriver()
        md.drive(system)

        assert np.linalg.norm(np.subtract(system.m['r1'].average,
                                          (Ms, 0, 0))) < 1

        assert np.linalg.norm(np.subtract(system.m['r2'].average,
                                          (0, 0, Ms))) < 1

        self.calculator.delete(system)

    def test_field(self):
        name = 'zeeman_field'

        def value_fun(pos):
            x, y, z = pos
            if x <= 0:
                return (1e6, 0, 0)
            else:
                return (0, 0, 1e6)

        mesh = df.Mesh(region=self.region, cell=self.cell)

        H = df.Field(mesh, dim=3, value=value_fun)
        Ms = 1e6

        system = mm.System(name=name)
        system.energy = mm.Zeeman(H=H)
        system.m = df.Field(mesh, dim=3, value=(0, 1, 0), norm=Ms)

        md = self.calculator.MinDriver()
        md.drive(system)

        value = system.m((-2e-9, -2e-9, -2e-9))
        assert np.linalg.norm(np.subtract(value, (Ms, 0, 0))) < 1e-3

        value = system.m((2e-9, 2e-9, 2e-9))
        assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) < 1e-3

        self.calculator.delete(system)
