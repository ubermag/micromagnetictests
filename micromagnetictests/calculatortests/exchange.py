import pytest
import random
import numpy as np
import discretisedfield as df
import micromagneticmodel as mm


class TestExchange:
    @pytest.fixture(autouse=True)
    def _setup_calculator(self, calculator):
        self.calculator = calculator

    def setup(self):
        p1 = (-5e-9, -5e-9, -3e-9)
        p2 = (5e-9, 5e-9, 3e-9)
        self.region = df.Region(p1=p1, p2=p2)
        self.n = (10, 10, 6)
        self.subregions = {'r1': df.Region(p1=(-5e-9, -5e-9, -3e-9),
                                           p2=(5e-9, 0, 3e-9)),
                           'r2': df.Region(p1=(-5e-9, 0, -3e-9),
                                           p2=(5e-9, 5e-9, 3e-9))}

    def random_m(self, pos):
        return [2*random.random()-1 for i in range(3)]

    def test_scalar(self):
        name = 'exchange_scalar'

        A = 1e-12
        Ms = 1e6

        system = mm.System(name=name)
        system.energy = mm.Exchange(A=A)

        mesh = df.Mesh(region=self.region, n=self.n)
        system.m = df.Field(mesh, dim=3, value=self.random_m, norm=Ms)

        md = self.calculator.MinDriver()
        md.drive(system)

        assert abs(np.linalg.norm(system.m.average) - Ms) < 1e-3

    def test_dict(self):
        name = 'exchange_dict'

        A = {'r1': 0, 'r2': 1e-12, 'r1:r2': 1e-12, 'default': 2e-12}
        Ms = 1e6

        system = mm.System(name=name)
        system.energy = mm.Exchange(A=A)

        mesh = df.Mesh(region=self.region, n=self.n,
                       subregions=self.subregions)
        system.m = df.Field(mesh, dim=3, value=self.random_m, norm=Ms)

        md = self.calculator.MinDriver()
        md.drive(system)

        # A=0 region
        value1 = system.m((1e-9, -4e-9, 2e-9))
        value2 = system.m((1e-9, -2e-9, 2e-9))
        assert np.linalg.norm(np.subtract(value1, value2)) > 1

        # A!=0 region
        value1 = system.m((1e-9, 4e-9, 2e-9))
        value2 = system.m((1e-9, 2e-9, 2e-9))
        assert np.linalg.norm(np.subtract(value1, value2)) < 1

    def test_field(self):
        name = 'exchange_field'

        def A_fun(pos):
            x, y, z = pos
            if x <= 0:
                return 1e-10  # for 0, OOMMF gives nan
            else:
                return 1e-12

        mesh = df.Mesh(region=self.region, n=self.n)
        A = df.Field(mesh, dim=1, value=A_fun)
        Ms = 1e6

        system = mm.System(name=name)
        system.energy = mm.Exchange(A=A)

        system.m = df.Field(mesh, dim=3, value=self.random_m, norm=Ms)

        md = self.calculator.MinDriver()
        md.drive(system)

        assert abs(np.linalg.norm(system.m.average) - Ms) < 1e-3