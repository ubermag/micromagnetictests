import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import pytest


class TestFixedSubregions:
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

    def test_fixed_subregions(self):
        name = 'fixed_subregions'

        H = (0, 0, 1e5)
        Ms = 1e6

        system = mm.System(name=name)
        system.energy = mm.Zeeman(H=H)

        mesh = df.Mesh(region=self.region, cell=self.cell,
                       subregions=self.subregions)
        system.m = df.Field(mesh, dim=3, value=(1, 0, 0), norm=Ms)

        md = self.calculator.MinDriver()
        md.drive(system, fixed_subregions=['r1'])

        assert np.linalg.norm(np.subtract(system.m['r1'].average,
                                          (Ms, 0, 0))) < 1

        assert np.linalg.norm(np.subtract(system.m['r2'].average,
                                          (0, 0, Ms))) < 1

        self.calculator.delete(system)
