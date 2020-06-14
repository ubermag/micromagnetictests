import pytest
import discretisedfield as df
import micromagneticmodel as mm


class TestDemag:
    @pytest.fixture(autouse=True)
    def _setup_calculator(self, calculator):
        self.calculator = calculator

    def setup(self):
        p1 = (-10e-9, 0, 0)
        p2 = (10e-9, 1e-9, 1e-9)
        cell = (1e-9, 1e-9, 1e-9)
        region = df.Region(p1=p1, p2=p2)
        self.mesh = df.Mesh(region=region, cell=cell)

    def test_demag(self):
        name = 'demag'

        Ms = 1e6

        system = mm.System(name=name)
        system.energy = mm.Demag()

        system.m = df.Field(self.mesh, dim=3, value=(1, 1, 1), norm=Ms)

        md = self.calculator.MinDriver()
        md.drive(system)

        # Check if it runs. Tests to be added here.

        self.calculator.delete(system)

    def test_demag_asymptotic_radius(self):
        name = 'demag_asymptotic_radius'

        Ms = 1e6

        system = mm.System(name=name)
        system.energy = mm.Demag(asymptotic_radius=6)

        system.m = df.Field(self.mesh, dim=3, value=(0, 0, 1), norm=Ms)

        md = self.calculator.MinDriver()
        md.drive(system)

        # Check if it runs. Tests to be added here.

        self.calculator.delete(system)
