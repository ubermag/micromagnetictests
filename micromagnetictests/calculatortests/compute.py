import sys

import discretisedfield as df
import micromagneticmodel as mm
import pytest


class TestCompute:
    @pytest.fixture(autouse=True)
    def _setup_calculator(self, calculator):
        self.calculator = calculator

    def setup(self):
        name = "compute_tests"
        p1 = (0, 0, 0)
        p2 = (10e-9, 2e-9, 2e-9)
        cell = (2e-9, 2e-9, 2e-9)
        region = df.Region(p1=p1, p2=p2)
        mesh = df.Mesh(region=region, cell=cell)

        self.system = mm.System(name=name)
        self.system.energy = (
            mm.Exchange(A=1e-12)
            + mm.Demag()
            + mm.Zeeman(H=(8e6, 0, 0))
            + mm.UniaxialAnisotropy(K=1e4, u=(0, 0, 1))
            + mm.CubicAnisotropy(K=1e3, u1=(1, 0, 0), u2=(0, 1, 0))
        )

        self.system.m = df.Field(mesh, dim=3, value=(0, 0, 1), norm=8e6)

    def test_energy(self):
        for term in self.system.energy:
            assert isinstance(self.calculator.compute(term.energy, self.system), float)
        assert isinstance(
            self.calculator.compute(self.system.energy.energy, self.system), float
        )
        self.calculator.delete(self.system)

    def test_energy_density(self):
        for term in self.system.energy:
            assert isinstance(
                self.calculator.compute(term.density, self.system), df.Field
            )
        assert isinstance(
            self.calculator.compute(self.system.energy.density, self.system), df.Field
        )
        self.calculator.delete(self.system)

    def test_effective_field(self):
        for term in self.system.energy:
            assert isinstance(
                self.calculator.compute(term.effective_field, self.system), df.Field
            )
        assert isinstance(
            self.calculator.compute(self.system.energy.effective_field, self.system),
            df.Field,
        )
        self.calculator.delete(self.system)

    def test_invalid_func(self):
        with pytest.raises(ValueError):
            self.calculator.compute(self.system.energy.__len__, self.system)

    def test_dmi(self):
        if sys.platform != "win32":
            self.system.energy += mm.DMI(D=5e-3, crystalclass="T")
            term = self.system.energy.dmi
            for crystalclass in [
                "T",
                "Cnv_x",
                "Cnv_y",
                "Cnv_z",
                "D2d_x",
                "D2d_y",
                "D2d_z",
            ]:
                term.crystalclass = crystalclass
                assert isinstance(
                    self.calculator.compute(term.energy, self.system), float
                )
                assert isinstance(
                    self.calculator.compute(term.density, self.system), df.Field
                )
                assert isinstance(
                    self.calculator.compute(term.effective_field, self.system), df.Field
                )
            assert isinstance(
                self.calculator.compute(self.system.energy.energy, self.system), float
            )
            self.calculator.delete(self.system)

    def test_slonczewski(self):
        self.system.dynamics = mm.Slonczewski(J=7.5e12, mp=(1, 0, 0), P=0.4, Lambda=2)
        assert isinstance(
            self.calculator.compute(self.system.energy.energy, self.system), float
        )
        self.calculator.delete(self.system)

    def test_zhang_li(self):
        self.system.dynamics = mm.ZhangLi(beta=0.01, u=5e6)
        assert isinstance(
            self.calculator.compute(self.system.energy.energy, self.system), float
        )
        self.calculator.delete(self.system)
