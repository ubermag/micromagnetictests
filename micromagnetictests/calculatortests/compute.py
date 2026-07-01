import sys
from types import SimpleNamespace

import discretisedfield as df
import micromagneticmodel as mm
import pytest


@pytest.fixture
def compute_case(calculator):
    case = SimpleNamespace()
    case.calculator = calculator
    name = "compute_tests"
    p1 = (0, 0, 0)
    p2 = (10e-9, 2e-9, 2e-9)
    cell = (2e-9, 2e-9, 2e-9)
    region = df.Region(p1=p1, p2=p2)
    subregions = {
        "a": df.Region(p1=(0, 0, 0), p2=(6e-9, 2e-9, 2e-9)),
        "b": df.Region(p1=(6e-9, 0, 0), p2=(10e-9, 2e-9, 2e-9)),
    }
    mesh = df.Mesh(region=region, cell=cell, subregions=subregions)
    case.subregions = subregions
    case.system = mm.System(name=name)
    case.system.energy = (
        mm.Exchange(A=1e-12)
        + mm.Demag()
        + mm.Zeeman(H=(8e6, 0, 0))
        + mm.UniaxialAnisotropy(K=1e4, u=(0, 0, 1))
        + mm.CubicAnisotropy(K=1e3, u1=(1, 0, 0), u2=(0, 1, 0))
    )

    case.system.m = df.Field(mesh, nvdim=3, value=(0, 0, 1), norm=8e6)

    return case


def test_compute_energy(compute_case):
    self = compute_case
    for term in self.system.energy:
        assert isinstance(self.calculator.compute(term.energy, self.system), float)
    assert isinstance(
        self.calculator.compute(self.system.energy.energy, self.system), float
    )
    self.calculator.delete(self.system)


def test_compute_energy_density(compute_case):
    self = compute_case
    for term in self.system.energy:
        e_density = self.calculator.compute(term.density, self.system)
        assert isinstance(e_density, df.Field)
        assert e_density.mesh.subregions == self.subregions
    e_density = self.calculator.compute(self.system.energy.density, self.system)
    assert isinstance(e_density, df.Field)
    assert e_density.mesh.subregions == self.subregions
    self.calculator.delete(self.system)


def test_compute_effective_field(compute_case):
    self = compute_case
    for term in self.system.energy:
        effective_field = self.calculator.compute(term.effective_field, self.system)
        assert isinstance(effective_field, df.Field)
        assert effective_field.mesh.subregions == self.subregions
    effective_field = self.calculator.compute(
        self.system.energy.effective_field, self.system
    )
    assert isinstance(effective_field, df.Field)
    assert effective_field.mesh.subregions == self.subregions
    self.calculator.delete(self.system)


def test_compute_invalid_func(compute_case):
    self = compute_case
    with pytest.raises(ValueError):
        self.calculator.compute(self.system.energy.__len__, self.system)


def test_compute_dmi(compute_case):
    self = compute_case
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
            assert isinstance(self.calculator.compute(term.energy, self.system), float)
            e_density = self.calculator.compute(term.density, self.system)
            assert isinstance(e_density, df.Field)
            assert e_density.mesh.subregions == self.subregions
            effective_field = self.calculator.compute(term.effective_field, self.system)
            assert isinstance(effective_field, df.Field)
            assert effective_field.mesh.subregions == self.subregions
        assert isinstance(
            self.calculator.compute(self.system.energy.energy, self.system), float
        )
        self.calculator.delete(self.system)


def test_compute_slonczewski(compute_case):
    self = compute_case
    self.system.dynamics = mm.Slonczewski(J=7.5e12, mp=(1, 0, 0), P=0.4, Lambda=2)
    assert isinstance(
        self.calculator.compute(self.system.energy.energy, self.system), float
    )
    self.calculator.delete(self.system)


def test_compute_zhang_li(compute_case):
    self = compute_case
    self.system.dynamics = mm.ZhangLi(beta=0.01, u=5e6)
    assert isinstance(
        self.calculator.compute(self.system.energy.energy, self.system), float
    )
    self.calculator.delete(self.system)
