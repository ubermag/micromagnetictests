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
    for term in compute_case.system.energy:
        assert isinstance(
            compute_case.calculator.compute(term.energy, compute_case.system), float
        )
    assert isinstance(
        compute_case.calculator.compute(
            compute_case.system.energy.energy, compute_case.system
        ),
        float,
    )
    compute_case.calculator.delete(compute_case.system)


def test_compute_energy_density(compute_case):
    for term in compute_case.system.energy:
        e_density = compute_case.calculator.compute(term.density, compute_case.system)
        assert isinstance(e_density, df.Field)
        assert e_density.mesh.subregions == compute_case.subregions
    e_density = compute_case.calculator.compute(
        compute_case.system.energy.density, compute_case.system
    )
    assert isinstance(e_density, df.Field)
    assert e_density.mesh.subregions == compute_case.subregions
    compute_case.calculator.delete(compute_case.system)


def test_compute_effective_field(compute_case):
    for term in compute_case.system.energy:
        effective_field = compute_case.calculator.compute(
            term.effective_field, compute_case.system
        )
        assert isinstance(effective_field, df.Field)
        assert effective_field.mesh.subregions == compute_case.subregions
    effective_field = compute_case.calculator.compute(
        compute_case.system.energy.effective_field, compute_case.system
    )
    assert isinstance(effective_field, df.Field)
    assert effective_field.mesh.subregions == compute_case.subregions
    compute_case.calculator.delete(compute_case.system)


def test_compute_invalid_func(compute_case):
    with pytest.raises(ValueError):
        compute_case.calculator.compute(
            compute_case.system.energy.__len__, compute_case.system
        )


def test_compute_dmi(compute_case):
    if sys.platform != "win32":
        compute_case.system.energy += mm.DMI(D=5e-3, crystalclass="T")
        term = compute_case.system.energy.dmi
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
                compute_case.calculator.compute(term.energy, compute_case.system), float
            )
            e_density = compute_case.calculator.compute(
                term.density, compute_case.system
            )
            assert isinstance(e_density, df.Field)
            assert e_density.mesh.subregions == compute_case.subregions
            effective_field = compute_case.calculator.compute(
                term.effective_field, compute_case.system
            )
            assert isinstance(effective_field, df.Field)
            assert effective_field.mesh.subregions == compute_case.subregions
        assert isinstance(
            compute_case.calculator.compute(
                compute_case.system.energy.energy, compute_case.system
            ),
            float,
        )
        compute_case.calculator.delete(compute_case.system)


def test_compute_slonczewski(compute_case):
    compute_case.system.dynamics = mm.Slonczewski(
        J=7.5e12, mp=(1, 0, 0), P=0.4, Lambda=2
    )
    assert isinstance(
        compute_case.calculator.compute(
            compute_case.system.energy.energy, compute_case.system
        ),
        float,
    )
    compute_case.calculator.delete(compute_case.system)


def test_compute_zhang_li(compute_case):
    compute_case.system.dynamics = mm.ZhangLi(beta=0.01, u=5e6)
    assert isinstance(
        compute_case.calculator.compute(
            compute_case.system.energy.energy, compute_case.system
        ),
        float,
    )
    compute_case.calculator.delete(compute_case.system)
