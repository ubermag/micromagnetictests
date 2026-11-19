from types import SimpleNamespace

import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import pytest


@pytest.fixture
def mesh_case(calculator):
    case = SimpleNamespace()
    case.calculator = calculator
    p1 = (-7e-9, -5e-9, -4e-9)
    p2 = (7e-9, 5e-9, 4e-9)
    case.region = df.Region(p1=p1, p2=p2)
    case.cell = (1e-9, 1e-9, 1e-9)
    case.bc = "xyz"
    case.subregions = {
        "r1": df.Region(p1=(-7e-9, -5e-9, -4e-9), p2=(7e-9, 0, 4e-9)),
        "r2": df.Region(p1=(-7e-9, 0, -4e-9), p2=(7e-9, 2e-9, 4e-9)),
        "r3": df.Region(p1=(-7e-9, 2e-9, -4e-9), p2=(7e-9, 5e-9, 4e-9)),
    }

    return case


def test_mesh_single_nopbc(mesh_case):
    name = "mesh_single_nopbc"

    Ms = 1e6
    H = (0, 0, 5e6)

    mesh = df.Mesh(region=mesh_case.region, cell=mesh_case.cell)

    system = mm.System(name=name)
    system.energy = mm.Zeeman(H=H)
    system.m = df.Field(mesh, nvdim=3, value=(1, 0, 0), norm=Ms)

    md = mesh_case.calculator.MinDriver()
    md.drive(system)

    value = system.m(mesh.region.center)
    assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) < 1e-3

    mesh_case.calculator.delete(system)


def test_mesh_multi_nopbc(mesh_case):
    name = "mesh_multi_nopbc"

    Ms = 1e6
    H = (0, 0, 5e6)

    mesh = df.Mesh(
        region=mesh_case.region, cell=mesh_case.cell, subregions=mesh_case.subregions
    )

    system = mm.System(name=name)
    system.energy = mm.Zeeman(H=H)
    system.m = df.Field(mesh, nvdim=3, value=(1, 0, 0), norm=Ms)

    md = mesh_case.calculator.MinDriver()
    md.drive(system)

    value = system.m(mesh.region.center)
    assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) < 1e-3

    mesh_case.calculator.delete(system)


def test_mesh_single_pbc(mesh_case):
    name = "mesh_single_pbc"

    Ms = 1e6
    H = (0, 0, 5e6)

    mesh = df.Mesh(region=mesh_case.region, cell=mesh_case.cell, bc=mesh_case.bc)

    system = mm.System(name=name)
    system.energy = mm.Zeeman(H=H)
    system.m = df.Field(mesh, nvdim=3, value=(1, 0, 0), norm=Ms)

    md = mesh_case.calculator.MinDriver()
    md.drive(system)

    value = system.m(mesh.region.center)
    assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) < 1e-3

    mesh_case.calculator.delete(system)


def test_mesh_multi_pbc(mesh_case):
    name = "mesh_multi_pbc"

    Ms = 1e6
    H = (0, 0, 5e6)

    mesh = df.Mesh(
        region=mesh_case.region,
        cell=mesh_case.cell,
        bc=mesh_case.bc,
        subregions=mesh_case.subregions,
    )

    system = mm.System(name=name)
    system.energy = mm.Zeeman(H=H)
    system.m = df.Field(mesh, nvdim=3, value=(1, 0, 0), norm=Ms)

    md = mesh_case.calculator.MinDriver()
    md.drive(system)

    value = system.m(mesh.region.center)
    assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) < 1e-3

    mesh_case.calculator.delete(system)
