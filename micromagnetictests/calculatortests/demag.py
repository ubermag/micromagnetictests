from types import SimpleNamespace

import discretisedfield as df
import micromagneticmodel as mm
import pytest


@pytest.fixture
def demag_case(calculator):
    case = SimpleNamespace()
    case.calculator = calculator
    p1 = (-5e-9, 0, 0)
    p2 = (5e-9, 5e-9, 1e-9)
    case.cell = (1e-9, 1e-9, 1e-9)
    case.region = df.Region(p1=p1, p2=p2)
    case.mesh = df.Mesh(region=case.region, cell=case.cell)

    return case


def test_demag_demag(demag_case):
    name = "demag"

    Ms = 1e6

    system = mm.System(name=name)
    system.energy = mm.Demag()

    system.m = df.Field(demag_case.mesh, nvdim=3, value=(1, 1, 1), norm=Ms)

    md = demag_case.calculator.MinDriver()
    md.drive(system)

    # Check if it runs. Tests to be added here.

    demag_case.calculator.delete(system)


def test_demag_demag_asymptotic_radius(demag_case):
    name = "demag_asymptotic_radius"

    Ms = 1e6

    system = mm.System(name=name)
    system.energy = mm.Demag(asymptotic_radius=6)

    system.m = df.Field(demag_case.mesh, nvdim=3, value=(0, 0, 1), norm=Ms)

    md = demag_case.calculator.MinDriver()
    md.drive(system)

    # Check if it runs. Tests to be added here.

    demag_case.calculator.delete(system)


def test_demag_demag_1_pbc(demag_case):
    name = "demag_pbc"

    Ms = 1e6

    system = mm.System(name=name)
    system.energy = mm.Demag()

    md = demag_case.calculator.MinDriver()

    # 1D pbc
    mesh = df.Mesh(region=demag_case.region, cell=demag_case.cell, bc="x")
    system.m = df.Field(mesh, nvdim=3, value=(0, 0, 1), norm=Ms)

    md.drive(system)
    demag_case.calculator.delete(system)


def test_demag_demag_2_pbc(demag_case):
    name = "demag_pbc"

    Ms = 1e6

    system = mm.System(name=name)
    system.energy = mm.Demag()

    md = demag_case.calculator.MinDriver()

    # 2D pbc
    mesh = df.Mesh(region=demag_case.region, cell=demag_case.cell, bc="xy")
    system.m = df.Field(mesh, nvdim=3, value=(0, 0, 1), norm=Ms)

    if not hasattr(demag_case.calculator, "RelaxDriver"):
        with pytest.raises(ValueError):
            md.drive(system)
    else:
        md.drive(system)

    demag_case.calculator.delete(system)


def test_demag_demag_3_pbc(demag_case):
    name = "demag_pbc"

    Ms = 1e6

    system = mm.System(name=name)
    system.energy = mm.Demag()

    md = demag_case.calculator.MinDriver()

    # 3D pbc
    mesh = df.Mesh(region=demag_case.region, cell=demag_case.cell, bc="xyz")
    system.m = df.Field(mesh, nvdim=3, value=(0, 0, 1), norm=Ms)

    if not hasattr(demag_case.calculator, "RelaxDriver"):
        with pytest.raises(ValueError):
            md.drive(system)
    else:
        md.drive(system)

    demag_case.calculator.delete(system)
