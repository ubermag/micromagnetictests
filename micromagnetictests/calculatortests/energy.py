from types import SimpleNamespace

import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import pytest


@pytest.fixture
def energy_case(calculator):
    case = SimpleNamespace()
    case.calculator = calculator
    p1 = (0, 0, 0)
    p2 = (10e-9, 5e-9, 3e-9)
    case.region = df.Region(p1=p1, p2=p2)
    case.cell = (1e-9, 1e-9, 1e-9)
    case.subregions = {
        "r1": df.Region(p1=(0, 0, 0), p2=(5e-9, 5e-9, 3e-9)),
        "r2": df.Region(p1=(5e-9, 0, 0), p2=(10e-9, 5e-9, 3e-9)),
    }

    return case


def test_energy_exchange_zeeman(energy_case):
    name = "energy_exchange_zeeman"

    A = 1e-12
    H = (1e6, 0, 0)
    Ms = 1e6

    mesh = df.Mesh(region=energy_case.region, cell=energy_case.cell)

    system = mm.System(name=name)
    system.energy = mm.Exchange(A=A) + mm.Zeeman(H=H)
    system.m = df.Field(mesh, nvdim=3, value=(0, 1, 0), norm=Ms)

    if hasattr(energy_case.calculator, "RelaxDriver"):
        system.dynamics = mm.Damping(alpha=0.5)
        md = energy_case.calculator.RelaxDriver()
    else:
        md = energy_case.calculator.MinDriver()
    md.drive(system)

    value = system.m(mesh.region.center)
    assert np.linalg.norm(np.subtract(value, (Ms, 0, 0))) < 1e-3

    energy_case.calculator.delete(system)


def test_energy_exchange_uniaxialanisotropy(energy_case):
    name = "exchange_uniaxialanisotropy"

    A = {"r1": 1e-12, "r2": 0}
    K = 1e5
    u = (1, 0, 0)
    Ms = 1e6

    mesh = df.Mesh(
        region=energy_case.region,
        cell=energy_case.cell,
        subregions=energy_case.subregions,
    )

    system = mm.System(name=name)
    system.energy = mm.Exchange(A=A) + mm.UniaxialAnisotropy(K=K, u=u)
    system.m = df.Field(mesh, nvdim=3, value=(0.5, 1, 0), norm=Ms)

    if hasattr(energy_case.calculator, "RelaxDriver"):
        system.dynamics = mm.Damping(alpha=0.5)
        md = energy_case.calculator.RelaxDriver()
    else:
        md = energy_case.calculator.MinDriver()
    md.drive(system)

    value = system.m(mesh.region.center)
    assert np.linalg.norm(np.subtract(value, (Ms, 0, 0))) < 1e-3

    energy_case.calculator.delete(system)


def test_energy_exchange_cubicanisotropy(energy_case):
    name = "exchange_cubicanisotropy"

    A = {"r1": 1e-12, "r2": 0}
    K = 1e5
    u1 = (1, 0, 0)
    u2 = (0, 1, 0)
    Ms = 1e6

    mesh = df.Mesh(
        region=energy_case.region,
        cell=energy_case.cell,
        subregions=energy_case.subregions,
    )

    system = mm.System(name=name)
    system.energy = mm.Exchange(A=A) + mm.CubicAnisotropy(K=K, u1=u1, u2=u2)
    system.m = df.Field(mesh, nvdim=3, value=(1, 0.3, 0), norm=Ms)

    md = energy_case.calculator.MinDriver()
    md.drive(system)

    value = system.m(mesh.region.center)
    assert np.linalg.norm(np.subtract(value, (Ms, 0, 0))) < 1e-3

    energy_case.calculator.delete(system)


def test_energy_exchange_dmi_zeeman(energy_case):
    name = "exchange_dmi_zeeman"

    mesh = df.Mesh(
        region=energy_case.region,
        cell=energy_case.cell,
        subregions=energy_case.subregions,
    )

    # Very weak DMI and strong Zeeman to make the final
    # magnetisation uniform.
    A = {"r1": 1e-12, "r2": 3e-12, "r1:r2": 2e-12}
    D = {"r1": 1e-9, "r2": 0, "r1:r2": 5e-9}
    H = df.Field(mesh, nvdim=3, value=(1e10, 0, 0))
    Ms = 1e6

    system = mm.System(name=name)
    system.energy = (
        mm.Exchange(A=A) + mm.DMI(D=D, crystalclass="Cnv_z") + mm.Zeeman(H=H)
    )
    system.m = df.Field(mesh, nvdim=3, value=(1, 0.3, 0), norm=Ms)

    md = energy_case.calculator.MinDriver()
    md.drive(system)

    value = system.m(mesh.region.center)
    assert np.linalg.norm(np.subtract(value, (Ms, 0, 0))) < 1

    energy_case.calculator.delete(system)


def test_energy_exchange_dmi_zeeman_uniaxialanisotropy_demag(energy_case):
    name = "exchange_dmi_zeeman_uniaxialanisotropy"

    mesh = df.Mesh(
        region=energy_case.region,
        cell=energy_case.cell,
        subregions=energy_case.subregions,
    )

    # Very weak DMI and strong Zeeman to make the final
    # magnetisation uniform.
    A = {"r1": 1e-12, "r2": 3e-12, "r1:r2": 2e-12}
    D = {"r1": 1e-9, "r2": 0, "r1:r2": 5e-9}  # Very weak DMI
    H = df.Field(mesh, nvdim=3, value=(1e12, 0, 0))
    K = 1e6
    u = (1, 0, 0)
    Ms = 1e5

    system = mm.System(name=name)
    system.energy = (
        mm.Exchange(A=A)
        + mm.DMI(D=D, crystalclass="Cnv_z")
        + mm.UniaxialAnisotropy(K=K, u=u)
        + mm.Zeeman(H=H)
        + mm.Demag()
    )
    system.m = df.Field(mesh, nvdim=3, value=(1, 0.3, 0), norm=Ms)

    md = energy_case.calculator.MinDriver()
    md.drive(system)

    value = system.m(mesh.region.center)
    assert np.linalg.norm(np.subtract(value, (Ms, 0, 0))) < 1

    energy_case.calculator.delete(system)


def test_energy_zeeman_zeeman(energy_case):
    name = "multi_zeeman_field"

    def value_fun(pos):
        x, _, _ = pos
        if x <= 0:
            return (1e6, 0, 0)
        else:
            return (0, 0, 1e6)

    mesh = df.Mesh(region=energy_case.region, cell=energy_case.cell)

    H1 = df.Field(mesh, nvdim=3, value=value_fun)
    H2 = df.Field(mesh, nvdim=3, value=(0, 1e6, 0))
    Ms = 1e6

    system = mm.System(name=name)
    system.energy = mm.Zeeman(H=H1, name="zeeman1") + mm.Zeeman(H=H2, name="zeeman2")
    system.m = df.Field(mesh, nvdim=3, value=(0, 1, 0), norm=Ms)

    md = energy_case.calculator.MinDriver()
    md.drive(system)

    # test if it runs

    energy_case.calculator.delete(system)
