from types import SimpleNamespace

import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import pytest


@pytest.fixture
def uniaxial_anisotropy_case(calculator):
    case = SimpleNamespace()
    case.calculator = calculator
    p1 = (-7e-9, -5e-9, -4e-9)
    p2 = (7e-9, 5e-9, 4e-9)
    case.region = df.Region(p1=p1, p2=p2)
    case.cell = (1e-9, 1e-9, 1e-9)
    case.subregions = {
        "r1": df.Region(p1=(-7e-9, -5e-9, -4e-9), p2=(0, 5e-9, 4e-9)),
        "r2": df.Region(p1=(0, -5e-9, -4e-9), p2=(7e-9, 5e-9, 4e-9)),
    }

    return case


def test_uniaxial_anisotropy_scalar_vector(uniaxial_anisotropy_case):
    name = "uniaxialanisotropy_scalar_vector"

    K = 1e5
    u = (0, 0, 1)
    Ms = 1e6

    system = mm.System(name=name)
    system.energy = mm.UniaxialAnisotropy(K=K, u=u)

    mesh = df.Mesh(
        region=uniaxial_anisotropy_case.region, cell=uniaxial_anisotropy_case.cell
    )
    system.m = df.Field(mesh, nvdim=3, value=(0, 0.3, 1), norm=Ms)

    md = uniaxial_anisotropy_case.calculator.MinDriver()
    md.drive(system)

    value = system.m(mesh.region.center)
    assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) < 1e-3

    uniaxial_anisotropy_case.calculator.delete(system)


def test_uniaxial_anisotropy_field_vector(uniaxial_anisotropy_case):
    name = "uniaxialanisotropy_field_vector"

    def value_fun(pos):
        x, y, z = pos
        if x <= 0:
            return 0
        else:
            return 1e5

    mesh = df.Mesh(
        region=uniaxial_anisotropy_case.region, cell=uniaxial_anisotropy_case.cell
    )

    K = df.Field(mesh, nvdim=1, value=value_fun)
    u = (0, 0, 1)
    Ms = 1e6

    system = mm.System(name=name)
    system.energy = mm.UniaxialAnisotropy(K=K, u=u)
    system.m = df.Field(mesh, nvdim=3, value=(0, 0.3, 1), norm=Ms)

    md = uniaxial_anisotropy_case.calculator.MinDriver()
    md.drive(system)

    value = system.m((-2e-9, -2e-9, -2e-9))
    assert np.linalg.norm(np.cross(value, (0, 0.3 * Ms, Ms))) < 1e-3

    value = system.m((2e-9, 2e-9, 2e-9))
    assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) < 1e-3

    uniaxial_anisotropy_case.calculator.delete(system)


def test_uniaxial_anisotropy_scalar_field(uniaxial_anisotropy_case):
    name = "uniaxialanisotropy_scalar_field"

    def value_fun(pos):
        x, y, z = pos
        if x <= 0:
            return (1, 0, 0)
        else:
            return (0, 1, 0)

    mesh = df.Mesh(
        region=uniaxial_anisotropy_case.region, cell=uniaxial_anisotropy_case.cell
    )

    K = 1e5
    u = df.Field(mesh, nvdim=3, value=value_fun)
    Ms = 1e6

    system = mm.System(name=name)
    system.energy = mm.UniaxialAnisotropy(K=K, u=u)
    system.m = df.Field(mesh, nvdim=3, value=(1, 1, 0), norm=Ms)

    md = uniaxial_anisotropy_case.calculator.MinDriver()
    md.drive(system)

    value = system.m((-2e-9, -2e-9, -2e-9))
    assert np.linalg.norm(np.subtract(value, (Ms, 0, 0))) < 1e-3

    value = system.m((2e-9, 2e-9, 2e-9))
    assert np.linalg.norm(np.subtract(value, (0, Ms, 0))) < 1e-3

    uniaxial_anisotropy_case.calculator.delete(system)


def test_uniaxial_anisotropy_field_field(uniaxial_anisotropy_case):
    name = "uniaxialanisotropy_field_field"

    def K_fun(pos):
        x, y, z = pos
        if -2e-9 <= x <= 2e-9:
            return 0
        else:
            return 1e5

    def u_fun(pos):
        x, y, z = pos
        if x <= 0:
            return (1, 0, 0)
        else:
            return (0, 1, 0)

    mesh = df.Mesh(
        region=uniaxial_anisotropy_case.region, cell=uniaxial_anisotropy_case.cell
    )

    K = df.Field(mesh, nvdim=1, value=K_fun)
    u = df.Field(mesh, nvdim=3, value=u_fun)
    Ms = 1e6

    system = mm.System(name=name)
    system.energy = mm.UniaxialAnisotropy(K=K, u=u)
    system.m = df.Field(mesh, nvdim=3, value=(1, 1, 0), norm=Ms)

    md = uniaxial_anisotropy_case.calculator.MinDriver()
    md.drive(system)

    value = system.m((-3e-9, -3e-9, -3e-9))
    assert np.linalg.norm(np.subtract(value, (Ms, 0, 0))) < 1e-3

    value = system.m((3e-9, 3e-9, 3e-9))
    assert np.linalg.norm(np.subtract(value, (0, Ms, 0))) < 1e-3

    value = system.m((0, 0, 0))
    assert np.linalg.norm(np.cross(value, (Ms, Ms, 0))) < 1e-3

    uniaxial_anisotropy_case.calculator.delete(system)


def test_uniaxial_anisotropy_dict_vector(uniaxial_anisotropy_case):
    name = "uniaxialanisotropy_dict_vector"

    mesh = df.Mesh(
        region=uniaxial_anisotropy_case.region,
        cell=uniaxial_anisotropy_case.cell,
        subregions=uniaxial_anisotropy_case.subregions,
    )
    K = {"r1": 0, "r2": 1e5}
    u = (0, 0, 1)
    Ms = 1e6

    system = mm.System(name=name)
    system.energy = mm.UniaxialAnisotropy(K=K, u=u)
    system.m = df.Field(mesh, nvdim=3, value=(0, 0.3, 1), norm=Ms)

    md = uniaxial_anisotropy_case.calculator.MinDriver()
    md.drive(system)

    value = system.m((-2e-9, -2e-9, -2e-9))
    assert np.linalg.norm(np.cross(value, (0, 0.3 * Ms, Ms))) < 1e-3

    value = system.m((2e-9, 2e-9, 2e-9))
    assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) < 1e-3

    uniaxial_anisotropy_case.calculator.delete(system)


def test_uniaxial_anisotropy_field_dict(uniaxial_anisotropy_case):
    name = "uniaxialanisotropy_field_dict"

    def K_fun(pos):
        x, y, z = pos
        if -2e-9 <= x <= 2e-9:
            return 0
        else:
            return 1e5

    def u_fun(pos):
        x, y, z = pos
        if x <= 0:
            return (1, 0, 0)
        else:
            return (0, 1, 0)

    mesh = df.Mesh(
        region=uniaxial_anisotropy_case.region, cell=uniaxial_anisotropy_case.cell
    )

    K = df.Field(mesh, nvdim=1, value=K_fun)
    u = df.Field(mesh, nvdim=3, value=u_fun)
    Ms = 1e6

    system = mm.System(name=name)
    system.energy = mm.UniaxialAnisotropy(K=K, u=u)
    system.m = df.Field(mesh, nvdim=3, value=(1, 1, 0), norm=Ms)

    md = uniaxial_anisotropy_case.calculator.MinDriver()
    md.drive(system)

    value = system.m((-3e-9, -3e-9, -3e-9))
    assert np.linalg.norm(np.subtract(value, (Ms, 0, 0))) < 1e-3

    value = system.m((3e-9, 3e-9, 3e-9))
    assert np.linalg.norm(np.subtract(value, (0, Ms, 0))) < 1e-3

    value = system.m((0, 0, 0))
    assert np.linalg.norm(np.cross(value, (Ms, Ms, 0))) < 1e-3

    uniaxial_anisotropy_case.calculator.delete(system)


def test_uniaxial_anisotropy_higher_order_scalar_vector(uniaxial_anisotropy_case):
    name = "uniaxialanisotropy_higher_order_scalar_vector"

    K1 = 1e5
    K2 = 2e3
    u = (0, 0, 1)
    Ms = 1e6

    system = mm.System(name=name)
    system.energy = mm.UniaxialAnisotropy(K1=K1, K2=K2, u=u)

    mesh = df.Mesh(
        region=uniaxial_anisotropy_case.region, cell=uniaxial_anisotropy_case.cell
    )
    system.m = df.Field(mesh, nvdim=3, value=(0, 0.3, 1), norm=Ms)

    md = uniaxial_anisotropy_case.calculator.MinDriver()
    md.drive(system)

    value = system.m(mesh.region.center)
    assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) < 1e-3

    uniaxial_anisotropy_case.calculator.delete(system)
