import math

import discretisedfield as df
import micromagneticdata as mdata
import micromagneticmodel as mm
import pytest


class TestZhangLi:
    """
    Each test defines a nanostrip with a domain wall in the left half. Current is
    applied to move the DW to the other side of the strip. Details of the current
    (direction, time dependence) are varied in the different tests.
    """

    @pytest.fixture(autouse=True)
    def _setup_calculator(self, calculator):
        self.calculator = calculator

    # inside setup_method it is not possible to use a fixture
    @pytest.fixture(autouse=True)
    def setup_method_as_fixture(self):
        """Prepare nano strip (l_x=200nm) with DW at x≈70nm."""
        p1 = (0, 0, 0)
        p2 = (200e-9, 20e-9, 5e-9)
        cell = (5e-9, 5e-9, 5e-9)
        subregions = {
            "r1": df.Region(p1=(0, 0, 0), p2=(100e-9, 20e-9, 5e-9)),
            "r2": df.Region(p1=(100e-9, 0, 0), p2=(200e-9, 20e-9, 5e-9)),
        }

        Ms = 5.8e5
        A = 15e-12
        K = 0.5e6
        anisotropy_axis = (0, 0, 1)

        def init_m(p):
            if p[0] < 70e-9:
                return (0.1, 0.1, -1)
            return (0.1, 0.1, 1)

        system = mm.System(name="strip_x")
        system.energy = mm.Exchange(A=A) + mm.UniaxialAnisotropy(K=K, u=anisotropy_axis)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)
        system.m = df.Field(mesh, nvdim=3, value=init_m, norm=Ms)

        md = self.calculator.MinDriver()
        md.drive(system)

        # ensure that the initial state is as expected,
        # a domain wall at x≈70 nm
        assert 0.2 < system.m.orientation.z.mean() < 0.4
        assert system.m.orientation.z.sel(x=(0, 60e-9)).mean() < -0.95
        assert system.m.orientation.z.sel(x=(80e-9, 200e-9)).mean() > 0.95

        self.u = 200
        self.beta = 0.5
        self.system = system

    def test_scalar_u(self):
        """
        Uniform current in x direction.
        """
        td = self.calculator.TimeDriver()

        system = self.system

        system.dynamics = (
            mm.Precession(gamma0=mm.consts.gamma0)
            + mm.Damping(alpha=0.3)
            + mm.ZhangLi(u=self.u, beta=self.beta)
        )
        td.drive(system, t=0.35e-9, n=1)

        # check that the domain wall moved to the right side of the strip to x≈130nm
        assert -0.4 < system.m.orientation.z.mean() < -0.2
        assert system.m.orientation.z.sel(x=(0, 120e-9)).mean() < -0.95
        assert system.m.orientation.z.sel(x=(140e-9, 200e-9)).mean() > 0.95

        self.calculator.delete(system)

    def test_time_func_scalar_u(self):
        """
        Uniform current in x direction with sin time dependence.
        """
        td = self.calculator.TimeDriver()

        system = self.system

        f = 2 * math.pi / 0.6e-9

        def time_dep(t):
            return math.sin(f * t)

        system.dynamics = (
            mm.Precession(gamma0=mm.consts.gamma0)
            + mm.Damping(alpha=0.3)
            + mm.ZhangLi(u=self.u, beta=self.beta, func=time_dep, dt=1e-13)
        )

        # drive for one period and save two steps
        td.drive(system, t=0.6e-9, n=2)

        # use micromagneticdata to read the two saved steps
        drive = mdata.Data(name=system.name)[-1]

        # check that the domain wall moved to the right
        # during the first half wave
        assert -0.1 < drive[0].orientation.z.mean() < 0.1
        assert drive[0].orientation.z.sel(x=(0, 90e-9)).mean() < -0.95
        assert drive[0].orientation.z.sel(x=(110e-9, 200e-9)).mean() > 0.95

        # check that the domain wall moved back to its initial position at x≈70nm
        # during the second half wave
        assert 0.2 < drive[1].orientation.z.mean() < 0.4
        assert drive[1].orientation.z.sel(x=(0, 60e-9)).mean() < -0.95
        assert drive[1].orientation.z.sel(x=(80e-9, 200e-9)).mean() > 0.95

        # self.calculator.delete(system)

    def test_time_tcl_scalar_u(self):
        """
        Uniform current in x direction with sin time dependence.
        """
        td = self.calculator.TimeDriver()
        system = self.system

        # time-dependence - tcl strings
        tcl_strings = {}
        # pre-compute frequency (pi not directly available)
        # f = 2 * pi / 0.6e-9 = 10471975511.965977
        tcl_strings["script"] = """proc TimeFunction { total_time } {
            return [expr sin(10471975511.965977 * $total_time)]
        }
        """
        tcl_strings["script_args"] = "total_time"
        tcl_strings["script_name"] = "TimeFunction"

        system.dynamics = (
            mm.Precession(gamma0=mm.consts.gamma0)
            + mm.Damping(alpha=0.3)
            + mm.ZhangLi(u=self.u, beta=self.beta, tcl_strings=tcl_strings)
        )

        # drive for one period and save two steps
        td.drive(system, t=0.6e-9, n=2)

        # use micromagneticdata to read the two saved steps
        drive = mdata.Data(name=system.name)[-1]

        # check that the domain wall moved to the right
        # during the first half wave
        assert -0.1 < drive[0].orientation.z.mean() < 0.1
        assert drive[0].orientation.z.sel(x=(0, 90e-9)).mean() < -0.95
        assert drive[0].orientation.z.sel(x=(110e-9, 200e-9)).mean() > 0.95

        # check that the domain wall moved back to its initial position at x≈70nm
        # during the second half wave
        assert 0.2 < drive[1].orientation.z.mean() < 0.4
        assert drive[1].orientation.z.sel(x=(0, 60e-9)).mean() < -0.95
        assert drive[1].orientation.z.sel(x=(80e-9, 200e-9)).mean() > 0.95

        self.calculator.delete(system)

    def test_dict_scalar_u(self):
        """
        Current only in left half of the strip, defined with a dict.
        """
        td = self.calculator.TimeDriver()

        system = self.system

        system.dynamics = (
            mm.Precession(gamma0=mm.consts.gamma0)
            + mm.Damping(alpha=0.3)
            + mm.ZhangLi(u={"r1": self.u, "r2": 0}, beta=self.beta)
        )
        td.drive(system, t=0.35e-9, n=1)

        # check that the domain wall stops moving when the current stops at x≈100nm
        assert -0.1 < system.m.orientation.z.mean() < 0.1
        assert system.m.orientation.z.sel(x=(0, 90e-9)).mean() < -0.95
        assert system.m.orientation.z.sel(x=(110e-9, 200e-9)).mean() > 0.95

        self.calculator.delete(system)

    def test_dict_vector_u(self):
        """
        Current only in left half of the strip, defined with a dict.
        """
        td = self.calculator.TimeDriver()

        system = self.system

        system.dynamics = (
            mm.Precession(gamma0=mm.consts.gamma0)
            + mm.Damping(alpha=0.3)
            + mm.ZhangLi(u={"r1": (self.u, 0, 0), "r2": (0, 0, 0)}, beta=self.beta)
        )
        td.drive(system, t=0.35e-9, n=1)

        # check that the domain wall stops moving when the current stops at x≈100nm
        assert -0.1 < system.m.orientation.z.mean() < 0.1
        assert system.m.orientation.z.sel(x=(0, 90e-9)).mean() < -0.95
        assert system.m.orientation.z.sel(x=(110e-9, 200e-9)).mean() > 0.95

        self.calculator.delete(system)

    def test_field_scalar_u(self):
        """
        Current only in left half of the strip, defined with a Field.
        """
        td = self.calculator.TimeDriver()

        system = self.system

        u_field = df.Field(
            mesh=system.m.mesh, nvdim=1, value=lambda p: self.u if p[0] < 100e-9 else 0
        )

        system.dynamics = (
            mm.Precession(gamma0=mm.consts.gamma0)
            + mm.Damping(alpha=0.3)
            + mm.ZhangLi(u=u_field, beta=self.beta)
        )
        td.drive(system, t=0.35e-9, n=1)

        # check that the domain wall stops moving when the current stops at x≈100nm
        assert -0.1 < system.m.orientation.z.mean() < 0.1
        assert system.m.orientation.z.sel(x=(0, 90e-9)).mean() < -0.95
        assert system.m.orientation.z.sel(x=(110e-9, 200e-9)).mean() > 0.95

        self.calculator.delete(system)

    def test_field_vector_u(self):
        """
        Current only in left half of the strip, defined with a Field.
        """
        td = self.calculator.TimeDriver()

        system = self.system

        u_field = df.Field(
            mesh=system.m.mesh,
            nvdim=3,
            value=lambda p: (self.u, 0, 0) if p[0] < 100e-9 else (0, 0, 0),
        )

        system.dynamics = (
            mm.Precession(gamma0=mm.consts.gamma0)
            + mm.Damping(alpha=0.3)
            + mm.ZhangLi(u=u_field, beta=self.beta)
        )
        td.drive(system, t=0.35e-9, n=1)

        # check that the domain wall stops moving when the current stops at x≈100nm
        assert -0.1 < system.m.orientation.z.mean() < 0.1
        assert system.m.orientation.z.sel(x=(0, 90e-9)).mean() < -0.95
        assert system.m.orientation.z.sel(x=(110e-9, 200e-9)).mean() > 0.95

        self.calculator.delete(system)

    def test_vector_u(self):
        """
        Strip oriented along y with uniform current applied in y direction.
        """
        md = self.calculator.MinDriver()
        td = self.calculator.TimeDriver()

        system = self.system

        # replace system m with strip along y
        p1 = (0, 0, 0)
        p2 = (20e-9, 200e-9, 5e-9)
        cell = (5e-9, 5e-9, 5e-9)

        def init_m(p):
            if p[1] < 70e-9:
                return (0.1, 0.1, -1)
            return (0.1, 0.1, 1)

        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        Ms = 5.8e5
        system.m = df.Field(mesh, nvdim=3, value=init_m, norm=Ms)

        md.drive(system)

        # ensure that the initial state is as expected,
        # a domain wall at y≈70 nm
        assert 0.2 < system.m.orientation.z.mean() < 0.4
        assert system.m.orientation.z.sel(y=(0, 60e-9)).mean() < -0.95
        assert system.m.orientation.z.sel(y=(80e-9, 200e-9)).mean() > 0.95

        system.dynamics = (
            mm.Precession(gamma0=mm.consts.gamma0)
            + mm.Damping(alpha=0.3)
            + mm.ZhangLi(u=(0, self.u, 0), beta=self.beta)
        )
        td.drive(system, t=0.35e-9, n=1)

        # check that the domain wall moved to the right side of the strip to y≈130nm
        assert -0.4 < system.m.orientation.z.mean() < -0.2
        assert system.m.orientation.z.sel(y=(0, 120e-9)).mean() < -0.95
        assert system.m.orientation.z.sel(y=(140e-9, 200e-9)).mean() > 0.95

        self.calculator.delete(system)
