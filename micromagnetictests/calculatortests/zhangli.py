import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import pytest


class TestZhangLi:
    @pytest.fixture(autouse=True)
    def _setup_calculator(self, calculator):
        self.calculator = calculator

    def setup_method(self):
        p1 = (-5e-9, -5e-9, -3e-9)
        p2 = (5e-9, 5e-9, 3e-9)
        self.region = df.Region(p1=p1, p2=p2)
        self.cell = (1e-9, 1e-9, 3e-9)
        self.subregions = {
            "r1": df.Region(p1=(-5e-9, -5e-9, -3e-9), p2=(5e-9, 0, 3e-9)),
            "r2": df.Region(p1=(-5e-9, 0, -3e-9), p2=(5e-9, 5e-9, 3e-9)),
        }

    def test_scalar_scalar(self):
        name = "zhangli_scalar_scalar"

        u = 0
        beta = 0.5
        H = (0, 0, 1e5)
        Ms = 1e6

        mesh = df.Mesh(region=self.region, cell=self.cell)

        system = mm.System(name=name)
        system.energy = mm.Zeeman(H=H)
        system.dynamics = mm.ZhangLi(u=u, beta=beta)
        system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.2e-9, n=50)

        # u is zero, nothing should change.
        value = system.m(mesh.region.center)
        assert np.linalg.norm(np.cross(value, (0, 0.1 * Ms, Ms))) < 1e-3

        system.dynamics -= mm.ZhangLi(u=u, beta=beta)
        # empty dynamics is not allowed
        system.dynamics += mm.Damping(alpha=1)

        td.drive(system, t=0.2e-9, n=50)

    def test_time_scalar_scalar(self):
        name = "zhangli_scalar_scalar"

        u = 0
        beta = 0.5
        H = (0, 0, 1e5)
        Ms = 1e6

        mesh = df.Mesh(region=self.region, cell=self.cell)

        system = mm.System(name=name)
        system.energy = mm.Zeeman(H=H)
        system.dynamics = mm.ZhangLi(u=u, beta=beta)
        system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.2e-9, n=50)

        # u is zero, nothing should change.
        value = system.m(mesh.region.center)
        assert np.linalg.norm(np.cross(value, (0, 0.1 * Ms, Ms))) < 1e-3

        system.dynamics -= mm.ZhangLi(u=u, beta=beta)
        # empty dynamics is not allowed
        system.dynamics += mm.Damping(alpha=1)

        td.drive(system, t=0.2e-9, n=50)

        # Check if it runs.

        # time-dependence - function
        def time_dep(t):
            return np.sin(t * 1e10)

        system.dynamics = mm.ZhangLi(u=u, beta=beta, func=time_dep, dt=1e-13)
        system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

        td.drive(system, t=0.2e-9, n=50)

        # u is zero, nothing should change.
        value = system.m(mesh.region.center)
        assert np.linalg.norm(np.cross(value, (0, 0.1 * Ms, Ms))) < 1e-3

        # time-dependence - tcl strings
        tcl_strings = {}
        tcl_strings["script"] = """proc TimeFunction { total_time } {
            return $total_time
        }
        """
        tcl_strings["script_args"] = "total_time"
        tcl_strings["script_name"] = "TimeFunction"

        system.dynamics = mm.ZhangLi(u=u, beta=beta, tcl_strings=tcl_strings)
        system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

        td.drive(system, t=0.2e-9, n=50)

        # u is zero, nothing should change.
        value = system.m(mesh.region.center)
        assert np.linalg.norm(np.cross(value, (0, 0.1 * Ms, Ms))) < 1e-3

        self.calculator.delete(system)

    def test_dict_scalar(self):
        name = "zhangli_dict_scalar"

        H = (0, 0, 1e6)
        u = {"r1": 0, "r2": 1}
        beta = 0.5
        Ms = 1e6

        mesh = df.Mesh(region=self.region, cell=self.cell, subregions=self.subregions)

        system = mm.System(name=name)
        system.energy = mm.Zeeman(H=H)
        system.dynamics = mm.ZhangLi(u=u, beta=beta)
        system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.2e-9, n=50)

        # u=0 region
        value = system.m((1e-9, -4e-9, 3e-9))
        assert np.linalg.norm(np.cross(value, (0, 0.1 * Ms, Ms))) < 1e-3

        # u!=0 region
        value = system.m((1e-9, 4e-9, 3e-9))
        assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) > 1

        # time-dependence - function
        def time_dep(t):
            return np.sin(t * 1e10)

        # - default value set to 0
        # - vector-valued dictionary elements
        u = {"r2": (1, 0, 0)}
        system.dynamics = mm.ZhangLi(u=u, beta=beta, func=time_dep, dt=1e-13)
        system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

        td.drive(system, t=0.2e-9, n=50)

        # u=0 region
        value = system.m((1e-9, -4e-9, 3e-9))
        assert np.linalg.norm(np.cross(value, (0, 0.1 * Ms, Ms))) < 1e-3

        # u!=0 region
        value = system.m((1e-9, 4e-9, 3e-9))
        assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) > 1

        # time-dependence - tcl strings
        tcl_strings = {}
        tcl_strings["script"] = """proc TimeFunction { total_time } {
            return $total_time
        }
        """
        tcl_strings["script_args"] = "total_time"
        tcl_strings["script_name"] = "TimeFunction"

        u = {"r1": 0, "r2": 1}
        system.dynamics = mm.ZhangLi(u=u, beta=beta, tcl_strings=tcl_strings)
        system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

        td.drive(system, t=0.2e-9, n=50)

        # u=0 region
        value = system.m((1e-9, -4e-9, 3e-9))
        assert np.linalg.norm(np.cross(value, (0, 0.1 * Ms, Ms))) < 1e-3

        # u!=0 region
        value = system.m((1e-9, 4e-9, 3e-9))
        assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) > 1

        self.calculator.delete(system)

    def test_field_scalar(self):
        name = "zhangli_field_scalar"

        mesh = df.Mesh(region=self.region, cell=self.cell)

        def u_fun(pos):
            x, y, z = pos
            if y <= 0:
                return 0
            else:
                return 1

        H = (0, 0, 1e6)
        u = df.Field(mesh, nvdim=1, value=u_fun)
        beta = 0.5
        Ms = 1e6

        system = mm.System(name=name)
        system.energy = mm.Zeeman(H=H)
        system.dynamics = mm.ZhangLi(u=u, beta=beta)
        system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.2e-9, n=50)

        # u=0 region
        value = system.m((1e-9, -4e-9, 3e-9))
        assert np.linalg.norm(np.cross(value, (0, 0.1 * Ms, Ms))) < 1e-3

        # u!=0 region
        value = system.m((1e-9, 4e-9, 3e-9))
        assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) > 1

        # time-dependence - function
        def time_dep(t):
            return np.sin(t * 1e10)

        system.dynamics = mm.ZhangLi(u=u, beta=beta, func=time_dep, dt=1e-13)
        system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

        td.drive(system, t=0.2e-9, n=50)

        # u=0 region
        value = system.m((1e-9, -4e-9, 3e-9))
        assert np.linalg.norm(np.cross(value, (0, 0.1 * Ms, Ms))) < 1e-3

        # u!=0 region
        value = system.m((1e-9, 4e-9, 3e-9))
        assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) > 1

        # time-dependence - tcl strings
        tcl_strings = {}
        tcl_strings["script"] = """proc TimeFunction { total_time } {
            return $total_time
        }
        """
        tcl_strings["script_args"] = "total_time"
        tcl_strings["script_name"] = "TimeFunction"

        def u_fun(pos):
            x, y, z = pos
            if y <= 0:
                return (0, 0, 0)
            else:
                return (1, 0, 0)

        u = df.Field(mesh, nvdim=3, value=u_fun)
        system.dynamics = mm.ZhangLi(u=u, beta=beta, tcl_strings=tcl_strings)
        system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

        td.drive(system, t=0.2e-9, n=50)

        # u=0 region
        value = system.m((1e-9, -4e-9, 3e-9))
        assert np.linalg.norm(np.cross(value, (0, 0.1 * Ms, Ms))) < 1e-3

        # u!=0 region
        value = system.m((1e-9, 4e-9, 3e-9))
        assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) > 1

        self.calculator.delete(system)

    def test_vector_scalar(self):
        """
        Domain wall in a strip oriented in x (y) direction. Current is applied in x (y)
        direction. The average mz component before and after moving the domain walls
        are compared. There is no check for mx and my because we do not ensure same
        chirality of the domain walls.
        """
        md = self.calculator.MinDriver()
        td = self.calculator.TimeDriver()

        pa = 200e-9
        pb = 20e-9
        Ms = 5.8e5

        def init_m(direction):
            def _inner(p):
                if p[direction] < 70e-9:
                    return (0.1, 0.1, -1)
                return (0.1, 0.1, 1)

            return _inner

        A = 15e-12
        K = 0.5e6
        u = (0, 0, 1)

        system_x = mm.System(name="strip_x")
        system_x.energy = mm.Exchange(A=A) + mm.UniaxialAnisotropy(K=K, u=u)
        mesh = df.Mesh(p1=(0, 0, 0), p2=(pa, pb, 5e-9), cell=(5e-9, 5e-9, 5e-9))
        system_x.m = df.Field(mesh, nvdim=3, value=init_m(0), norm=Ms)
        md.drive(system_x)

        system_y = mm.System(name="strip_y")
        system_y.energy = mm.Exchange(A=A) + mm.UniaxialAnisotropy(K=K, u=u)
        mesh = df.Mesh(p1=(0, 0, 0), p2=(pb, pa, 5e-9), cell=(5e-9, 5e-9, 5e-9))
        system_y.m = df.Field(mesh, nvdim=3, value=init_m(1), norm=Ms)
        md.drive(system_y)

        assert system_x.m.orientation.z.mean() > 0.25
        assert np.allclose(
            system_x.m.orientation.z.mean(), system_y.m.orientation.z.mean()
        )

        system_x.dynamics = (
            mm.Precession(gamma0=mm.consts.gamma0)
            + mm.Damping(alpha=0.3)
            + mm.ZhangLi(u=(200, 0, 0), beta=0.5)
        )
        td.drive(system_x, t=0.4e-9, n=1)

        system_y.dynamics = (
            mm.Precession(gamma0=mm.consts.gamma0)
            + mm.Damping(alpha=0.3)
            + mm.ZhangLi(u=(0, 200, 0), beta=0.5)
        )
        td.drive(system_y, t=0.4e-9, n=1)

        assert system_x.m.orientation.z.mean() < -0.25
        assert np.allclose(
            system_x.m.orientation.z.mean(), system_y.m.orientation.z.mean()
        )

        self.calculator.delete(system_x)
        self.calculator.delete(system_y)
