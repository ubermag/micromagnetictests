import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import pytest


class TestZhangLi:
    @pytest.fixture(autouse=True)
    def _setup_calculator(self, calculator):
        self.calculator = calculator

    def setup(self):
        p1 = (-5e-9, -5e-9, -3e-9)
        p2 = (5e-9, 5e-9, 3e-9)
        self.region = df.Region(p1=p1, p2=p2)
        self.cell = (1e-9, 1e-9, 3e-9)
        self.subregions = {'r1': df.Region(p1=(-5e-9, -5e-9, -3e-9),
                                           p2=(5e-9, 0, 3e-9)),
                           'r2': df.Region(p1=(-5e-9, 0, -3e-9),
                                           p2=(5e-9, 5e-9, 3e-9))}

    def test_scalar_scalar(self):
        name = 'zhangli_scalar_scalar'

        u = 0
        beta = 0.5
        H = (0, 0, 1e5)
        Ms = 1e6

        mesh = df.Mesh(region=self.region, cell=self.cell)

        system = mm.System(name=name)
        system.energy = mm.Zeeman(H=H)
        system.dynamics = mm.ZhangLi(u=u, beta=beta)
        system.m = df.Field(mesh, dim=3, value=(0, 0.1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.2e-9, n=50)

        # u is zero, nothing should change.
        value = system.m(mesh.region.random_point())
        assert np.linalg.norm(np.cross(value, (0, 0.1*Ms, Ms))) < 1e-3

        system.dynamics -= mm.ZhangLi(u=u, beta=beta)
        td.drive(system, t=0.2e-9, n=50)

        # Check if it runs.

        # time-dependence - function
        def time_dep(t):
            return np.sin(t * 1e10)

        system.dynamics = mm.ZhangLi(u=u, beta=beta, func=time_dep, dt=1e-13)
        system.m = df.Field(mesh, dim=3, value=(0, 0.1, 1), norm=Ms)

        td.drive(system, t=0.2e-9, n=50)

        # u is zero, nothing should change.
        value = system.m(mesh.region.random_point())
        assert np.linalg.norm(np.cross(value, (0, 0.1 * Ms, Ms))) < 1e-3

        # time-dependence - tcl strings
        tcl_strings = {}
        tcl_strings['script'] = '''proc TimeFunction { total_time } {
            return $total_time
        }
        '''
        tcl_strings['script_args'] = 'total_time'
        tcl_strings['script_name'] = 'TimeFunction'

        system.dynamics = mm.ZhangLi(u=u, beta=beta, tcl_strings=tcl_strings)
        system.m = df.Field(mesh, dim=3, value=(0, 0.1, 1), norm=Ms)

        td.drive(system, t=0.2e-9, n=50)

        # u is zero, nothing should change.
        value = system.m(mesh.region.random_point())
        assert np.linalg.norm(np.cross(value, (0, 0.1*Ms, Ms))) < 1e-3

        self.calculator.delete(system)

    def test_dict_scalar(self):
        name = 'zhangli_dict_scalar'

        H = (0, 0, 1e6)
        u = {'r1': 0, 'r2': 1}
        beta = 0.5
        Ms = 1e6

        mesh = df.Mesh(region=self.region, cell=self.cell,
                       subregions=self.subregions)

        system = mm.System(name=name)
        system.energy = mm.Zeeman(H=H)
        system.dynamics = mm.ZhangLi(u=u, beta=beta)
        system.m = df.Field(mesh, dim=3, value=(0, 0.1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.2e-9, n=50)

        # u=0 region
        value = system.m((1e-9, -4e-9, 3e-9))
        assert np.linalg.norm(np.cross(value, (0, 0.1*Ms, Ms))) < 1e-3

        # u!=0 region
        value = system.m((1e-9, 4e-9, 3e-9))
        assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) > 1

        # time-dependence - function
        def time_dep(t):
            return np.sin(t * 1e10)

        system.dynamics = mm.ZhangLi(u=u, beta=beta, func=time_dep, dt=1e-13)
        system.m = df.Field(mesh, dim=3, value=(0, 0.1, 1), norm=Ms)

        td.drive(system, t=0.2e-9, n=50)

        # u=0 region
        value = system.m((1e-9, -4e-9, 3e-9))
        assert np.linalg.norm(np.cross(value, (0, 0.1*Ms, Ms))) < 1e-3

        # u!=0 region
        value = system.m((1e-9, 4e-9, 3e-9))
        assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) > 1

        # time-dependence - tcl strings
        tcl_strings = {}
        tcl_strings['script'] = '''proc TimeFunction { total_time } {
            return $total_time
        }
        '''
        tcl_strings['script_args'] = 'total_time'
        tcl_strings['script_name'] = 'TimeFunction'

        system.dynamics = mm.ZhangLi(u=u, beta=beta, tcl_strings=tcl_strings)
        system.m = df.Field(mesh, dim=3, value=(0, 0.1, 1), norm=Ms)

        td.drive(system, t=0.2e-9, n=50)

        # u=0 region
        value = system.m((1e-9, -4e-9, 3e-9))
        assert np.linalg.norm(np.cross(value, (0, 0.1*Ms, Ms))) < 1e-3

        # u!=0 region
        value = system.m((1e-9, 4e-9, 3e-9))
        assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) > 1

        self.calculator.delete(system)

    def test_field_scalar(self):
        name = 'zhangli_field_scalar'

        mesh = df.Mesh(region=self.region, cell=self.cell)

        def u_fun(pos):
            x, y, z = pos
            if y <= 0:
                return 0
            else:
                return 1

        H = (0, 0, 1e6)
        u = df.Field(mesh, dim=1, value=u_fun)
        beta = 0.5
        Ms = 1e6

        system = mm.System(name=name)
        system.energy = mm.Zeeman(H=H)
        system.dynamics = mm.ZhangLi(u=u, beta=beta)
        system.m = df.Field(mesh, dim=3, value=(0, 0.1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.2e-9, n=50)

        # u=0 region
        value = system.m((1e-9, -4e-9, 3e-9))
        assert np.linalg.norm(np.cross(value, (0, 0.1*Ms, Ms))) < 1e-3

        # u!=0 region
        value = system.m((1e-9, 4e-9, 3e-9))
        assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) > 1

        # time-dependence - function
        def time_dep(t):
            return np.sin(t * 1e10)

        system.dynamics = mm.ZhangLi(u=u, beta=beta, func=time_dep, dt=1e-13)
        system.m = df.Field(mesh, dim=3, value=(0, 0.1, 1), norm=Ms)

        td.drive(system, t=0.2e-9, n=50)

        # u=0 region
        value = system.m((1e-9, -4e-9, 3e-9))
        assert np.linalg.norm(np.cross(value, (0, 0.1*Ms, Ms))) < 1e-3

        # u!=0 region
        value = system.m((1e-9, 4e-9, 3e-9))
        assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) > 1

        # time-dependence - tcl strings
        tcl_strings = {}
        tcl_strings['script'] = '''proc TimeFunction { total_time } {
            return $total_time
        }
        '''
        tcl_strings['script_args'] = 'total_time'
        tcl_strings['script_name'] = 'TimeFunction'

        system.dynamics = mm.ZhangLi(u=u, beta=beta, tcl_strings=tcl_strings)
        system.m = df.Field(mesh, dim=3, value=(0, 0.1, 1), norm=Ms)

        td.drive(system, t=0.2e-9, n=50)

        # u=0 region
        value = system.m((1e-9, -4e-9, 3e-9))
        assert np.linalg.norm(np.cross(value, (0, 0.1*Ms, Ms))) < 1e-3

        # u!=0 region
        value = system.m((1e-9, 4e-9, 3e-9))
        assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) > 1

        self.calculator.delete(system)
