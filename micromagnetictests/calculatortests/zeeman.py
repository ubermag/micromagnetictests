import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import pytest


class TestZeeman:
    @pytest.fixture(autouse=True)
    def _setup_calculator(self, calculator):
        self.calculator = calculator

    def setup(self):
        p1 = (-10e-9, -5e-9, -3e-9)
        p2 = (10e-9, 5e-9, 3e-9)
        self.region = df.Region(p1=p1, p2=p2)
        self.cell = (1e-9, 1e-9, 1e-9)
        self.subregions = {'r1': df.Region(p1=(-10e-9, -5e-9, -3e-9),
                                           p2=(10e-9, 0, 3e-9)),
                           'r2': df.Region(p1=(-10e-9, 0, -3e-9),
                                           p2=(10e-9, 5e-9, 3e-9))}

    def test_vector(self):
        name = 'zeeman_vector'

        H = (0, 0, 1e6)
        Ms = 1e6

        system = mm.System(name=name)

        # time-independent
        system.energy = mm.Zeeman(H=H)

        mesh = df.Mesh(region=self.region, cell=self.cell)
        system.m = df.Field(mesh, dim=3, value=(1, 1, 1), norm=Ms)

        md = self.calculator.MinDriver()
        md.drive(system)

        value = system.m(mesh.region.random_point())
        assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) < 1e-3

        # time-dependent - sin
        system.energy = mm.Zeeman(H=H, func='sin', f=1e9, t0=1e-12)

        mesh = df.Mesh(region=self.region, cell=self.cell)
        system.m = df.Field(mesh, dim=3, value=(1, 1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.1e-9, n=20)

        # time-dependent - sinc
        system.energy = mm.Zeeman(H=H, func='sinc', f=1e9, t0=0)

        mesh = df.Mesh(region=self.region, cell=self.cell)
        system.m = df.Field(mesh, dim=3, value=(1, 1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.1e-9, n=20)

        self.calculator.delete(system)

        # time-dependent - function
        def t_func(t):
            if t < 1e-10:
                return 1
            elif t < 5e-10:
                return (5e-10 - t) / 4e-10
            else:
                return 0

        system.energy = mm.Zeeman(H=H, func=t_func, dt=1e-13)

        mesh = df.Mesh(region=self.region, cell=self.cell)
        system.m = df.Field(mesh, dim=3, value=(1, 1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.1e-9, n=20)

        # time-dependent - two terms
        f = 10e9

        def cos_wave(t):
            return np.cos(2 * np.pi * f * t)

        def sin_wave(t):
            return np.sin(2 * np.pi * f * t)

        H_x = (1e6, 0, 0)
        H_y = (0, 1e6, 0)
        system.energy = (
            mm.Zeeman(H=H_x, func=cos_wave, dt=5e-12, name='xdir')
            + mm.Zeeman(H=H_y, func='sin', t0=0, f=f, name='ydir'))
        td.drive(system, t=0.1e-9, n=100)

        assert not np.allclose(system.table.data['Bx_xdir'], 0)
        assert np.allclose(system.table.data['By_xdir'], 0)

        assert np.allclose(system.table.data['Bx_ydir'], 0)
        assert not np.allclose(system.table.data['By_ydir'], 0)

        assert np.isclose(np.max(system.table.data['Bx_xdir']),
                          np.max(system.table.data['By_ydir']))

        H_x = (1e6, 0, 0)
        H_y = (0, 1e6, 0)
        system.energy = (
            mm.Zeeman(H=H_x, func=cos_wave, dt=5e-12, name='xdir')
            + mm.Zeeman(H=H_y, func=sin_wave, dt=5e-12, name='ydir'))
        td.drive(system, t=0.1e-9, n=100)

        assert not np.allclose(system.table.data['Bx_xdir'], 0)
        assert np.allclose(system.table.data['By_xdir'], 0)

        assert np.allclose(system.table.data['Bx_ydir'], 0)
        assert not np.allclose(system.table.data['By_ydir'], 0)

        assert np.isclose(np.max(system.table.data['Bx_xdir']),
                          np.max(system.table.data['By_ydir']))

        # time-dependent - tcl strings
        tcl_strings = {}
        tcl_strings['script'] = '''proc TimeFunction { total_time } {
            set Hx [expr {sin($total_time * 1e10)}]
            set dHx [expr {1e10 * cos($total_time * 1e10)}]
            return [list $Hx 0 0 $dHx 0 0]
        }
        '''
        tcl_strings['energy'] = 'Oxs_ScriptUZeeman'
        tcl_strings['script_args'] = 'total_time'
        tcl_strings['script_name'] = 'TimeFunction'

        system.energy = mm.Zeeman(H=H, tcl_strings=tcl_strings)

        mesh = df.Mesh(region=self.region, cell=self.cell)
        system.m = df.Field(mesh, dim=3, value=(1, 1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.1e-9, n=20)

        self.calculator.delete(system)

    def test_dict(self):
        name = 'zeeman_dict'

        H = {'r1': (1e5, 0, 0), 'r2': (0, 0, 1e5)}
        Ms = 1e6

        system = mm.System(name=name)
        system.energy = mm.Zeeman(H=H)

        mesh = df.Mesh(region=self.region, cell=self.cell,
                       subregions=self.subregions)
        system.m = df.Field(mesh, dim=3, value=(1, 1, 1), norm=Ms)

        md = self.calculator.MinDriver()
        md.drive(system)

        assert np.linalg.norm(np.subtract(system.m['r1'].average,
                                          (Ms, 0, 0))) < 1

        assert np.linalg.norm(np.subtract(system.m['r2'].average,
                                          (0, 0, Ms))) < 1

        # time-dependent - sin
        system.energy = mm.Zeeman(H=H, func='sin', f=1e9, t0=1e-12)

        mesh = df.Mesh(region=self.region, cell=self.cell,
                       subregions=self.subregions)
        system.m = df.Field(mesh, dim=3, value=(1, 1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.1e-9, n=20)

        # time-dependent - sinc
        system.energy = mm.Zeeman(H=H, func='sinc', f=1e9, t0=0)

        mesh = df.Mesh(region=self.region, cell=self.cell,
                       subregions=self.subregions)
        system.m = df.Field(mesh, dim=3, value=(1, 1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.1e-9, n=20)

        # time-dependent - function
        def t_func(t):
            if t < 1e-10:
                return 1
            elif t < 5e-10:
                return (5e-10 - t) / 4e-10
            else:
                return 0

        system.energy = mm.Zeeman(H=H, func=t_func, dt=1e-13)

        mesh = df.Mesh(region=self.region, cell=self.cell,
                       subregions=self.subregions)
        system.m = df.Field(mesh, dim=3, value=(1, 1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.1e-9, n=20)

        self.calculator.delete(system)

    def test_field(self):
        name = 'zeeman_field'

        def value_fun(pos):
            x, y, z = pos
            if x <= 0:
                return (1e6, 0, 0)
            else:
                return (0, 0, 1e6)

        mesh = df.Mesh(region=self.region, cell=self.cell)

        H = df.Field(mesh, dim=3, value=value_fun)
        Ms = 1e6

        system = mm.System(name=name)
        system.energy = mm.Zeeman(H=H)
        system.m = df.Field(mesh, dim=3, value=(0, 1, 0), norm=Ms)

        md = self.calculator.MinDriver()
        md.drive(system)

        value = system.m((-2e-9, -2e-9, -2e-9))
        assert np.linalg.norm(np.subtract(value, (Ms, 0, 0))) < 1e-3

        value = system.m((2e-9, 2e-9, 2e-9))
        assert np.linalg.norm(np.subtract(value, (0, 0, Ms))) < 1e-3

        # time-dependent - sin
        system.energy = mm.Zeeman(H=H, func='sin', f=1e9, t0=1e-12)

        mesh = df.Mesh(region=self.region, cell=self.cell)
        system.m = df.Field(mesh, dim=3, value=(1, 1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.1e-9, n=20)

        # time-dependent - sinc
        system.energy = mm.Zeeman(H=H, func='sinc', f=1e9, t0=0)

        mesh = df.Mesh(region=self.region, cell=self.cell)
        system.m = df.Field(mesh, dim=3, value=(1, 1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.1e-9, n=20)

        # time-dependent - function
        def t_func(t):
            omega = 2*np.pi * 1e9
            return [np.cos(omega * t), -np.sin(omega * t), 0,
                    np.sin(omega * t), np.cos(omega * t), 0,
                    0, 0, 1]

        system.energy = mm.Zeeman(H=H, func=t_func, dt=1e-13)

        mesh = df.Mesh(region=self.region, cell=self.cell)
        system.m = df.Field(mesh, dim=3, value=(1, 1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.1e-9, n=20)

        # time-dependent - tcl strings
        tcl_strings = {}
        tcl_strings['script'] = '''proc TimeFunction { total_time } {
            set PI [expr {4*atan(1.)}]
            set w [expr {1e9*2*$PI}]
            set ct [expr {cos($w*$total_time)}]
            set mct [expr {-1*$ct}]      ;# "mct" is "minus cosine (w)t"
            set st [expr {sin($w*$total_time)}]
            set mst [expr {-1*$st}]      ;# "mst" is "minus sine (w)t"
            return [list  $ct $mst  0 \
                          $st $ct   0 \
                          0   0   1 \
                          [expr {$w*$mst}] [expr {$w*$mct}] 0 \
                          [expr {$w*$ct}]  [expr {$w*$mst}] 0 \
                              0                0         0]
        }'''
        tcl_strings['energy'] = 'Oxs_TransformZeeman'
        tcl_strings['type'] = 'general'
        tcl_strings['script_args'] = 'total_time'
        tcl_strings['script_name'] = 'TimeFunction'

        system.energy = mm.Zeeman(H=H, tcl_strings=tcl_strings)

        mesh = df.Mesh(region=self.region, cell=self.cell)
        system.m = df.Field(mesh, dim=3, value=(1, 1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.1e-9, n=20)

        self.calculator.delete(system)
