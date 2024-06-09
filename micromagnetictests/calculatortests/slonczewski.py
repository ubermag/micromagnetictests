import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import pytest


class TestSlonczewski:
    @pytest.fixture(autouse=True)
    def _setup_calculator(self, calculator):
        self.calculator = calculator

    def setup_method(self):
        p1 = (-5e-9, -5e-9, -3e-9)
        p2 = (5e-9, 5e-9, 3e-9)
        self.region = df.Region(p1=p1, p2=p2)
        self.n = (2, 2, 2)
        self.subregions = {
            "r1": df.Region(p1=(-5e-9, -5e-9, -3e-9), p2=(5e-9, 0, 3e-9)),
            "r2": df.Region(p1=(-5e-9, 0, -3e-9), p2=(5e-9, 5e-9, 3e-9)),
        }

    def test_single_values(self):
        name = "slonczewski_scalar_values"

        J = 1e12
        mp = (1, 0, 0)
        P = 0.4
        Lambda = 2
        eps_prime = 0
        H = (0, 0, 1e6)
        Ms = 1e6

        mesh = df.Mesh(region=self.region, n=self.n)

        system = mm.System(name=name)
        system.energy = mm.Zeeman(H=H)
        system.dynamics = mm.Slonczewski(
            J=J, mp=mp, P=P, Lambda=Lambda, eps_prime=eps_prime
        )
        system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.2e-9, n=20)

        # Check if it runs.

        # remove current -> needs different evolver
        system.dynamics -= mm.Slonczewski(
            J=J, mp=mp, P=P, Lambda=Lambda, eps_prime=eps_prime
        )
        # Damping factor is introduced because of the dynamics
        # check function as empty system.dynamics will fail the test
        system.dynamics += mm.Damping(alpha=1)

        td.drive(system, t=0.2e-9, n=20)

        # Check if it runs.

        # time-dependence - function
        def time_dep(t):
            return np.sin(t * 1e10)

        system.dynamics = mm.Slonczewski(
            J=J, mp=mp, P=P, Lambda=Lambda, eps_prime=eps_prime, dt=1e-13, func=time_dep
        )
        system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

        td.drive(system, t=0.2e-9, n=50)

        # Check if it runs.

        # time-dependence - tcl strings
        tcl_strings = {}
        tcl_strings["script"] = """proc TimeFunction { total_time } {
            return $total_time
        }
        """
        tcl_strings["script_args"] = "total_time"
        tcl_strings["script_name"] = "TimeFunction"

        system.dynamics = mm.Slonczewski(
            J=J, mp=mp, P=P, Lambda=Lambda, eps_prime=eps_prime, tcl_strings=tcl_strings
        )
        system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

        td.drive(system, t=0.2e-9, n=50)

        # Check if it runs.

        self.calculator.delete(system)

    def test_single_values_finite_temperature(self):
        name = "slonczewski_scalar_values_finite_temperature"

        J = 1e12
        mp = (1, 0, 0)
        P = 0.4
        Lambda = 2
        eps_prime = 0
        H = (0, 0, 1e6)
        Ms = 1e6

        mesh = df.Mesh(region=self.region, n=self.n)

        system = mm.System(name=name)
        system.energy = mm.Zeeman(H=H)
        system.dynamics = mm.Slonczewski(
            J=J, mp=mp, P=P, Lambda=Lambda, eps_prime=eps_prime
        )
        system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)
        system.T = 10

        evolver = self.calculator.Xf_ThermSpinXferEvolver()
        td = self.calculator.TimeDriver(evolver=evolver)
        td.drive(system, t=0.2e-11, n=20)

        # Check if it runs.

        # time-dependence - function
        def time_dep(t):
            return np.sin(t * 1e10)

        system.dynamics = mm.Slonczewski(
            J=J, mp=mp, P=P, Lambda=Lambda, eps_prime=eps_prime, dt=1e-13, func=time_dep
        )
        system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

        td.drive(system, t=0.2e-11, n=50)

        # Check if it runs.

        # time-dependence - tcl strings
        tcl_strings = {}
        tcl_strings["script"] = """proc TimeFunction { total_time } {
            return $total_time
        }
        """
        tcl_strings["script_args"] = "total_time"
        tcl_strings["script_name"] = "TimeFunction"

        system.dynamics = mm.Slonczewski(
            J=J, mp=mp, P=P, Lambda=Lambda, eps_prime=eps_prime, tcl_strings=tcl_strings
        )
        system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

        td.drive(system, t=0.2e-11, n=50)

        # Check if it runs.

        self.calculator.delete(system)

    def test_dict_values(self):
        name = "slonczewski_scalar_values"

        J = {"r1": 1e12, "r2": 5e12}
        mp = {"r1": (0, 0, 1), "r2": (0, 1, 0)}
        P = {"r1": 0.4, "r2": 0.35}
        Lambda = {"r1": 2, "r2": 1.5}
        eps_prime = {"r1": 0, "r2": 1}
        H = (0, 0, 1e6)
        Ms = 1e6

        mesh = df.Mesh(region=self.region, n=self.n, subregions=self.subregions)

        system = mm.System(name=name)
        system.energy = mm.Zeeman(H=H)
        system.dynamics = mm.Slonczewski(
            J=J, mp=mp, P=P, Lambda=Lambda, eps_prime=eps_prime
        )
        system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.2e-9, n=20)

        # Check if it runs.

        # time-dependence - function
        def time_dep(t):
            return np.sin(t * 1e10)

        system.dynamics = mm.Slonczewski(
            J=J, mp=mp, P=P, Lambda=Lambda, eps_prime=eps_prime, dt=1e-13, func=time_dep
        )
        system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

        td.drive(system, t=0.2e-9, n=50)

        # Check if it runs.

        # time-dependence - tcl strings
        tcl_strings = {}
        tcl_strings["script"] = """proc TimeFunction { total_time } {
            return $total_time
        }
        """
        tcl_strings["script_args"] = "total_time"
        tcl_strings["script_name"] = "TimeFunction"

        system.dynamics = mm.Slonczewski(
            J=J, mp=mp, P=P, Lambda=Lambda, eps_prime=eps_prime, tcl_strings=tcl_strings
        )
        system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

        td.drive(system, t=0.2e-9, n=50)

        # Check if it runs.

        self.calculator.delete(system)

    def test_field_values(self):
        name = "slonczewski_scalar_values"

        mesh = df.Mesh(region=self.region, n=self.n)

        J = df.Field(mesh, nvdim=1, value=0.5e12)
        mp = df.Field(mesh, nvdim=3, value=(1, 0, 0))
        P = df.Field(mesh, nvdim=1, value=0.5)
        Lambda = df.Field(mesh, nvdim=1, value=2)
        eps_prime = df.Field(mesh, nvdim=1, value=1)
        H = (0, 0, 1e6)
        Ms = 1e6

        system = mm.System(name=name)
        system.energy = mm.Zeeman(H=H)
        system.dynamics = mm.Slonczewski(
            J=J, mp=mp, P=P, Lambda=Lambda, eps_prime=eps_prime
        )
        system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

        td = self.calculator.TimeDriver()
        td.drive(system, t=0.2e-9, n=20)

        # Check if it runs.

        # time-dependence - function
        def time_dep(t):
            return np.sin(t * 1e10)

        system.dynamics = mm.Slonczewski(
            J=J, mp=mp, P=P, Lambda=Lambda, eps_prime=eps_prime, dt=1e-13, func=time_dep
        )
        system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

        td.drive(system, t=0.2e-9, n=50)

        # Check if it runs.

        # time-dependence - tcl strings
        tcl_strings = {}
        tcl_strings["script"] = """proc TimeFunction { total_time } {
            return $total_time
        }
        """
        tcl_strings["script_args"] = "total_time"
        tcl_strings["script_name"] = "TimeFunction"

        system.dynamics = mm.Slonczewski(
            J=J, mp=mp, P=P, Lambda=Lambda, eps_prime=eps_prime, tcl_strings=tcl_strings
        )
        system.m = df.Field(mesh, nvdim=3, value=(0, 0.1, 1), norm=Ms)

        td.drive(system, t=0.2e-9, n=50)

        # Check if it runs.

        self.calculator.delete(system)
