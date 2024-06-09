import random

import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import pytest


class TestDMI:
    @pytest.fixture(autouse=True)
    def _setup_calculator(self, calculator):
        self.calculator = calculator

    def setup_method(self):
        p1 = (-100e-9, 0, 0)
        p2 = (100e-9, 1e-9, 1e-9)
        self.region = df.Region(p1=p1, p2=p2)
        self.cell = (1e-9, 1e-9, 1e-9)
        self.subregions = {
            "r1": df.Region(p1=(-100e-9, 0, 0), p2=(0, 1e-9, 1e-9)),
            "r2": df.Region(p1=(0, 0, 0), p2=(100e-9, 1e-9, 1e-9)),
        }

    def random_m(self, pos):
        return [2 * random.random() - 1 for i in range(3)]

    def test_scalar(self):
        name = "dmi_scalar"

        D = 1e-3
        Ms = 1e6

        system = mm.System(name=name)
        system.energy = mm.DMI(D=D, crystalclass="Cnv_z")

        mesh = df.Mesh(region=self.region, cell=self.cell)
        system.m = df.Field(mesh, nvdim=3, value=self.random_m, norm=Ms)

        if hasattr(self.calculator, "RelaxDriver"):
            system.dynamics = mm.Damping(alpha=0.5)
            md = self.calculator.RelaxDriver()
            with pytest.raises(RuntimeError):
                md.drive(system)
            system.energy += mm.Exchange(A=1e-21)
        else:
            md = self.calculator.MinDriver()

        md.drive(system)

        # There are 4N cells in the mesh. Because of that the average should be
        # 0.
        assert np.linalg.norm(system.m.mean()) < 1

        self.calculator.delete(system)

    def test_dict(self):
        name = "dmi_dict"

        D = {"r1": 0, "r2": 1e-3, "default": 2e-3}
        Ms = 1e6

        system = mm.System(name=name)
        system.energy = mm.DMI(D=D, crystalclass="Cnv_z")

        mesh = df.Mesh(region=self.region, cell=self.cell, subregions=self.subregions)
        system.m = df.Field(mesh, nvdim=3, value=self.random_m, norm=Ms)

        if hasattr(self.calculator, "RelaxDriver"):
            system.dynamics = mm.Damping(alpha=0.5)
            md = self.calculator.RelaxDriver()
            with pytest.raises(RuntimeError):
                md.drive(system)
            system.energy += mm.Exchange(A=1e-21)
        else:
            md = self.calculator.MinDriver()
        md.drive(system)

        assert np.linalg.norm(system.m["r1"].mean()) > 1
        # There are 4N cells in the region with D!=0. Because of that
        # the average should be 0.
        assert np.linalg.norm(system.m["r2"].mean()) < 1

        self.calculator.delete(system)

    @pytest.mark.filterwarnings("ignore:Use of `Cnv` is deprecated:FutureWarning")
    @pytest.mark.filterwarnings("ignore:Use of `D2d` is deprecated:FutureWarning")
    def test_crystalclass(self):
        name = "dmi_crystalclass"

        D = 1e-3
        Ms = 1e6

        for crystalclass in [
            "Cnv_x",
            "Cnv_y",
            "Cnv_z",
            "T",
            "O",
            "D2d_x",
            "D2d_y",
            "D2d_z",
            "Cnv",
            "D2d",  # legacy crystalclass names
        ]:
            system = mm.System(name=name)
            system.energy = mm.DMI(D=D, crystalclass=crystalclass)

            if crystalclass.endswith(("x", "y")):
                mesh = df.Mesh(
                    p1=(0, 0, -100e-9), p2=(1e-9, 1e-9, 100e-9), cell=self.cell
                )
            else:
                mesh = df.Mesh(region=self.region, cell=self.cell)

            system.m = df.Field(mesh, nvdim=3, value=self.random_m, norm=Ms)

            md = self.calculator.MinDriver()
            md.drive(system)

            # There are 4N cells in the mesh. Because of that the
            # average should be 0.
            assert np.linalg.norm(system.m.mean()) < 1

        self.calculator.delete(system)

    @pytest.mark.filterwarnings("ignore:Use of `Cnv` is deprecated:FutureWarning")
    @pytest.mark.filterwarnings("ignore:Use of `D2d` is deprecated:FutureWarning")
    def test_crystalclass_init(self):
        name = "dmi_crystalclass"

        D = 1e-3
        Ms = 1e6

        mumax3_cc = ["Cnv_z", "T", "O", "Cnv"]

        for crystalclass in [
            "Cnv_x",
            "Cnv_y",
            "Cnv_z",
            "T",
            "O",
            "D2d_x",
            "D2d_y",
            "D2d_z",
            "Cnv",
            "D2d",  # legacy crystalclass names
        ]:
            system = mm.System(name=name)
            system.energy = mm.DMI(D=D, crystalclass=crystalclass)

            if crystalclass.endswith(("x", "y")):
                mesh = df.Mesh(
                    p1=(0, 0, -100e-9), p2=(1e-9, 1e-9, 100e-9), cell=self.cell
                )
            else:
                mesh = df.Mesh(region=self.region, cell=self.cell)

            system.m = df.Field(mesh, nvdim=3, value=self.random_m, norm=Ms)

            md = self.calculator.MinDriver()
            if hasattr(self.calculator, "RelaxDriver"):
                system.energy += mm.Exchange(A=1e-21)
                if crystalclass not in mumax3_cc:
                    with pytest.raises(ValueError):
                        md.drive(system)
                else:
                    md.drive(system)
            else:
                md.drive(system)

        self.calculator.delete(system)
