import os

import discretisedfield as df
import micromagneticmodel as mm


def test_outputstep(calculator):
    name = 'output_step'

    p1 = (0, 0, 0)
    p2 = (5e-9, 5e-9, 5e-9)
    n = (2, 2, 2)
    Ms = 1e6
    A = 1e-12
    H = (0, 0, 1e6)
    region = df.Region(p1=p1, p2=p2)
    mesh = df.Mesh(region=region, n=n)

    system = mm.System(name=name)
    system.energy = mm.Exchange(A=A) + mm.Zeeman(H=H)
    system.m = df.Field(mesh, dim=3, value=(0, 0.1, 1), norm=Ms)

    md = calculator.MinDriver()
    md.drive(system, output_step=True)

    dirname = os.path.join(name, 'drive-0')
    assert os.path.exists(dirname)

    assert len(system.table.data.index) > 1

    calculator.delete(system)
