import discretisedfield as df
import micromagneticmodel as mm
import pytest


def test_format(calculator):
    name = 'output_format'

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

    # test if it runs
    md.drive(system)  # 'bin8' (default)
    md.drive(system, ovf_format='bin4')
    md.drive(system, ovf_format='txt')
    with pytest.raises(ValueError):
        md.drive(system, ovf_format='unknown')

    calculator.delete(system)
