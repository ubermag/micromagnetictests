import discretisedfield as df
import micromagneticmodel as mm


def test_stdprob5(calculator):
    name = 'stdprob5'

    # Geometry
    lx = 100e-9  # x dimension of the sample(m)
    ly = 100e-9  # y dimension of the sample (m)
    lz = 10e-9  # sample thickness (m)

    # Material (permalloy) parameters
    Ms = 8e5  # saturation magnetisation (A/m)
    A = 1.3e-11  # exchange energy constant (J/m)

    # Dynamics (LLG + ZhangLi equation) parameters
    gamma0 = 2.211e5  # gyromagnetic ratio (m/As)
    alpha = 0.1  # Gilbert damping
    ux = -72.35  # velocity in x direction
    beta = 0.05  # non-adiabatic STT parameter

    system = mm.System(name=name)
    p1 = (0, 0, 0)
    p2 = (lx, ly, lz)
    cell = (5e-9, 5e-9, 5e-9)
    region = df.Region(p1=p1, p2=p2)
    mesh = df.Mesh(region=region, cell=cell)
    system.energy = mm.Exchange(A=A) + mm.Demag()

    def m_vortex(pos):
        x, y, _ = pos[0]/1e-9-50, pos[1]/1e-9-50, pos[2]/1e-9
        return (-y, x, 10)

    system.m = df.Field(mesh, dim=3, value=m_vortex, norm=Ms)

    md = calculator.MinDriver()
    md.drive(system)

    system.dynamics += (mm.Precession(gamma0=gamma0) +
                        mm.Damping(alpha=alpha) + mm.ZhangLi(u=ux, beta=beta))

    td = calculator.TimeDriver()
    td.drive(system, t=8e-9, n=100)

    mx = system.table.data['mx'].values

    assert -0.35 < mx.min() < -0.30
    assert -0.03 < mx.max() < 0

    calculator.delete(system)
