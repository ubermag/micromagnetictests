"""Calculator tests"""
from .compute import TestCompute
from .cubicanisotropy import TestCubicAnisotropy
from .damping import TestDamping
from .demag import TestDemag
from .dirname import test_dirname
from .dmi import TestDMI
from .dynamics import TestDynamics
from .energy import TestEnergy
from .exchange import TestExchange
from .fixedsubregions import TestFixedSubregions
from .hysteresisdriver import (
    test_check_for_energy,
    test_simple_hysteresis_loop,
    test_stepped_hysteresis_loop,
)
from .info_file import test_info_file
from .mesh import TestMesh
from .mindriver import TestMinDriver
from .multiple_drives import test_multiple_drives
from .outputformat import test_format
from .outputstep import test_outputstep
from .precession import TestPrecession
from .relaxdriver import test_relax_check_for_energy, test_relaxdriver
from .rkky import TestRKKY
from .schedule import test_schedule
from .skyrmion import test_skyrmion
from .slonczewski import TestSlonczewski
from .stdprob3 import test_stdprob3
from .stdprob4 import test_stdprob4
from .stdprob5 import test_stdprob5
from .threads import TestThreads
from .timedriver import TestTimeDriver
from .uniaxialanisotropy import TestUniaxialAnisotropy
from .zeeman import TestZeeman
from .zhangli import TestZhangLi
