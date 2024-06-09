"""Calculator tests"""

from .compute import TestCompute as TestCompute
from .cubicanisotropy import TestCubicAnisotropy as TestCubicAnisotropy
from .damping import TestDamping as TestDamping
from .demag import TestDemag as TestDemag
from .dirname import test_dirname as test_dirname
from .dmi import TestDMI as TestDMI
from .dynamics import TestDynamics as TestDynamics
from .energy import TestEnergy as TestEnergy
from .exchange import TestExchange as TestExchange
from .fixedsubregions import TestFixedSubregions as TestFixedSubregions
from .hysteresisdriver import Ms as Ms
from .hysteresisdriver import system as system
from .hysteresisdriver import (
    test_hysteresis_check_for_energy as test_hysteresis_check_for_energy,
)
from .hysteresisdriver import test_simple_hysteresis_loop as test_simple_hysteresis_loop
from .hysteresisdriver import (
    test_stepped_hysteresis_loop as test_stepped_hysteresis_loop,
)
from .info_file import test_info_file as test_info_file
from .mesh import TestMesh as TestMesh
from .mindriver import TestMinDriver as TestMinDriver
from .multiple_drives import test_multiple_drives as test_multiple_drives
from .outputformat import test_format as test_format
from .outputstep import test_outputstep as test_outputstep
from .precession import TestPrecession as TestPrecession
from .relaxdriver import test_relax_check_for_energy as test_relax_check_for_energy
from .relaxdriver import test_relaxdriver as test_relaxdriver
from .rkky import TestRKKY as TestRKKY
from .schedule import test_schedule as test_schedule
from .skyrmion import test_skyrmion as test_skyrmion
from .slonczewski import TestSlonczewski as TestSlonczewski
from .stdprob3 import test_stdprob3 as test_stdprob3
from .stdprob4 import test_stdprob4 as test_stdprob4
from .stdprob5 import test_stdprob5 as test_stdprob5
from .threads import TestThreads as TestThreads
from .timedriver import TestTimeDriver as TestTimeDriver
from .uniaxialanisotropy import TestUniaxialAnisotropy as TestUniaxialAnisotropy
from .zeeman import TestZeeman as TestZeeman
from .zhangli import TestZhangLi as TestZhangLi
