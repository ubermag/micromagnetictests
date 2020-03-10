import pytest
import pkg_resources
from .exchange import TestExchange
from .dmi import TestDMI
from .uniaxialanisotropy import TestUniaxialAnisotropy
from .cubicanisotropy import TestCubicAnisotropy
from .demag import TestDemag
from .magnetoelastic import TestMagnetoElastic
from .zeeman import TestZeeman
from .energy import TestEnergy
from .precession import TestPrecession
from .damping import TestDamping
from .zhangli import TestZhangLi
from .slonczewski import TestSlonczewski
from .dynamics import TestDynamics
from .mindriver import TestMinDriver
from .timedriver import TestTimeDriver
from .mesh import TestMesh
from .compute import TestCompute
from .info_file import test_info_file
from .save_delete import test_save_delete
from .multiple_drives import test_multiple_drives
from .stdprob3 import test_stdprob3
from .stdprob4 import test_stdprob4
from .stdprob5 import test_stdprob5


def test():
    return pytest.main(['-v', '--pyargs',
                        'micromagnetictests'])  # pragma: no cover


__version__ = pkg_resources.get_distribution(__name__).version
__dependencies__ = pkg_resources.require(__name__)
