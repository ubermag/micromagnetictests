import sys

import micromagneticmodel as mm
import pytest


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Calling the oommf executable without a full path does not properly work.",
)
def test_schedule(calculator, tmp_path):
    system = mm.examples.macrospin()

    td = calculator.TimeDriver()
    with pytest.raises(RuntimeError):
        # We have no test system with a job scheduling system such as slurm.
        # Instead, we use oommf to test that the mif file creation and the
        # subprocess call to schedule the run work as expected.
        # OOMMF will raise an error because we call it with a "command" job.sh (the
        # submission system script)
        td.schedule(
            system,
            "oommf",
            "scheduling resources",
            dirname=str(tmp_path),
            t=0.2e-9,
            n=50,
        )
        assert len(list(tmp_path.glob("**/job.sh"))) == 1
