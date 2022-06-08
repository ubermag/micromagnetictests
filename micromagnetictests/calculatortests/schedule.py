import micromagneticmodel as mm


def test_schedule(calculator, tmp_path):
    system = mm.examples.macrospin()

    td = calculator.TimeDriver()
    # We have no test system with a job scheduling system such as slurm.
    # Instead, we use dir to test that the mif file and job script creation and the
    # subprocess call to schedule the run work as expected.
    # `dir` is expected to work on all operating systems.
    td.schedule(
        system,
        "dir",
        "scheduling resources",
        dirname=str(tmp_path),
        t=0.2e-9,
        n=50,
    )
    assert len(list(tmp_path.glob("**/job.sh"))) == 1
