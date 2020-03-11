import os
import pytest
import micromagneticmodel as mm


def test_save_delete(calculator):
    system = mm.examples.macrospin()

    td = calculator.TimeDriver()
    td.drive(system, t=1e-12, n=5, save=True, overwrite=True)

    assert os.path.exists(os.path.join(system.name, 'drive-0'))

    with pytest.raises(FileExistsError):
        system.drive_number = 0
        td.drive(system, t=1e-12, n=5, save=True)

    assert os.path.exists(os.path.join(system.name, 'drive-0'))
    assert not os.path.exists(os.path.join(system.name, 'drive-1'))

    system.drive_number = 0
    td.drive(system, t=1e-12, n=5, save=False)

    assert os.path.exists(os.path.join(system.name, 'drive-0'))
    assert not os.path.exists(os.path.join(system.name, 'drive-1'))

    td.drive(system, t=1e-12, n=5, save=True, overwrite=True)

    assert os.path.exists(os.path.join(system.name, 'drive-0'))
    assert os.path.exists(os.path.join(system.name, 'drive-1'))

    calculator.delete(system)

    assert not os.path.exists(system.name)

    with pytest.raises(FileNotFoundError):
        calculator.delete(system)
