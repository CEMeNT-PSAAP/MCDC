import numpy as np
import h5py
import pytest
import os

import mcdc


@pytest.mark.regression
@pytest.mark.fixed_source
@pytest.mark.serial
@pytest.mark.parametrize(("mode"), ["python", "numba"])
def test_serial(get_file, mode):
    file = get_file("problems/fixed_source/cooper2_iqmc_input.py")
    os.system("python {} --particles=20 --mode={}".format(file, mode))

    output = h5py.File("output.h5", "r")
    answer = h5py.File("answer.h5", "r")
    a = answer["tally/iqmc_flux"][:]
    b = output["tally/iqmc_flux"][:]
    output.close()
    answer.close()

    assert np.allclose(a, b)


@pytest.mark.regression
@pytest.mark.fixed_source
@pytest.mark.processor
@pytest.mark.parametrize(("mode"), ["python", "numba"])
def test_processor(get_file, nprocess, mode):
    file = get_file("problems/fixed_source/cooper2_iqmc_input.py")

    mpi = "mpiexec" if os.name == "nt" else "srun"
    os.system(
        "{} -n {} python {} --particles=20 --mode={} \
              --file=output_mpi".format(
            mpi, nprocess, file, mode
        )
    )

    os.system(
        "python {} --particles=20 --mode={} --file=output_serial".format(file, mode)
    )

    mpi_output = h5py.File("output_mpi.h5", "r")
    serial_output = h5py.File("output_serial.h5", "r")
    answer = h5py.File("answer.h5", "r")
    a = mpi_output["tally/iqmc_flux"][:]
    b = serial_output["tally/iqmc_flux"][:]
    c = answer["tally/iqmc_flux"][:]
    mpi_output.close()
    serial_output.close()
    answer.close()

    assert np.isclose(a, c).all() & np.isclose(b, c).all()
