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
    file = get_file("problems/fixed_source/dsm_lattice_input.py")
    os.system("python {} --particles=5 --mode={}".format(file, mode))

    scores = ["flux"]

    output = h5py.File("output.h5", "r")
    answer = h5py.File("answer.h5", "r")
    for score in scores:
        name = "tally/" + score + "/mean"
        a = output[name][:]
        b = answer[name][:]
        assert np.isclose(a, b).all()

        name = "tally/" + score + "/sdev"
        a = output[name][:]
        b = answer[name][:]
        assert np.isclose(a, b).all()

    output.close()
    answer.close()


@pytest.mark.regression
@pytest.mark.fixed_source
@pytest.mark.processor
@pytest.mark.parametrize(("mode"), ["python", "numba"])
def test_processor(get_file, nprocess, mode):
    file = get_file("problems/fixed_source/dsm_lattice_input.py")

    mpi = "mpiexec" if os.name == "nt" else "srun"
    os.system(
        "{} -n {} python {} --particles=5 --mode={} \
            --file=output_mpi".format(
            mpi, nprocess, file, mode
        )
    )

    os.system(
        "python {} --particles=5 --mode={} --file=output_serial".format(file, mode)
    )

    scores = ["flux"]

    mpi_output = h5py.File("output_mpi.h5", "r")
    serial_output = h5py.File("output_serial.h5", "r")
    answer = h5py.File("answer.h5", "r")
    for score in scores:
        name = "tally/" + score + "/mean"
        a = mpi_output[name][:]
        b = serial_output[name][:]
        c = answer[name][:]
        assert np.isclose(a, c).all() & np.isclose(b, c).all()

        name = "tally/" + score + "/sdev"
        a = mpi_output[name][:]
        b = serial_output[name][:]
        c = answer[name][:]
        assert np.isclose(a, c).all() & np.isclose(b, c).all()

    mpi_output.close()
    serial_output.close()
    answer.close()
