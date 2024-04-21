import pytest


# Moves working directory to test file directory
@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


# Set mode (python, numba), mpi (srun, mpiexec), and number of ranks
def pytest_addoption(parser):
    parser.addoption(
        "--mode", action="store", default="python", choices=("python", "numba")
    )
    parser.addoption(
        "--mpi", action="store", default="srun", choices=("srun", "mpiexec")
    )
    parser.addoption("--N_rank", action="store", default="1")


@pytest.fixture(scope="session")
def mode(pytestconfig):
    return pytestconfig.getoption("mode")


@pytest.fixture(scope="session")
def mpi(pytestconfig):
    return pytestconfig.getoption("mpi")


@pytest.fixture(scope="session")
def N_rank(pytestconfig):
    return pytestconfig.getoption("N_rank")
