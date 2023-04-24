from pathlib import Path
import pytest

# This fixture moves working directory to test file directory


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


@pytest.fixture(scope="function", autouse=True)
def check_verification(request):
    yield
    if (request.session.testsfailed) and (
        request.node.get_closest_marker("regression")
    ):
        N_min = 3
        N_max = 7
        if request.config.getoption("--verify"):
            print("Running Verification...")
            # verify.run(N_min, N_max, request.config.getoption("--nprocess"))
        # For testing
        else:
            print("No!")


@pytest.fixture(scope="session")
def get_file():
    def _(file_path: str):
        return Path(__file__).parent / file_path

    return _


@pytest.fixture(scope="session")
def nprocess(request):
    return request.config.getoption("--nprocess")


def pytest_addoption(parser):
    parser.addoption(
        "--verify", action="store_true", default=False, help="Runs Verification if True"
    )
    parser.addoption(
        "--nprocess",
        action="store",
        default=2,
        type=int,
        help="Number of processors to use",
    )
