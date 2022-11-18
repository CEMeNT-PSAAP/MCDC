import pytest

# This fixture moves working directory to test file directory
@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
        monkeypatch.chdir(request.fspath.dirname)
