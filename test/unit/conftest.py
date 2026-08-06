import os
import subprocess
import sys

import pytest


@pytest.fixture
def prepare_simulation():
    """Compile and pack an explicit simulation for kernel-level unit tests."""
    from mcdc.main import prepare
    from mcdc.object_.cell import Cell
    from mcdc.object_.simulation import Simulation

    def _prepare(
        *,
        cells=(),
        tallies=(),
        sources=(),
        objects=(),
        configure=None,
    ):
        simulation = Simulation()
        simulation.set_model(cells or (Cell(),))
        simulation.set_tallies(tallies)
        simulation.set_sources(sources)
        if configure is not None:
            configure(simulation)

        simulation.compile()
        for object_ in objects:
            object_._compile_into_simulation(simulation)

        return prepare(simulation)

    return _prepare


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config):
    # Outer invocation: `pytest test/unit` should run python+numba automatically.
    if os.environ.get("MCDC_PYTEST_INNER") == "1":
        return None

    args = list(config.invocation_params.args)
    has_unit_target = any(
        arg == "test/unit" or arg.startswith("test/unit/")
        for arg in args
        if not arg.startswith("-")
    )
    has_mode = any(arg == "--mode" or arg.startswith("--mode=") for arg in args)

    # Only orchestrate when unit tests are targeted and mode is not explicitly chosen.
    if (not has_unit_target) or has_mode:
        return None

    base_cmd = [sys.executable, "-m", "pytest", *args]
    env = os.environ.copy()
    env["MCDC_PYTEST_INNER"] = "1"

    print("\n=== MCDC unit tests: python mode ===", flush=True)
    result_python = subprocess.run([*base_cmd, "--mode=python"], env=env)
    if result_python.returncode != 0:
        print("\nMCDC unit tests failed in python mode.", flush=True)
        return result_python.returncode

    print("\n=== MCDC unit tests: numba mode ===", flush=True)
    result_numba = subprocess.run([*base_cmd, "--mode=numba"], env=env)
    if result_numba.returncode != 0:
        print("\nMCDC unit tests failed in numba mode.", flush=True)
        return result_numba.returncode

    return 0


def pytest_configure(config):
    mode = config.getoption("--mode")
    if mode == "python":
        os.environ["NUMBA_DISABLE_JIT"] = "1"
    else:
        os.environ.pop("NUMBA_DISABLE_JIT", None)


def pytest_report_header(config):
    return f"MCDC mode: {config.getoption('--mode')}"
