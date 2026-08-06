import runpy
from pathlib import Path

import pytest

import mcdc

EXAMPLES_ROOT = Path(__file__).parents[2] / "examples"
EXAMPLE_INPUTS = tuple(sorted(EXAMPLES_ROOT.glob("**/input.py")))
assert EXAMPLE_INPUTS, "No example inputs found"


def example_id(input_path):
    return str(input_path.relative_to(EXAMPLES_ROOT).parent)


@pytest.mark.parametrize("input_path", EXAMPLE_INPUTS, ids=example_id)
def test_example_constructs_and_compiles_simulation(input_path, monkeypatch):
    """Validate every example input without running particle transport."""
    compiled_simulations = []

    def compile_only(simulation, *args, **kwargs):
        simulation.compile()
        compiled_simulations.append(simulation)

    monkeypatch.setattr(mcdc.Simulation, "run", compile_only)
    monkeypatch.setattr(mcdc.Simulation, "visualize_model", compile_only)
    monkeypatch.chdir(input_path.parent)

    runpy.run_path(input_path.name, run_name="__main__")

    assert compiled_simulations
    unique_simulations = {
        id(simulation): simulation for simulation in compiled_simulations
    }
    assert len(unique_simulations) == 1
    simulation = next(iter(unique_simulations.values()))
    assert simulation.compiled
    assert simulation.cells
    assert simulation.sources
    assert simulation.tallies
