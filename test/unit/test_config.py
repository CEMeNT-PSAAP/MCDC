from pathlib import Path
from types import SimpleNamespace

import mcdc
import mcdc.config as config

from mcdc.config import override_settings


class SingleRankCommunicator:
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1


def test_cache_cleanup_does_not_remove_python_cache(monkeypatch):
    removed = []
    communicator = SingleRankCommunicator()

    monkeypatch.setattr(config, "MPI", SimpleNamespace(COMM_WORLD=communicator))
    monkeypatch.setattr(config, "caching", False)
    monkeypatch.setattr(config, "clear_cache", False)
    monkeypatch.setattr(config.Path, "exists", lambda path: True)
    monkeypatch.setattr(config.shutil, "rmtree", removed.append)

    config._manage_runtime_caches()

    assert removed == [Path.cwd() / "__harmonize_cache__"]
    assert all(path.name != "__pycache__" for path in removed)


def test_compilation_applies_command_line_overrides(monkeypatch):
    simulation = mcdc.Simulation()
    simulation.set_model([mcdc.Cell()])
    monkeypatch.setattr(config, "target", "cpu")
    monkeypatch.setattr(config.args, "N_particle", 100)
    monkeypatch.setattr(config.args, "N_batch", None)
    monkeypatch.setattr(config.args, "output", None)
    monkeypatch.setattr(
        config.args, "progress_bar", simulation.settings.use_progress_bar
    )

    simulation.compile()

    assert simulation.settings.N_particle == 100
    assert not override_settings(simulation)
