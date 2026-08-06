import mcdc
import mcdc.config as config

from mcdc.config import override_settings


def test_compilation_applies_command_line_overrides(monkeypatch):
    simulation = mcdc.Simulation()
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
