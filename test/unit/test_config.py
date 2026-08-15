import mcdc
import mcdc.config as config

from mcdc.config import _build_parser, override_settings


def test_parser_accepts_active_cycle_override():
    args = _build_parser().parse_args(["--N_active", "25"])

    assert args.N_active == 25


def test_compilation_applies_command_line_overrides(monkeypatch):
    simulation = mcdc.Simulation()
    simulation.set_model([mcdc.Cell()])
    monkeypatch.setattr(config, "target", "cpu")
    monkeypatch.setattr(config.args, "N_particle", 100)
    monkeypatch.setattr(config.args, "N_batch", None)
    monkeypatch.setattr(config.args, "N_active", None)
    monkeypatch.setattr(config.args, "output", None)
    monkeypatch.setattr(
        config.args, "progress_bar", simulation.settings.use_progress_bar
    )

    simulation.compile()

    assert simulation.settings.N_particle == 100
    assert not override_settings(simulation)


def test_active_cycle_override_updates_total_cycles(monkeypatch):
    simulation = mcdc.Simulation()
    simulation.settings.set_eigenmode(N_inactive=5, N_active=10)

    monkeypatch.setattr(config, "target", "cpu")
    monkeypatch.setattr(config.args, "N_particle", None)
    monkeypatch.setattr(config.args, "N_batch", None)
    monkeypatch.setattr(config.args, "N_active", 25)
    monkeypatch.setattr(config.args, "output", None)
    monkeypatch.setattr(
        config.args, "progress_bar", simulation.settings.use_progress_bar
    )

    assert override_settings(simulation)
    assert simulation.settings.N_active == 25
    assert simulation.settings.N_cycle == 30
