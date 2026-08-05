import numpy as np
import pytest

import mcdc.print_ as print_module


def test_print_1d_array():
    assert print_module.print_1d_array(np.array([])) == "(size=0): []"
    assert print_module.print_1d_array(np.array([1.0, 2.0])) == "(size=2): [1, 2]"
    assert print_module.print_1d_array(np.arange(6.0)) == "(size=6): [0, 1, ..., 4, 5]"


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        (2.0, "2.00 seconds"),
        (120.0, "2.00 minutes"),
        (7_200.0, "2.00 hours"),
        (172_800.0, "2.00 days"),
    ],
)
def test_print_time(duration, expected, capsys):
    print_module.print_time("Stage", duration, 25.0)

    assert capsys.readouterr().out == f"   Stage | {expected} (25.0%)\n"


def test_print_runtime_handles_zero_total(monkeypatch, capsys):
    monkeypatch.setattr(print_module, "_IS_MASTER", True)
    simulation = {
        "runtime_total": 0.0,
        "runtime_preparation": 0.0,
        "runtime_simulation": 0.0,
        "runtime_output": 0.0,
    }

    print_module.print_runtime(simulation)

    output = capsys.readouterr().out
    assert "Runtime report:" in output
    assert "Preparation | 0.00 seconds (0.0%)" in output


def test_master_only_message(monkeypatch, capsys):
    monkeypatch.setattr(print_module, "_IS_MASTER", False)

    print_module.print_msg("hidden")
    print_module.print_warning("hidden")

    assert capsys.readouterr().out == ""


def test_print_error_exits_unsuccessfully(capsys):
    with pytest.raises(SystemExit) as error:
        print_module.print_error("invalid model")

    assert error.value.code == 1
    assert "[ERROR]: invalid model" in capsys.readouterr().out
