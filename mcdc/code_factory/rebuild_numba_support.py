from __future__ import annotations

import sys
from pathlib import Path

# Allow this file to be run directly from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    from mcdc.code_factory.numba_layers_generator import rebuild_numba_support

    rebuild_numba_support()


if __name__ == "__main__":
    main()
