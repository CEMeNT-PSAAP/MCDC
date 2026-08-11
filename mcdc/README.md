# MC/DC Package Map

This directory contains the main MC/DC Python package.

- `__init__.py` defines the public Python interface.
- `object_/` defines the user-facing model objects and is the model-side home for methods development.
- `config.py`, `constant.py`, and `literals.py` provide shared execution configuration and values.
- `main.py` coordinates runtime preparation, transport execution, and output.
- `code_factory/` compiles Python models into runtime data and backend-specific code.
- `mcdc_get/` and `mcdc_set/` contain runtime-data accessors generated during preparation and used during transport.
- `transport/` implements the shared Monte Carlo algorithms and is the primary place to study or extend particle transport.
- `output.py` processes and serializes simulation results.

See the [architecture guide](../docs/source/developer_guide/architecture/index.rst) for the detailed design and component responsibilities.
