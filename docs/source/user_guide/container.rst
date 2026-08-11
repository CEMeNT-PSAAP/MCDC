MC/DC Container Guide
=====================

What Are Containers?
--------------------

A container is a lightweight, portable package that bundles an application with its code, libraries, system tools, and settings.
Like a shipping container, it keeps its contents consistent across host systems.

**Why does this matter for MC/DC?**

Installing MC/DC requires Python, MPI, Numba, and other dependencies.
Coordinating these dependencies can be difficult on HPC systems without administrator access.
A container provides a pre-built environment where the dependencies are already installed and tested.

Tested Platforms
----------------

+-------------+------------+--------+--------------------+--------+
| System      | OS         | Arch   | Container Tool     | Status |
+=============+============+========+====================+========+
| MacBook Pro | macOS 26.3 | arm64  | Docker 29.2.0      | ✓      |
+-------------+------------+--------+--------------------+--------+
| Tuolumne    | RHEL 8.10  | x86_64 | Podman 4.9.4       | ✓      |
+-------------+------------+--------+--------------------+--------+
| Dane        | RHEL 8.10  | x86_64 | Podman 4.9.4       | ✓      |
+-------------+------------+--------+--------------------+--------+
| Tioga       | RHEL 8.10  | x86_64 | Podman 4.9.4       | ✓      |
+-------------+------------+--------+--------------------+--------+
| COE (OSU)   | Rocky 8.10 | x86_64 | Apptainer 1.4.5    | ✓      |
+-------------+------------+--------+--------------------+--------+

The published CPU image contains the current MC/DC branch build, Python 3.13, MPICH, and the development tools needed to run the test suites.

Getting Started (New Users)
---------------------------

This section is for anyone who wants to **run MC/DC** in a container.
No prior container experience needed.

Step 1: Pull the Pre-Built Image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You do not need to build the image because a ready-to-use image is available on the GitHub Container Registry.

.. rubric:: Local Machine (Docker)

Install Docker Desktop, open a terminal, and run:

.. code-block:: bash

    docker pull ghcr.io/mcdc-project/mcdc:dev
    docker run --rm -it ghcr.io/mcdc-project/mcdc:dev

You are now inside the container.
Try importing MC/DC:

.. code-block:: bash

    python -c "import mcdc; print('MC/DC OK')"

Type ``exit`` to leave the container.

.. rubric:: LLNL Systems — Tuolumne, Tioga, Dane (Podman)

Podman is already installed on LLNL systems.
It uses the same commands as Docker in these examples.

.. code-block:: bash

    podman pull ghcr.io/mcdc-project/mcdc:dev
    podman run --rm -it ghcr.io/mcdc-project/mcdc:dev

.. note::

    If you see ``lsetxattr: operation not supported``, see *LLNL Storage Setup* in Part 2.

.. rubric:: OSU Systems — COE (Apptainer)

Apptainer is already installed on COE.

.. code-block:: bash

    apptainer build --sandbox mcdc_sandbox docker://ghcr.io/mcdc-project/mcdc:dev
    apptainer exec mcdc_sandbox python -c "import mcdc; print('MC/DC OK')"

.. note::

    If ``apptainer pull`` fails with "Out of memory", use ``--sandbox``.

Step 2: Run Your Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. rubric:: Docker / Podman

.. code-block:: bash

    docker run --rm -v $(pwd):/work -w /work ghcr.io/mcdc-project/mcdc:dev python input.py
    docker run --rm ghcr.io/mcdc-project/mcdc:dev mpirun -n 4 python input.py

For Podman, replace ``docker`` with ``podman``.

**Flags explanation**

- ``--rm``: Automatically clean up container.
- ``-it``: Interactive terminal.
- ``-v $(pwd):/work``: Share current folder.
- ``-w /work``: Start inside shared folder.

.. rubric:: Apptainer (OSU)

.. code-block:: bash

    apptainer exec mcdc_sandbox python input.py
    apptainer exec mcdc_sandbox mpirun -launcher fork -n 4 python input.py

.. note::

    Apptainer automatically shares your home directory.

Step 3: Docker Compose (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From MC/DC repo root:

.. code-block:: bash

    docker compose -f containers/docker-compose.yml run --rm dev bash
    docker compose -f containers/docker-compose.yml run --rm test
    docker compose -f containers/docker-compose.yml run --rm mpi mpirun -n 4 python input.py
