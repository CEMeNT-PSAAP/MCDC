#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess


def test():
    """
    This test runs the specified regression tests with a specified number
    of MPI processes. This way, we can compare the numba results directly
    to the python mode results.

    This test will not work on Windows machines

    """

    tests = [
        "eigenvalue/slab_kornreich_iqmc/",
        "fixed_source/reed/",
    ]

    for i in range(len(tests)):
        cwd = tests[i]
        result = subprocess.run(["mpiexec", "-np", "2", "python", "test.py"], cwd=cwd)
        assert result.returncode == 0


if __name__ == "__main__":
    test()
