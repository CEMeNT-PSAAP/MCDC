#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess


def test():
    """
    This test runs the specified regression tests in numba mode. This way,
    we can compare the numba results directly to the python mode results.

    This test will not work on Windows machines

    """

    tests = [
        "../regression/eigenvalue/slab_kornreich/",
        "../regression/eigenvalue/slab_kornreich_iqmc/",
        "../regression/fixed_source/reed/",
    ]

    for i in range(len(tests)):
        cwd = tests[i]
        result = subprocess.run(["python", "test.py", "--mode=numba"], cwd=cwd)
        assert result.returncode == 0


if __name__ == "__main__":
    test()
