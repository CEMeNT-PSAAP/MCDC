from setuptools import setup, find_packages

kwargs = {
    "name": "mcdc",
    "version": "0.1.0",
    "packages": find_packages(include=["mcdc"]),
}

setup(**kwargs)
