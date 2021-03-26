from setuptools import setup

setup(
    name='mcdc',
    version='0.1dev',
    packages=['mcdc',],
    setup_requires=['pytest-runner',],
    tests_require=['pytest',],
)
