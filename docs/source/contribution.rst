.. _contributions:

=================
Contibution Guide
=================

Thank you for looking to contirbute to MC/DC! 
We are really exctied to see what you bring to this exciting open source project!
Wheater you are here to make a single PR and never return, or want to become a maintiner we are pumped to work with you.
We have regular devlopers meetings for any and all who are interested to discss contributions to this code base.

This describes the processes of contributing to MC/DC for both internal (CEMeNT) and external devlopers.

Please note our `code of conduct <>` which we take seriously

------------
Code Styling
------------

Our code is autolinted for the `Black code style <https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html>`.
Your contirbutions will not be merged unless you follow this code style.
It's pretty easy to do this locally, just run,

.. code-block:: sh
    pip install black
    black .

in the top level MC/DC direcory and all nessacary changes will be automatically made for you.

-------
Testing
-------

MC/DC has a robust testing suite that your changes must be able to pass before a PR is accepted.
Unit tests for functions that have them are ran in a pure python from.
Mostly this is for ensuring input operability
A regression test suite (including models with anylitical and expermental soultions) is provided to ensure accuracy and precision of MC/DC.

Our test suite runs on every PR, and Push.
Our github based CI runs for, 
* linux-64 (x86)
* osx-64 (x86, intel based macs)
* win-64 (x86)
* osx-arm64 (apple sillicon)
while we do not have continous integration we have validated MC/DC on
* linux-aarch64 (IBM POWER9)
* linux-nvidia-cuda
* linux-amd-gcn

To run the regression tests locally, navigate to ``\MCDC\tests\regression`` and run,

.. code-block:: sh
    python run.py <OPTION_FLAG(s)>

and all the tests will run. Various option ``OPTION_FLAG`` are accepted to control the tests ran,

* Run a specific test (with wildcard `*` support): ``--name=<test_name>`` 
* Run in Numba mode: ``--mode=numba``
* Run in multiple MPI ranks (currently support `mpiexec` and `srun`): ``--mpiexec=<number of ranks>``

Note that flags can be combined. To add a new test:

#. Create a folder. The name of the folder will be the test name.
#. Add the input file. Name it`input.py`.
#. Add the answer key file. Name it `answer.h5`.
#. Make sure that the number of particles run is large enough for a good test.
#. If the test runs longer than 5 seconds, consider decreasing the number of particles.

When adding a new hardware backend a new enstantioation of the test suit should be made.
This is done with github actions. 
See the (.github/workflows) for examples.

If a new simulation type is added (e.g. quasi montecarlo w/ davidson's method, residual monte carlo, intrisuve uq) more regression tests shuold be added with your PR.
If you are wondering accomidations.


-------------
Documentaiton
-------------


It's not everything it needs to be but we are trying!
If your contirbution changes the behavior of the input deck, instilation process, or testing infrastructure your contribution must include alteration to this documentaiton.
That can be done by editing the RST files in ``/MCDC/docs/source/<FILENAME>.rst``.


-------------
Pull Requstes
-------------


MC/DC works off of a fork workflow in which contributors fork our repo, make allterations, and submit a pull requests.
You should only submit a pull request once your code passes all tests, is properly linted, you have edited documentaiton (if nessacary), and added any new tests (if needed).

Within your pull request documetnation please list:
#. Type of PR (e.g. Enhancment, bugfix, etc);
#. Link to any theroy to understand what you are doing;
#. Link to any open/closed issues if applicable;
#. New functionalties implemented
#. Depreicated functionalities
#. New dependincies needed (we don't add these lightly)
#. Anything else we need to give you the thorough code review you deserve!

If these things aren't listed we will ask for clarifying qustions!