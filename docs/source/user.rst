.. _user:

============
User's Guide
============

Once you have a working copy of MC/DC and your ready to devlope

This guide presupposes you are familiar with modeling nuclear systems using a Monte Carlo method.
If you are completely new, we suggest checking out `OpenMC's theory guide <https://docs.openmc.org/en/stable/methods/introduction.htmll>`_ as most the basic underlying algorithms and core concepts are the same.
Our input decks and keyword phrases are designed so that if you are familiar with tools like OpenMC or MCNP you should be able to get up and running quick.

While this guide is a great place to start, the  next best place to look when getting started is our ``MCDC/examples`` or ``MCDC/testing`` directories.
Run a few problems there, change a few inputs around see if it works and keep looking around till you get the general hang of what we are doing.
Believe it or not, there is a method to all this madness.

A note on testing:
Just because something seems right doesn't mean it is.
Care must be taken to ensure that you are running the problem you think you are.
The software only knows what you tell it.

Workflows in MC/DC
------------------

MC/DC uses an ``input`` -> ``run`` -> ``post-process`` work flow where users,

#. build input decks using scripts that import ``mcdc`` as a package and call functions to build geometries, tally meshes, and set other simulation parameters
#. define a runtime sequence in the terminal executing ``input`` script (terminal operations are required for MPI calls)
#. exporting the results from ``.h5`` files and either using the visualizer or tools like ``matplotlib`` to view results.

----------------------
Building an input deck
----------------------

Building an input deck can be a complicated and nuanced process. Depending on the type of simulation you need to build you might touch most of the functions in MC/DC, or very few.
Again the best way to start building input decks is to look at what we have already done in the ``MCDC/examples`` or ``MCDC/testing`` directories.

To see more on input functions look through pythonapi/index.
As an example we are going to be building the ``MCDC/examples/fixed_source/slab_absorbium`` problem in which a three region purely absorbing mono-energetic transient slab wall problem is modeled.

First we start with our inputs,

.. code-block:: python3

    import numpy as np

    import mcdc

You may require more packages depending on the methods you are constructing but most of what you need will be in these two.
Now we define the materials that are going into this problem

.. code-block:: python3

    # Set materials
    m1 = mcdc.material(capture=np.array([1.0]))
    m2 = mcdc.material(capture=np.array([1.5]))
    m3 = mcdc.material(capture=np.array([2.0]))

In this problem we only have mono-energetic capture but MC/DC has support for multi-group (capture, scatter, fission) and contentious energy (capture, scatter, fission).
When in multi-group simulation modes, a Numpy array has the different .
These functions can get quite complicated.

For continues energy transport if you are a member of CEMeNT we have internal repos which can be auto-configured with the required data.
Unfortunately due to export controls we can not publicly distribute this data.
If you are looking for cross-section data to plug into MC/DC we recommend you look at OpenMC or `NJOY <http://www.njoy21.io/>`_.

After setting material data we set surfaces to define the problem space. 

.. code-block:: python3

    # Set surfaces
    s1 = mcdc.surface("plane-z", z=0.0, bc="vacuum")
    s2 = mcdc.surface("plane-z", z=2.0)
    s3 = mcdc.surface("plane-z", z=4.0)
    s4 = mcdc.surface("plane-z", z=6.0, bc="vacuum")

note that these surface also contain information about boundary conditions.
Remember that we are solving a 7 independent partial-intrgro differential equation so we need both initial and boundary conditions.
While we have tried to include warnings and errors if a ill-posed problem is detected we cannot forecast all the ways in which things might go hay-wire.
Ideally MC/DC will fail before it even runs but possibly you could get in an infinite tracking problem or your data will look very odd when you go to visualize it.

If this is a transient simulation initial conditions are assumed as 0 everywhere.
Once we set surfaces we can define the cells that exist between surfaces and the material that fills them. 
Here we use the materials we defined earlier distributing them between our four surface planes.

.. code-block:: python3

    mcdc.cell([+s1, -s2], m2)
    mcdc.cell([+s2, -s3], m3)
    mcdc.cell([+s3, -s4], m1)

Uniform isotropic source throughout the domain

.. code-block:: python3

    mcdc.source(z=[0.0, 6.0], isotropic=True)

Next we set tallies and specify the specific parameters of interest. Here its the time and space averaged flux and the time and space averaged current across the whole problem and direction space,

.. code-block:: python3

    # Tally: cell-average fluxes and currents
    mcdc.tally(
        scores=["flux", "current"],
        z=np.linspace(0.0, 6.0, 61),
        mu=np.linspace(-1.0, 1.0, 32 + 1),
    )

Here you can see again that we are using Numpy arrays to construct our tally mesh. Monte Carlo results are really just a histogram over relevant tallies.
In fact regardless of the specific problem particles are always flying through space direction and time, we just disable most of the tallying in those dimensions for a problem this simple.

Next we set simulation settings, primarily the number of particles.
If you where running a k-eigenvalue type problem there would be a number of different setting to put in here as well.
You can also control weather the MC/DC title mast displays, something you might want to disable if MC/DC transport is part of an inner loop.

.. code-block:: python3

    mcdc.setting(N_particle=1e3)

Finally execute the problem.

.. code-block:: python3

    mcdc.run()

When you string this all together it should look something like this,

.. code-block:: python3

    import numpy as np
    import mcdc

    # =============================================================================
    # Set model
    # =============================================================================
    # Three slab layers with different purely-absorbing materials

    # Set materials
    m1 = mcdc.material(capture=np.array([1.0]))
    m2 = mcdc.material(capture=np.array([1.5]))
    m3 = mcdc.material(capture=np.array([2.0]))

    # Set surfaces
    s1 = mcdc.surface("plane-z", z=0.0, bc="vacuum")
    s2 = mcdc.surface("plane-z", z=2.0)
    s3 = mcdc.surface("plane-z", z=4.0)
    s4 = mcdc.surface("plane-z", z=6.0, bc="vacuum")

    # Set cells
    mcdc.cell([+s1, -s2], m2)
    mcdc.cell([+s2, -s3], m3)
    mcdc.cell([+s3, -s4], m1)

    # =============================================================================
    # Set source
    # =============================================================================
    # Uniform isotropic source throughout the domain

    mcdc.source(z=[0.0, 6.0], isotropic=True)

    # =============================================================================
    # Set tally, setting, and run mcdc
    # =============================================================================

    # Tally: cell-average fluxes and currents
    mcdc.tally(
        scores=["flux", "current"],
        z=np.linspace(0.0, 6.0, 61),
        mu=np.linspace(-1.0, 1.0, 32 + 1),
    )

    # Setting
    mcdc.setting(N_particle=1e3)

    # Run
    mcdc.run()

Now that we have a script to run how do we actually run it?

----------------
Running a Script
----------------



Pure Python Mode
----------------

Numba Mode
----------

MPI Mode
--------


----------------------
Postprocessing Outputs
----------------------

While the whole workflow of MC/DC can be done in one script for anything but very simple inputs
(or if you know what your doing) it is recommended to keep the simulation and post-processing/ visualization scripts separate.

