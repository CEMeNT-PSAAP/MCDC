.. _user:

============
User's Guide
============

This guide presupposes you are familiar with modeling nuclear systems using a Monte Carlo method.
If you are completely new, we suggest checking out `OpenMC's theory guide <https://docs.openmc.org/en/stable/methods/introduction.htmll>`_ as most the basic underlying algorithms and core concepts are the same.
Our input decks and keyword phrases are designed so that if you are familiar with tools like OpenMC or MCNP you should be able to get up and running quick.

While this guide is a great place to start, the  next best place to look when getting started is our ``MCDC/examples`` or ``MCDC/testing`` directories.
Run a few problems there, change a few inputs around see if it works and keep looking around till you get the general hang of what we are doing.
Believe it or not, there is a method to all this madness.
If you find your self with errors you really don't know what to do with take look at our `GitHub issues page <https://github.com/CEMeNT-PSAAP/MCDC/issues>`_.
If it looks like you are the first to have a given problem feel free to submit a new ticket!

A note on testing:
Just because something seems right doesn't mean it is.
Care must be taken to ensure that you are running the problem you think you are.
The software only knows what you tell it.

------------------
Workflows in MC/DC
------------------

MC/DC uses an ``input`` -> ``run`` -> ``post-process`` work flow where users,

#. build input decks using scripts that import ``mcdc`` as a package and call functions to build geometries, tally meshes, and set other simulation parameters
#. define a runtime sequence in the terminal executing ``input`` script (terminal operations are required for MPI calls)
#. exporting the results from ``.h5`` files and either using the visualizer or tools like ``matplotlib`` to view results.


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

``#input.py``

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


Running a Script
----------------

While running MC/DC in something like a jupyter note book is possible it is not recommended,
epically when doing MPI and/or Numba type runs.
This assumes you have a correctly configured MC/DC instillation.
MPI can be quite tricky to get configured if your on an HPC and if you are having trouble consult our :ref:`install`, your HPC admin, or our `GitHub issues page <https://github.com/CEMeNT-PSAAP/MCDC/issues>`_.

----------------
Pure Python Mode
----------------

To run in pure Python mode (slow with no acceleration) 

.. code-block:: python3

    python input.py

----------
Numba Mode
----------

.. code-block:: python3

    python input.py --mode=numba

--------
MPI Mode
--------

.. code-block:: python3

    srun python input.py --mode=<PYTHON/NUMBA>

MPI mode can be run with or without Numba acceleration.
If ``numba-mode`` is enabled the ``jit`` compilation (which can take between 30s-2min) will be executed on all threads.
This is why we allow users to use Python mode to avoid this for intermediate sized problems.





Postprocessing Outputs
----------------------

While the whole workflow of MC/DC can be done in one script---for anything but very simple inputs
(or if you know what your doing)---it is recommended to keep the simulation and post-processing/visualization scripts separate.

When a problem is executed tallied results are compiled, compressed, and saved in ``.h5`` files.
The size of these files can vary widely depending on your tally settings, 
the geometric size of the problem (e.g. number of surfaces, and the number of particles tracked.
Expect sizes as small as ``kB`` or as large as ``TB``.

These result files can be exported manipulated and visualized.
Data can be pulled from an ``.h5`` file using something like,

.. code-block:: python3

    import h5py
    import numpy as np
    # Load results
    with h5py.File("output.h5", "r") as f:
        z = f["tally/grid/z"][:]
        dz = z[1:] - z[:-1]
        z_mid = 0.5 * (z[:-1] + z[1:])

        mu = f["tally/grid/mu"][:]
        dmu = mu[1:] - mu[:-1]
        mu_mid = 0.5 * (mu[:-1] + mu[1:])

        psi = f["tally/flux/mean"][:]
        psi_sd = f["tally/flux/sdev"][:]
        J = f["tally/current/mean"][:, 2]
        J_sd = f["tally/current/sdev"][:, 2]  

While there can be some nuance to the dimensions of these data arrays the folder structures should be pretty evident from your tally settings.
If needed to look in and browse around a ``.h5`` file you can use something like `h5Viewer <https://www.hdfgroup.org/downloads/>`_ (which on linux can be installed with ``sudo apt-get install hdfview``).
Otherwise these arrays can then be manipulated and modified like any other.
They are just arrays of numbers so tools like SciPy, NumPy, Pandas, .etc are all on the table for use to further compute information from these simulations.

When plotting results if the problem is small enough a tool like ``matplotlib`` will work great.
For more complex simulations open source professional visualization softwares like
`Paraview <https://www.paraview.org/>`_  or `Visit <https://sd.llnl.gov/simulation/computer-codes/visit>`_ are available.

As the problem we ran above is pretty simple and has no scattering or fission we actually have an analytic solution we can import

.. code-block:: python3

    from reference import reference

From here we can go about plotting our problem like any other in ``matplotlib``. 
In this script we plot the spatial averaged flux and current as separate figures 
(including the statistical noise from the Monte Carlo solution).
Remember that a Monte Carlo solution must always include a report a statical error!
We then use those terms to compute a new term, spatial averaged angular flux, and plot
that over it's dimensions (angle and distance) in a heat map.

.. code-block:: python3

    import matplotlib.pyplot as plt
    import numpy as np

    I = len(z) - 1
    N = len(mu) - 1

    # Scalar flux
    phi = np.zeros(I)
    phi_sd = np.zeros(I)
    for i in range(I):
        phi[i] += np.sum(psi[i, :])
        phi_sd[i] += np.linalg.norm(psi_sd[i, :])

    # Normalize
    phi /= dz
    phi_sd /= dz
    J /= dz
    J_sd /= dz
    for n in range(N):
        psi[:, n] = psi[:, n] / dz / dmu[n]
        psi_sd[:, n] = psi_sd[:, n] / dz / dmu[n]

    # Reference solution
    phi_ref, J_ref, psi_ref = reference(z, mu)

    # Flux - spatial average
    plt.plot(z_mid, phi, "-b", label="MC")
    plt.fill_between(z_mid, phi - phi_sd, phi + phi_sd, alpha=0.2, color="b")
    plt.plot(z_mid, phi_ref, "--r", label="Ref.")
    plt.xlabel(r"$z$, cm")
    plt.ylabel("Flux")
    plt.ylim([0.06, 0.16])
    plt.grid()
    plt.legend()
    plt.title(r"$\bar{\phi}_i$")
    plt.show()

    # Current - spatial average
    plt.plot(z_mid, J, "-b", label="MC")
    plt.fill_between(z_mid, J - J_sd, J + J_sd, alpha=0.2, color="b")
    plt.plot(z_mid, J_ref, "--r", label="Ref.")
    plt.xlabel(r"$z$, cm")
    plt.ylabel("Current")
    plt.ylim([-0.03, 0.045])
    plt.grid()
    plt.legend()
    plt.title(r"$\bar{J}_i$")
    plt.show()

    # Angular flux - spatial average
    vmin = min(np.min(psi_ref), np.min(psi))
    vmax = max(np.max(psi_ref), np.max(psi))
    fig, ax = plt.subplots(1, 2, sharey=True)
    Z, MU = np.meshgrid(z_mid, mu_mid)
    im = ax[0].pcolormesh(MU.T, Z.T, psi_ref, vmin=vmin, vmax=vmax)
    ax[0].set_xlabel(r"Polar cosine, $\mu$")
    ax[0].set_ylabel(r"$z$")
    ax[0].set_title(r"\psi")
    ax[0].set_title(r"$\bar{\psi}_i(\mu)$ [Ref.]")
    ax[1].pcolormesh(MU.T, Z.T, psi, vmin=vmin, vmax=vmax)
    ax[1].set_xlabel(r"Polar cosine, $\mu$")
    ax[1].set_ylabel(r"$z$")
    ax[1].set_title(r"$\bar{\psi}_i(\mu)$ [MC]")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Angular flux")
    plt.show()

While this script does look rather long, most of these commands are controlling things like axis labels and what not.
But at the end we have something like this.

.. image:: images/user/sf_slab_1.png
   :width: 266
   :alt: Reference v computed scalar flux, 1e3 particles
.. image:: images/user/j_slab_1.png
   :width: 266
   :alt: Reference v computed current, 1e3 particles
.. image:: images/user/af_slab_1.png
   :width: 266
   :alt: Reference v computed angular flux, 1e3 particles

Notice how crappy these solutions are? We only ran 1e3 particles.
We need more particles to get a less statistically noise, more converged solution.
Here is the same simulation with 1e6 particles.

.. image:: images/user/sf_slab_2.png
   :width: 266
   :alt: Reference v computed scalar flux, 1e6 particles
.. image:: images/user/j_slab_2.png
   :width: 266
   :alt: Reference v computed current, 1e6 particles
.. image:: images/user/af_slab_2.png
   :width: 266
   :alt: Reference v computed angular flux, 1e6 particles

Its matching much closer with the analytic solution.
As with everything else, the best way to see what you can do is sniff around the examples.
We have examples with animated solutions, subplots, moving regions and more!

-------------------------------------
MC/DC's built in model ``visualizer``
-------------------------------------