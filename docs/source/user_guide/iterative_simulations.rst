.. _iterative_simulations:

=====================
Iterative Simulations
=====================

Use an iterative simulation when several calculations share most of a model
but change selected inputs between runs. Common applications include parameter
sweeps, source updates, feedback iterations, and optimization studies.

The workflow reuses one :class:`mcdc.Simulation` and its Python model objects.
Each iteration still prepares a complete, consistent simulation for transport;
MC/DC does not patch only the changed value into a previous execution.

Build the Shared Model Once
---------------------------

Construct the geometry, materials, sources, tallies, settings, and techniques
as usual. Attach them to one simulation before beginning the iteration. Objects
that do not change can remain attached throughout the study.

Use separate simulations only when the calculations require independent object
graphs. See :doc:`simulation_lifecycle` for the normal construction workflow
and process-ownership rules.

Update, Compile, and Run
------------------------

Change the relevant Python objects at the beginning of each iteration. For
example, the relative probabilities of two sources can be varied while the
geometry, material, and tally remain unchanged:

.. code-block:: python

   source_mixes = (
       ("left_20", 0.2),
       ("left_50", 0.5),
       ("left_80", 0.8),
   )

   for case_name, left_fraction in source_mixes:
       source_left.probability = left_fraction
       source_right.probability = 1.0 - left_fraction
       simulation.settings.output_name = f"source_mix_{case_name}"

       simulation.compile()
       simulation.run()

The explicit ``compile`` call establishes a fresh model snapshot after each
partial update. It is particularly useful when the updated model will be
inspected or visualized before execution.

``simulation.run()`` automatically compiles an unprepared simulation and marks
it unprepared again after execution. A simple loop that only updates and runs
may therefore omit the explicit ``compile`` call. If an attached object is
modified after an explicit compilation or visualization, compile again before
inspecting, visualizing, or running the updated model.

Keep Outputs Distinct
---------------------

Assign a unique ``simulation.settings.output_name`` before every run so that a
later iteration does not replace an earlier result. Keep tally names stable
when the same post-processing operation will be applied to every output.

The driver or post-processing script can record the iteration parameters next
to each output name. This makes plots and comparisons reproducible without
depending on internal object identifiers.

Complete Example
----------------

See :ref:`example_iterative_source_reweighting` for a complete runnable input.
Its post-processing script overlays the flux profiles from three source
mixtures and checks their expected symmetry.
