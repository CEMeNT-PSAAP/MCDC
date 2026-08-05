.. _writing_numba_compatible_transport_code:

=======================================
Writing Numba-Compatible Transport Code
=======================================

Use this page when verified Python transport behavior is being prepared for Numba-CPU execution and long-term maintenance in MC/DC.
Read :doc:`../architecture/python_first_numba_accelerated_design` for the development rationale.
If the change introduces new model state, begin with :doc:`extending_the_object_model`.

Numba-CPU Development
---------------------

Complete the Python and Numba-CPU implementation before considering additional execution targets.

Choose the Host or Transport Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Place work on the host when it changes the model or prepares execution:

- Validate and normalize user input.
- Traverse Python objects or inspect their classes.
- Add, remove, or resize model data.
- Read files, construct tables, or derive run-wide configuration.
- Allocate the packed state needed by transport.

Place work in transport when it must be performed during particle execution:

- Inspect or mutate particle records.
- Evaluate geometry or physics from prepared numerical data.
- Sample distributions and update the random-number state.
- Score tallies or update preallocated banks and counters.

Do as much irregular work as practical before transport.
A small amount of host-side preparation can turn a dynamic operation into simple indexed access inside a frequently called kernel.

For example, derive a reusable coefficient once while preparing the model instead of recomputing it for every particle event:

.. code-block:: python

   # On the model class
   inverse_dx: float

   def _prepare_spacing(self):
       self.inverse_dx = 1.0 / self.dx

   # In transport
   index = int((particle["x"] - mesh["x0"]) * mesh["inverse_dx"])

Port a Verified Method
^^^^^^^^^^^^^^^^^^^^^^

Keep the verified Python result as the behavioral baseline while adapting the method to MC/DC's packed runtime inputs and compiler-compatible operations.
Decorate the maintained function with ``@njit`` like the surrounding transport functions; in Python mode, MC/DC disables JIT compilation and calls the same function as Python.

.. code-block:: python

   from numba import njit


   @njit
   def apply_weight_factor(particle_container, factor):
       particle = particle_container[0]
       particle["w"] *= factor

Keep the function small enough that its inputs, outputs, and mutations are clear.
Exercise the maintained function in Python mode against the baseline, then use Numba-CPU to resolve typing and compiled-runtime issues.

Use Runtime Data, Not Model Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An integrated transport function should accept scalars, NumPy arrays, structured records, one-element record containers, and the packed ``simulation`` and ``data`` state.
It should not accept an ``MCDCObject`` instance or follow Python object references.

Use the representation appropriate to each field:

- Read fixed-size values directly from a structured record.
- Follow registered relationships with their integer IDs.
- Use ``sub_type`` and ``sub_ID`` for polymorphic dispatch.
- Use generated ``mcdc_get`` and ``mcdc_set`` helpers for variable-length fields in ``data``.
- Mutate only state allocated during runtime preparation.

If the required value is unavailable in this form, extend the object model and generated runtime layer before writing the transport behavior.
Do not create a parallel Python-only lookup inside the kernel.

For example, recover a prepared distribution by simulation-local ID instead of passing its Python model object into transport:

.. code-block:: python

   # Python model construction
   source_object.energy_group_pmf = distribution_object

   # Portable transport representation
   distribution = simulation["distributions"][source["energy_group_pmf_ID"]]

Keep Types Stable
^^^^^^^^^^^^^^^^^

Numba determines a compiled function's types from its arguments and control flow.
Make those types unambiguous:

- Initialize local variables on every path before use.
- Return the same number and compatible types of values from every branch.
- Avoid changing a variable from a scalar to an array or from an integer to an unrelated object.
- Use explicit integer and floating-point constants when their width or signedness affects an operation.
- Keep structured-record field names fixed; do not compute field names at run time.
- Avoid heterogeneous Python lists, dictionaries, sets, generators, and dynamically created classes in transport.

For example, initialize a scalar result before control flow so every path returns the same type:

.. code-block:: python

   @njit
   def find_energy_bin(E, grid):
       index = -1
       for i in range(len(grid) - 1):
           if grid[i] <= E < grid[i + 1]:
               index = i
               break
       return index

Represent Variable-Length Data Explicitly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Do not recover variable-length model data by allocating a new Python container.
Read it from ``data`` using the metadata in its owning record.
Prefer the generated accessor because it preserves the layout convention:

.. code-block:: python

   value = mcdc_get.tally.energy(index, tally, data)

An ``*_all`` or ``*_chunk`` helper may expose a view when an algorithm needs a range.
Element access is preferable when the kernel only needs one value.
Do not resize the returned view or retain it beyond the prepared run.

For a new variable-length field, declare the field on the Python model class and regenerate its accessors as described in :doc:`extending_the_object_model`.
Do not hand-maintain offset arithmetic in several transport modules.

Use Explicit Dispatch
^^^^^^^^^^^^^^^^^^^^^

Dispatch on integer constants through a small interface function rather than using Python ``isinstance`` checks or methods on runtime records:

.. code-block:: python

   @njit
   def get_mesh_x(index, mesh, simulation, data):
       sub_ID = mesh["sub_ID"]
       if mesh["sub_type"] == MESH_STRUCTURED:
           concrete = simulation["structured_meshes"][sub_ID]
           return mcdc_get.structured_mesh.x(index, concrete, data)
       if mesh["sub_type"] == MESH_UNIFORM:
           concrete = simulation["uniform_meshes"][sub_ID]
           return concrete["x0"] + index * concrete["dx"]
       return 0.0

Use the actual category interface and constants already defined for that family.
When adding a subtype, update every exhaustive dispatch site and test an unsupported value deliberately if the interface defines fallback behavior.

Control Allocation and Mutation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Particle histories repeatedly execute transport functions, so temporary allocation can be expensive and difficult for Numba to optimize.

- Reuse particle containers, banks, tally arrays, and scratch state allocated during preparation.
- Update structured fields or ``data`` through explicit assignments and generated setters.
- Prefer scalar calculations or bounded loops over constructing intermediate Python collections.
- Do not append a new model entity or change a field's shape during transport.
- Preserve the one-element container convention when a mutable record must be shared across function boundaries.

The owner of each mutation should be evident from the function signature.
Avoid hidden module-level mutable state.

For example, select a minimum with scalar state instead of building a temporary list:

.. code-block:: python

   best_distance = INF
   for i in range(cell["N_surface"]):
       distance = distance_to_surface(i, particle, cell, simulation, data)
       if distance < best_distance:
           best_distance = distance

When a scalar structured record must be mutated across a function boundary, pass its one-element container and recover the record inside the function:

.. code-block:: python

   @njit
   def increment_counter(counter_container):
       counter = counter_container[0]
       counter["value"] += 1

Debug Python and Numba-CPU
^^^^^^^^^^^^^^^^^^^^^^^^^^

When a change fails, isolate the layer:

#. **Python construction** -- confirm that the model compiles and the expected objects, IDs, offsets, and data are present.
#. **Python transport** -- verify the algorithm and state mutation with JIT disabled.
#. **Numba-CPU** -- resolve typing, unsupported-operation, and compiled-runtime failures.

Resolve each layer before moving to the next.
A passing Python test establishes behavior but does not establish Numba type compatibility.
For example, investigate an unexpected physical result in Python transport and a Numba ``TypingError`` during Numba-CPU porting.

Verify Numba-CPU Support
^^^^^^^^^^^^^^^^^^^^^^^^

Before considering the Numba-CPU implementation complete:

- The model compiles into the expected packed representation.
- Focused unit tests exercise the numerical behavior in Python.
- The same tests or a representative regression case pass in Numba-CPU mode.
- Python and Numba-CPU results agree within the test's numerical tolerance.
- Existing examples still construct successfully when the public API or model compilation changed.
- User and developer documentation describe the new behavior and any CPU limitation.

Numba-GPU Development
---------------------

Numba-CPU is a valid final implementation when it satisfies the intended workloads and project requirements.
Add Numba-GPU support as a later phase when those requirements call for accelerator execution.
Begin this phase from a verified Numba-CPU implementation and retain its tests as the behavioral baseline.
Read :doc:`../architecture/transport_execution` for MC/DC's GPU compilation and runtime architecture.

Apply Additional GPU Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Numba-CPU supports Python and Numba features that may not be available in device code.
For transport code that will execute on a GPU:

- Use numerical control flow and operations established in nearby GPU-compatible transport modules.
- Avoid Python exceptions as ordinary control flow.
- Keep file access, printing, timing, MPI orchestration, and other host services outside device functions.
- Do not use Numba ``objmode`` in a device path.
- Replace CPU-only Numba features with device-compatible operations.
- Keep target-specific atomics, memory operations, and scheduling behind the existing GPU adaptation layer.

Separate Shared and GPU-Specific Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Keep the shared numerical operation in ``mcdc/transport`` when possible, and place only the required device adaptation under ``mcdc/code_factory/gpu``.
Document why a GPU path differs and test its physical equivalence with Python and Numba-CPU.

For example, keep reporting on the host while the numerical operation remains available to compiled transport:

.. code-block:: python

   @njit
   def apply_survival_biasing(particle, survival_probability):
       particle["w"] *= survival_probability


   def report_survival_biasing(survival_probability):
       print(f"Survival probability: {survival_probability}")

Debug Numba-GPU
^^^^^^^^^^^^^^^

Start GPU diagnosis only after the Python and Numba-CPU tests pass.
Then isolate device compilation, memory placement, atomics, and Harmonize scheduling.
For example, investigate a device-link or unsupported-atomic error in this phase without reopening already verified CPU behavior.

Verify Numba-GPU Support
^^^^^^^^^^^^^^^^^^^^^^^^

Before claiming Numba-GPU support:

- The Python and Numba-CPU verification remains passing.
- A supported GPU environment exercises every device path being claimed.
- GPU results preserve the same physical behavior within appropriate numerical and statistical tolerances.
- GPU-specific adaptations and limitations are documented.

Use the :doc:`../../contributing/index` for repository commands, continuous-integration coverage, and regression-test options.
For changes affecting public inputs, follow :doc:`../../contributing/example_validation`.
