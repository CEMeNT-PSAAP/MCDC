.. _writing_numba_compatible_transport_code:

=======================================
Writing Numba-Compatible Transport Code
=======================================

Use this page when verified Python transport behavior is being prepared for Numba-CPU, Numba-GPU, and long-term maintenance in MC/DC.
Read :doc:`../architecture/python_first_numba_accelerated_design` for the development rationale and :doc:`../architecture/transport_execution` for the execution mechanisms.
If the change introduces new model state, begin with :doc:`extending_the_object_model`.

Choose the Host or Transport Layer
----------------------------------

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
----------------------

Keep the verified Python result as the behavioral baseline while adapting the method to MC/DC's packed runtime inputs and compiler-compatible operations.
Decorate the maintained function with ``@njit`` like the surrounding transport functions; in Python mode, MC/DC disables JIT compilation and calls the same function as Python.

.. code-block:: python

   from numba import njit


   @njit
   def apply_weight_factor(particle_container, factor):
       particle = particle_container[0]
       particle["w"] *= factor

Keep the function small enough that its inputs, outputs, and mutations are clear.
Exercise the maintained function in Python mode against the baseline, then use Numba-CPU to resolve typing issues, and finally validate device compatibility in Numba-GPU.

Use Runtime Data, Not Model Objects
-----------------------------------

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
-----------------

Numba determines a compiled function's types from its arguments and control flow.
Make those types unambiguous:

- Initialize local variables on every path before use.
- Return the same number and compatible types of values from every branch.
- Avoid changing a variable from a scalar to an array or from an integer to an unrelated object.
- Use explicit integer and floating-point constants when their width or signedness affects an operation.
- Keep structured-record field names fixed; do not compute field names at run time.
- Avoid heterogeneous Python lists, dictionaries, sets, generators, and dynamically created classes in transport.

Numba may accept a construct on the CPU without making it available on the GPU.
When a function is shared, compatibility with the stricter target is the relevant standard.

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
-----------------------------------------

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
---------------------

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
-------------------------------

Particle histories repeatedly execute transport functions, so temporary allocation can be both expensive and difficult to support consistently across targets.

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

Stay Within the CPU/GPU Common Subset
-------------------------------------

For code intended to run on both targets:

- Use numerical control flow and operations already established in nearby shared transport modules.
- Avoid Python exceptions as ordinary control flow.
- Keep file access, printing, timing, MPI orchestration, and other host services outside device functions.
- Do not use Numba ``objmode`` in a path that must execute on the GPU.
- Avoid relying on a CPU-only Numba feature merely because Numba-CPU compiles it.
- Keep target-specific atomics, memory operations, and scheduling behind the existing GPU adaptation layer.

Backend-specific code is appropriate when the execution model truly differs.
Keep the shared numerical operation in ``mcdc/transport`` when possible, and put only the required adaptation under ``mcdc/code_factory/gpu``.
Document why the paths differ and test their physical equivalence.

For example, keep the shared numerical operation free of reporting or device-management code:

.. code-block:: python

   @njit
   def apply_survival_biasing(particle, survival_probability):
       particle["w"] *= survival_probability


   def report_survival_biasing(survival_probability):
       print(f"Survival probability: {survival_probability}")

Debug in Layers
---------------

When a change fails, isolate the layer:

#. **Python construction** -- confirm that the model compiles and the expected objects, IDs, offsets, and data are present.
#. **Python transport** -- verify the algorithm and state mutation with JIT disabled.
#. **Numba-CPU** -- resolve typing, unsupported-operation, and compiled-runtime failures.
#. **Numba-GPU** -- resolve device compilation, memory, atomic, and scheduling failures.

Do not begin by diagnosing a GPU compiler error if the same calculation is already incorrect in Python.
Conversely, a passing Python test does not prove that the function is type-stable or device compatible.

For example, investigate an unexpected physical result in Python transport, a Numba ``TypingError`` during Numba-CPU porting, and a device-link or unsupported-atomic error in the Numba-GPU layer.

Verification Checklist
----------------------

Before considering a transport extension complete:

- The model compiles into the expected packed representation.
- Focused unit tests exercise the new numerical behavior in Python.
- The same tests or representative regression case pass in Numba-CPU mode.
- Python and Numba-CPU results agree within the test's numerical tolerance.
- A supported GPU environment exercises the path.
- GPU results preserve the same physical behavior within appropriate numerical and statistical tolerances.
- Existing examples still construct successfully when the public API or model compilation changed.
- User and developer documentation describe any new behavior or limitation.

Use the :doc:`../../contributing/index` for repository commands, continuous-integration coverage, and regression-test options.
For changes affecting public inputs, follow :doc:`../../contributing/example_validation`.
