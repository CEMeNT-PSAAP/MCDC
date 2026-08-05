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

Choose Between Model Compilation and Particle Transport
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First decide whether the change belongs to the Python model or to the algorithms executed during particle transport.

Place model definition and model-finalization work in ``mcdc/object_/`` when it changes or completes the model before execution:

- Validate and normalize user input.
- Traverse Python objects or inspect their classes.
- Add, remove, or resize model data.
- Read files, construct tables, or derive run-wide configuration.
- Define the state that must later be available to transport.

Use an object's ``_compile_into_simulation`` hook when the work belongs to that object and use ``Simulation._finalize_compilation`` when it requires the complete discovered model.
``compile_simulation`` in ``mcdc/code_factory/`` coordinates those phases and should change only when the compilation framework itself gains a new phase or registered category.

``mcdc.main.prepare`` and the runtime generators in ``mcdc/code_factory/`` are framework-level machinery that pack the finalized model, allocate execution resources, and configure execution backends.
Most scientific-method additions should not change them; extend them only when the runtime representation or execution framework cannot express the required behavior.
See :doc:`extending_the_object_model` for the model-compilation workflow.

Place event-time numerical work in ``mcdc/transport/`` when it must be performed during particle execution:

- Inspect or mutate particle records.
- Evaluate geometry or physics from prepared numerical data.
- Sample distributions and update the random-number state.
- Score tallies or update preallocated banks and counters.

Do as much irregular work as practical during model compilation.
A small amount of object-side finalization can turn a dynamic operation into simple indexed access inside a frequently called kernel.

For example, derive a reusable coefficient in the model object's compilation hook instead of recomputing it for every particle event:

.. code-block:: python

   # On the model class
   inverse_dx: float

   def _compile_into_simulation(self, simulation):
       if not super()._compile_into_simulation(simulation):
           return False

       self.inverse_dx = 1.0 / self.dx
       return True

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

Use Named Constants and Explicit Dispatch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The packed runtime representation expresses categories, events, and other discrete states as primitive numerical values rather than Python types.
``mcdc.constant`` gives those values shared names and also defines common numerical limits and tolerances.
These definitions form a static implementation contract and retain the same meaning across simulations.
Use the named constants instead of repeating their numerical values in model, transport, scoring, or output code.

Dispatch on those constants through a small interface function rather than using Python ``isinstance`` checks or methods on runtime records:

.. code-block:: python

   @njit
   def get_mesh_x(index, mesh, simulation, data):
       sub_ID = mesh["sub_ID"]
       if mesh["sub_type"] == MESH_STRUCTURED:
           structured_mesh = simulation["structured_meshes"][sub_ID]
           return mcdc_get.structured_mesh.x(index, structured_mesh, data)
       if mesh["sub_type"] == MESH_UNIFORM:
           uniform_mesh = simulation["uniform_meshes"][sub_ID]
           return uniform_mesh["x0"] + index * uniform_mesh["dx"]
       return 0.0

Use the actual category interface and constants already defined for that family.
When introducing a new subtype, event, or score, give its constant a unique value within the corresponding family and update every consumer of that family.
Test an unsupported value deliberately when the interface defines fallback behavior.
Do not place a size or setting derived from one simulation in ``constant.py``; simulation-specific values that must be visible to Numba as compile-time values use the generated literals described in :ref:`simulation_specific_literals`.

Control Allocation and Mutation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Transport functions execute repeatedly for every particle history, so they should normally operate on storage prepared before transport begins.
Reuse particle banks, tally arrays, and other prepared runtime state instead of rebuilding them inside the particle loop.
Update structured fields or ``data`` through explicit assignments and generated setters, and do not append model entities or change field shapes during transport.

Prefer scalar state and bounded loops when an intermediate container is unnecessary.
For example, select a minimum directly instead of building a temporary list of distances:

.. code-block:: python

   best_distance = INF
   for i in range(cell["N_surface"]):
       distance = distance_to_surface(i, particle, cell, simulation, data)
       if distance < best_distance:
           best_distance = distance

Some transport operations genuinely require new local storage.
Fission and scattering create secondary-particle records during transport, while geometry and physics routines may require small bounded work arrays.
Use ``util.local_array`` with a known shape and stable dtype for these cases.
In Numba-CPU mode, it provides a NumPy array that Numba can compile and mutate within the transport function.

For example, fission allocates one reusable record for newly generated particles, initializes it for each secondary, and passes its container to a particle bank:

.. code-block:: python

   particle_container_new = util.local_array(1, type_.particle_data)
   particle_new = particle_container_new[0]

   for _ in range(N):
       particle_module.copy_as_child(
           particle_container_new, particle_container
       )
       particle_new["w"] = weight_product
       particle_bank_module.bank_census_particle(
           particle_container_new, program
       )

Use generated structured dtypes from ``mcdc.numba_types`` for local transport records.
Keep local work arrays small and predictable, and reuse the same allocation within a loop when its contents can be overwritten.
Make every mutation visible through function arguments rather than hidden module-level mutable state.
The one-element container preserves the mutable record across function boundaries; the next section explains when functions should receive such a container and when they should receive the record itself.

Use Runtime Arguments Deliberately
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MC/DC uses ``particle_container``, ``simulation_container``, and ``program`` for different purposes.
They are calling conventions that preserve mutation and backend portability, not interchangeable names for simulation state.

``particle_container``
   A one-element structured array that owns one mutable particle record.
   Pass the container when a function must mutate the particle and recover the record locally for field access.
   The container may be caller-owned, a one-element view into a particle bank, or local storage created with ``util.local_array``.

   .. code-block:: python

      @njit
      def reduce_weight(particle_container, factor):
          particle = particle_container[0]
          particle["w"] *= factor

``simulation_container``
   A one-element structured array that owns the mutable top-level ``simulation`` record.
   It establishes storage and lifetime at an execution entry point; ordinary transport functions normally receive the recovered record rather than the container.

   .. code-block:: python

      def fixed_source_simulation(simulation_container, data):
          simulation = simulation_container[0]
          settings = simulation["settings"]
          for idx_batch in range(settings["N_batch"]):
              simulation["idx_batch"] = idx_batch

``program``
   A backend-neutral execution handle used when an operation requires execution services such as particle banking or scheduling.
   In Python and Numba-CPU modes, ``program`` is the ``simulation`` record and ``util.access_simulation(program)`` returns it unchanged.
   Treat the handle as opaque even in these modes so the same transport function can later use a GPU implementation.

   .. code-block:: python

      @njit
      def bank_active_particle(particle_container, program):
          simulation = util.access_simulation(program)
          bank = simulation["bank_active"]
          _bank_particle(particle_container, bank)

Accept ``simulation`` when a function only needs prepared simulation state.
Accept ``program`` when it needs backend-dependent execution services, and recover ``simulation`` through ``util.access_simulation``.
Create one-element containers only at ownership or local-storage boundaries; do not wrap every structured record passed between helpers.

Cross the Python Boundary with ``objmode``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Numba's ``objmode`` temporarily returns from compiled Numba-CPU execution to the Python interpreter for one bounded block.
MC/DC uses this boundary when compiled orchestration must invoke a Python service that Numba cannot compile and moving the complete operation outside the compiled driver would obscure its ownership or timing.

Current uses include:

- MPI collectives and particle-bank communication.
- Wall-clock timing through ``MPI.Wtime``.
- Progress and fatal diagnostics provided by ``mcdc.print_``.
- Census-based HDF5 output performed between transport stages.

Keep the ``objmode`` block as small and infrequent as possible because entering the interpreter interrupts compiled execution and adds conversion and dispatch overhead.
Perform the surrounding numerical work in compiled code, and do not use ``objmode`` merely to avoid expressing a maintained numerical algorithm in Numba-compatible form.

If a value produced in Python is used after the block, declare its Numba type on the context manager:

.. code-block:: python

   time_start = 0.0
   with objmode(time_start="float64"):
       time_start = MPI.Wtime()

No output declaration is needed when the block only performs a side effect or mutates an array that was allocated before entering it:

.. code-block:: python

   local_total = np.array([particle_weight], dtype=np.float64)
   global_total = np.zeros(1, dtype=np.float64)
   with objmode():
       MPI.COMM_WORLD.Allreduce(local_total, global_total, MPI.SUM)

``objmode`` is a deliberate Numba-CPU escape hatch, not part of the GPU execution model.
Code intended for Numba-GPU must keep the Python service outside device execution or provide a backend-specific implementation.

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

Use the GPU Program Handle
^^^^^^^^^^^^^^^^^^^^^^^^^^

In Numba-GPU execution, ``program`` is a Harmonize program handle rather than the ``simulation`` record itself.
The GPU adaptation replaces ``util.access_simulation`` so shared transport functions can recover the device-resident simulation state without knowing how the handle stores it.
Do not index ``program`` directly or assume its CPU representation.
The same adaptation replaces ``util.local_array`` with device-local allocation, which is why shared code should use the utility consistently before GPU porting begins.

For example, a GPU entry point recovers state from the handle before calling shared transport behavior:

.. code-block:: python

   def step(program: nb.uintp, particle_input: particle_gpu):
       simulation = access_simulation(program)
       data_ptr = access_data_ptr(program)
       data = harmonize.array_from_ptr(data_ptr, shape, nb.float64)

       particle_container = util.local_array(1, type_.particle)
       particle_container[0] = particle_input
       step_particle(particle_container, program, data)

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
