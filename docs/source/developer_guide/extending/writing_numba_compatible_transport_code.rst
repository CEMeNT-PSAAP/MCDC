.. _writing_numba_compatible_transport_code:

=======================================
Writing Numba-Compatible Transport Code
=======================================

Maintained transport code in MC/DC is written in Python but may execute as Python, Numba-compiled CPU code, or Numba-compiled GPU device code.
A method may be prototyped within the MC/DC Python backend before it is adapted for the two accelerated backends.

Read :doc:`../architecture/python_first_numba_accelerated_design` first for the reason behind this development model.
If the change introduces new model state, also read :doc:`extending_the_object_model` and :doc:`../architecture/runtime_data_layout`.

The rules below apply when an MC/DC Python prototype is being made portable and maintained as part of MC/DC.
They are not restrictions on temporary Python-only behavior during initial exploration.

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

Work Python First
-----------------

An exploratory method may begin in MC/DC Python, using whichever Python data structures make the algorithm easiest to understand and verify.
It still runs through MC/DC's normal model preparation and transport path.
This Python-only implementation is a valid stopping point for a small study that does not need compiled performance.

At this stage, use module-level global state, arbitrary Python objects, ad hoc imports, dynamic behavior, file I/O, callbacks, visualization, hard-coded assumptions, or direct transport edits whenever they help answer the research question.
Calls to packages such as SciPy and Matplotlib are acceptable even during transport.
A formal prototype service or abstraction is not required.

Keep prototype-only dependencies local to the experiment rather than adding them to MC/DC's required dependencies.
The prototype is allowed to be disposable and is not expected to compile unchanged.

Unrestricted prototyping is optional.
A developer already familiar with MC/DC and Numba may begin with packed runtime inputs, stable types, and supported operations, producing an implementation that is nearly Numba-compatible from the start.
This can minimize the deliberate porting work while preserving the freedom to use ordinary Python whenever it helps establish the method.

Porting for accelerated execution begins by adapting the verified method so that the required information and behavior use the packed runtime inputs available to transport.
Decorate the maintained function with ``@njit`` like the surrounding transport functions; in Python mode, MC/DC disables JIT compilation and calls the same function as Python.

.. code-block:: python

   from numba import njit


   @njit
   def apply_weight_factor(particle_container, factor):
       particle = particle_container[0]
       particle["w"] *= factor

Keep the function small enough that its inputs, outputs, and mutations are clear.
First exercise it in Python mode to verify the algorithm and obtain an ordinary traceback.
Then run it in Numba-CPU mode to verify type inference.
Finally, adapt and validate it for Numba-GPU.
Each stage may reveal constraints that were irrelevant to the preceding implementation.

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

Debug in Layers
---------------

When a change fails, isolate the layer:

#. **Python construction** -- confirm that the model compiles and the expected objects, IDs, offsets, and data are present.
#. **Python transport** -- verify the algorithm and state mutation with JIT disabled.
#. **Numba-CPU** -- resolve typing, unsupported-operation, and compiled-runtime failures.
#. **Numba-GPU** -- resolve device compilation, memory, atomic, and scheduling failures.

Do not begin by diagnosing a GPU compiler error if the same calculation is already incorrect in Python.
Conversely, a passing Python test does not prove that the function is type-stable or device compatible.

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

A general feature merged and maintained as part of MC/DC must complete all three execution stages unless the maintainers explicitly accept and document a backend-specific limitation.
Experimental or study-specific work may stop at Python or Numba-CPU when that stage already satisfies its purpose.

Repository commands, continuous-integration coverage, and regression-test options belong in the :doc:`../../contributing/index`.
For changes affecting public inputs, follow :doc:`../../contributing/example_validation`.
