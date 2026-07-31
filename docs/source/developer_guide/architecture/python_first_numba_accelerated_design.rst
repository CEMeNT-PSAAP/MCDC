.. _python_first_numba_accelerated_design:

======================================
Python-First, Numba-Accelerated Design
======================================

MC/DC is designed for Monte Carlo methods development in Python while retaining
an execution path suitable for large transport calculations. Python is the
language in which users construct models and developers express transport
algorithms. Numba is the acceleration layer that specializes those algorithms
for CPUs and helps make them available to GPU execution.

*Python-first* describes the order in which MC/DC is understood and developed:
establish correct behavior in Python, accelerate the same typed transport path
with Numba-CPU, and then satisfy the additional constraints of Numba-GPU and
Harmonize. It does not mean that transport operates directly on the rich
Python object graph.

Where Each Layer Runs
---------------------

MC/DC deliberately separates flexible model construction from constrained
transport execution:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Stage
     - Representation
     - Execution environment
   * - Model definition
     - User-facing Python objects and references
     - Python
   * - Model compilation
     - Simulation-owned, ordered object graph
     - Python
   * - Runtime preparation
     - Structured ``simulation`` state and flat ``data``
     - Python and NumPy
   * - Transport
     - Typed functions operating on packed state
     - Python, Numba-CPU, or Numba-GPU
   * - Output and postprocessing
     - HDF5 output and analysis objects
     - Python

The Python stages may use object-oriented interfaces, variable-length
collections, validation, and other features that make model construction
clear. The transport stage uses stable numerical types, arrays, integer IDs,
and explicit function calls that compilation targets can understand.

Why MC/DC Uses Numba
--------------------

Numba lets MC/DC preserve a Python implementation of its numerical algorithms
while compiling type-stable functions for a hardware target at run time. This
supports several project goals:

- Researchers can implement and inspect transport methods in Python.
- Python mode provides a direct path for debugging algorithm and data-flow
  errors.
- Numba-CPU can specialize the same transport functions without maintaining a
  separate implementation in another language.
- The typed runtime representation and transport interfaces provide a
  foundation for GPU code generation and Harmonize scheduling.

The result is one conceptual transport implementation, with backend-specific
adaptation where the hardware requires it. GPU support is not automatic for
arbitrary Python or arbitrary Numba-CPU code; transport intended for all three
modes must stay within their supported common subset.

Staged Methods Development
--------------------------

Python-first also describes how new methods are typically developed. A method
usually begins as a pure-Python prototype, either independently or in MC/DC's
Python execution mode. At this stage, the priority is to establish the
method's correctness, understand its behavior, and test it on problems small
enough that compiled performance is unnecessary.

Once the method is verified and larger calculations require more performance,
the implementation can advance to Numba-CPU. This step is more than enabling
JIT compilation: Python constructs that Numba cannot type or compile may need
to be replaced with explicit numerical types, stable control flow, packed
runtime data, and supported operations.

Numba-GPU is a further stage. Device execution typically imposes additional
constraints involving memory placement, allocation, host interaction, atomics,
and Harmonize scheduling. Some code that works with Numba-CPU therefore
requires further adaptation before it can execute on a GPU.

Each stage is a useful endpoint:

- **Python** is sufficient for prototyping methods, diagnosing behavior, and
  running small problems.
- **Numba-CPU** is sufficient when compiled CPU performance meets the
  calculation's needs.
- **Numba-GPU** provides the final portability stage when accelerator
  performance is required.

This staged workflow allows researchers to stop when their immediate objective
has been achieved. A general feature intended to be merged and maintained as
part of MC/DC, however, must work in Python, Numba-CPU, and Numba-GPU unless
the maintainers explicitly accept and document a backend-specific limitation.

An independent pure-Python prototype may use data structures chosen only for
exploration. Once integrated into MC/DC, however, Python mode still performs
model compilation and runtime preparation. It is the reference execution mode
for the same packed transport path that later stages accelerate.

Architectural Consequences
--------------------------

Using Numba shapes the boundary between MC/DC's model and execution layers.
The following are architectural requirements rather than incidental
implementation details.

Model Objects Are Not Runtime Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Materials, cells, sources, tallies, and related objects are convenient
descriptions of a model. Before transport, MC/DC discovers the complete object
graph and converts Python references into simulation-local IDs. Transport
receives packed records and arrays, not instances of ``MCDCObject``.

See :doc:`simulation_compilation` for the owning ``Simulation`` context,
recursive discovery, and the ``MCDCBase``--``MCDCObject``--
``MCDCPolymorphic`` hierarchy.

Execution Data Has a Stable Layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The types and collection sizes needed by a run are finalized before transport.
Fixed-layout values are stored in the structured ``simulation`` record.
Variable-length and ragged numerical values are flattened into ``data`` and
described by offsets, lengths, and shapes.

This representation avoids dynamic Python containers in transport and gives
all backends the same logical state. Generated ``mcdc_get`` and ``mcdc_set``
helpers centralize the corresponding indexing rules. See
:doc:`runtime_data_layout` for the complete representation.

Dispatch Is Explicit
^^^^^^^^^^^^^^^^^^^^

Python dynamic dispatch is replaced by integer type codes and explicit
branches in transport interfaces. A polymorphic base record stores
``sub_type`` and ``sub_ID``; transport uses them to select the concrete packed
record and implementation.

This makes the possible execution paths visible to Numba and keeps runtime
behavior independent of Python object identity.

Mutation Is Bounded
^^^^^^^^^^^^^^^^^^^

Transport updates particle state, banks, tallies, and counters in storage
allocated during preparation. It does not add model objects or resize the
model's runtime layout. A model change therefore produces a new compilation
snapshot and a newly prepared runtime representation.

CPU and GPU Share a Deliberate Subset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Numba-CPU supports some operations that are unavailable or unsuitable inside a
GPU device function. Code shared by the two targets must use explicit numeric
types, supported NumPy operations, predictable control flow, and preallocated
state. Host callbacks, Python objects, and object-mode escapes must remain
outside GPU transport paths.

MC/DC may use target-specific modules when execution, memory, atomics, or
scheduling genuinely differs. The common transport path should remain the
default so that backend implementations do not drift apart.

Design Tradeoffs
----------------

The architecture makes methods development and compiled execution share a
language and much of an implementation, but the boundary has costs:

- Runtime-visible state must be declared and packed before execution.
- Dynamic Python conveniences cannot be used inside compiled transport paths.
- New model concepts often require coordinated object, layout, accessor,
  transport, and test changes.
- Just-in-time compilation adds startup work and can produce typing errors that
  do not appear in Python mode.
- GPU support may require additional code generation, memory management, and
  backend-specific implementations.

These constraints are accepted at the execution boundary so that the
user-facing model and host-side development workflow can remain expressive.

Related Design Choices
----------------------

Other designs emphasize different tradeoffs. A pure-Python solver would
maximize dynamism but would not provide MC/DC's compiled execution path.
Handwritten C, C++, or Fortran kernels could provide direct low-level control
but would split methods development across languages and require a binding
boundary. Passing the object graph directly to execution would retain its
structure but make types, ownership, and accelerator data movement harder to
control. Padding every ragged field into a rectangular array would simplify
some indexing while increasing storage and obscuring the field's natural
shape.

MC/DC instead uses rich Python model objects on the host and a generated,
typed, flat representation for execution. This section describes the current
design space; it should not be read as a claim that every alternative was
implemented and benchmarked by the project.

Future Evolution
----------------

The Python-first principle does not require the present implementation to
remain fixed. Future work may revise compilation-state ownership, generated
layer lifecycles, typed-container support, or the boundary between common and
backend-specific transport. Such changes should preserve three properties:

- A clear Python path for developing and diagnosing transport behavior.
- An explicit conversion from user model to execution state.
- Equivalent physical behavior across supported execution backends.

For the mechanisms behind this boundary, see :doc:`simulation_compilation` and
:doc:`runtime_data_layout`. If following the main Architecture path, return to
the Numba-CPU section of :doc:`python_numba_cpu_execution`, then continue with
:doc:`numba_gpu_execution`. When implementing a transport change, use
:doc:`../extending/writing_numba_compatible_transport_code`.
