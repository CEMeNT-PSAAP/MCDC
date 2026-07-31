.. _python_first_numba_accelerated_design:

======================================
Python-First, Numba-Accelerated Design
======================================

MC/DC is a Python-based environment for developing Monte Carlo transport
methods, but large calculations also require compiled performance.
*Python-first* means that developers can build and validate a method with the
full Python language and ecosystem before addressing the requirements of
accelerated execution. Numba provides the bridge between these goals—but what
is Numba, and why is it a good fit for MC/DC?

What Is Numba?
--------------

`Numba <https://numba.readthedocs.io/en/stable/user/5minguide.html>`_ is a
just-in-time (JIT) compiler for numerical Python. When a function marked with a
Numba decorator such as ``@njit`` is called, Numba examines the argument types
and translates supported Python, NumPy, and loop operations into machine code
specialized for those types. Compilation happens as the program runs rather
than in a separate build step.

Numba does not compile the entire Python language. Efficient compilation
requires numerical types, predictable control flow, and supported operations.
Arbitrary Python objects, dynamic behavior, and calls into unsupported
packages remain useful during prototyping but must be removed, replaced, or
moved outside compiled functions.

MC/DC uses Numba as the bridge between Python development and accelerated
transport. Its common transport path supports three *execution backends*, or
environments that run the integrated functions:

- **Python** executes the functions through the interpreter with JIT
  compilation disabled.
- **Numba-CPU** compiles the functions into machine code for the host CPU.
- **Numba-GPU** compiles device functions for an accelerator, while Harmonize
  provides the GPU runtime that schedules particle work.

Why MC/DC Uses Numba
--------------------

Numba lets MC/DC keep its maintained transport algorithms in Python while
specializing them for the model and hardware used by a calculation. This
supports several project goals:

- New transport methods can begin in the language and scientific ecosystem
  already used to construct MC/DC models.
- The integrated Python backend provides ordinary Python tracebacks while
  exercising the same transport functions and data flow used by acceleration.
- Numba-CPU accelerates those functions without requiring a separate
  implementation in C, C++, or Fortran.
- The typed transport interfaces provide a common foundation for CPU and GPU
  execution, with backend-specific adaptation where the hardware requires it.

The choice therefore preserves an expressive starting point while providing a
path to compiled execution. It also introduces the constraints discussed in
the remainder of this page.

Staged Methods Development
--------------------------

Pure-Python Prototype
^^^^^^^^^^^^^^^^^^^^^

A method may begin as a pure-Python prototype, either independently or through
direct modifications to MC/DC. At this stage, the priority is to establish the
method's correctness, understand its behavior, and test it on problems small
enough that compiled performance is unnecessary.

The prototype may use arbitrary Python objects, module-level global state,
manual imports, dynamic dispatch, callbacks, hard-coded assumptions, or direct
edits to transport routines. It may call packages such as SciPy or Matplotlib
during transport, perform file I/O, or visualize intermediate state. These
choices are acceptable because the prototype prioritizes scientific
expressiveness and rapid validation over performance, encapsulation,
dependency discipline, and compiler compatibility.

No prototype framework or service is required. Researchers may use the most
direct implementation that answers the question being studied. Prototype-only
packages do not need to become MC/DC dependencies, and the prototype is not
expected to remain in its original form.

This freedom is permission, not a required style. A developer familiar with
MC/DC and Numba may voluntarily use typed data, the packed representation, and
compiler-compatible operations from the beginning. A nearly Numba-compatible
prototype can substantially reduce later porting work without turning
Python-first development into a compiler-first requirement.

MC/DC Integration and Python Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Integration begins when the method adopts MC/DC's standard execution path.
User-facing model objects are compiled into an ordered, simulation-owned model
and then packed into numerical runtime state. Here, *model compilation* means
discovering connected Python objects and assigning simulation-local integer
IDs; it is distinct from Numba JIT compilation.

The packed runtime has two main parts: a structured ``simulation`` record for
fixed-layout state and a flat ``data`` array for variable-length values. MC/DC
Python mode runs the integrated transport functions against this
representation with JIT compilation disabled. It therefore provides ordinary
Python tracebacks while exercising the same data flow and function interfaces
used by the accelerated backends.

Numba-CPU Compilation
^^^^^^^^^^^^^^^^^^^^^

Once the method is verified and larger calculations require more performance,
Numba JIT compilation can be enabled for the CPU. Unless the prototype was
already written against the compatible subset, reaching this stage requires
deliberate porting rather than simply changing an execution option. Python
constructs that Numba cannot type or compile are removed or replaced with
explicit numerical types, stable control flow, packed runtime data, and
supported operations.

Numba-GPU Compilation
^^^^^^^^^^^^^^^^^^^^^

GPU execution is a further stage. Device code typically imposes additional
constraints involving memory placement, allocation, interaction with the host
CPU, synchronized updates, and Harmonize scheduling. Some code that works with
Numba-CPU therefore requires further adaptation before it can execute on a
GPU.

Valid Stopping Points
^^^^^^^^^^^^^^^^^^^^^

Each stage can be a useful endpoint:

- **Pure-Python prototype** is sufficient for exploring a method and answering
  a focused research question.
- **MC/DC Python** is sufficient for integrated debugging and small
  calculations.
- **Numba-CPU** is sufficient when compiled CPU performance meets the
  calculation's needs.
- **Numba-GPU** provides the final portability stage when accelerator
  performance is required.

This staged workflow allows researchers to stop when their immediate objective
has been achieved. A general feature intended to be merged and maintained as
part of MC/DC, however, must work in MC/DC Python, Numba-CPU, and Numba-GPU
unless the maintainers explicitly accept and document a backend-specific
limitation.

The Standard Execution Path
---------------------------

After prototyping, MC/DC's standard path separates flexible model construction
from constrained transport execution:

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

Model construction and preparation may use object-oriented interfaces,
variable-length collections, validation, and other expressive Python features.
The boundary occurs at integrated transport, where compilation targets require
stable numerical types, arrays, integer IDs, and explicit function calls.

Architectural Consequences
--------------------------

Once a method enters the integrated path, Numba shapes the boundary between
MC/DC's model and execution layers. The following requirements do not constrain
the initial prototype, but they are fundamental to maintained transport code.

Model Objects Are Not Runtime Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Materials, cells, sources, tallies, and related objects are convenient
descriptions of a model. Before integrated transport, MC/DC discovers the
complete object graph and converts Python references into simulation-local
IDs. The maintained transport path receives packed records and arrays, not
instances of ``MCDCObject``, the base class for independently registered model
entities.

See :doc:`simulation_compilation` for the owning ``Simulation`` context,
recursive discovery, and the ``MCDCBase``--``MCDCObject``--
``MCDCPolymorphic`` hierarchy.

Execution Data Has a Stable Layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The types and collection sizes needed by a run are finalized before transport.
Fixed-layout values are stored in the structured ``simulation`` record.
Variable-length numerical values, including nested arrays whose lengths differ,
are flattened into ``data`` and described by offsets, lengths, and shapes.

This representation avoids dynamic Python containers in transport and gives
all backends the same logical state. Generated ``mcdc_get`` and ``mcdc_set``
helpers centralize the corresponding indexing rules. See
:doc:`runtime_data_layout` for the complete representation.

Dispatch Is Explicit
^^^^^^^^^^^^^^^^^^^^

Python dynamic dispatch is replaced by integer type codes and explicit
branches in transport interfaces. For a model category with several concrete
representations, its common record stores ``sub_type`` and ``sub_ID``.
Transport uses those values to select the concrete packed record and
implementation.

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
state. Host callbacks, Python objects, and Numba ``objmode`` blocks—which
temporarily return compiled CPU code to the Python interpreter—must remain
outside GPU transport paths.

MC/DC may use target-specific modules when execution, memory, atomics, or
scheduling genuinely differs. The common transport path should remain the
default so that backend implementations do not drift apart.

Design Tradeoffs
----------------

The staged approach does not eliminate development cost; it defers some of that
cost until a method has demonstrated enough value to justify integration.

An unrestricted Python prototype maximizes scientific expressiveness and
minimizes the effort required to test an idea. The tradeoff is that parts of
the prototype may need to be redesigned or rewritten for Numba-CPU. Beginning
with MC/DC's packed runtime representation and Numba-compatible operations can
reduce that porting effort, but introduces implementation constraints earlier
in the research process.

Once integrated, the shared execution architecture introduces additional
costs:

- Runtime-visible state must be declared and packed before execution.
- Dynamic Python behavior must be removed from compiled transport paths.
- New model concepts may require coordinated changes to objects, layouts,
  accessors, transport functions, and tests.
- Supporting Python, Numba-CPU, and Numba-GPU increases validation work.
- Just-in-time compilation adds startup time and compiler-specific failure
  modes.
- GPU execution may require additional memory management, code generation,
  atomics, and scheduling support.

MC/DC accepts possible prototype rework so that early methods development
remains unconstrained. It accepts the stricter integrated architecture so that
maintained features can share transport logic across execution backends rather
than becoming separate Python, CPU, and GPU implementations.

Related Design Choices
----------------------

Alternative designs place the boundary between expressiveness and portability
elsewhere. Requiring every prototype to follow the Numba-compatible subset
would reduce later porting, but constrain exploration before the method's value
is known. Maintaining a complete Python solver alongside separate optimized
backends would preserve Python freedom indefinitely, but duplicate transport
logic and allow implementations to drift.

Handwritten C, C++, or Fortran kernels would provide direct low-level control,
but split methods development across languages and introduce a binding
boundary. Passing rich Python objects directly to every execution backend
would preserve the prototype representation, but make typing, ownership, and
accelerator data movement difficult to control.

MC/DC instead uses staged convergence. A prototype may begin anywhere on the
spectrum from an ad hoc Python experiment to a nearly Numba-compatible
implementation. When the method needs compiled performance or becomes a
maintained MC/DC capability, it converges on the shared typed transport path.
The generated structured and flat runtime representation provides that common
execution boundary without dictating how the original prototype must be
written.

These are design-space comparisons, not a claim that every alternative was
implemented and benchmarked by the project.

Future Evolution
----------------

The Python-first principle does not require the present implementation to
remain fixed. Future work should make movement between stages easier without
narrowing the freedom of pure-Python exploration.

Potential improvements fall into three areas:

- **Prototype access** -- optionally make the active Python model and arbitrary
  prototype state easier to reach during Python execution without requiring a
  formal prototype framework.
- **Integration assistance** -- improve inspection and conversion of model
  state, generated layouts and accessors, and diagnostics that identify
  compiler-incompatible code.
- **Execution infrastructure** -- evolve where compilation state is stored,
  when generated runtime layers are created and refreshed, how Numba typed
  containers are used, and the boundary between shared and backend-specific
  transport.

Future changes should preserve the ability to stop at any development stage,
keep the transition into maintained execution state explicit and reviewable,
and retain equivalent physical behavior across the supported backends of an
integrated feature.

For the mechanisms behind this boundary, see :doc:`simulation_compilation` and
:doc:`runtime_data_layout`. If following the main Architecture path, return to
the Numba-CPU section of :doc:`python_numba_cpu_execution`, then continue with
:doc:`numba_gpu_execution`. When implementing a transport change, use
:doc:`../extending/writing_numba_compatible_transport_code`.
