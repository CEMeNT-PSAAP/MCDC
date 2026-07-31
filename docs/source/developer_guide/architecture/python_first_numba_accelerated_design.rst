.. _python_first_numba_accelerated_design:

======================================
Python-First, Numba-Accelerated Design
======================================

MC/DC is a Python-based environment for developing Monte Carlo transport
methods, but large calculations also require compiled performance.
*Python-first* means that developers can build and validate a method through
MC/DC's normal Python execution while using the full Python language and
ecosystem. They can address the requirements of accelerated execution later.
Numba provides the bridge between these goals—but what is Numba, and why is it
a good fit for MC/DC?

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
packages remain useful while prototyping in MC/DC Python. They must be removed,
replaced, or moved outside compiled functions before the method can use an
accelerated backend.

MC/DC uses Numba as the bridge between Python development and accelerated
transport. Its common transport path supports three *execution backends*, or
environments that run the transport functions:

- **Python** executes the functions through the interpreter with JIT
  compilation disabled.
- **Numba-CPU** compiles the functions into machine code for the host CPU.
- **Numba-GPU** compiles device functions for an accelerator, while a
  supporting runtime called Harmonize schedules particle work.

Why MC/DC Uses Numba
--------------------

Numba lets MC/DC keep its maintained transport algorithms in Python while
specializing them for the model and hardware used by a calculation. This
supports several project goals:

- New transport methods can begin in the language and scientific ecosystem
  already used to construct MC/DC models.
- The Python backend provides ordinary Python tracebacks while exercising the
  same MC/DC model lifecycle used by acceleration.
- Numba-CPU accelerates those functions without requiring a separate
  implementation in C, C++, or Fortran.
- The typed transport interfaces provide a common foundation for CPU and GPU
  execution, with backend-specific adaptation where the hardware requires it.

The choice therefore preserves an expressive starting point while providing a
path to compiled execution. It also introduces the constraints discussed in
the remainder of this page.

Staged Methods Development
--------------------------

MC/DC Python: Prototyping and Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A new method can be developed through MC/DC's Python backend. It still follows
the normal MC/DC path: the user defines a valid model, MC/DC prepares the
complete model and its numerical execution data, and transport runs through
the Python interpreter with JIT compilation disabled. The preparation process
is described in :doc:`simulation_compilation` and
:doc:`runtime_data_layout`.

Within those minimum requirements, prototype transport code may use arbitrary
Python objects, module-level global state, manual imports, dynamic dispatch,
callbacks, hard-coded assumptions, or direct edits to transport routines. It
may call packages such as SciPy or Matplotlib during transport, perform file
I/O, or visualize intermediate state. It may also reach beyond MC/DC's prepared
transport data through ordinary Python mechanisms. These choices are
acceptable because this stage prioritizes scientific expressiveness and rapid
validation over performance, encapsulation, dependency discipline, and
compiler compatibility.

No special prototype framework or service is required. Researchers may use
the most direct implementation within MC/DC Python that answers the question
being studied. Prototype-only packages do not need to become MC/DC
dependencies, and temporary implementation choices are not expected to remain
in their original form.

This freedom is permission, not a required style. A developer familiar with
MC/DC and Numba may voluntarily begin with simple numerical data and
predictable operations that are already close to the compiled form. This can
substantially reduce later porting work without turning Python-first
development into a compiler-first requirement. Practical guidance is provided
in :doc:`../extending/writing_numba_compatible_transport_code`.

Numba-CPU Compilation
^^^^^^^^^^^^^^^^^^^^^

Once the method is verified and larger calculations require more performance,
it can be adapted for Numba-CPU. Unless the Python implementation was already
written against the compatible subset, reaching this stage requires deliberate
porting rather than simply changing an execution option. Python features that
Numba cannot compile are replaced with supported numerical forms and
predictable control flow. See :doc:`python_numba_cpu_execution` for the
execution path and
:doc:`../extending/writing_numba_compatible_transport_code` for coding
guidance.

Numba-GPU Compilation
^^^^^^^^^^^^^^^^^^^^^

GPU execution is a further stage. Device code typically imposes additional
constraints on how data is stored and moved, how the GPU interacts with the
host CPU, and how concurrent particle work is coordinated. Some code that
works with Numba-CPU therefore requires further adaptation before it can
execute on a GPU. These additional layers are described in
:doc:`numba_gpu_execution`.

Valid Stopping Points
^^^^^^^^^^^^^^^^^^^^^

Each stage can be a useful endpoint:

- **MC/DC Python** is sufficient for prototyping, debugging, and small
  calculations. Prototype code may use unrestricted Python where needed.
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

Every backend follows MC/DC's standard model and execution path. What changes
between development stages is how transport runs and which Python features it
may use:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Stage
     - Representation
     - Execution environment
   * - Model definition
     - Flexible Python objects and relationships
     - Python
   * - Model compilation
     - Complete, internally consistent model snapshot
     - Python
   * - Runtime preparation
     - Compact numerical execution data
     - Python and NumPy
   * - Transport
     - Shared numerical algorithms
     - Python, Numba-CPU, or Numba-GPU
   * - Output and postprocessing
     - HDF5 output and analysis objects
     - Python

Model construction and preparation may use object-oriented interfaces,
variable-length collections, validation, and other expressive Python features.
Transport in the Python backend may also use ordinary Python facilities beyond
the prepared execution data. The accelerated backends require predictable
numerical data and explicit behavior. See
:doc:`simulation_compilation` and :doc:`runtime_data_layout` for the two
preparation stages, then :doc:`python_numba_cpu_execution` and
:doc:`numba_gpu_execution` for backend execution.

Architectural Consequences
--------------------------

All MC/DC backends follow the standard model and execution path. Preparing a
method for acceleration narrows the freedom available inside transport. A
prototype in MC/DC Python may temporarily work around the following
requirements, but portable, maintained transport code must satisfy them.

Model Description Becomes Execution Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Python objects are well suited for describing materials, geometry, sources,
tallies, and their relationships. MC/DC turns this model into a
simulation-specific numerical representation before every execution. A
prototype in MC/DC Python may also access other Python state, but accelerated
transport needs data with predictable types and connections.

The ownership and finalization of the Python model are explained in
:doc:`simulation_compilation`. Its numerical representation is explained in
:doc:`runtime_data_layout`.

The Execution Structure Is Fixed Before Transport
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Preparation determines which model entities and data arrays exist for a run.
Transport can update particle state, scores, banks, and counters, but it does
not add new model entities or resize the model structure. Changing the model
requires preparing a new execution snapshot.

See :doc:`simulation_compilation` for the snapshot lifecycle and
:doc:`runtime_data_layout` for what transport may read and update.

Dynamic Choices Become Explicit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ordinary Python can choose behavior dynamically from object types and methods.
Compiled transport makes those choices visible through numerical tags,
explicit branches, and stable function interfaces. This allows Numba to see
the possible execution paths before generating machine code.

The object relationships behind this conversion are covered in
:doc:`simulation_compilation`. Contributors adding a new representation should
follow :doc:`../extending/extending_the_object_model` and
:doc:`../extending/writing_numba_compatible_transport_code`.

CPU and GPU Share a Deliberate Subset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some operations accepted by Numba-CPU are unavailable or unsuitable in GPU
device code. Transport shared by both targets therefore uses a common set of
numerical operations and predictable control flow. Hardware-specific memory,
synchronization, and scheduling behavior is isolated where the targets truly
differ.

Read :doc:`python_numba_cpu_execution` and :doc:`numba_gpu_execution` for the
two compiled backends. Practical compatibility rules belong in
:doc:`../extending/writing_numba_compatible_transport_code`.

Design Tradeoffs
----------------

The staged approach does not eliminate development cost; it defers some of that
cost until a method has demonstrated enough value to justify acceleration and
long-term maintenance.

Prototyping through MC/DC Python maximizes scientific expressiveness and
minimizes the effort required to test an idea. The tradeoff is that parts of
the Python implementation may need to be redesigned or rewritten for
Numba-CPU. Beginning with simple numerical data and operations that Numba
supports can reduce that porting effort, but introduces implementation
constraints earlier in the research process. See
:doc:`../extending/writing_numba_compatible_transport_code` for the practical
constraints.

Making the method portable across the accelerated backends introduces
additional costs:

- Information used by accelerated transport must be expressible in the
  prepared execution data.
- Dynamic Python behavior cannot remain inside accelerated transport code.
- A new model concept may require coordinated changes to the user interface,
  preparation steps, transport algorithms, and tests.
- Supporting Python, Numba-CPU, and Numba-GPU increases validation work.
- Just-in-time compilation adds startup time and compiler-specific failure
  modes.
- GPU execution may require additional work for data movement, synchronized
  updates, and particle scheduling.

MC/DC accepts possible porting work so that early methods development in
Python remains unconstrained. It accepts stricter requirements for accelerated
execution so that maintained features can share transport logic across
backends rather than becoming separate Python, CPU, and GPU implementations.

The preparation costs are detailed in :doc:`simulation_compilation` and
:doc:`runtime_data_layout`. The backend-specific costs are detailed in
:doc:`python_numba_cpu_execution` and :doc:`numba_gpu_execution`.

Related Design Choices
----------------------

Alternative designs place the boundary between expressiveness and portability
elsewhere. Requiring all transport development to follow the Numba-compatible
subset from the beginning would reduce later porting, but constrain
exploration before the method's value is known. Maintaining a complete Python
solver alongside separate optimized backends would preserve Python freedom
indefinitely, but duplicate transport logic and allow implementations to
drift.

Handwritten C, C++, or Fortran kernels would provide direct low-level control,
but split methods development across languages and introduce a binding
boundary. Passing rich Python objects directly to every execution backend
would preserve Python-mode flexibility, but make typing, ownership, and
accelerator data movement difficult to control.

MC/DC instead uses staged convergence within its standard execution path. A
method in MC/DC Python may range from an ad hoc implementation to one that is
already nearly Numba-compatible. When the method needs compiled performance or
becomes a maintained MC/DC capability, it converges on common numerical data
and transport interfaces. This preserves a shared execution path without
dictating how the Python implementation must begin. The data boundary is
described in :doc:`runtime_data_layout`; contributor guidance is provided in
:doc:`../extending/writing_numba_compatible_transport_code`.

These are design-space comparisons, not a claim that every alternative was
implemented and benchmarked by the project.

Future Evolution
----------------

The Python-first principle does not require the present implementation to
remain fixed. Future work should make movement between stages easier without
narrowing the freedom of prototyping in MC/DC Python.

Potential improvements fall into three areas:

- **Python-mode flexibility** -- make the active Python model and arbitrary
  development state easier to reach during Python execution without requiring
  a formal prototype framework.
- **Porting assistance** -- provide better tools for inspecting a Python
  implementation and identifying code that an accelerated backend cannot run.
- **Execution infrastructure** -- make the creation and ownership of prepared
  data clearer, and keep hardware-specific behavior separated from the common
  transport algorithms.

Future changes should preserve the ability to stop at any development stage,
keep the transition from an MC/DC-Python-only implementation to a portable
maintained implementation explicit and reviewable, and retain equivalent
physical behavior across the supported backends of a maintained feature.

For the preparation process, see :doc:`simulation_compilation` and
:doc:`runtime_data_layout`. For execution, continue with
:doc:`python_numba_cpu_execution` and :doc:`numba_gpu_execution`. When
implementing a transport change, use
:doc:`../extending/writing_numba_compatible_transport_code`.
