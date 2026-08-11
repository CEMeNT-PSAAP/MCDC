.. _python_first_numba_accelerated_design:

======================================
Python-First, Numba-Accelerated Design
======================================

MC/DC is a Python-based environment for developing Monte Carlo transport methods, but large calculations also require compiled performance.
*Python-first* means that developers can build and validate a method through MC/DC's normal Python execution while using the full Python language and ecosystem.
They can address the requirements of accelerated execution later.
Numba provides the bridge between these goals—but what is Numba, and why is it a good fit for MC/DC?

What Is Numba?
--------------

`Numba <https://numba.readthedocs.io/en/stable/user/5minguide.html>`_ is a just-in-time (JIT) compiler for numerical Python.
It translates compatible Python functions into machine code as a calculation runs, allowing MC/DC to accelerate Python transport algorithms without maintaining a second implementation in a lower-level language.

Numba does not support every feature of Python.
A prototype in MC/DC Python may use the full Python language and ecosystem, but an accelerated implementation must express the relevant data and behavior in forms that Numba can compile.

MC/DC uses Numba as the bridge between Python development and accelerated transport.
Its common transport path supports three *execution backends*, or environments that run the transport functions:

- **Python** executes the functions through the interpreter with JIT compilation disabled.
- **Numba-CPU** compiles the functions into machine code for the host CPU.
- **Numba-GPU** compiles functions for an accelerator, with Harmonize providing the supporting GPU runtime.

Why MC/DC Uses Numba
--------------------

Numba lets MC/DC keep its maintained transport algorithms in Python while specializing them for the model and hardware used by a calculation.
This supports several project goals:

- New transport methods can begin in the language and scientific ecosystem already used to construct MC/DC models.
- The Python backend supports unrestricted methods development and ordinary Python debugging within MC/DC's normal model lifecycle.
- Numba-CPU accelerates those functions without requiring a separate implementation in C, C++, or Fortran.
- Common transport interfaces provide a shared foundation for CPU and GPU execution, with backend-specific adaptation where the hardware requires it.

The choice therefore preserves an expressive starting point while providing a path to compiled execution.
Those goals introduce the constraints discussed below.

Staged Methods Development
--------------------------

MC/DC Python: Prototyping and Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A new method can be developed through MC/DC's Python backend.
It still follows the normal MC/DC path: the user defines a valid model, MC/DC prepares the complete model and its numerical execution data, and transport runs through the Python interpreter with JIT compilation disabled.
The preparation process is described in :doc:`simulation_compilation` and :doc:`runtime_data_layout`.

Within those minimum requirements, prototype transport code may use arbitrary Python objects, module-level global state, manual imports, dynamic dispatch, callbacks, hard-coded assumptions, or direct edits to transport routines.
It may call packages such as SciPy or Matplotlib during transport, perform file I/O, or visualize intermediate state.
It may also reach beyond MC/DC's prepared transport data through ordinary Python mechanisms.
These choices are acceptable because this stage prioritizes scientific expressiveness and rapid validation over performance, encapsulation, dependency discipline, and compiler compatibility.

No special prototype framework or service is required.
Researchers may use the most direct implementation within MC/DC Python that answers the question being studied.
Prototype-only packages do not need to become MC/DC dependencies, and temporary implementation choices are not expected to remain in their original form.

This freedom is permission, not a required style.
A developer familiar with MC/DC and Numba may begin with an implementation already close to the compiled form, reducing later porting work without making compiler compatibility a requirement for initial exploration.
Practical guidance is provided in :doc:`../extending/writing_numba_compatible_transport_code`.

Numba-CPU Compilation
^^^^^^^^^^^^^^^^^^^^^

Once the method is verified and larger calculations require more performance, it can be adapted for Numba-CPU.
Unless the Python implementation was already written against the compatible subset, reaching this stage requires deliberate porting rather than simply changing an execution option.
See :doc:`transport_execution` for the execution architecture and :doc:`../extending/writing_numba_compatible_transport_code` for the practical porting requirements.

Numba-GPU Compilation
^^^^^^^^^^^^^^^^^^^^^

GPU execution is a further stage with additional hardware constraints.
Some code that works with Numba-CPU therefore requires further adaptation before it can execute on a GPU.
These additional layers are described in :doc:`transport_execution`.

Valid Stopping Points
^^^^^^^^^^^^^^^^^^^^^

Each stage can be a useful endpoint:

- **MC/DC Python** is sufficient for prototyping, debugging, and small calculations.
  Prototype code may use unrestricted Python where needed.
- **Numba-CPU** is sufficient when compiled CPU performance meets the calculation's needs.
- **Numba-GPU** provides the final portability stage when accelerator performance is required.

This staged workflow allows researchers to stop when their immediate objective has been achieved.
A general feature intended to be merged and maintained as part of MC/DC, however, must work in MC/DC Python, Numba-CPU, and Numba-GPU unless the maintainers explicitly accept and document a backend-specific limitation.

The Standard Execution Path
---------------------------

Every backend follows MC/DC's standard model and execution path.
What changes between development stages is how transport runs and which Python features it may use:

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

Model construction and preparation may use object-oriented interfaces, variable-length collections, validation, and other expressive Python features.
Transport in the Python backend may also use ordinary Python facilities beyond the prepared execution data.
The accelerated backends require predictable numerical data and explicit behavior.
See :doc:`simulation_compilation` and :doc:`runtime_data_layout` for the two preparation stages, then :doc:`transport_execution` for execution.

Architectural Consequences
--------------------------

Preparing a method for acceleration narrows the freedom available inside transport and has three broad architectural consequences.

**Flexible models become predictable execution data.**
Python objects remain the natural way to describe a problem, but accelerated transport operates on a numerical representation prepared for one simulation.
MC/DC therefore replaces Python's implicit object machinery with a purpose-built runtime object model based on structured records, simulation-local IDs, offsets, and generated accessors.
Mutable runtime records are passed in one-element array containers, providing stable shared storage across Python, Numba-CPU, and Numba-GPU execution.
See :doc:`simulation_compilation` and :doc:`runtime_data_layout` for this transformation.

**Transport behavior becomes explicit.**
Dynamic Python techniques that are useful during prototyping must be expressed in forms the compiler can understand when the method is accelerated.
Use :doc:`../extending/writing_numba_compatible_transport_code` and :doc:`../extending/extending_the_object_model` when implementing these changes.

**CPU and GPU share a portable core.**
Common transport algorithms use behavior supported by both accelerated targets, while hardware-specific concerns remain isolated.
The execution modes are described in :doc:`transport_execution`.

Design Tradeoffs
----------------

The staged approach does not eliminate development cost; it defers some of that cost until a method has demonstrated enough value to justify acceleration and long-term maintenance.

Prototyping through MC/DC Python maximizes scientific expressiveness and minimizes the effort required to test an idea.
The tradeoff is that parts of the Python implementation may need to be redesigned or rewritten for Numba-CPU.
Beginning with simple numerical data and operations that Numba supports can reduce that porting effort, but introduces implementation constraints earlier in the research process.

Portability also increases compiler, backend, and validation work.
MC/DC accepts these costs so that maintained features can share transport logic across Python, CPU, and GPU execution rather than becoming separate implementations that may drift apart.
Use :doc:`../extending/writing_numba_compatible_transport_code` for practical constraints and :doc:`transport_execution` for mode-specific execution mechanisms.

Related Design Choices
----------------------

Alternative designs place the boundary between expressiveness and portability elsewhere.
Requiring Numba compatibility from the beginning would reduce later porting but constrain early exploration.
Maintaining separate Python and optimized solvers would preserve unrestricted Python behavior but duplicate transport logic.
Handwritten kernels would provide more low-level control but split methods development across languages.

MC/DC instead uses staged convergence within its standard execution path.
A method in MC/DC Python may range from an ad hoc implementation to one that is already nearly Numba-compatible.
When the method needs compiled performance or becomes a maintained MC/DC capability, it converges on common numerical data and transport interfaces.
This preserves a shared execution path without dictating how the Python implementation must begin.
The data boundary is described in :doc:`runtime_data_layout`; contributor guidance is provided in :doc:`../extending/writing_numba_compatible_transport_code`.

Future Evolution
----------------

The Python-first principle does not require the present implementation to remain fixed.
Future work should make movement between stages easier without narrowing the freedom of prototyping in MC/DC Python.

Potential improvements fall into three areas:

- **Python-mode flexibility** -- make the active Python model and arbitrary development state easier to reach during Python execution without requiring a formal prototype framework.
- **Porting assistance** -- provide better tools for inspecting a Python implementation and identifying code that an accelerated backend cannot run.
- **Execution infrastructure** -- make the creation and ownership of prepared data clearer, and keep hardware-specific behavior separated from the common transport algorithms.

Future changes should preserve the ability to stop at any development stage, keep the transition from an MC/DC-Python-only implementation to a portable maintained implementation explicit and reviewable, and retain equivalent physical behavior across the supported backends of a maintained feature.

Where to Go Next
----------------

For model compilation and runtime preparation, continue with :doc:`simulation_compilation` and :doc:`runtime_data_layout`.
For execution, read :doc:`transport_execution`.
Contributors implementing a transport change should use :doc:`../extending/writing_numba_compatible_transport_code`.
