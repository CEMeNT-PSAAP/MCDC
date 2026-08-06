.. _extending_the_object_model:

==========================
Extending the Object Model
==========================

Use this page when an extension changes a model class, introduces runtime-visible state, or adds a registered or polymorphic object type.
It assumes the hierarchy and data representation described in :doc:`../architecture/simulation_compilation` and :doc:`../architecture/runtime_data_layout` and applies those designs as implementation recipes.

When the new state is consumed during particle transport, continue with :doc:`writing_numba_compatible_transport_code` for type, dispatch, allocation, CPU/GPU compatibility, and verification guidance.

Choose the Extension Type
-------------------------

Choose the narrowest extension that represents the new concept:

.. list-table::
   :header-rows: 1
   :widths: 20 20 36 24

   * - Change
     - Starting point
     - Use when
     - Example
   * - Add a field
     - Existing class
     - The concept already belongs to an existing model or configuration object.
     - Add a new source parameter to ``Source``.
   * - Add embedded state
     - ``MCDCBase``
     - The state belongs to one parent and does not need an independently addressable entry in a simulation registry.
     - Add technique settings owned by ``Simulation``.
   * - Add a registered category
     - ``MCDCObject``
     - Transport must refer to independently registered instances by simulation-local ``ID``.
     - Add a new Surface-like category with its own collection.
   * - Add a representation to an existing category
     - ``MCDCPolymorphic``
     - The extension shares a category interface but needs a distinct packed layout and dispatch code.
     - Add a concrete ``MeshBase`` representation.

Prefer adding a subtype to an existing polymorphic family over creating a new registered category when the new object has the same conceptual role.
For example, implement a new tally estimator as a ``Tally`` subtype.

The Common Class Contract
-------------------------

Every MC/DC model class must follow the conventions used by the compiler and Numba-layer generator.

``label``
   Provide a unique, stable, lower-case label such as ``structured_mesh``.
   The label names generated structured layouts and the corresponding modules under ``mcdc_get`` and ``mcdc_set``.

   .. code-block:: python

      class MeshStructured(MeshBase):
          label = "structured_mesh"

``sub_type``
   Give every concrete ``MCDCPolymorphic`` subclass a unique named integer constant within its family.
   The shared base uses ``sub_type = -1``.

   .. code-block:: python

      class MeshStructured(MeshBase):
          sub_type = MESH_STRUCTURED

Type annotations
   Annotate every field that must be represented at runtime.
   Annotations define scalar fields, embedded structures, object-ID references, and variable-length payloads.

   .. code-block:: python

      active: bool
      translation: Annotated[NDArray[float64], (3,)]
      move_velocities: Annotated[NDArray[float64], ("N_move", 3)]
      surfaces: list[Surface]

Initialization
   Assign every runtime-visible field a valid initial value.
   Subclasses of ``MCDCObject`` must call ``super().__init__()`` so ``ID`` is initialized; subclasses of ``MCDCPolymorphic`` must do the same so both ``ID`` and ``sub_ID`` are initialized.

   .. code-block:: python

      def __init__(self, name, boundaries):
          super().__init__()
          self.name = name
          self.boundaries = np.asarray(boundaries, dtype=float64)

``non_numba``
   List Python-only fields that should not be traversed or packed automatically.
   The class must explicitly convert any required information from those fields into annotated runtime-visible fields before packing.

   For example, ``Cell`` keeps its expressive ``region`` and ``fill`` objects on the Python side, then derives RPN tokens, a fill-type code, and a fill ID for transport:

   .. code-block:: python

      non_numba = ["region", "fill"]

      region: Region
      fill: Material | Universe | Lattice | None
      region_RPN_tokens: list[int]
      fill_type: int
      fill_ID: int

Compilation hook
   Use the inherited ``_compile_into_simulation`` implementation unless the class must canonicalize objects, compile excluded references, derive fields from assigned object IDs, or otherwise finalize state owned by that object.

   .. code-block:: python

      def _compile_into_simulation(self, simulation):
          if not super()._compile_into_simulation(simulation):
              return False

          self.reference._compile_into_simulation(simulation)
          self.reference_ID = self.reference.ID
          return True

   If Python-only members must be canonicalized before ordinary traversal, guard the work with ``compile_ID`` and then call ``super`` exactly once:

   .. code-block:: python

      def _compile_into_simulation(self, simulation):
          if self.compile_ID == simulation.compile_ID:
              return False

          self._resolve_python_inputs(simulation)

          if not super()._compile_into_simulation(simulation):
              return False

          self._derive_post_registration_fields()
          return True

   Do not assign ``compile_ID``, ``ID``, or ``sub_ID`` manually.

Model-wide finalization
   Object hooks should not normalize or coordinate unrelated registries.
   When a value requires the complete discovered model, coordinate it once in ``Simulation._finalize_compilation`` instead.
   Source-probability normalization, particle-bank capacities, and settings derived from the complete material or tally collections are examples of model-wide finalization.
   Explicitly compile any new runtime-visible object introduced during this phase because ordinary recursive discovery has already occurred.

   ``compile_simulation`` orchestrates recursive discovery and calls this model-wide finalization phase.
   Change the compiler orchestration only when adding a new compilation phase or registered category.
   Do not place model-specific finalization in ``mcdc.main.prepare``; that function is reserved for framework-level packing, resource allocation, backend configuration, and external runtime state.

Represent Fields Deliberately
-----------------------------

The layer generator interprets annotations according to the field's role:

.. list-table::
   :header-rows: 1
   :widths: 28 27 45

   * - Example annotation
     - Runtime representation
     - Transport access
   * - ``active: bool``
     - Scalar structured field
     - ``simulation["technique"]["new_technique"]["active"]``
   * - ``translation: Annotated[NDArray[float64], (3,)]``
     - Fixed-size embedded array
     - ``cell["translation"][axis]``
   * - ``energy: NDArray[float64]``
     - Offset and length plus values in ``data``
     - ``mcdc_get.tally.energy(index, tally, data)``
   * - ``move_velocities: Annotated[NDArray[float64], ("N_move", 3)]``
     - Offset, length, and shape metadata plus flattened values
     - ``mcdc_get.surface.move_velocities(move, axis, surface, data)``
   * - ``energy_group_pmf: DistributionPMF``
     - Simulation-local object ID
     - ``simulation["distributions"][source["energy_group_pmf_ID"]]``
   * - ``collision_tallies: list[TallyCollision]``
     - Count and offset to IDs stored in ``data``
     - ``mcdc_get.cell.collision_tally_IDs(index, cell, data)``

Use an integer-only shape when an array is always the same size.
Use symbolic dimensions when a shape depends on the model.
Do not store a Python reference in transport-visible state; annotate it as an ``MCDCObject`` or polymorphic base so the packed layer records an ID.

Adding a Field to an Existing Class
-----------------------------------

#. Add the annotation to the class that owns the concept.
#. Initialize the field for every construction path.
#. Decide whether it is fixed-size, variable-length, an embedded ``MCDCBase``, or an ``MCDCObject`` reference.
#. Update any compile hook that derives the field or converts a Python-only input into its runtime representation.
#. Consume the field through direct structured access or its generated ``mcdc_get`` and ``mcdc_set`` helpers.
#. Update the public docstring and API documentation when users can configure the field.

For example, a new variable-length ``energy_bias`` field on ``Source`` requires an annotation and initialized array on the model class:

.. code-block:: python

   # In Source annotations
   energy_bias: NDArray[float64]

   # In Source.__init__
   self.energy_bias = np.asarray(energy_bias, dtype=float64)

Transport then reads one value through the generated accessor:

.. code-block:: python

   bias = mcdc_get.source.energy_bias(group, source, data)

Avoid adding parallel state in several classes.
If a value belongs to the simulation as a whole, place it in ``Simulation`` or one of its embedded configuration objects and pass or access that representation consistently.

Adding Embedded ``MCDCBase`` State
----------------------------------

Use ``MCDCBase`` for configuration or runtime state that is owned by one parent.
A minimal class has a label, annotated fields, and initialized values:

.. code-block:: python

   class NewTechnique(MCDCBase):
       label = "new_technique"

       active: bool
       strength: float

       def __init__(self) -> None:
           super().__init__()
           self.active = False
           self.strength = 1.0

Add an annotated field for the object to its owner and instantiate it with the owner.
Simulation-wide transport techniques belong to the ``Technique`` aggregate,
which is itself owned by ``Simulation``.
The embedded object participates in recursive compilation but does not require a registry branch or object ID.
This Python ownership hierarchy is preserved under the packed
``simulation["technique"]`` record.

.. code-block:: python

   class Technique(MCDCBase):
       new_technique: NewTechnique

       def __init__(self):
           self.new_technique = NewTechnique()

The corresponding user and runtime interfaces are
``simulation.technique.new_technique(...)`` and
``simulation["technique"]["new_technique"]``.

If the embedded object refers to an ``MCDCObject``, annotate that reference.
The default traversal will register the referenced object when compilation reaches the embedded configuration.

Adding a New ``MCDCObject`` Category
------------------------------------

A genuinely new registered category requires coordinated changes:

#. Define the class with a unique ``label``, annotations, initialized fields, and a call to ``super().__init__()``.
#. Add the category collection to ``Simulation`` annotations and initialize or reset it in ``Simulation._reset_model``.
#. Import the category in ``mcdc/code_factory/python_objects_compiler.py`` and add an ``isinstance`` branch in ``register_object`` that selects its simulation collection.
#. Ensure the object is reachable from an existing simulation root or add an explicit root and compilation step.
#. Ensure the class's module is imported before ``numba_layers_generator.py`` discovers the classes.
   A public class is normally imported through ``mcdc/__init__.py``; an internal class must be imported by another module in the compilation path.
#. Add the collection and lookup behavior required by transport.

For example, the class and its simulation collection begin with:

.. code-block:: python

   class Detector(MCDCObject):
       label = "detector"

       name: str
       response: NDArray[float64]

       def __init__(self, name, response):
           super().__init__()
           self.name = name
           self.response = np.asarray(response, dtype=float64)


   class Simulation(MCDCBase):
       detectors: list[Detector]

The compiler then selects that collection explicitly:

.. code-block:: python

   elif isinstance(object_, Detector):
       object_list = simulation.detectors

Do not add a fallback registration branch that silently accepts unknown objects.
An explicit category branch keeps registry ownership and runtime layout reviewable.

Adding a Polymorphic Subtype
----------------------------

Adding a concrete subtype to an existing family is more localized than adding a category:

#. Add a unique integer constant for the subtype.
#. Inherit from the existing polymorphic base, such as ``MeshBase`` or ``Tally``.
#. Set a unique concrete ``label`` and the new ``sub_type`` constant.
#. Call the base initializer so shared fields, ``ID``, and ``sub_ID`` are initialized.
#. Annotate and initialize subtype-specific fields.
#. Import the subtype before layer generation and expose it from ``mcdc/__init__.py`` when it is public.
#. Add transport dispatch for the new ``sub_type`` and implement the subtype-specific behavior.

For example, the structural part of a mesh subtype follows this pattern:

.. code-block:: python

   class MeshNew(MeshBase):
       label = "new_mesh"
       sub_type = MESH_NEW

       boundaries: NDArray[float64]

       def __init__(self, boundaries, name="") -> None:
           super().__init__(name)
           self.boundaries = np.asarray(boundaries, dtype=float64)

No new ``register_object`` branch is needed for a subtype of an already registered family.
The existing ``isinstance(..., MeshBase)`` or corresponding category check places it in the base collection, while ``sub_type`` and ``sub_ID`` connect it to its concrete packed collection.

Generated Runtime Layers and Accessors
--------------------------------------

The annotation is the source of truth for generated runtime fields and accessors.
Do not edit ``mcdc/numba_types.py``, ``mcdc_get``, or ``mcdc_set`` to introduce a field.
Prepare a representative simulation so ``numba_layers_generator.py`` regenerates those files, then verify the access pattern predicted by the field representation chosen above.

For example, a variable-length ``Detector.response`` field produces element accessors associated with the ``detector`` label:

.. code-block:: python

   value = mcdc_get.detector.response(index, detector, data)
   mcdc_set.detector.response(index, detector, data, new_value)

A fixed-size field such as ``Cell.translation`` remains embedded and is accessed directly:

.. code-block:: python

   value = cell["translation"][axis]

Public API and Documentation
----------------------------

For a user-facing class or constructor:

#. Export the class from ``mcdc/__init__.py``.
#. Add it to the appropriate autosummary group in ``docs/source/reference/python_api/index.rst``.
#. Document parameters, units, defaults, constraints, and at least one usable example in the class docstring.
#. Update the User Guide when the extension changes how users construct or run a model.

Keep internal helper classes under ``mcdc.object_`` and import them explicitly in the compilation path.
Export only classes that form part of the public API.

For example, a public ``Detector`` is re-exported from the package and listed by its qualified name in the API autosummary:

.. code-block:: python

   from mcdc.object_.detector import Detector

.. code-block:: rst

   ~mcdc.Detector

Verification Checklist
----------------------

An object-model extension should verify all affected layers:

- Construction accepts valid input and rejects invalid shapes or types.
- Compilation discovers the object from the intended root.
- Shared references register once, and recompilation produces a valid new snapshot.
- Packed fields, object IDs, offsets, and generated accessors contain the expected values.
- Python and Numba-CPU modes produce equivalent behavior.
- GPU execution is covered when the changed transport path supports GPUs.
- Public examples compile under the example validator when the API changes.
- API and developer documentation build without warnings.

Add focused unit tests near ``test/unit/test_object_compilation.py`` for compilation behavior and near the relevant transport tests for runtime behavior.
Use :doc:`../../contributing/example_validation` when an extension changes public examples.
