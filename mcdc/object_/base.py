from mcdc.object_.util import check_type
from mcdc.print_ import print_error


class MCDCBase:
    """Base class for Python-side MC/DC model and runtime objects.

    Subclasses declare a :attr:`label` and type-annotated fields. Assignments to
    annotated fields are checked at runtime so invalid model data is rejected
    before the simulation is packed for transport. ``compile_ID`` records the
    simulation compilation in which an object most recently participated,
    providing shared recompilation and cycle-prevention behavior for both
    embedded configuration objects and registered model objects.
    """

    label: str
    compile_ID: int = 0
    non_numba = ()

    def __init_subclass__(cls):
        # Require metadata used by the object and Numba-layer factories
        if not hasattr(cls, "label"):
            raise NotImplementedError(
                f"MC/DC class '{cls.__name__}' must have 'label' class attribute."
            )

    def __setattr__(self, key, value):
        # Validate annotated fields before updating the object
        hints = getattr(self.__class__, "__annotations__", {})
        if key in hints and not check_type(value, hints[key], self.__class__, self):
            print_error(f"{key} must be {hints[key]!r}, got {value!r}")
        super().__setattr__(key, value)

    def _compile_into_simulation(self, simulation) -> bool:
        """Compile an embedded object for the current simulation.

        Embedded ``MCDCBase`` objects participate in compilation without being
        registered in a simulation object collection. Registered
        :class:`MCDCObject` subclasses extend this lifecycle with an object ID.
        Subclasses may extend this hook to validate or normalize their state,
        compile excluded references, and derive fields that require the owning
        simulation.

        Returns
        -------
        bool
            ``True`` if the object was compiled for the current simulation
            compilation, or ``False`` if it had already been compiled and was
            skipped.
        """
        # Compile each embedded object once per simulation compilation
        if self.compile_ID == simulation.compile_ID:
            return False
        self.compile_ID = simulation.compile_ID

        # Compile all members represented in the Numba layer
        self._compile_members_into_simulation(simulation)
        return True

    def _compile_members_into_simulation(self, simulation) -> None:
        """Compile object members represented in the Numba layer.

        ``MCDCObject`` members are registered with ``simulation``. Embedded
        ``MCDCBase`` members and lists are compiled recursively. Members listed
        in ``non_numba`` are intentionally left to the owning class because
        they generally require a custom packed representation.
        """
        # Compile members represented in the Numba layer
        excluded = getattr(self, "non_numba", ())
        for name, value in vars(self).items():
            if name in excluded:
                continue
            self._compile_member_value(value, simulation)

    @staticmethod
    def _compile_member_value(value, simulation) -> None:
        # Register direct object members
        if isinstance(value, MCDCBase):
            value._compile_into_simulation(simulation)

        # Compile object members stored in lists
        elif isinstance(value, list):
            for item in value:
                MCDCBase._compile_member_value(item, simulation)

        # Scalar, array, and other non-object members require no compilation.
        else:
            return


class MCDCObject(MCDCBase):
    """Base class for model objects registered during simulation compilation.

    ``ID`` identifies an object in its heterogeneous simulation collection.
    The inherited ``compile_ID`` prevents duplicate registration when an
    object is shared by multiple parts of a model.
    """

    # MC/DC framework metadata
    label = "object"

    ID: int

    def __init__(self):
        # Initialize the object as unregistered
        self.ID = -1

    def __repr__(self) -> str:
        # Build the shared object-registration summary
        nice_label = self.label.replace("_", " ").title()
        text = "\n"
        text += f"{nice_label}\n"
        if self.compile_ID > 0:
            text += f"  (compile_ID={self.compile_ID}, ID={self.ID})\n"

        return text

    def _compile_into_simulation(self, simulation) -> bool:
        from mcdc.code_factory.python_objects_compiler import register_object

        # Register once for the current simulation compilation
        if not register_object(self, simulation):
            return False

        # Compile all members represented in the Numba layer
        self._compile_members_into_simulation(simulation)
        return True


class MCDCPolymorphic(MCDCObject):
    """Base class for model objects with multiple packed representations.

    In addition to the global :attr:`~MCDCObject.ID`, polymorphic objects carry
    a subtype code and a subtype-local ``sub_ID``. The transport kernels use
    these values to dispatch to the correct packed object representation.
    """

    # MC/DC framework metadata
    label = "polymorphic"

    sub_type: int
    sub_ID: int

    def __init_subclass__(cls):
        # Apply the common object metadata requirements
        super().__init_subclass__()

        # Require the code used for polymorphic Numba dispatch
        if not hasattr(cls, "sub_type"):
            raise NotImplementedError(
                f"MC/DC class '{cls.__name__}' must have 'sub_type' class attribute."
            )

    def __init__(self):
        # Initialize common object registration state
        super().__init__()

        # Initialize the object as unregistered within its subtype
        self.sub_ID = -1

    def __repr__(self) -> str:
        # Build the shared polymorphic registration summary
        nice_label = self.label.replace("_", " ").title()
        text = "\n"
        text += f"{nice_label}\n"
        if self.compile_ID > 0:
            text += f"  (compile_ID={self.compile_ID}, ID={self.ID}, sub_ID={self.sub_ID})\n"

        return text
