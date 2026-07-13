from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcdc.object_.surface import Surface

####

import numpy as np
import sympy

from numpy import float64
from numpy.typing import NDArray
from operator import attrgetter
from types import NoneType
from typing import Annotated, Sequence
from sympy.logic.boolalg import Boolean

####

from mcdc.constant import (
    BOOL_AND,
    BOOL_NOT,
    BOOL_OR,
    FILL_LATTICE,
    FILL_MATERIAL,
    FILL_NONE,
    FILL_UNIVERSE,
    PI,
)
from mcdc.object_.base import MCDCObject
from mcdc.object_.material import MaterialBase
from mcdc.object_.tally import TallyCollision, TallyTracklength
from mcdc.object_.universe import Universe, Lattice
from mcdc.print_ import print_error

# ======================================================================================
# Region
# ======================================================================================


class Region(MCDCObject):
    type: str
    A: Surface | Region | NoneType
    B: Region | int | NoneType

    def __init__(self, type_, A, B):
        super().__init__()

        self.type = type_
        self.A = A
        self.B = B

    @classmethod
    def make_halfspace(cls, surface, sense):
        region = Region("halfspace", surface, sense)
        return region

    def __and__(self, other):
        return Region("intersection", self, other)

    def __or__(self, other):
        return Region("union", self, other)

    def __invert__(self):
        return Region("complement", self, None)

    def __repr__(self):
        return f"Region: {str.capitalize(self.type)}"


# ======================================================================================
# Cell
# ======================================================================================


class Cell(MCDCObject):
    """
    Define a cell from a region and a fill.

    Parameters
    ----------
    region : Region, optional
        The spatial region defining the cell boundaries.
        Constructed using ``+surface`` / ``-surface`` half-space operators.
    fill : Material or MaterialMG or Universe or Lattice, optional
        The material or universe that fills the cell.
    name : str, optional
        User label.
    translation : array_like of float, optional
        Translation vector ``[tx, ty, tz]`` in cm.
    rotation : array_like of float, optional
        Rotation angles ``[rx, ry, rz]`` in degrees.

    See Also
    --------
    mcdc.Surface : Creates surfaces that can be used to define cell regions.
    mcdc.Universe : Groups cells into a universe.
    """

    name: str
    region: Region  # Non-numba
    fill: MaterialBase | Universe | Lattice | NoneType  # Non-numba
    fill_translated: bool
    fill_rotated: bool
    translation: Annotated[NDArray[float64], (3,)]
    rotation: Annotated[NDArray[float64], (3,)]
    region_RPN_tokens: list[int]
    region_RPN: Boolean  # Non-numba
    surfaces: list[Surface]
    collision_tallies: list[TallyCollision]
    tracklength_tallies: list[TallyTracklength]
    #
    fill_type: int
    fill_ID: int

    def __init__(
        self,
        region: Region | NoneType = None,
        fill: MaterialBase | Universe | Lattice | NoneType = None,
        name: str = "",
        translation: Sequence[float] = [0.0, 0.0, 0.0],
        rotation: Sequence[float] = [0.0, 0.0, 0.0],
    ):
        # MC/DC framework metadata
        super().__init__()
        self.label = "cell"
        self.non_numba += ["region", "fill", "region_RPN"]

        # Set name
        if name == "":
            self.name = "(Unnamed cell)"
        else:
            self.name = name

        # Set region
        if region is None:
            self.region = Region("all", None, None)
        else:
            self.region = region

        # Set fill
        self.fill = fill

        # Local coordinate modifier
        self.translation = np.array(translation, dtype=float)
        self.rotation = np.array(rotation, dtype=float)
        self.fill_translated = False
        self.fill_rotated = False
        if (self.translation != 0.0).any():
            self.fill_translated = True
        if (self.rotation != 0.0).any():
            self.fill_rotated = True
            # Convert ritation
            self.rotation *= PI / 180.0

        # Set region Reversed Polished Notation (RPN)
        # TODO
        """
        if self.region.type != "all":
            self.region_RPN_tokens = generate_RPN_tokens(self.region)
            self.region_RPN = generate_RPN(self.region_RPN_tokens)
        else:
            self.region_RPN_tokens = []
            self.region_RPN = Boolean(True)
        """

        # List surfaces
        # TODO
        # self.surfaces = list_surfaces(self.region_RPN_tokens)

        # Cell tallies
        self.collision_tallies = []
        self.tracklength_tallies = []

        # ==============================================================================
        # Numba attribute manual set up
        # ==============================================================================

        # Numba representation of the cell fill
        #   (Because polymorphic Ffill object is not supported)
        if isinstance(fill, MaterialBase):
            self.fill_type = FILL_MATERIAL
            self.fill_ID = fill.ID
        elif isinstance(fill, Universe):
            self.fill_type = FILL_UNIVERSE
            self.fill_ID = fill.ID
        elif isinstance(fill, Lattice):
            self.fill_type = FILL_LATTICE
            self.fill_ID = fill.ID
        elif fill == None:
            self.fill_type = FILL_NONE
            self.fill_ID = -1
        else:
            print_error(f"Unsupported cell fill: {fill}")

    def __repr__(self):
        text = "\n"
        text += f"Cell\n"
        text += f"  - Name: {self.name}\n"
        text += f"  - {self.region}\n"
        if isinstance(self.fill, MaterialBase):
            text += f"  - Fill (material): {self.fill.name}\n"
        elif isinstance(self.fill, Lattice):
            text += f"  - Fill (lattice): {self.fill.name}\n"
        elif isinstance(self.fill, Universe):
            text += f"  - Fill (universe): {self.fill.name}\n"
        if self.fill_translated:
            text += f"  - Translation: {self.translation}\n"
        if self.fill_rotated:
            text += f"  - Rotation: {self.rotation * 180 / PI}\n"
        # text += f"  - Bounding surfaces: {[x.ID for x in self.surfaces]}\n"
        if len(self.collision_tallies) > 0:
            text += f"  - Collision tallies: {[x.ID for x in self.collision_tallies]}\n"
        if len(self.tracklength_tallies) > 0:
            text += (
                f"  - Tracklength tallies: {[x.ID for x in self.tracklength_tallies]}\n"
            )
        return text


def generate_RPN_tokens(region):
    # The RPN tokens
    rpn_tokens = []

    # Build RPN based on recursive evaluation of the region
    stack = [region]
    while len(stack) > 0:
        token = stack.pop()
        if isinstance(token, Region):
            if token.type == "halfspace":
                rpn_tokens.append(token.A.ID)
                if token.B < 0:
                    rpn_tokens.append(BOOL_NOT)
            elif token.type == "intersection":
                stack += ["&", token.A, token.B]
            elif token.type == "union":
                stack += ["|", token.A, token.B]
            elif token.type == "complement":
                stack += ["~", token.A]
        else:
            if token == "&":
                rpn_tokens.append(BOOL_AND)
            elif token == "|":
                rpn_tokens.append(BOOL_OR)
            elif token == "~":
                rpn_tokens.append(BOOL_NOT)
            else:
                print_error(f"Unrecognized token in the generating region RPN: {token}")

    return rpn_tokens


def generate_RPN(rpn_tokens):
    stack = []

    for token in rpn_tokens:
        if token >= 0:
            stack.append(sympy.symbols(f"s{token}"))
        else:
            if token == BOOL_AND or token == BOOL_OR:
                item_1 = stack.pop()
                item_2 = stack.pop()
                if token == BOOL_AND:
                    stack.append(item_1 & item_2)
                else:
                    stack.append(item_1 | item_2)

            elif token == BOOL_NOT:
                item = stack.pop()
                if isinstance(item, Region):
                    item = sympy.symbols(str(item)[8:])

                stack.append(~item)

    return sympy.logic.boolalg.simplify_logic(stack[0])


def list_surfaces(rpn_tokens):
    surfaces = []

    for token in rpn_tokens:
        if token >= 0:
            surface = simulation.surfaces[token]
            if surface not in surfaces:
                surfaces.append(surface)

    return sorted(surfaces, key=attrgetter("ID"))
