from collections.abc import Sequence
from numbers import Integral
from typing import Annotated

import numpy as np
from numpy import float64, int64
from numpy.typing import NDArray

from mcdc.constant import (
    DATA_NONE,
    DATA_POLYNOMIAL,
    DATA_TABLE,
    INTERPOLATION_HISTOGRAM,
    INTERPOLATION_LINEAR,
    INTERPOLATION_LOG,
    INTERPOLATION_SEMILOGX,
    INTERPOLATION_SEMILOGY,
)
from mcdc.object_.base import MCDCPolymorphic
from mcdc.print_ import print_1d_array, print_error

# ======================================================================================
# Data base class
# ======================================================================================


class DataBase(MCDCPolymorphic):
    # MC/DC framework metadata
    label = "data"
    sub_type = -1


# ======================================================================================
# None
# ======================================================================================
# Placeholder for data that does not need to store anything:
#   - Fission multiplicity and delayed precursor data for non-fissionable nuclides


class DataNone(DataBase):
    # MC/DC framework metadata
    label = "none_data"
    sub_type = DATA_NONE


# ======================================================================================
# Table data
# ======================================================================================


class DataTable(DataBase):
    # MC/DC framework metadata
    label = "table_data"
    sub_type = DATA_TABLE

    # Main data
    N: int
    x: NDArray[float64]
    y: NDArray[float64]

    # Interpolation rules
    interpolations: NDArray[int64]
    interpolation_boundaries: NDArray[int64]

    # Auxiliary data
    N_aux: int
    aux: Annotated[NDArray[float64], ("N_aux", "N")]

    def __init__(
        self,
        x: NDArray[float64],
        y: NDArray[float64],
        interpolations: int | Sequence[int],
        interpolation_boundaries: Sequence[int] | None = None,
        aux: NDArray[float64] | None = None,
    ) -> None:
        super().__init__()

        # Set primary data
        self.x = np.asarray(x, dtype=float64)
        self.y = np.asarray(y, dtype=float64)

        if self.x.ndim != 1:
            print_error("x must be one-dimensional.")
        if self.y.ndim != 1:
            print_error("y must be one-dimensional.")

        self.N = len(self.x)

        # Basic size checks
        if self.N == 0:
            print_error("x and y must contain at least one value.")
        if len(self.y) != self.N:
            print_error("x and y must have the same length.")

        # Set auxiliary data
        if aux is None:
            self.N_aux = 0
            self.aux = np.zeros((0, self.N), dtype=float64)
        else:
            aux_array = np.asarray(aux, dtype=float64)

            if aux_array.ndim == 1:
                if len(aux_array) != self.N:
                    print_error("One-dimensional aux must have the same length as x.")

                self.N_aux = 1
                self.aux = aux_array.reshape(1, self.N)

            elif aux_array.ndim == 2:
                if aux_array.shape[1] != self.N:
                    print_error("Two-dimensional aux must have shape (N_aux, len(x)).")

                self.N_aux = aux_array.shape[0]
                self.aux = aux_array

            else:
                print_error("aux must be None, one-dimensional, or two-dimensional.")

        # Set interpolation rules and boundaries
        if isinstance(interpolations, Integral):
            self.interpolations = np.array([interpolations], dtype=int64)
            self.interpolation_boundaries = np.array([self.N], dtype=int64)
        else:
            self.interpolations = np.asarray(interpolations, dtype=int64)

            if self.interpolations.ndim != 1:
                print_error("interpolations must be one-dimensional.")

            if interpolation_boundaries is None:
                print_error(
                    "interpolation_boundaries is required when multiple "
                    "interpolation laws are provided."
                )

            self.interpolation_boundaries = np.asarray(
                interpolation_boundaries,
                dtype=int64,
            )

            if self.interpolation_boundaries.ndim != 1:
                print_error("interpolation_boundaries must be one-dimensional.")

        # Interpolation-region checks
        if len(self.interpolations) == 0:
            print_error("At least one interpolation law is required.")

        if len(self.interpolations) != len(self.interpolation_boundaries):
            print_error(
                "interpolations and interpolation_boundaries must have the same length."
            )

        if self.interpolation_boundaries[-1] != self.N:
            print_error("The last interpolation boundary must equal len(x).")

        previous = 0
        for boundary in self.interpolation_boundaries:
            if boundary <= previous:
                print_error("interpolation_boundaries must be strictly increasing.")
            if boundary > self.N:
                print_error("interpolation_boundaries cannot exceed len(x).")

            previous = boundary

        # Validate interpolation codes
        for interpolation in self.interpolations:
            decode_interpolation(interpolation)

    def __repr__(self) -> str:
        text = super().__repr__()

        text += f"  - x {print_1d_array(self.x)}\n"
        text += f"  - y {print_1d_array(self.y)}\n"

        if self.N_aux > 0:
            text += f"  - aux shape: {self.aux.shape}\n"
            for i, values in enumerate(self.aux):
                text += f"    - aux[{i}]: {print_1d_array(values)}\n"

        if len(self.interpolations) == 1:
            text += (
                "  - Interpolation: "
                f"{decode_interpolation(self.interpolations[0])}\n"
            )
        else:
            text += "  - Interpolation regions:\n"

            start = 0
            for interpolation, end in zip(
                self.interpolations,
                self.interpolation_boundaries,
            ):
                text += (
                    f"    - [{start}, {end}): "
                    f"{decode_interpolation(interpolation)}\n"
                )
                start = end

        return text


def decode_interpolation(type_: int) -> str:
    if type_ == INTERPOLATION_HISTOGRAM:
        return "histogram"
    if type_ == INTERPOLATION_LINEAR:
        return "linear"
    if type_ == INTERPOLATION_SEMILOGX:
        return "semilog-x"
    if type_ == INTERPOLATION_SEMILOGY:
        return "semilog-y"
    if type_ == INTERPOLATION_LOG:
        return "log"

    raise ValueError(f"Unknown interpolation type: {type_}")


def encode_interpolation(name: str) -> int:
    if name == "histogram":
        return INTERPOLATION_HISTOGRAM
    if name == "linear":
        return INTERPOLATION_LINEAR
    if name == "semilog-x":
        return INTERPOLATION_SEMILOGX
    if name == "semilog-y":
        return INTERPOLATION_SEMILOGY
    if name == "log":
        return INTERPOLATION_LOG

    raise ValueError(f"Unknown interpolation name: {name}")


# ======================================================================================
# Polynomial data
# ======================================================================================


class DataPolynomial(DataBase):
    # MC/DC framework metadata
    label = "polynomial_data"
    sub_type = DATA_POLYNOMIAL

    coefficients: NDArray[float64]

    def __init__(self, coefficients: NDArray[float64]) -> None:
        super().__init__()

        self.coefficients = np.asarray(coefficients, dtype=float64)

        if self.coefficients.ndim != 1:
            print_error("coefficients must be one-dimensional.")

    def __repr__(self) -> str:
        text = super().__repr__()

        text += f"  - coefficients {print_1d_array(self.coefficients)}\n"
        return text
