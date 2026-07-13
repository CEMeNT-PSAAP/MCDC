import numpy as np

from collections.abc import Sequence
from numpy import float64, int64
from numpy.typing import ArrayLike, NDArray

from mcdc.constant import (
    DISTRIBUTION_NONE,
    DISTRIBUTION_PMF,
    DISTRIBUTION_TABULATED,
    DISTRIBUTION_MULTITABLE,
    DISTRIBUTION_LEVEL_SCATTERING,
    DISTRIBUTION_EVAPORATION,
    DISTRIBUTION_MAXWELLIAN,
    DISTRIBUTION_KALBACH_MANN,
    DISTRIBUTION_TABULATED_ENERGY_ANGLE,
    DISTRIBUTION_N_BODY,
    INTERPOLATION_HISTOGRAM,
    INTERPOLATION_LINEAR,
)
from mcdc.object_.base import MCDCPolymorphic
from mcdc.object_.data import DataTable
from mcdc.object_.util import (
    cdf_from_pdf,
    pdf_from_cdf,
    multi_cdf_from_pdf,
    cmf_from_pmf,
)
from mcdc.print_ import print_1d_array, print_error

# ======================================================================================
# Distribution base class
# ======================================================================================


class DistributionBase(MCDCPolymorphic):
    """Base class for probability distributions.

    Distribution objects describe probability laws used throughout MC/DC,
    including source distributions and secondary-particle distributions.

    Parameters
    ----------
    child_label : str
        Framework label identifying the concrete distribution subtype.

    child_type : int
        Integer identifier specifying the concrete distribution subtype.

    non_numba : list of str
        Additional attribute names excluded from the compiled representation.
    """

    def __init__(
        self,
        child_label: str,
        child_type: int,
        non_numba: list[str],
    ) -> None:
        """Initialize distribution framework metadata."""
        super().__init__(
            label="distribution",
            child_label=child_label,
            child_type=child_type,
            non_numba=non_numba,
        )

    def __repr__(self) -> str:
        """Return a human-readable description of the distribution."""
        text = "\n"
        text += f"{decode_type(self.child_type)}\n"
        text += f"  - ID: {self.ID}\n"
        return text


def decode_type(child_type: int) -> str:
    """Convert a distribution type code to a human-readable name.

    Parameters
    ----------
    child_type : int
        Distribution type code.

    Returns
    -------
    str
        Human-readable distribution name.

    Raises
    ------
    ValueError
        If the distribution type code is unknown.
    """
    if child_type == DISTRIBUTION_NONE:
        return "Distribution (None)"
    if child_type == DISTRIBUTION_PMF:
        return "Distribution (PMF)"
    if child_type == DISTRIBUTION_TABULATED:
        return "Distribution (Tabulated)"
    if child_type == DISTRIBUTION_MULTITABLE:
        return "Distribution (Multi-table)"
    if child_type == DISTRIBUTION_LEVEL_SCATTERING:
        return "Distribution (Level scattering)"
    if child_type == DISTRIBUTION_EVAPORATION:
        return "Distribution (Evaporation)"
    if child_type == DISTRIBUTION_MAXWELLIAN:
        return "Distribution (Maxwellian spectrum)"
    if child_type == DISTRIBUTION_KALBACH_MANN:
        return "Distribution (Kalbach-Mann)"
    if child_type == DISTRIBUTION_TABULATED_ENERGY_ANGLE:
        return "Distribution (Tabulated energy-angle)"
    if child_type == DISTRIBUTION_N_BODY:
        return "Distribution (N-body)"

    raise ValueError(f"Unknown distribution type: {child_type}")


# ======================================================================================
# None
# ======================================================================================
# Placeholder for a distribution that does not need to store data:
#   - Isotropic distributions
#   - Energy-correlated angles stored in the energy distribution


class DistributionNone(DistributionBase):
    """Placeholder for the absence of a stored distribution."""

    def __init__(self) -> None:
        """Create an empty distribution object."""
        super().__init__(
            child_label="none_distribution",
            child_type=DISTRIBUTION_NONE,
            non_numba=[],
        )


# ======================================================================================
# Probability mass function
# ======================================================================================


class DistributionPMF(DistributionBase):
    """Discrete probability mass function."""

    value: NDArray[float64]
    pmf: NDArray[float64]
    cmf: NDArray[float64]

    def __init__(
        self,
        value: ArrayLike,
        pmf: ArrayLike,
    ) -> None:
        """Create a discrete probability mass function.

        Parameters
        ----------
        value : array_like
            Values sampled by the distribution.

        pmf : array_like
            Probability masses corresponding to ``value``. The masses are
            normalized internally.
        """
        super().__init__(
            child_label="pmf_distribution",
            child_type=DISTRIBUTION_PMF,
            non_numba=[],
        )

        self.value = np.asarray(value, dtype=float64)
        pmf_array = np.asarray(pmf, dtype=float64)

        if self.value.ndim != 1:
            print_error("value must be one-dimensional.")
        if pmf_array.ndim != 1:
            print_error("pmf must be one-dimensional.")
        if len(self.value) == 0:
            print_error("value and pmf must contain at least one entry.")
        if len(self.value) != len(pmf_array):
            print_error("value and pmf must have the same length.")

        self.pmf, self.cmf = cmf_from_pmf(pmf_array)

    def __repr__(self) -> str:
        """Return a human-readable summary of the PMF."""
        text = super().__repr__()
        text += f"  - value {print_1d_array(self.value)}\n"
        text += f"  - pmf {print_1d_array(self.pmf)}\n"
        return text


# ======================================================================================
# Tabulated
# ======================================================================================


class DistributionTabulated(DistributionBase):
    """One-dimensional tabulated probability distribution.

    The distribution is stored as a :class:`DataTable` whose independent
    variable contains sample values and whose dependent variable contains the
    normalized probability density. The cumulative distribution is stored as
    auxiliary table data for sampling.

    A PDF input is treated as piecewise linear. A CDF input is treated as
    piecewise linear, producing a histogram PDF.
    """

    pdf: DataTable

    def __init__(
        self,
        value: ArrayLike,
        pdf: ArrayLike | None = None,
        cdf: ArrayLike | None = None,
    ) -> None:
        """Create a tabulated distribution from either a PDF or CDF.

        Parameters
        ----------
        value : array_like
            Sample values.

        pdf : array_like, optional
            Probability densities at the sample values. The PDF is normalized
            and its CDF is calculated by trapezoidal integration.

        cdf : array_like, optional
            Cumulative probabilities at the sample values. The CDF is
            normalized and a histogram PDF is derived from it.

        Notes
        -----
        Exactly one of ``pdf`` or ``cdf`` must be provided.
        """
        super().__init__(
            child_label="tabulated_distribution",
            child_type=DISTRIBUTION_TABULATED,
            non_numba=[],
        )

        if (pdf is None) == (cdf is None):
            print_error("Exactly one of pdf or cdf must be provided.")

        value_array = np.asarray(value, dtype=float64)

        if value_array.ndim != 1:
            print_error("value must be one-dimensional.")
        if len(value_array) == 0:
            print_error("value must contain at least one entry.")

        if pdf is not None:
            pdf_array = np.asarray(pdf, dtype=float64)

            if pdf_array.ndim != 1:
                print_error("pdf must be one-dimensional.")
            if len(pdf_array) != len(value_array):
                print_error("value and pdf must have the same length.")

            interpolation = INTERPOLATION_LINEAR
            pdf_normalized, cdf_normalized = cdf_from_pdf(
                value_array,
                pdf_array,
            )
        else:
            cdf_array = np.asarray(cdf, dtype=float64)

            if cdf_array.ndim != 1:
                print_error("cdf must be one-dimensional.")
            if len(cdf_array) != len(value_array):
                print_error("value and cdf must have the same length.")

            interpolation = INTERPOLATION_HISTOGRAM
            pdf_normalized, cdf_normalized = pdf_from_cdf(
                value_array,
                cdf_array,
            )

        self.pdf = DataTable(
            value_array,
            pdf_normalized,
            interpolation,
            aux=cdf_normalized,
        )

    def __repr__(self) -> str:
        """Return a human-readable summary of the distribution."""
        text = super().__repr__()
        text += f"  - value {print_1d_array(self.pdf.x)}\n"
        text += f"  - probability density {print_1d_array(self.pdf.y)}\n"
        text += "  - cumulative distribution " f"{print_1d_array(self.pdf.aux[0])}\n"
        return text


# ======================================================================================
# Multi-table
# ======================================================================================


class DistributionMultiTable(DistributionBase):
    """Distribution represented by tabulated distributions on a grid.

    Each grid point owns one :class:`DistributionTabulated`. The flattened
    ``value`` array is divided into individual tables using ``offset``.
    """

    grid: NDArray[float64]
    tables: list[DistributionTabulated]

    def __init__(
        self,
        grid: ArrayLike,
        offset: ArrayLike,
        value: ArrayLike,
        pdf: ArrayLike | None = None,
        cdf: ArrayLike | None = None,
    ) -> None:
        """Create a multi-table distribution.

        Parameters
        ----------
        grid : array_like
            Grid values associated with the tabulated distributions.

        offset : array_like
            Starting index of each table in the flattened arrays.

        value : array_like
            Flattened sample values for all tables.

        pdf : array_like, optional
            Flattened PDF values.

        cdf : array_like, optional
            Flattened CDF values.

        Notes
        -----
        Exactly one of ``pdf`` or ``cdf`` must be provided.
        """
        super().__init__(
            child_label="multi_table_distribution",
            child_type=DISTRIBUTION_MULTITABLE,
            non_numba=[],
        )

        if (pdf is None) == (cdf is None):
            print_error("Exactly one of pdf or cdf must be provided.")

        self.grid = np.asarray(grid, dtype=float64)
        offset_array = np.asarray(offset, dtype=int64)
        value_array = np.asarray(value, dtype=float64)

        if self.grid.ndim != 1:
            print_error("grid must be one-dimensional.")
        if offset_array.ndim != 1:
            print_error("offset must be one-dimensional.")
        if value_array.ndim != 1:
            print_error("value must be one-dimensional.")

        if len(self.grid) == 0:
            print_error("grid must contain at least one value.")
        if len(self.grid) != len(offset_array):
            print_error("grid and offset must have the same length.")
        if len(value_array) == 0:
            print_error("value must contain at least one value.")
        if offset_array[0] != 0:
            print_error("offset[0] must be zero.")
        if np.any(offset_array[1:] <= offset_array[:-1]):
            print_error("offset must be strictly increasing.")
        if offset_array[-1] >= len(value_array):
            print_error("Every offset must refer to an element in value.")

        pdf_array = None
        cdf_array = None

        if pdf is not None:
            pdf_array = np.asarray(pdf, dtype=float64)

            if pdf_array.ndim != 1:
                print_error("pdf must be one-dimensional.")
            if len(pdf_array) != len(value_array):
                print_error("pdf and value must have the same length.")

        if cdf is not None:
            cdf_array = np.asarray(cdf, dtype=float64)

            if cdf_array.ndim != 1:
                print_error("cdf must be one-dimensional.")
            if len(cdf_array) != len(value_array):
                print_error("cdf and value must have the same length.")

        stop = np.empty(len(offset_array), dtype=int64)
        stop[:-1] = offset_array[1:]
        stop[-1] = len(value_array)

        self.tables = []

        for start_i, stop_i in zip(offset_array, stop):
            if stop_i <= start_i:
                print_error("Each table must contain at least one value.")

            if pdf_array is not None:
                table = DistributionTabulated(
                    value_array[start_i:stop_i],
                    pdf=pdf_array[start_i:stop_i],
                )
            else:
                table = DistributionTabulated(
                    value_array[start_i:stop_i],
                    cdf=cdf_array[start_i:stop_i],
                )

            self.tables.append(table)

    def __repr__(self) -> str:
        """Return a human-readable summary of the distribution."""
        text = super().__repr__()
        text += f"  - grid: {print_1d_array(self.grid)}\n"
        text += f"  - tables: {len(self.tables)}\n"

        for i, table in enumerate(self.tables):
            text += f"    - table[{i}] " f"(grid={self.grid[i]:.6g}, N={table.pdf.N})\n"

        return text


# ======================================================================================
# Level scattering
# ======================================================================================


class DistributionLevelScattering(DistributionBase):
    """Level-scattering energy distribution."""

    C1: float
    C2: float

    def __init__(self, C1: float, C2: float) -> None:
        """Create a level-scattering distribution."""
        super().__init__(
            child_label="level_scattering_distribution",
            child_type=DISTRIBUTION_LEVEL_SCATTERING,
            non_numba=[],
        )

        self.C1 = C1
        self.C2 = C2

    def __repr__(self) -> str:
        """Return a human-readable summary of the distribution."""
        text = super().__repr__()
        text += f"  - C1: {self.C1} [/eV^l]\n"
        text += f"  - C2: {self.C2}\n"
        return text


# ======================================================================================
# Evaporation
# ======================================================================================


class DistributionEvaporation(DistributionBase):
    """Evaporation energy distribution."""

    nuclear_temperature: DataTable
    restriction_energy: float

    def __init__(
        self,
        nuclear_temperature_energy_grid: ArrayLike,
        nuclear_temperature_value: ArrayLike,
        restriction_energy: float,
        temperature_interpolations: int | Sequence[int],
        interpolation_boundaries: Sequence[int] | None,
    ) -> None:
        """Create an evaporation distribution."""
        super().__init__(
            child_label="evaporation_distribution",
            child_type=DISTRIBUTION_EVAPORATION,
            non_numba=[],
        )

        self.restriction_energy = restriction_energy
        self.nuclear_temperature = DataTable(
            nuclear_temperature_energy_grid,
            nuclear_temperature_value,
            temperature_interpolations,
            interpolation_boundaries,
        )

    def __repr__(self) -> str:
        """Return a human-readable summary of the distribution."""
        text = super().__repr__()
        text += f"  - Restriction energy: {self.restriction_energy} [eV]\n"
        text += (
            "  - Nuclear temperature "
            f"{print_1d_array(self.nuclear_temperature.y)} [eV]\n"
        )
        text += (
            "  - Nuclear temperature energy grid "
            f"{print_1d_array(self.nuclear_temperature.x)} [eV]\n"
        )
        return text


# ======================================================================================
# Maxwellian
# ======================================================================================


class DistributionMaxwellian(DistributionBase):
    """Maxwellian energy distribution."""

    nuclear_temperature: DataTable
    restriction_energy: float

    def __init__(
        self,
        nuclear_temperature_energy_grid: ArrayLike,
        nuclear_temperature_value: ArrayLike,
        restriction_energy: float,
        temperature_interpolations: int | Sequence[int],
        interpolation_boundaries: Sequence[int] | None,
    ) -> None:
        """Create a Maxwellian distribution."""
        super().__init__(
            child_label="maxwellian_distribution",
            child_type=DISTRIBUTION_MAXWELLIAN,
            non_numba=[],
        )

        self.restriction_energy = restriction_energy
        self.nuclear_temperature = DataTable(
            nuclear_temperature_energy_grid,
            nuclear_temperature_value,
            temperature_interpolations,
            interpolation_boundaries,
        )

    def __repr__(self) -> str:
        """Return a human-readable summary of the distribution."""
        text = super().__repr__()
        text += f"  - Restriction energy: {self.restriction_energy} [eV]\n"
        text += (
            "  - Nuclear temperature "
            f"{print_1d_array(self.nuclear_temperature.y)} [eV]\n"
        )
        text += (
            "  - Nuclear temperature energy grid "
            f"{print_1d_array(self.nuclear_temperature.x)} [eV]\n"
        )
        return text


# ======================================================================================
# Kalbach-Mann
# ======================================================================================


class DistributionKalbachMann(DistributionBase):
    """Kalbach-Mann correlated energy-angle distribution."""

    energy: NDArray[float64]
    offset: NDArray[int64]
    energy_out: NDArray[float64]
    pdf: NDArray[float64]
    cdf: NDArray[float64]
    precompound_factor: NDArray[float64]
    angular_slope: NDArray[float64]

    def __init__(
        self,
        energy: ArrayLike,
        offset: ArrayLike,
        energy_out: ArrayLike,
        pdf: ArrayLike,
        precompound_factor: ArrayLike,
        angular_slope: ArrayLike,
    ) -> None:
        """Create a Kalbach-Mann distribution."""
        super().__init__(
            child_label="kalbach_mann_distribution",
            child_type=DISTRIBUTION_KALBACH_MANN,
            non_numba=[],
        )

        self.energy = np.asarray(energy, dtype=float64)
        self.offset = np.asarray(offset, dtype=int64)
        self.energy_out = np.asarray(energy_out, dtype=float64)
        pdf_array = np.asarray(pdf, dtype=float64)
        self.precompound_factor = np.asarray(
            precompound_factor,
            dtype=float64,
        )
        self.angular_slope = np.asarray(angular_slope, dtype=float64)

        self.pdf, self.cdf = multi_cdf_from_pdf(
            self.offset,
            self.energy_out,
            pdf_array,
        )

    def __repr__(self) -> str:
        """Return a human-readable summary of the distribution."""
        text = super().__repr__()
        text += f"  - grid {print_1d_array(self.energy)} [eV]\n"
        text += f"  - offset {print_1d_array(self.offset)}\n"
        text += f"  - energy {print_1d_array(self.energy_out)} [eV]\n"
        text += f"  - energy-pdf {print_1d_array(self.pdf)} [/eV]\n"
        text += "  - precompound factor " f"{print_1d_array(self.precompound_factor)}\n"
        text += f"  - angular slope {print_1d_array(self.angular_slope)}\n"
        return text


# ======================================================================================
# Tabulated energy-angle
# ======================================================================================


class DistributionTabulatedEnergyAngle(DistributionBase):
    """Tabulated correlated energy-angle distribution."""

    energy: NDArray[float64]
    offset: NDArray[int64]
    energy_out: NDArray[float64]
    pdf: NDArray[float64]
    cdf: NDArray[float64]
    cosine_offset_: NDArray[int64]
    cosine: NDArray[float64]
    cosine_pdf: NDArray[float64]
    cosine_cdf: NDArray[float64]

    def __init__(
        self,
        energy: ArrayLike,
        offset: ArrayLike,
        energy_out: ArrayLike,
        pdf: ArrayLike,
        cosine_offset: ArrayLike,
        cosine: ArrayLike,
        cosine_pdf: ArrayLike,
    ) -> None:
        """Create a tabulated correlated energy-angle distribution."""
        super().__init__(
            child_label="tabulated_energy_angle_distribution",
            child_type=DISTRIBUTION_TABULATED_ENERGY_ANGLE,
            non_numba=[],
        )

        self.energy = np.asarray(energy, dtype=float64)
        self.offset = np.asarray(offset, dtype=int64)
        self.energy_out = np.asarray(energy_out, dtype=float64)
        pdf_array = np.asarray(pdf, dtype=float64)
        self.cosine_offset_ = np.asarray(cosine_offset, dtype=int64)
        self.cosine = np.asarray(cosine, dtype=float64)
        cosine_pdf_array = np.asarray(cosine_pdf, dtype=float64)

        self.pdf, self.cdf = multi_cdf_from_pdf(
            self.offset,
            self.energy_out,
            pdf_array,
        )

        self.cosine_pdf = cosine_pdf_array.copy()
        self.cosine_cdf = np.zeros_like(self.cosine_pdf)

        for i in range(len(self.offset)):
            energy_start = self.offset[i]

            if i + 1 < len(self.offset):
                energy_stop = self.offset[i + 1]
            else:
                energy_stop = len(self.energy_out)

            inner_offset = self.cosine_offset_[energy_start:energy_stop]

            if len(inner_offset) == 0:
                print_error(
                    "Each incident-energy table must reference at least one "
                    "cosine distribution."
                )

            cosine_start = inner_offset[0]

            if i + 1 < len(self.offset):
                cosine_stop = self.cosine_offset_[energy_stop]
            else:
                cosine_stop = len(self.cosine)

            inner_offset_local = inner_offset - cosine_start

            (
                self.cosine_pdf[cosine_start:cosine_stop],
                self.cosine_cdf[cosine_start:cosine_stop],
            ) = multi_cdf_from_pdf(
                inner_offset_local,
                self.cosine[cosine_start:cosine_stop],
                cosine_pdf_array[cosine_start:cosine_stop],
            )

    def __repr__(self) -> str:
        """Return a human-readable summary of the distribution."""
        text = super().__repr__()
        text += f"  - grid {print_1d_array(self.energy)} [eV]\n"
        text += f"  - offset {print_1d_array(self.offset)}\n"
        text += f"  - energy {print_1d_array(self.energy_out)} [eV]\n"
        text += f"  - energy-pdf {print_1d_array(self.pdf)} [/eV]\n"
        text += "  - cosine-offset " f"{print_1d_array(self.cosine_offset_)}\n"
        text += f"  - cosine {print_1d_array(self.cosine)}\n"
        text += f"  - cosine-pdf {print_1d_array(self.cosine_pdf)}\n"
        return text


# ======================================================================================
# N-body
# ======================================================================================


class DistributionNBody(DistributionBase):
    """N-body energy distribution represented by a tabulated PDF and CDF.

    The input probability density is normalized internally, and the
    corresponding cumulative distribution is generated for sampling.
    """

    pdf: DataTable

    def __init__(
        self,
        values: ArrayLike,
        probabilities: ArrayLike,
    ) -> None:
        """Create an N-body distribution.

        Parameters
        ----------
        values : array_like
            Tabulated sample values.

        probabilities : array_like
            Probability densities at the sample values. The values do not need
            to be normalized.
        """
        super().__init__(
            child_label="nbody_distribution",
            child_type=DISTRIBUTION_N_BODY,
            non_numba=[],
        )

        value_array = np.asarray(values, dtype=float64)
        probability_array = np.asarray(probabilities, dtype=float64)

        if value_array.ndim != 1:
            print_error("values must be one-dimensional.")
        if probability_array.ndim != 1:
            print_error("probabilities must be one-dimensional.")
        if len(value_array) != len(probability_array):
            print_error("values and probabilities must have the same length.")

        pdf_normalized, cdf_normalized = cdf_from_pdf(
            value_array,
            probability_array,
        )

        self.pdf = DataTable(
            value_array,
            pdf_normalized,
            INTERPOLATION_LINEAR,
            aux=cdf_normalized,
        )

    def __repr__(self) -> str:
        """Return a human-readable summary of the distribution."""
        text = super().__repr__()
        text += f"  - value {print_1d_array(self.pdf.x)}\n"
        text += f"  - pdf {print_1d_array(self.pdf.y)}\n"
        return text
