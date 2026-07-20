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
    def __init__(
        self,
        child_label: str,
        child_type: int,
        non_numba: list[str],
    ) -> None:
        # MC/DC framework metadata
        super().__init__("distribution", child_label, child_type, non_numba)


# ======================================================================================
# None
# ======================================================================================
# Placeholder for a distribution that does not need to store data:
#   - Isotropic distributions
#   - Energy-correlated angles stored in the energy distribution


class DistributionNone(DistributionBase):
    def __init__(self) -> None:
        # MC/DC framework metadata
        super().__init__("none_distribution", DISTRIBUTION_NONE, [])


# ======================================================================================
# Probability mass function
# ======================================================================================


class DistributionPMF(DistributionBase):
    value: NDArray[float64]
    pmf: NDArray[float64]
    cmf: NDArray[float64]

    def __init__(self, value: ArrayLike, pmf: ArrayLike) -> None:
        # MC/DC framework metadata
        super().__init__("pmf_distribution", DISTRIBUTION_PMF, [])

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
        text = super().__repr__()

        text += f"  - value {print_1d_array(self.value)}\n"
        text += f"  - pmf {print_1d_array(self.pmf)}\n"
        return text


# ======================================================================================
# Tabulated
# ======================================================================================


class DistributionTabulated(DistributionBase):
    pdf: DataTable

    def __init__(
        self,
        value: ArrayLike,
        pdf: ArrayLike | None = None,
        cdf: ArrayLike | None = None,
    ) -> None:
        # MC/DC framework metadata
        super().__init__("tabulated_distribution", DISTRIBUTION_TABULATED, [])

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
        text = super().__repr__()

        text += f"  - value {print_1d_array(self.pdf.x)}\n"
        text += f"  - probability density {print_1d_array(self.pdf.y)}\n"
        text += "  - cumulative distribution " f"{print_1d_array(self.pdf.aux[0])}\n"
        return text


# ======================================================================================
# Multi-table
# ======================================================================================


class DistributionMultiTable(DistributionBase):
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
        # MC/DC framework metadata
        super().__init__("multi_table_distribution", DISTRIBUTION_MULTITABLE, [])

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
            elif cdf_array is not None:
                table = DistributionTabulated(
                    value_array[start_i:stop_i],
                    cdf=cdf_array[start_i:stop_i],
                )
            else:
                table = DistributionTabulated([])  # Unachievable

            self.tables.append(table)

    def __repr__(self) -> str:
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
    C1: float
    C2: float

    def __init__(self, C1: float, C2: float) -> None:
        # MC/DC framework metadata
        super().__init__(
            "level_scattering_distribution",
            DISTRIBUTION_LEVEL_SCATTERING,
            [],
        )

        self.C1 = C1
        self.C2 = C2

    def __repr__(self) -> str:
        text = super().__repr__()

        text += f"  - C1: {self.C1} [/eV^l]\n"
        text += f"  - C2: {self.C2}\n"
        return text


# ======================================================================================
# Evaporation
# ======================================================================================


class DistributionEvaporation(DistributionBase):
    nuclear_temperature: DataTable
    restriction_energy: float

    def __init__(
        self,
        nuclear_temperature_energy_grid: NDArray[float64],
        nuclear_temperature_value: NDArray[float64],
        restriction_energy: float,
        temperature_interpolations: int | Sequence[int],
        interpolation_boundaries: Sequence[int] | None,
    ) -> None:
        # MC/DC framework metadata
        super().__init__("evaporation_distribution", DISTRIBUTION_EVAPORATION, [])

        self.restriction_energy = restriction_energy
        self.nuclear_temperature = DataTable(
            nuclear_temperature_energy_grid,
            nuclear_temperature_value,
            temperature_interpolations,
            interpolation_boundaries,
        )

    def __repr__(self) -> str:
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
    nuclear_temperature: DataTable
    restriction_energy: float

    def __init__(
        self,
        nuclear_temperature_energy_grid: NDArray[float64],
        nuclear_temperature_value: NDArray[float64],
        restriction_energy: float,
        temperature_interpolations: int | Sequence[int],
        interpolation_boundaries: Sequence[int] | None,
    ) -> None:
        # MC/DC framework metadata
        super().__init__("maxwellian_distribution", DISTRIBUTION_MAXWELLIAN, [])

        self.restriction_energy = restriction_energy
        self.nuclear_temperature = DataTable(
            nuclear_temperature_energy_grid,
            nuclear_temperature_value,
            temperature_interpolations,
            interpolation_boundaries,
        )

    def __repr__(self) -> str:
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
        # MC/DC framework metadata
        super().__init__("kalbach_mann_distribution", DISTRIBUTION_KALBACH_MANN, [])

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
        # MC/DC framework metadata
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
    pdf: DataTable

    def __init__(
        self,
        values: ArrayLike,
        probabilities: ArrayLike,
    ) -> None:
        # MC/DC framework metadata
        super().__init__("nbody_distribution", DISTRIBUTION_N_BODY, [])

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
        text = super().__repr__()

        text += f"  - value {print_1d_array(self.pdf.x)}\n"
        text += f"  - pdf {print_1d_array(self.pdf.y)}\n"
        return text
