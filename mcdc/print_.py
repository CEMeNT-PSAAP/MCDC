"""Terminal diagnostics and progress reporting for MC/DC."""

import sys

import numba as nb
from colorama import Fore, Style
from mpi4py import MPI

_IS_MASTER = MPI.COMM_WORLD.Get_rank() == 0


# ======================================================================================
# General formatting and diagnostics
# ======================================================================================


def print_1d_array(array):
    """Return a compact representation of a one-dimensional array."""
    size = len(array)
    if size > 5:
        return (
            f"(size={size}): "
            f"[{array[0]:.5g}, {array[1]:.5g}, ..., "
            f"{array[-2]:.5g}, {array[-1]:.5g}]"
        )

    values = ", ".join(f"{value:.5g}" for value in array)
    return f"(size={size}): [{values}]"


def print_msg(message):
    """Print a framework message on the MPI master rank."""
    if not _IS_MASTER:
        return
    print(message)
    sys.stdout.flush()


def print_error(message):
    """Print a fatal error and terminate the current process unsuccessfully."""
    print(f"{Fore.RED}[ERROR]: {message}{Style.RESET_ALL}")
    sys.stdout.flush()
    raise SystemExit(1)


def print_warning(message):
    """Print a warning on the MPI master rank."""
    if not _IS_MASTER:
        return
    print(f"{Fore.YELLOW}[WARNING]: {message}{Style.RESET_ALL}")
    sys.stdout.flush()


def print_structure(structure):
    """Print every field in a NumPy structured record."""
    for name in structure.dtype.names:
        print(f"{name} = {structure[name]}")


def print_bank(bank, show_content=False):
    """Print summary information for a particle bank."""
    size_field = bank["size"]
    size = int(size_field[0]) if getattr(size_field, "ndim", 0) else int(size_field)
    particles = bank["particle_data"]

    print("\n=============")
    print("Particle bank")
    print("  tag  :", bank["tag"])
    print("  size :", size, "of", len(particles))
    if show_content:
        for index in range(size):
            print(" ", particles[index])
    print()


# ======================================================================================
# Calculation headers
# ======================================================================================


def print_banner():
    """Print the MC/DC banner on the MPI master rank."""
    if not _IS_MASTER:
        return
    print(
        "\n"
        + r"  __  __  ____  __ ____   ____ "
        + "\n"
        + r" |  \/  |/ ___|/ /_  _ \ / ___|"
        + "\n"
        + r" | |\/| | |   /_  / | | | |    "
        + "\n"
        + r" | |  | | |___ / /| |_| | |___ "
        + "\n"
        + r" |_|  |_|\____|// |____/ \____|"
        + "\n"
    )
    sys.stdout.flush()


def print_configuration():
    """Print the active execution configuration on the MPI master rank."""
    if not _IS_MASTER:
        return
    mode = "Python" if nb.config.DISABLE_JIT else "Numba"
    mpi_size = MPI.COMM_WORLD.Get_size()
    print(f"           Mode | {mode}\n  MPI Processes | {mpi_size}\n")
    sys.stdout.flush()


def print_eigenvalue_header(simulation):
    """Print the eigenvalue-cycle table header."""
    if not _IS_MASTER:
        return
    if simulation["settings"]["use_gyration_radius"]:
        print("\n #     k        GyRad.  k (avg)            ")
        print(" ====  =======  ======  ===================")
    else:
        print("\n #     k        k (avg)            ")
        print(" ====  =======  ===================")
    sys.stdout.flush()


def print_header_batch(index, size):
    """Print a one-based batch header."""
    if not _IS_MASTER:
        return
    print(f"\nBatch {index + 1}/{size}")
    sys.stdout.flush()


# ======================================================================================
# Calculation progress
# ======================================================================================


def print_progress(percent, simulation):
    """Update the fixed-source or eigenvalue progress bar."""
    if not _IS_MASTER:
        return

    sys.stdout.write("\r")
    settings = simulation["settings"]
    if not settings["neutron_eigenvalue_mode"]:
        if settings["N_census"] == 1:
            sys.stdout.write(
                " [%-28s] %d%%" % ("=" * int(percent * 28), percent * 100.0)
            )
        else:
            index = simulation["idx_census"] + 1
            size = settings["N_census"]
            sys.stdout.write(
                " Census %i/%i: [%-28s] %d%%"
                % (index, size, "=" * int(percent * 28), percent * 100.0)
            )
    elif settings["use_gyration_radius"]:
        sys.stdout.write(" [%-40s] %d%%" % ("=" * int(percent * 40), percent * 100.0))
    else:
        sys.stdout.write(" [%-32s] %d%%" % ("=" * int(percent * 32), percent * 100.0))
    sys.stdout.flush()


def print_progress_eigenvalue(simulation, data):
    """Print one eigenvalue-cycle result."""
    if not _IS_MASTER:
        return

    import mcdc.mcdc_get as mcdc_get

    index = simulation["idx_cycle"]
    k_effective = simulation["k_eff"]
    k_average = simulation["k_avg_running"]
    k_standard_deviation = simulation["k_sdv_running"]
    settings = simulation["settings"]

    if settings["use_progress_bar"]:
        sys.stdout.write("\r\033[K")

    if settings["use_gyration_radius"]:
        gyration_radius = mcdc_get.simulation.gyration_radius(index, simulation, data)
        if simulation["cycle_active"]:
            print(
                " %-4i  %.5f  %6.2f  %.5f +/- %.5f"
                % (
                    index + 1,
                    k_effective,
                    gyration_radius,
                    k_average,
                    k_standard_deviation,
                )
            )
        else:
            print(" %-4i  %.5f  %6.2f" % (index + 1, k_effective, gyration_radius))
    elif simulation["cycle_active"]:
        print(
            " %-4i  %.5f  %.5f +/- %.5f"
            % (index + 1, k_effective, k_average, k_standard_deviation)
        )
    else:
        print(" %-4i  %.5f" % (index + 1, k_effective))
    sys.stdout.flush()


# ======================================================================================
# Runtime report
# ======================================================================================


def print_time(label, duration, percent):
    """Print one duration using an appropriate time unit."""
    if duration >= 24 * 60 * 60:
        value = duration / (24 * 60 * 60)
        unit = "days"
    elif duration >= 60 * 60:
        value = duration / (60 * 60)
        unit = "hours"
    elif duration >= 60:
        value = duration / 60
        unit = "minutes"
    else:
        value = duration
        unit = "seconds"
    print(f"   {label} | {value:.2f} {unit} ({percent:.1f}%)")


def print_runtime(simulation):
    """Print preparation, transport, and output runtimes."""
    if not _IS_MASTER:
        return

    total = simulation["runtime_total"]
    preparation = simulation["runtime_preparation"]
    transport = simulation["runtime_simulation"]
    output = simulation["runtime_output"]

    def percentage(duration):
        return duration / total * 100.0 if total > 0.0 else 0.0

    print("\n Runtime report:")
    print_time("Total      ", total, 100.0)
    print_time("Preparation", preparation, percentage(preparation))
    print_time("Simulation ", transport, percentage(transport))
    print_time("Output     ", output, percentage(output))
    print()
    sys.stdout.flush()
