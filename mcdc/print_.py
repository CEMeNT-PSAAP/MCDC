import numba as nb
import numpy as np
import sys
from mpi4py import MPI


master = MPI.COMM_WORLD.Get_rank() == 0


def print_msg(msg):
    if master:
        print(msg)
        sys.stdout.flush()


def print_error(msg):
    print("ERROR: %s\n" % msg)
    sys.stdout.flush()
    sys.exit()


def print_warning(msg):
    if master:
        print("Warning: %s\n" % msg)
        sys.stdout.flush()


def print_banner(mcdc):
    size = MPI.COMM_WORLD.Get_size()
    if master:
        banner = (
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
            + "\n"
        )
        if nb.config.DISABLE_JIT:
            banner += "           Mode | Python\n"
        else:
            banner += "           Mode | Numba\n"
        if mcdc["technique"]["iQMC"]:
            banner += "      Algorithm | iQMC\n"
            rng = mcdc["technique"]["iqmc_generator"]
            if mcdc["setting"]["mode_eigenvalue"]:
                solver = mcdc["technique"]["iqmc_eigenmode_solver"]
            else:
                solver = mcdc["technique"]["iqmc_fixed_source_solver"]
            banner += "            RNG | " + rng + "\n"
            banner += "         Solver | " + solver + "\n"
        else:
            banner += "      Algorithm | History-based\n"
        banner += "  MPI Processes | %i\n" % size
        banner += " OpenMP Threads | 1"
        print(banner)
        sys.stdout.flush()


def print_progress(percent, mcdc):
    if master:
        sys.stdout.write("\r")
        if not mcdc["setting"]["mode_eigenvalue"]:
            if mcdc["setting"]["N_census"] == 1:
                sys.stdout.write(
                    " [%-28s] %d%%" % ("=" * int(percent * 28), percent * 100.0)
                )
            else:
                idx = mcdc["idx_census"] + 1
                N = len(mcdc["setting"]["census_time"])
                sys.stdout.write(
                    " Census %i/%i: [%-28s] %d%%"
                    % (idx, N, "=" * int(percent * 28), percent * 100.0)
                )
        else:
            if mcdc["setting"]["gyration_radius"]:
                sys.stdout.write(
                    " [%-40s] %d%%" % ("=" * int(percent * 40), percent * 100.0)
                )
            else:
                sys.stdout.write(
                    "[%-32s] %d%%" % ("=" * int(percent * 32), percent * 100.0)
                )
        sys.stdout.flush()


def print_header_eigenvalue(mcdc):
    if master:
        if mcdc["setting"]["gyration_radius"]:
            print("\n #     k        GyRad.  k (avg)            ")
            print(" ====  =======  ======  ===================")
        elif mcdc["technique"]["iQMC"]:
            print("\n #     k        Residual         ")
            print(" ==== ======= ===================")
        else:
            print("\n #     k        k (avg)            ")
            print(" ====  =======  ===================")


def print_header_batch(mcdc):
    idx_batch = mcdc["idx_batch"]
    if master:
        print("\nBatch %i/%i" % (idx_batch + 1, mcdc["setting"]["N_batch"]))


def print_progress_eigenvalue(mcdc):
    if master:
        idx_cycle = mcdc["idx_cycle"]
        k_eff = mcdc["k_eff"]
        k_avg = mcdc["k_avg_running"]
        k_sdv = mcdc["k_sdv_running"]
        gr = mcdc["gyration_radius"][idx_cycle]
        if mcdc["setting"]["progress_bar"]:
            sys.stdout.write("\r")
            sys.stdout.write("\033[K")
        if mcdc["setting"]["gyration_radius"]:
            if not mcdc["cycle_active"]:
                print(" %-4i  %.5f  %6.2f" % (idx_cycle + 1, k_eff, gr))
            else:
                print(
                    " %-4i  %.5f  %6.2f  %.5f +/- %.5f"
                    % (idx_cycle + 1, k_eff, gr, k_avg, k_sdv)
                )
        else:
            if not mcdc["cycle_active"]:
                print(" %-4i  %.5f" % (idx_cycle + 1, k_eff))
            else:
                print(
                    " %-4i  %.5f  %.5f +/- %.5f" % (idx_cycle + 1, k_eff, k_avg, k_sdv)
                )


def print_iqmc_eigenvalue_progress(mcdc):
    if master:
        k_eff = mcdc["k_eff"]
        itt = mcdc["technique"]["iqmc_itt_outter"]
        res = mcdc["technique"]["iqmc_res_outter"]
        print("\n ", itt, " ", np.round(k_eff, 5), " ", np.round(res, 9))


def print_iqmc_eigenvalue_exit_code(mcdc):
    if master:
        maxit = mcdc["technique"]["iqmc_maxitt"]
        itt = mcdc["technique"]["iqmc_itt_outter"]
        tol = mcdc["technique"]["iqmc_tol"]
        res = mcdc["technique"]["iqmc_res_outter"]
        solver = mcdc["technique"]["iqmc_eigenmode_solver"]
        if itt >= maxit:
            print("\n ================================\n ")
            print(
                solver
                + " convergence to tolerance not achieved: Maximum number of iterations."
            )
        elif res <= tol:
            print("\n ================================\n ")
            print("Successful " + solver + " convergence.")


def print_runtime(mcdc):
    total = mcdc["runtime_total"]
    preparation = mcdc["runtime_preparation"]
    simulation = mcdc["runtime_simulation"]
    output = mcdc["runtime_output"]
    if master:
        print("\n Runtime report:")
        print_time("Total      ", total, 100)
        print_time("Preparation", preparation, preparation / total * 100)
        print_time("Simulation ", simulation, simulation / total * 100)
        print_time("Output     ", output, output / total * 100)
        print("\n")
        sys.stdout.flush()


def print_time(tag, t, percent):
    if t >= 24 * 60 * 60:
        print("   %s | %.2f days (%.1f%%)" % (tag, t / 24 / 60 / 60), percent)
    elif t >= 60 * 60:
        print("   %s | %.2f hours (%.1f%%)" % (tag, t / 60 / 60, percent))
    elif t >= 60:
        print("   %s | %.2f minutes (%.1f%%)" % (tag, t / 60, percent))
    else:
        print("   %s | %.2f seconds (%.1f%%)" % (tag, t, percent))


def print_bank(bank, show_content=False):
    tag = bank["tag"]
    size = bank["size"]
    particles = bank["particles"]

    print("\n=============")
    print("Particle bank")
    print("  tag  :", tag)
    print("  size :", size, "of", len(bank["particles"]))
    if show_content and size > 0:
        for i in range(size):
            print(" ", particles[i])
    print("\n")


def print_progress_iqmc(mcdc):
    # TODO: function was not working with numba when structured like the
    # other print_progress functions
    if master:
        if mcdc["setting"]["progress_bar"]:
            itt = mcdc["technique"]["iqmc_itt"]
            res = mcdc["technique"]["iqmc_res"]
            print("\n*******************************")
            print("Iteration ", itt)
            print("Residual ", res)
            print("*******************************\n")
