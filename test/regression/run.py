import h5py, os, sys, argparse, fnmatch
import numpy as np
from colorama import Fore, Style

# Option parser
parser = argparse.ArgumentParser(description="MC/DC regression test")
parser.add_argument("--mode", type=str, choices=["python", "numba"], default="python")
parser.add_argument("--mpiexec", type=int, default=0)
parser.add_argument("--srun", type=int, default=0)
parser.add_argument("--name", type=str, default="ALL")
parser.add_argument("--skip", type=str, default="NONE")
args, unargs = parser.parse_known_args()

# Parse
mode = args.mode
mpiexec = args.mpiexec
srun = args.srun
name = args.name
skip = args.skip

# Get test names
if name == "ALL":
    names = []
    for item in os.listdir():
        if os.path.isdir(item):
            names.append(item)
else:
    names = [item for item in os.listdir() if fnmatch.fnmatch(item, name)]
names.sort()

# Remove skipped if specified
if skip != "NONE":
    skips = [item for item in os.listdir() if fnmatch.fnmatch(item, skip)]
    for name in skips:
        names.remove(name)

# Data for each test
printouts = []
runtimes = []
flags = []
error_msgs = []
crashes = []
all_pass = True

# Run all tests
for i, name in enumerate(names):
    # Skip cache if any
    if name == "__pycache__":
        continue

    print("\n[%i/%i] " % (i + 1, len(names)) + name)
    error_msgs.append([])
    crashes.append(False)
    runtimes.append(-1)

    # Change directory
    os.chdir(name)

    # Check test setup
    if not os.path.exists("input.py"):
        print(Fore.RED + "  input.py is missing\n" + Style.RESET_ALL)
        sys.exit()
    if not os.path.exists("answer.h5"):
        print(Fore.RED + "  answer.h5 is missing\n" + Style.RESET_ALL)
        sys.exit()

    # Delete output if exists
    if os.path.exists("output.h5"):
        os.remove("output.h5")

    # Run the test problem (redirect the stdout)
    if mpiexec > 1:
        os.system(
            "mpiexec -n %i python input.py --mode=%s --output=output --no-progress-bar > tmp 2>&1"
            % (mpiexec, mode)
        )
    elif srun > 1:
        os.system(
            "srun -n %i python input.py --mode=%s --output=output --no-progress-bar > tmp 2>&1"
            % (srun, mode)
        )
    else:
        os.system(
            "python input.py --mode=%s --output=output --no-progress-bar > tmp 2>&1"
            % (mode)
        )
    with open("tmp") as f:
        printouts.append(f.read())
    os.remove("tmp")

    # Check if crashed
    if not os.path.exists("output.h5"):
        print(Fore.RED + "  Failed: Run crashed" + Style.RESET_ALL)
        all_pass = False
        crashes[-1] = True
        os.chdir("..")
        continue

    # Get the output and the answer key
    output = h5py.File("output.h5", "r")
    answer = h5py.File("answer.h5", "r")

    runtimes[-1] = output["runtime/total"][()]
    print("  (%.2f seconds)" % runtimes[-1])

    # Compare all scores
    for score in [key for key in output["tally"].keys() if key != "grid"]:
        # Compare mean, sdev, and uq_var (if available)
        for result in [key for key in output["tally/" + score].keys()]:
            result_name = "tally/" + score + "/" + result
            a = output[result_name][:]
            b = answer[result_name][:]

            # Passed?
            if np.isclose(a, b).all():
                print(
                    Fore.GREEN
                    + "  {}: Passed".format(score + "/" + result)
                    + Style.RESET_ALL
                )
            else:
                all_pass = False
                error_msgs[-1].append(
                    "Differences in %s"
                    % (name + "/" + score + "/" + result + "\n" + "{}".format(a - b))
                )
                print(
                    Fore.RED
                    + "  {}: Failed".format(score + "/" + result)
                    + Style.RESET_ALL
                )

    # Other quantities
    for result_name in ["k_mean", "k_sdev", "k_cycle", "k_eff"]:
        if result_name not in output.keys():
            continue

        a = output[result_name][()]
        b = answer[result_name][()]

        # Passed?
        if np.isclose(a, b).all():
            print(Fore.GREEN + "  {}: Passed".format(result_name) + Style.RESET_ALL)
        else:
            all_pass = False
            error_msgs[-1].append("Differences in {}\n{}".format(result_name, a - b))
            print(Fore.RED + "  {}: Failed".format(result_name) + Style.RESET_ALL)

    # iQMC flux
    if "iqmc" in output.keys():
        for score in [key for key in output["iqmc/tally/"].keys()]:
            result_name = "iqmc/tally/" + score
            a = output[result_name][:]
            b = answer[result_name][:]
            if a.size == 0:
                continue
            # Passed?
            if np.isclose(a, b).all():
                print(Fore.GREEN + "  {}: Passed".format(score) + Style.RESET_ALL)
            else:
                all_pass = False
                error_msgs[-1].append("Differences in {}\n{}".format(score, a - b))
                print(Fore.RED + "  {}: Failed".format(score) + Style.RESET_ALL)

    # Close files
    output.close()
    answer.close()

    # Move back up
    os.chdir("..")

# Report test results
N_fails = 0
for i in range(len(names)):
    if crashes[i] or len(error_msgs[i]) > 0:
        N_fails += 1

print(
    "\nTests passed: "
    + Fore.GREEN
    + "%i/%i" % (len(names) - N_fails, len(names))
    + Style.RESET_ALL
)
print("Tests failed: " + Fore.RED + "%i/%i" % (N_fails, len(names)) + Style.RESET_ALL)
print("  (%.2f seconds)\n" % np.sum(np.array(runtimes)))
for i in range(len(names)):
    if crashes[i]:
        print("\n" + "=" * 80)
        print("\n## {} crashed:".format(names[i]))
        print(printouts[i])
    if len(error_msgs[i]) > 0:
        print("\n" + "=" * 80)
        print("\n## {} failed:".format(names[i]))
        print(printouts[i])
        print("\n===\n")
        for msg in error_msgs[i]:
            print("\n# " + msg + "\n")

assert all_pass
