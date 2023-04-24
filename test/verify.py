import numpy as np
import argparse
import os
import glob

FILE_PATH = "./verification/analytic/fixed_source"
PROBLEM_PATH = "../../../../problems/fixed_source"


def run(N_min=3, N_max=7, N_proc=2):
    mpi = "mpiexec" if os.name == "nt" else "srun"
    for task in os.scandir(FILE_PATH):
        os.chdir(task)
        problem = os.path.join(PROBLEM_PATH, task.name + "_input.py")
        for N_hist in np.logspace(N_min, N_max, (N_max - N_min) * 2 + 1):
            particles = int(N_hist)
            save_file = "output_" + str(particles)
            if N_proc == 1:
                os.system(
                    "python {} --mode=python --particles={} \
                        --file={}".format(
                        problem, particles, save_file
                    )
                )
            else:
                os.system(
                    "{} -n {} python {} --mode=numba --particles={} \
                    --file={}".format(
                        mpi, N_proc, problem, particles, save_file
                    )
                )
        # os.system("python process.py")
        os.chdir(r"../../../..")


def display(folder=""):
    figures = glob.glob(FILE_PATH + folder + "/**/*.png", recursive=True)
    for figure in figures:
        os.system("display {}".format(figure))


def clean(check=True):
    figures = glob.glob(FILE_PATH + "/**/*.png", recursive=True)
    data = glob.glob(FILE_PATH + "/**/output*.h5", recursive=True)
    files = np.array(figures + data)
    ans = "y"
    if check:
        ans = input("{}\n\nRemove these files (y/n)? ".format(files)).lower()
    if ans == "y":
        for file in files:
            os.remove(file)
        print("Removed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Setting parameters to run verification from command line"
    )
    parser.add_argument("--run", action="store_true", dest="run")
    parser.add_argument("-Nmin", action="store", dest="N_min", type=int, default=1)
    parser.add_argument("-Nmax", action="store", dest="N_max", type=int, default=3)
    parser.add_argument("-nprocess", action="store", dest="N_proc", type=int, default=1)

    parser.add_argument("--display", action="store_true", dest="display")
    parser.add_argument("-folder", action="store", dest="folder", type=str, default="")

    parser.add_argument("--clean", action="store_true", dest="clean", default=False)
    parser.add_argument("-check", action="store_true", dest="check", default=True)

    args = parser.parse_args()

    if args.run:
        run(args.N_min, args.N_max, args.N_proc)

    if args.display:
        display(args.folder)

    if args.clean:
        clean(args.check)
