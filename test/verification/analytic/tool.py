import matplotlib.pyplot as plt
import numpy as np


def error(val, ref):
    return np.sqrt(np.average((val - ref) ** 2) / np.sum(ref**2))


def error_max(val, ref):
    return np.max(val - ref) / np.max(ref)


def rerror(val, ref):
    return np.sqrt(np.sum(((val - ref) / ref) ** 2)) / ref.size


def rerror_max(val, ref):
    return np.max(np.abs((val - ref) / ref))


def plot_convergence(name, N_particle, error, error_max):
    mid = int(len(N_particle) / 2)

    plt.plot(N_particle, error, "bo", fillstyle="none", label="2-norm")

    plt.plot(N_particle, error_max, "gD", fillstyle="none", label="max")

    line = 1.0 / np.sqrt(N_particle)
    line *= error[mid] / line[mid]
    plt.plot(N_particle, line, "r--", label=r"$O(N^{-0.5})$")

    line = 1.0 / np.sqrt(N_particle)
    line *= error_max[mid] / line[mid]
    plt.plot(N_particle, line, "r--")

    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("Relative error")
    plt.xlabel(r"# of histories, $N$")
    plt.legend()
    plt.grid()
    plt.title(name)
    plt.savefig("../results/" + name + ".png")
    # plt.show()
    plt.clf()
