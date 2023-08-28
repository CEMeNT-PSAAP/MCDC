import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
import sys

from reference import reference

output = sys.argv[1]

# Reference solution
with h5py.File(output, "r") as f:
    x = f["tally/grid/x"][:]
    t = f["tally/grid/t"][:]
phi_t_ref = reference(x, t)

# Get results
with h5py.File(output, "r") as f:
    x = f["tally/grid/x"][:]
    t = f["tally/grid/t"][:]
    dx = x[1] - x[0]
    x_mid = 0.5 * (x[:-1] + x[1:])
    dt = t[1:] - t[:-1]
    K = len(dt)
    J = len(x_mid)

    phi_t = f["tally/flux-t/mean"][1:]
    phi_t_sd = f["tally/flux-t/sdev"][1:]

# Normalize
for k in range(K):
    phi_t[k] *= 0.5 / dx
    phi_t_sd[k] *= 0.5 / dx

# Flux - t
fig = plt.figure()
ax = plt.axes(ylim=(5e-5, 4e-1))
ax.grid()
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"Flux")
ax.set_title(r"$\bar{\phi}_{j}(t)$")
ax.set_yscale('log')
(line1,) = ax.plot([], [], "-b", label="MC")
(line2,) = ax.plot([], [], "--r", label="Ref.")
fb = ax.fill_between([], [], [], [], alpha=0.2, color="b")
text = ax.text(0.02, 0.9, "", transform=ax.transAxes)
ax.legend()


def animate(k):
    global fb
    fb.remove()
    line1.set_data(x_mid, phi_t[k, :])
    fb = ax.fill_between(
        x_mid,
        phi_t[k, :] - phi_t_sd[k, :],
        phi_t[k, :] + phi_t_sd[k, :],
        alpha=0.2,
        color="b",
    )
    line2.set_data(x_mid, phi_t_ref[k - 1, :])
    text.set_text(r"$t = %.1f$ s" % (t[k]))
    return line1, line2, text


simulation = animation.FuncAnimation(fig, animate, frames=K)
writervideo = animation.FFMpegWriter(fps=6)
plt.show()
