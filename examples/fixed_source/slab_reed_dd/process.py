import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.integrate import quad

# =============================================================================
# Reference solution (not accurate enough for N_hist > 1E7)
# =============================================================================


def phi1(x):
    return (
        1.0
        - 5.96168047527760 * 10 ** (-47) * np.cosh(52.06761235859028 * x)
        - 6.78355315350872 * 10 ** (-56) * np.cosh(62.76152118553390 * x)
        - 7.20274049646598 * 10 ** (-84) * np.cosh(95.14161078659372 * x)
        - 6.34541150517664 * 10 ** (-238) * np.cosh(272.5766481169758 * x)
    )


def phi2(x):
    return (
        1.685808767651539 * 10**3 * np.exp(-5.206761235859028 * x)
        + 3.143867366942945 * 10**4 * np.exp(-6.276152118553390 * x)
        + 2.879977113018352 * 10**7 * np.exp(-9.514161078659372 * x)
        + 8.594190506002560 * 10**22 * np.exp(-27.25766481169758 * x)
        + 1.298426035202193 * 10 ** (-36) * np.exp(27.25766481169758 * x)
        + 1.432344656303454 * 10 ** (-13) * np.exp(9.514161078659372 * x)
        + 1.514562265056083 * 10 ** (-9) * np.exp(6.276152118553390 * x)
        + 1.594431209450755 * 10 ** (-8) * np.exp(5.206761235859028 * x)
    )


def phi3(x):
    return 1.105109108062394


def phi4(x):
    return (
        10.0
        - 0.1983746883968300 * np.exp(0.5254295183311557 * x)
        - 7.824765332896027 * 10 ** (-5) * np.exp(1.108937229227813 * x)
        - 9.746660212187006 * 10 ** (-6) * np.exp(1.615640334315550 * x)
        - 2.895098351422132 * 10 ** (-13) * np.exp(4.554850586269065 * x)
        - 75.34793864805979 * np.exp(-0.5254295183311557 * x)
        - 20.42874998426011 * np.exp(-1.108937229227813 * x)
        - 7.129175418204712 * 10 ** (2) * np.exp(-1.615640334315550 * x)
        - 2.716409367577795 * 10 ** (9) * np.exp(-4.554850586269065 * x)
    )


def phi5(x):
    return (
        31.53212162577067 * np.exp(-0.5254295183311557 * x)
        + 26.25911060454856 * np.exp(-1.108937229227813 * x)
        + 1.841223066417334 * 10 ** (3) * np.exp(-1.615640334315550 * x)
        + 1.555593549394869 * 10 ** (11) * np.exp(-4.554850586269065 * x)
        - 3.119310353653182 * 10 ** (-3) * np.exp(0.5254295183311557 * x)
        - 6.336401143340483 * 10 ** (-7) * np.exp(1.108937229227813 * x)
        - 3.528757679361232 * 10 ** (-8) * np.exp(1.615640334315550 * x)
        - 4.405514335746888 * 10 ** (-18) * np.exp(4.554850586269065 * x)
    )


def f_phi(x1, x2):
    dx = x2 - x1
    if x2 <= 2.0:
        return quad(phi1, x1, x2)[0] / dx
    if x2 <= 3.0:
        return quad(phi2, x1, x2)[0] / dx
    if x2 <= 5.0:
        return quad(phi3, x1, x2)[0] / dx
    if x2 <= 6.0:
        return quad(phi4, x1, x2)[0] / dx
    return quad(phi5, x1, x2)[0] / dx


def f_phi_x(x):
    if x <= 2.0:
        return phi1(x)
    if x <= 3.0:
        return phi2(x)
    if x <= 5.0:
        return phi3(x)
    if x <= 6.0:
        return phi4(x)
    return phi5(x)


x = np.linspace(0.0, 8.0, 81)
dx = x[1] - x[0]
x_mid = 0.5 * (x[:-1] + x[1:])

phi_ref = np.zeros_like(x_mid)
phi_x_ref = np.zeros_like(x)

for i in range(len(x)):
    phi_x_ref[i] = f_phi_x(x[i])

for i in range(len(x_mid)):
    phi_ref[i] = f_phi(x[i], x[i + 1])

# =============================================================================
# Plot
# =============================================================================

# Load output
with h5py.File("output.h5", "r") as f:
    # Note the spatial (dx) and source strength (100+1) normalization
    phi = f["tallies/mesh_tally_0/flux/mean"][:] / dx * 101.0
    phi_sd = f["tallies/mesh_tally_0/flux/sdev"][:] / dx * 101.0

# Flux - spatial average
plt.plot(x_mid, phi, "-b", label="MC")
plt.fill_between(x_mid, phi - phi_sd, phi + phi_sd, alpha=0.2, color="b")
plt.plot(x_mid, phi_ref, "--r", label="ref.")
plt.xlabel(r"$x$, cm")
plt.ylabel("Flux")
plt.grid()
plt.legend()
plt.title(r"$\bar{\phi}_i$")
plt.show()
