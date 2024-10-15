import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.integrate import quad

# =============================================================================
# Import data
# =============================================================================

with h5py.File("output.h5", "r") as f:
    phi = f["iqmc/tally/flux/mean"][:]
    x = f["iqmc/grid/x"][:]
    mesh = f["iqmc/grid/x"][:]
    dx = x[1] - x[0]
    x_mid = 0.5 * (x[:-1] + x[1:])
    f.close()

# =============================================================================
# Reference solution (not accurate enough for N_hist > 1E7)
# =============================================================================


def reeds_sol(Nx=80, LB=-8.0, RB=8.0):
    # =========================================================================
    # Reference solution
    # =========================================================================
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
        midpoint = (x2 + x1) * 0.5
        x1 = abs(x1)
        x2 = abs(x2)
        dx = abs(x2 - x1)
        if x1 > x2:
            a = x2
            b = x1
        else:
            a = x1
            b = x2
        if midpoint < -6.0:
            return quad(phi5, a, b)[0] / dx
        elif -6.0 < midpoint < -5.0:
            return quad(phi4, a, b)[0] / dx
        elif -5.0 < midpoint < -3.0:  # vacuum region 1
            return quad(phi3, a, b)[0] / dx
        elif -3.0 < midpoint < -2.0:
            return quad(phi2, a, b)[0] / dx
        elif -2.0 < midpoint < 0.0:
            return quad(phi1, a, b)[0] / dx
        elif 0.0 < midpoint < 2.0:
            return quad(phi1, a, b)[0] / dx
        elif 2.0 < midpoint < 3.0:
            return quad(phi2, a, b)[0] / dx
        elif 3.0 < midpoint < 5.0:  # vacuum region 2
            return quad(phi3, a, b)[0] / dx
        elif 5.0 < midpoint < 6.0:
            return quad(phi4, a, b)[0] / dx
        elif 6.0 < midpoint:
            return quad(phi5, a, b)[0] / dx

    phi_ref = np.zeros(Nx)
    dx = (RB - LB) / Nx
    left_edges = np.linspace(LB, RB - dx, num=Nx)
    right_edges = left_edges + dx

    for i in range(Nx):
        phi_ref[i] = f_phi(left_edges[i], right_edges[i])
    phi_ref = np.reshape(phi_ref, (Nx, 1))

    return phi_ref


phi_ref = reeds_sol(Nx=x_mid.size, LB=-8.0, RB=8.0)


# =============================================================================
# Plot
# =============================================================================

# Flux - spatial average
plt.plot(x_mid, phi_ref, label="Sol")
plt.plot(x_mid, phi, label="iQMC")
plt.ylabel(r"$\phi(x)$")
plt.xlabel(r"$x$")
plt.grid()
plt.legend()
plt.show()
