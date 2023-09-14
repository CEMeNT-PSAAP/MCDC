import numpy as np
from scipy.integrate import quad
from scipy.special import exp1

# Parameters
SigmaT = 1.0
v = 1.0
T = 5.0

# Point-wise solution


def phi_(x, t):
    if x > v * t:
        return 0.0
    else:
        return (
            1.0
            / T
            * (
                SigmaT * x * (exp1(SigmaT * v * t) - exp1(SigmaT * x))
                + np.e ** (-SigmaT * x)
                - x / (v * t) * np.e ** (-SigmaT * v * t)
            )
        )


def phi_X(t, x1, x2):
    return quad(phi_, x1, x2, args=(t))[0] / (x2 - x1)


def reference(x, t):
    dx = x[1] - x[0]
    x_mid = 0.5 * (x[:-1] + x[1:])
    dt = t[1:] - t[:-1]
    K = len(dt)
    J = len(x_mid)

    phi = np.zeros([K, J])
    for k in range(K):
        for j in range(J):
            x1 = x[j]
            x2 = x[j + 1]
            t1 = t[k]
            t2 = t[k + 1]
            phi[k, j] = quad(phi_X, t1, t2, args=(x1, x2))[0] / (t2 - t1)
            # phi[k, j] = quad(phi_, x1, x2, args=(t[k + 1]))[0] / (x2 - x1)

    return phi
