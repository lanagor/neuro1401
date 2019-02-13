import numpy as np
import itertools
import matplotlib.pyplot as plt


def exponential(R, gamma, x):
    return np.power(gamma, x) * R


def hyperbolic(R, ITI, x):
    return R / (ITI + x)


def plot_exponentials(D1, Rs, gammas, maxY=None):
    maxY = max(D1) if maxY is None else maxY
    for R_ratio, gamma in itertools.product(Rs, gammas):
        label = f'E: g={gamma}, r={R_ratio:.2f}'
        D2 = np.log(R_ratio) / np.log(gamma) + D1
        D1 = D1[np.nonzero(D2 < maxY)]
        D2 = D2[np.nonzero(D2 < maxY)]
        plt.plot(D1, D2, label=label)


def plot_hyperbolics(D1, Rs, ITIs, maxY=None):
    maxY = max(D1) if maxY is None else maxY
    for R_ratio, ITI in itertools.product(Rs, ITIs):
        label = f'H: ITI={ITI}, r={R_ratio:.2f}'
        D2 = (ITI + D1) / R_ratio
        D1 = D1[np.nonzero(D2 < maxY)]
        D2 = D2[np.nonzero(D2 < maxY)]
        plt.plot(D1, D2, label=label)

D = np.linspace(20, 100, 1000)
plot_exponentials(D, [64, 1 / 64], [2, 4])
plot_hyperbolics(D, [2, 0.5], [1, 5, 9])
plt.legend(loc='best')
plt.show()
