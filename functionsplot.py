from functions.functions import function as f
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import itertools
import numpy as np


NPPOINTS = 100

PARAMS = {
    "Sphere": {
        "semiwidth": 2.5,
        "min": r"$x^*=(0,...,0)$",
        "min2D": r"$x^*=(0,0)$",
        "fopt": r"$f(x^*)=0$"
    },
    "Rosenbrock": {
        "semiwidth": 2.5,
        "min": r"$x^*=(1,...,1)$",
        "min2D": r"$x^*=(1,1)$",
        "fopt": r"$f(x^*)=0$"
    },
    "StyblinskiTang": {
        "semiwidth": 5,
        "min": r"$x^*=(-2.903534,...,-2.903534)$",
        "min2D": r"$x^*=(-2.903534,-2.903534)$",
        "fopt": r"$f(x^*) \approx -39.16617n$"
    },
    # "Easom": {
    #     "semiwidth": 5,
    #     "min2D": r"$x^*=(\pi,\pi)$",
    #     "fopt": r"$f(x^*)=-1$"
    # },
    "Rastigrin": {
        "semiwidth": 5,
        "min": r"$x^*=(0,...,0)$",
        "min2D": r"$x^*=(0,0)$",
        "fopt": r"$f(x^*)=0$"
    },
    # "Levy13": {
    #     "semiwidth": 5,
    #     "min2D": r"$x^*=(1,1)$",
    #     "fopt": r"$f(x^*)=0$"
    # },
    # "Holder": {
    #     "semiwidth": 5,
    #     "min2D": r"$x^*=(\pm 8.05502,\pm 9.66459)$",
    #     "fopt": r"$f(x^*)=-19.2085$"
    # },
    # "CrossInTray": {
    #     "semiwidth": 5,
    #     "min2D": r"$x^*=(\pm 1.34941,\pm 1.34941)$",
    #     "fopt": r"$f(x^*)=-2.06261$"
    # },
}


fig = plt.figure()
# ncols = len(PARAMS.keys()) // 2 if len(PARAMS.keys()) % 2 == 0 else 1 + len(PARAMS.keys()) // 2
# nrows = len(PARAMS.keys()) // ncols if len(PARAMS.keys()) % ncols == 0 else 1 + len(PARAMS.keys()) // ncols
ncols = 4
nrows = 1

for idx, name in enumerate(PARAMS.keys()):
    semiwidth = PARAMS[name]["semiwidth"]
    ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
    X = np.linspace(-semiwidth, semiwidth, NPPOINTS)
    Y = np.linspace(-semiwidth, semiwidth, NPPOINTS)
    X, Y = np.meshgrid(X, Y)
    XY = np.dstack((X, Y))
    Z = np.zeros(X.shape)
    for i in range(len(XY)):
        for j in range(len(XY[i])):
            Z[i, j] = f(name, XY[i, j].reshape(-1, 1))

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.Spectral,
                           linewidth=0, antialiased=False)
    ax.set_title(name + "\n" + PARAMS[name]["min"] + "\n" + PARAMS[name]["fopt"], fontsize=7)
plt.show()
