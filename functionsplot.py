from functions.functions import function as f
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import itertools
import numpy as np


NPPOINTS = 100

PARAMS = {
    "Rosenbrock": {
        "semiwidth": 2.5
    },
    "Sphere": {
        "semiwidth": 2.5
    },
    "Rastigrin": {
        "semiwidth": 5
    },
    "Easom": {
        "semiwidth": 5
    },
    "Holder": {
        "semiwidth": 5
    },
    "CrossInTray": {
        "semiwidth": 5
    },
    "StyblinskiTang": {
        "semiwidth": 5
    }
}


fig = plt.figure()
ncols = len(PARAMS.keys()) // 2 if len(PARAMS.keys()) % 2 == 0 else 1 + len(PARAMS.keys()) // 2
nrows = len(PARAMS.keys()) // ncols if len(PARAMS.keys()) % ncols == 0 else 1 + len(PARAMS.keys()) // ncols


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
    ax.set_title(name)
plt.show()
