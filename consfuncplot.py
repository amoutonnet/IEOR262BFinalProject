from functions.functions import function as f
from functions.functions import getcons
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import itertools
import numpy as np


NPPOINTS = 100

PARAMS = {
    "SphereWithLinCons": {
        "semiwidth": 5,
        "min": r"$x^*=(-\frac{1}{n},...,-\frac{1}{n})$",
        "fopt": r"$f(x^*)=\frac{1}{n}$"
    },
    "Norm1SphereWithSphereCons": {
        "semiwidth": 5,
        "min": r"$x^*=(-\sqrt{3},...,-\sqrt{3})$",
        "fopt": r"$f(x^*)=-n\sqrt{3}$"
    },
    "StyblinskiTangWithPosCons": {
        "semiwidth": 5,
        "min": r"$x^*=(2.7468, -2.90353,...,-2.90353)$",
        "fopt": r"$f(x^*)\approx-25.02946-39.16617(n-1)$"
    },
}


fig = plt.figure()
ncols = 3
nrows = 1

for idx, name in enumerate(PARAMS.keys()):
    semiwidth = PARAMS[name]["semiwidth"]
    ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
    X = np.linspace(-semiwidth, semiwidth, NPPOINTS)
    Y = np.linspace(-semiwidth, semiwidth, NPPOINTS)
    X, Y = np.meshgrid(X, Y)
    cons = getcons(name, 2)
    XY = np.dstack((X, Y))
    Z = np.zeros(X.shape)
    vmin = float('inf')
    vmax = -float('inf')
    for i in range(len(XY)):
        for j in range(len(XY[i])):
            if cons.test(XY[i, j].reshape(-1, 1)):
                Z[i, j] = f(name, XY[i, j].reshape(-1, 1))
            else:
                Z[i, j] = np.nan
            if Z[i, j] > vmax:
                vmax = Z[i, j]
            if Z[i, j] < vmin:
                vmin = Z[i, j]

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.Spectral,
                           linewidth=0, antialiased=False, vmin=vmin, vmax=vmax)
    ax.set_title(name + "\n" + PARAMS[name]["min"] + "\n" + PARAMS[name]["fopt"], fontsize=12)
plt.show()
