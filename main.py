from optimizers import DFoptimizers as dfo
from optimizers import GBoptimizers as gbo
from functions.functions import function as f
from functions.functions import gradient as g
from functions.functions import hessian as h
from functions.functions import optinfo
from functions.functions import getcons
from functools import partial
from utils import graph
import matplotlib.pyplot as plt
import numpy as np
from time import time
import seaborn as sns
import pandas as pd
from tqdm import tqdm

np.random.seed(100)


# Optimization algorithm stop parameters
MAX_ITER = 5000
FTOL = 1e-14
GTOL = 1e-14
XTOL = 1e-8


def init_optimizers(n, name, opts_list, **kwargs):
    fct = partial(f, name)
    gdt = partial(g, name)
    hes = partial(h, name)
    optinf = partial(optinfo, name)
    cons = getcons(name, n)
    opts = {}
    if "MADS" in opts_list:
        opts["MADS"] = dfo.MADSOptimizer(
            dim=n,
            function=fct,
            constraints=cons,
            getoptinfo=optinf,
            max_iter=MAX_ITER,
            ftol=FTOL,
            xtol=XTOL,
            mu_min=1 / (4**12),
        )
    if "CMAES" in opts_list:
        opts["CMAES"] = dfo.CMAESOptimizer(
            dim=n,
            function=fct,
            constraints=cons,
            getoptinfo=optinf,
            max_iter=MAX_ITER,
            ftol=FTOL,
            xtol=XTOL,
            learning_rate=10,
            lambd=None,
        )
    if "Newton Line Search" in opts_list:
        opts["Newton Line Search"] = gbo.NewtonLineSearchOptimizer(
            dim=n,
            function=fct,
            gradient=gdt,
            hessian=hes,
            constraints=cons,
            getoptinfo=optinf,
            max_iter=MAX_ITER,
            ftol=FTOL,
            gtol=GTOL,
            xtol=XTOL,
            epsilon=1e-8
        )
    if "Newton Log Barrier" in opts_list:
        opts["Newton Log Barrier"] = gbo.NewtonLogBarrierOptimizer(
            dim=n,
            function=fct,
            gradient=gdt,
            hessian=hes,
            constraints=cons,
            getoptinfo=optinf,
            max_iter=MAX_ITER,
            ftol=FTOL,
            gtol=GTOL,
            xtol=XTOL,
            mu=1.01,
            theta0=10000,
            epsilon=1e-8
        )
    if "GD" in opts_list:
        opts["GD"] = gbo.GradientDescentOptimizer(
            dim=n,
            function=fct,
            gradient=gdt,
            hessian=hes,
            constraints=cons,
            getoptinfo=optinf,
            max_iter=MAX_ITER,
            ftol=FTOL,
            gtol=GTOL,
            xtol=XTOL,
            epsilon=1e-8,
            learning_rate = 1e-2
        )
    return opts, cons


def plot_optimization(name, opts_list):
    # Problem definition
    x_0 = np.array([
        10, 10
    ]).reshape(-1, 1)
    n = x_0.shape[0]
    opts, _ = init_optimizers(n, name, opts_list)

    # Optimize and plot
    res = {}
    for opt in opts.keys():
        res[opt] = opts[opt].optimize(
            x0=x_0,
            verbose=True
        )
        graph.plot_track(res[opt], opt, name)
    plt.show()


def plot_box(opts_list, nb_iter=10, dim=2, names=['Sphere']):
    data = []
    for name in names:
        print(name)
        opts, cons = init_optimizers(dim, name, opts_list=opts_list)
        for _ in tqdm(range(nb_iter), total=nb_iter):
            x_0 = np.random.uniform(low=-50, high=50, size=dim).reshape(-1, 1)
            while not cons.test(x_0):
                x_0 = np.random.uniform(low=-50, high=50, size=dim).reshape(-1, 1)
            for opt in opts_list:
                track = opts[opt].optimize(
                    x0=x_0,
                    verbose=False
                )
                _, _, timeperiter, _, _, xoptdiffs, foptdiffs = map(np.array, zip(*track))

                data.append([opt, np.sum(timeperiter[1:]), len(track), xoptdiffs[-1], foptdiffs[-1]])
    data = pd.DataFrame(data, columns=['Optimizer', 'Timestamp', 'NbIter', 'Finalxoptdiff', 'Finalfoptdiff'])
    graph.plot_box(data, names, nb_iter)
    plt.show()


if __name__ == "__main__":
    # Objective function
    names = []
    names += ["Sphere"]
    # names += ["Rosenbrock"]
    # names += ["Rastigrin"]
    # names += ["Levy13"]
    # names += ["Easom"]
    # names += ["StyblinskiTang"]
    # names += ["CrossInTray"]
    # names += ["Norm1SphereWithSphereCons"]

    opts_list = []
    opts_list += ["MADS"]
    opts_list += ["CMAES"]
    # opts_list += ["Newton Line Search"]
    # opts_list += ["Newton Log Barrier"]
    opts_list += ["GD"]

    plot_optimization(names[0], opts_list)
    # plot_box(opts_list=opts_list, nb_iter=100, dim=2, names=names)
