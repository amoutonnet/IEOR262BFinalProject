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

MAX_ITER = 1000
FTOL = 1e-14
XTOL = 1e-14
GTOL = 0

# Optimization algorithm stop parameters
params = {}

params["SphereWithLinCons"] = {
    "MADS": {
        "max_iter": MAX_ITER,
        "ftol": 0,
        "xtol": 0,
        "mu_min": 1 / (4**14),
    },
    "CMAES": {
        "max_iter": MAX_ITER,
        "ftol": FTOL,
        "xtol": XTOL,
        "learning_rate": 10,
        "lambd": None,
    },
    "Newton Line Search": {
        "max_iter": MAX_ITER,
        "ftol": FTOL,
        "gtol": GTOL,
        "xtol": XTOL,
        "epsilon": 1e-8
    },
    "Newton Log Barrier": {
        "max_iter": MAX_ITER,
        "ftol": FTOL,
        "gtol": GTOL,
        "xtol": XTOL,
        "mu": 1.01,
        "theta0": 10,
        "epsilon": 1e-8
    }
}

params["Norm1SphereWithSphereCons"] = {
    "MADS": {
        "max_iter": MAX_ITER,
        "ftol": 0,
        "xtol": 0,
        "mu_min": 1 / (4**12),
    },
    "CMAES": {
        "max_iter": MAX_ITER,
        "ftol": FTOL,
        "xtol": XTOL,
        "learning_rate": 10,
        "lambd": None,
    },
    "Newton Line Search": {
        "max_iter": MAX_ITER,
        "ftol": FTOL,
        "gtol": GTOL,
        "xtol": XTOL,
        "epsilon": 1e-8
    },
    "Newton Log Barrier": {
        "max_iter": MAX_ITER,
        "ftol": FTOL,
        "gtol": GTOL,
        "xtol": XTOL,
        "mu": 1.01,
        "theta0": 1000,
        "epsilon": 1e-8
    }

}

params["StyblinskiTangWithPosCons"] = {
    "MADS": {
        "max_iter": MAX_ITER,
        "ftol": 0,
        "xtol": 0,
        "mu_min": 1 / (4**14),
    },
    "CMAES": {
        "max_iter": MAX_ITER,
        "ftol": FTOL,
        "xtol": XTOL,
        "learning_rate": 10,
        "lambd": None,
    },
    "Newton Line Search": {
        "max_iter": MAX_ITER,
        "ftol": FTOL,
        "gtol": GTOL,
        "xtol": XTOL,
        "epsilon": 1e-8
    },
    "Newton Log Barrier": {
        "max_iter": MAX_ITER,
        "ftol": FTOL,
        "gtol": GTOL,
        "xtol": XTOL,
        "mu": 1.01,
        "theta0": 10,
        "epsilon": 1e-8
    }

}


def init_optimizers(n, name, opts_list):
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
            **params[name]["MADS"]
        )
    if "CMAES" in opts_list:
        opts["CMAES"] = dfo.CMAESOptimizer(
            dim=n,
            function=fct,
            constraints=cons,
            getoptinfo=optinf,
            **params[name]["CMAES"]
        )
    if "Newton Line Search" in opts_list:
        opts["Newton Line Search"] = gbo.NewtonLineSearchOptimizer(
            dim=n,
            function=fct,
            gradient=gdt,
            hessian=hes,
            constraints=cons,
            getoptinfo=optinf,
            **params[name]["Newton Line Search"]
        )
    if "Newton Log Barrier" in opts_list:
        opts["Newton Log Barrier"] = gbo.NewtonLogBarrierOptimizer(
            dim=n,
            function=fct,
            gradient=gdt,
            hessian=hes,
            constraints=cons,
            getoptinfo=optinf,
            **params[name]["Newton Log Barrier"]
        )
    return opts, cons


def plot_optimization(name, opts_list, x0):
    # Problem definition
    n = x0.shape[0]
    opts, _ = init_optimizers(n, name, opts_list)

    # Optimize and plot
    res = {}
    for opt in opts.keys():
        res[opt] = opts[opt].optimize(
            x0=x0,
            verbose=True
        )
        graph.plot_track(res[opt], opt, name)
    plt.show()


def plot_box(opts_list, nb_iter=10, dim=2, names=['Sphere']):
    data = []
    for name in names:
        print(name)
        for _ in tqdm(range(nb_iter), total=nb_iter):
            opts, cons = init_optimizers(dim, name, opts_list=opts_list)
            if name == "Norm1SphereWithSphereCons":
                x_0 = np.random.uniform(low=-np.sqrt(dim * 3), high=np.sqrt(dim * 3), size=dim).reshape(-1, 1)
                while not cons.test(x_0):
                    x_0 = np.random.uniform(low=-np.sqrt(dim * 3), high=np.sqrt(dim * 3), size=dim).reshape(-1, 1)
            else:
                x_0 = np.random.uniform(low=-50, high=50, size=dim).reshape(-1, 1)
                while not cons.test(x_0):
                    x_0 = np.random.uniform(low=-50, high=50, size=dim).reshape(-1, 1)
            for opt in opts_list:
                if opt == "Newton Line Search" and name == "Norm1SphereWithSphereCons":
                    continue
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
    # names += ["Sphere"]
    # names += ["Rosenbrock"]
    # names += ["Rastigrin"]
    # names += ["Levy13"]
    # names += ["Easom"]
    # names += ["StyblinskiTang"]
    # names += ["CrossInTray"]
    names += ["SphereWithLinCons"]
    names += ["Norm1SphereWithSphereCons"]
    names += ["StyblinskiTangWithPosCons"]

    opts_list = []
    opts_list += ["MADS"]
    opts_list += ["CMAES"]
    opts_list += ["Newton Line Search"]
    opts_list += ["Newton Log Barrier"]

    x0 = np.array([
        1, 1, 0, 0, 0
    ]).reshape(-1, 1)

    # plot_optimization(names[0], opts_list, x0)
    plot_box(opts_list=opts_list, nb_iter=100, dim=5, names=names)
