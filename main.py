from optimizers import DFoptimizers as dfo
from optimizers import GBoptimizers as gbo
from functions.functions import function as f
from functions.functions import gradient as g
from functions.functions import hessian as h
from functions.functions import optinfo
from functions.functions import getcons
from functions import constraints
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
MAX_ITER = 1000
FTOL = 1e-14
GTOL = 1e-14
XTOL = 1e-14
EPSILON = 1e-8
CONVERGENCE_THRESHOLD = 1e-3  # for plot_box


def init_optimizers(n, name, opts_list, **kwargs):
    fct = partial(f, name)
    gdt = partial(g, name)
    hes = partial(h, name)
    optinf = partial(optinfo, name)
    cons = getcons(name, n)
    opts = {}

    # Tolerance hyperparameters
    max_iter = kwargs.pop('max_iter') if 'max_iter' in kwargs.keys() else MAX_ITER
    ftol = kwargs.pop('ftol') if 'ftol' in kwargs.keys() else FTOL
    gtol = kwargs.pop('gtol') if 'gtol' in kwargs.keys() else GTOL
    xtol = kwargs.pop('xtol') if 'xtol' in kwargs.keys() else XTOL
    epsilon = kwargs.pop('epsilon') if 'epsilon' in kwargs.keys() else EPSILON

    if "MADS" in opts_list:
        params = kwargs.pop('MADS') if 'MADS' in kwargs.keys() else {}
        opts["MADS"] = dfo.MADSOptimizer(
            dim=n,
            function=fct,
            constraints=cons,
            getoptinfo=optinf,
            **params[name]["MADS"]
        )
    if "CMAES" in opts_list:
        params = kwargs.pop('CMAES') if 'CMAES' in kwargs.keys() else {}
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
    if "GD" in opts_list:
        params = kwargs.pop('CMAES') if 'CMAES' in kwargs.keys() else {}
        opts["GD"] = gbo.GradientDescentOptimizer(
            dim=n,
            function=fct,
            gradient=gdt,
            hessian=hes,
            constraints=cons,
            getoptinfo=optinf,
            max_iter=max_iter,
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
            epsilon=epsilon,
            learning_rate=params.pop('learning_rate') if 'learning_rate' in params.keys() else 1e-2
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


def plot_box(opts_list, nb_iter=10, dim=2, names=['Sphere'], hp={}, cv_threshold=CONVERGENCE_THRESHOLD):
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

                data.append([opt, name, np.sum(timeperiter[1:]), len(track), xoptdiffs[-1], foptdiffs[-1]])
    data = pd.DataFrame(data, columns=['Optimizer', 'Function', 'Timestamp', 'NbIter', 'Finalxoptdiff', 'Finalfoptdiff'])
    graph.plot_box(data, names, nb_iter, cv_threshold=CONVERGENCE_THRESHOLD)
    plt.show()


if __name__ == "__main__":
    # Objective function
    names = []
    names += ["Sphere"]
    names += ["Rosenbrock"]
    names += ["Rastigrin"]
    names += ["Levy13"]
    names += ["Easom"]
    names += ["StyblinskiTang"]
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
    opts_list += ['Newton Basic']
    # opts_list += ["Newton Line Search"]
    # opts_list += ["Newton Log Barrier"]
    opts_list += ["GD"]

    dim = 2
    # hp template
    hyperparameters = {
        "Sphere": {},
        "Rosenbrock": {
            'max_iter': 500
        },
        "Rastigrin": {
            'CMAES': {
                'lambd': int((4 + int(3 * np.log(dim))) * 2 * np.log(dim + 1))
            }
        },
        "StyblinskiTang": {
            'xtol': -1
        },
        "Easom": {
            'ftol': -1,
            'CMAES': {
                'lambd': int((4 + int(3 * np.log(dim))) * 2 * np.log(dim + 1))
            }
        },
        "CrossInTray": {},
        "Holder": {}
    }

    # hp = hyperparameters[names[0]]
    # plot_optimization(names[0], opts_list, hp)
    plot_box(opts_list=opts_list, nb_iter=100, dim=dim, names=names, hp=hyperparameters)
