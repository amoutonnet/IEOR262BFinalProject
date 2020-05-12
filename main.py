from optimizers import DFoptimizers as dfo
from optimizers import GBoptimizers as gbo
from optimizers import constraints
from functions.functions import function as f
from functions.functions import gradient as g
from functions.functions import hessian as h
from functools import partial
from utils import graph
import matplotlib.pyplot as plt
import numpy as np
from time import time
import seaborn as sns
import pandas as pd

np.random.seed(100)


# Optimization algorithm stop parameters
MAX_ITER = 1000
FTOL = 0
GTOL = 1e-8
XTOL = 1e-8

def init_optimizers(n, fct, gdt, hes, opts_list=[], cons=[]):
    opts={}
    if "MADS" in opts_list:
        opts["MADS"] = dfo.MADSOptimizer(
            dim=n,
            function=fct,
            constraints=cons,
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
        max_iter=MAX_ITER,
        ftol=FTOL,
        xtol=XTOL,
        learning_rate=10,
        lambd=10,
        MSR=False,
    )
    if "Newton Line Search" in opts_list:
        opts["Newton Line Search"] = gbo.NewtonLineSearchOptimizer(
            dim=n,
            function=fct,
            gradient=gdt,
            hessian=hes,
            constraints=cons,
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
            max_iter=MAX_ITER,
            ftol=FTOL,
            gtol=GTOL,
            xtol=XTOL,
            mu=1.01,
            theta0=10000,
            epsilon=1e-8
        )
    return opts

def plot_optimization():
    # Problem definition
    x_0 = np.array([
        -10, -10
    ]).reshape(-1, 1)
    n = x_0.shape[0]

    # Constraints definitions
    cons = constraints.Constraints()
    A = np.array([
        [1, 1]
    ])

    b = np.array([
        -1
    ]).reshape(-1, 1)
    # cons.addineqcons(A, b)

    # Objective function
    # name = "Sphere"
    # name = "Rosenbrock"
    # name = "Rastigrin"
    name = "Easom"
    # name = "StyblinskiTang"
    # name = "CrossInTray"

    fct = partial(f, name)
    gdt = partial(g, name)
    hes = partial(h, name)

    opts_list=[]
    opts_list += ["MADS"]
    opts_list += ["CMAES"]
    # opts_list += ["Newton Line Search"]
    # opts_list += ["Newton Log Barrier"]
    opts = init_optimizers(n, fct, gdt, hes, opts_list=opts_list, cons=cons)

    # Optimize and plot
    res = {}
    for opt in opts.keys():
        res[opt] = opts[opt].optimize(
            x0=x_0,
            verbose=True
        )
        graph.plot_track(res[opt], opt, name)
    plt.show()

def plot_box(opts_list=[], nb_iter = 10, dim=2, name='Sphere'):
    # Initialize Objective function
    fct = partial(f, name)
    gdt = partial(g, name)
    hes = partial(h, name)

    # Constraints definition
    cons = constraints.Constraints()
    A = np.array([
        [1, 1]
    ])
    b = np.array([
        -1
    ]).reshape(-1, 1)
    cons.addineqcons(A, b)

    data =[]
    res = {}
    for opt in opts_list:
        res[opt] = []
    for _ in range(nb_iter):
        x_0 = np.random.uniform(low=-50, high=50, size=dim).reshape(-1, 1)
        opts = init_optimizers(dim, fct, gdt, hes, opts_list=opts_list, cons=cons)
        for opt in opts_list:
            start_time = time()
            res[opt].append(opts[opt].optimize(
                x0=x_0,
                verbose=False
            ))
            data.append([opt, time() - start_time, len(res[opt][-1])])
    data = pd.DataFrame(data, columns=['Optimizer', 'Timestamp', 'NbIter'])
    graph.plot_box(data, name)
    plt.show()


if __name__ == "__main__":
    plot_optimization()
    # plot_box(opts_list=["MADS", "CMAES", "Newton Line Search", "Newton Log Barrier"], nb_iter=100, dim=2, name='Sphere')
