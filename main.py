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

np.random.seed(100)


def main():
    A = np.array([
        [1, 1]
    ])

    b = np.array([
        50
    ]).reshape(-1, 1)

    x_0 = np.array([
        -50, -50
    ]).reshape(-1, 1)
    n = x_0.shape[0]

    cons = constraints.Constraints()
    cons.addineqcons(A, b)

    name = "StyblinskiTang"

    fct = partial(f, name)
    gdt = partial(g, name)
    hes = partial(h, name)
    max_iter = 1000
    ftol = 0
    gtol = 0
    xtol = 0

    opts = {}
    res = {}

    # opts["MADS"] = dfo.MADSOptimizer(
    #     dim=n,
    #     function=fct,
    #     constraints=cons,
    #     max_iter=max_iter,
    #     ftol=ftol,
    #     xtol=xtol,
    #     mu_min=1 / (4**12),
    # )

    opts["CMAES"] = dfo.CMAESOptimizer(
        dim=n,
        function=fct,
        constraints=cons,
        max_iter=max_iter,
        ftol=ftol,
        xtol=xtol,
        learning_rate=1,
        lambd=None,
        MSR=False,
        constrained_problem=True  # careful constrained are inverted (dont know why)
    )

    opts["Newton Line Search"] = gbo.NewtonLineSearchOptimizer(
        dim=n,
        function=fct,
        gradient=gdt,
        hessian=hes,
        constraints=cons,
        max_iter=max_iter,
        ftol=ftol,
        gtol=gtol,
        xtol=xtol,
        epsilon=1e-8
    )

    # opts["Newton Log Barrier"] = gbo.NewtonLogBarrierOptimizer(
    #     dim=n,
    #     function=fct,
    #     gradient=gdt,
    #     hessian=hes,
    #     constraints=cons,
    #     max_iter=max_iter,
    #     ftol=ftol,
    #     gtol=gtol,
    #     xtol=xtol,
    #     mu=1.01,
    #     theta0=10000,
    #     epsilon=1e-8
    # )

    for opt in opts.keys():
        res[opt] = opts[opt].optimize(
            x0=x_0,
            verbose=True
        )
        graph.plot_track(res[opt], opt)

    plt.show()


if __name__ == "__main__":
    main()
