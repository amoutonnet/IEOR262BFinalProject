from optimizers import DFoptimizers as dfo
from optimizers import GBoptimizers as gbo
from functions.functions import function as f
from functions.functions import gradient as g
from functions.functions import hessian as h
from functions.functions import optinfo
from functions.functions import getcons
from collections import defaultdict
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


IMPOSSIBLEPAIRS = {
    ("Newton Line Search", "Norm1SphereWithSphereCons"),
    ("Newton Basic", "CrossInTray"),
    ("Newton Basic", "Holder"),
    ("GD", "CrossInTray"),
    ("GD", "Holder"),
}


def init_optimizers(n, name, opts_list, **kwargs):
    fct = partial(f, name)
    gdt = partial(g, name)
    hes = partial(h, name)
    optinf = partial(optinfo, name)
    cons = getcons(name, n)
    opts = {}

    # Tolerance hyperparameters
    max_iter = kwargs.pop('max_iter', MAX_ITER)
    ftol = kwargs.pop('ftol', FTOL)
    gtol = kwargs.pop('gtol', GTOL)
    xtol = kwargs.pop('xtol', XTOL)
    epsilon = kwargs.pop('epsilon', EPSILON)

    if "MADS" in opts_list:
        params = kwargs.pop("MADS", {})
        opts["MADS"] = dfo.MADSOptimizer(
            dim=n,
            function=fct,
            constraints=cons,
            getoptinfo=optinf,
            max_iter=max_iter,
            ftol=ftol,
            xtol=xtol,
            mu_min=params.pop("mu_min", 1 / (4**14)),
            use_minibasis=params.pop("use_minibasis", False),
        )
    if "CMAES" in opts_list:
        params = kwargs.pop("CMAES", {})
        opts["CMAES"] = dfo.CMAESOptimizer(
            dim=n,
            function=fct,
            constraints=cons,
            getoptinfo=optinf,
            max_iter=max_iter,
            ftol=ftol,
            xtol=xtol,
            learning_rate=params.pop("learning_rate", 10),
            lambd=params.pop("lambd", None),
        )
    if "Newton Line Search" in opts_list:
        params = kwargs.pop("Newton Line Search", {})
        opts["Newton Line Search"] = gbo.NewtonLineSearchOptimizer(
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
            epsilon=params.pop("epsilon", epsilon),
        )
    if "Newton Log Barrier" in opts_list:
        params = kwargs.pop("Newton Log Barrier", {})
        opts["Newton Log Barrier"] = gbo.NewtonLogBarrierOptimizer(
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
            mu=params.pop("mu", 1.05),
            theta0=params.pop("theta0", 100),
        )
    if "Newton Basic" in opts_list:
        opts["Newton Basic"] = gbo.NewtonBasicOptimizer(
            dim=n,
            function=fct,
            gradient=gdt,
            hessian=hes,
            constraints=cons,
            getoptinfo=optinf,
            max_iter=max_iter,
            ftol=ftol,
            gtol=gtol,
            xtol=xtol
        )

    if "GD" in opts_list:
        params = kwargs.pop("GD", {})
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
            learning_rate=params.pop("learning_rate", 1e-2)
        )
    return opts, cons


def plot_optimization(name, opts_list, x0, hp):
    # Problem definition
    n = x0.shape[0]
    opts, _ = init_optimizers(n, name, opts_list, **hp[name])

    # Optimize and plot
    res = {}
    for opt in opts.keys():
        res[opt] = opts[opt].optimize(
            x0=x0,
            verbose=True
        )
        graph.plot_track(res[opt], opt, name)
    plt.show()


def draw_x0(name, cons, xmin=-50, xmax=50):
    if len(cons):
        if name == "Norm1SphereWithSphereCons":
            xmin, xmax = -np.sqrt(dim * 3), np.sqrt(dim * 3)
        x0 = np.random.uniform(low=xmin, high=xmax, size=dim).reshape(-1, 1)
        while not cons.test(x0):
            x0 = np.random.uniform(low=xmin, high=xmax, size=dim).reshape(-1, 1)
    else:
        x0 = np.random.uniform(low=xmin, high=xmax, size=dim).reshape(-1, 1)
    return x0


def plot_box(opts_list, hp, n_iter=10, dim=2, names=['Sphere'], cv_threshold=CONVERGENCE_THRESHOLD, latex=False):
    data = []
    for name in names:
        print("\n%s\n" % "Optimization for {:s} function starting".format(name).center(150, "-"))
        for _ in tqdm(range(n_iter), total=n_iter):
            opts, cons = init_optimizers(dim, name, opts_list=opts_list, **hp[name])
            x0 = draw_x0(name, cons)
            for opt in opts_list:
                if (opt, name) in IMPOSSIBLEPAIRS:
                    continue
                track = opts[opt].optimize(
                    x0=x0,
                    verbose=False
                )
                _, _, timeperiter, _, _, xoptdiffs, foptdiffs = map(np.array, zip(*track))

                data.append([opt, name, np.sum(timeperiter[1:]), len(track) - 1, xoptdiffs[-1], foptdiffs[-1]])
    data = pd.DataFrame(data, columns=['Optimizer', 'Function', 'Timestamp', 'NbIter', 'Finalxoptdiff', 'Finalfoptdiff'])
    graph.print_statistics(data, opts_list, names, n_iter, cv_threshold=CONVERGENCE_THRESHOLD, latex=latex)
    graph.plot_box(data, names, n_iter, cv_threshold=CONVERGENCE_THRESHOLD)
    plt.show()


if __name__ == "__main__":
    # Main parameters
    dim = 2
    constrained = False

    # Objective function
    names = []
    if constrained:
        names += ["SphereWithLinCons"]
        names += ["Norm1SphereWithSphereCons"]
        names += ["StyblinskiTangWithPosCons"]
    else:
        names += ["Sphere"]
        names += ["Rosenbrock"]
        names += ["Rastigrin"]
        names += ["StyblinskiTang"]
        if dim == 2:
            names += ["Levy13"]
            names += ["Easom"]
            names += ["CrossInTray"]
            names += ["Holder"]
            pass

    opts_list = []
    if constrained:
        opts_list += ["MADS"]
        opts_list += ["CMAES"]
        opts_list += ["Newton Line Search"]
        opts_list += ["Newton Log Barrier"]
    else:
        opts_list += ["MADS"]
        opts_list += ["CMAES"]
        opts_list += ['Newton Basic']
        opts_list += ["GD"]

    # hp template
    hyperparameters = defaultdict(lambda: {})

    hyperparameters["Rosenbrock"] = {
        'max_iter': 1000
    }

    hyperparameters["Rastigrin"] = {
        'CMAES': {
            'lambd': int(2 * np.log(dim) * 4 + int(3 * np.log(dim)))
        }
    }

    hyperparameters["StyblinskiTang"] = {
        'xtol': -1,
        'CMAES': {
            'lambd': int(2 * np.log(dim) * 4 + int(3 * np.log(dim)))
        }
    }

    hyperparameters["Easom"] = {
        'max_iter': 2000,
        'ftol': -1,
        'CMAES': {
            'lambd': int(2 * np.log(dim) * 4 + int(3 * np.log(dim)))
        }
    }

    hyperparameters["Holder"] = {
        'ftol': -1,
        'CMAES': {
            'lambd': int(2 * np.log(dim) * 4 + int(3 * np.log(dim)))
        }
    }

    hyperparameters["SphereWithLinCons"] = {
        "MADS": {
            "mu_min": 1 / (4**12),
        },
    }

    hyperparameters["Norm1SphereWithSphereCons"] = {
        "MADS": {
            "mu_min": 1 / (4**12),
        },
    }

    hyperparameters["StyblinskiTangWithPosCons"] = {
        'xtol': -1,
        'CMAES': {
            'lambd': int(2 * np.log(dim) * 4 + int(3 * np.log(dim)))
        },
        "Newton Line Search": {
            "epsilon": 1e-8
        }
    }

    plot_box(opts_list=opts_list, hp=hyperparameters, n_iter=100, dim=dim, names=names, latex=False)

    # x0 = np.array([-50, -50]).reshape(-1, 1)
    # plot_optimization(names[0], opts_list, x0, hyperparameters)
