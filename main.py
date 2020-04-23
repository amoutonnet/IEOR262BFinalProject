from optimizers import optimizers
from optimizers import constraints
import numpy as np


def main():
    A = np.array([
        -1, -1, -1, 0
    ])

    b = np.array([
        -50
    ])

    # c = np.array([
    #     -9, -10, 0, 0
    # ]).reshape(-1, 1)

    x_0 = np.array([
        100, 0, 0, 0
    ]).reshape(-1, 1)
    n = x_0.shape[0]

    def fRosenbrock(x):
        assert x.shape[0] > 1, 'dimension must be greater one'
        fx = 100 * np.sum(np.square(x[1:] - np.square(x[:-1])), axis=0) + np.sum(np.square(1 - x[:-1]), axis=0)
        return fx

    def func(x):
        fx = np.sum(x)
        return np.squeeze(fx)

    def fsphere(x):
        fx = np.sqrt(np.sum(np.square(x), axis = 0))
        return fx

    cons = [
        # constraints.IneqConstraint(lambda x: np.dot(x.T, x) - 3 * x.shape[0])
        constraints.AffIneqConstraint(A, b)
    ]

    optimizer = 'CMAES'  # 'MADS' or 'CMAES'
    if optimizer == 'MADS':
        opt = optimizers.MADSOptimizer(
            dim=n,
            function=func,
            constraints=cons,
        )
    elif optimizer == 'CMAES':
        opt = optimizers.CMAESOptimizer(
            dim=n,
            function=fRosenbrock,
            constraints=cons,
            learning_rate=0.5,
            lambd=None,
            MSR=True,
            constrained_problem=True, # careful constrained are inverted (dont know why)
            stop_eigenvalue=1e8
        )
    else:
        print('This optimizer is not implemented.')

    opt.optimize(
        x0=x_0,
        max_iter=1000,
        ftol=0,
        xtol=0,
        plot=True,
        verbose=True
    )


if __name__ == "__main__":
    main()
