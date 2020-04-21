from optimizers import optimizers
from optimizers import constraints
import numpy as np


def main():
    # A = np.array([
    #     [1, 1, 1, 0],
    #     [1, -1, 0, 1]
    # ])

    # b = np.array([
    #     [100],
    #     [50]
    # ])

    # c = np.array([
    #     -9, -10, 0, 0
    # ]).reshape(-1, 1)

    x_0 = np.array([
        0, 0, 0, 0, 0
    ]).reshape(-1, 1)
    n = x_0.shape[0]

    def func(x):
        fx = np.sum(x)
        return np.squeeze(fx)

    cons = [
        constraints.IneqConstraint(lambda x: np.dot(x.T, x) - 3 * x.shape[0])
    ]

    optimizer = 'MADS'  # 'MADS' or 'CMAES'
    if optimizer == 'MADS':
        opt = optimizers.MADSOptimizer(
            dim=n,
            function=func,
            constraints=cons,
        )
    elif optimizer == 'CMAES':
        pass
    else:
        print('This optimizer is not implemented.')

    opt.optimize(
        x0=x_0,
        max_iter=2000,
        ftol=0,
        xtol=0,
        plot=True,
        verbose=True
    )


if __name__ == "__main__":
    main()
