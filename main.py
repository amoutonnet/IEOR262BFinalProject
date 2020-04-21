from optimizers import optimizers
from optimizers import constraints
import numpy as np


def main():
    A = np.array([
        [1, 1, 1, 0],
        [1, -1, 0, 1]
    ])

    b = np.array([
        [100],
        [50]
    ])

    _, n = A.shape

    c = np.array([
        -9, -10, 0, 0
    ]).reshape(-1, 1)

    x_0 = np.array([
        50, 50, 0, 50
    ]).reshape(-1, 1)

    def func(x):
        return np.dot((x - 1).T, (x - 1))[0][0]

    cons = [
        # constraints.AffIneqConstraint(A, b),
        constraints.PosConstraint()
    ]

    optimizer = 'MADS'  # 'MADS' or 'CMAES'
    if optimizer == 'MADS':
        opt = optimizers.MADSOptimizer(
            dim=n,
            function=func,
            constraints=cons,
            alpha=1e-2,
            beta_1=0.8,
            beta_2=0.9,
            gamma=1.1,
            forcing_function=lambda x: x * x,
        )
    elif optimizer == 'CMAES':
        pass
    else:
        print('This optimizer is not implemented.')

    opt.optimize(
        x0=x_0,
        max_iter=1000,
        ftol=1e-12,
        xtol=0,
        plot=True,
        verbose=True
    )


if __name__ == "__main__":
    main()
