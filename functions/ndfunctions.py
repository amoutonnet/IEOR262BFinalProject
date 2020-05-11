import numpy as np

PI = np.pi

LISTF = ["Rosenbrock", "Sphere", "Rastigrin", "StyblinskiTang"]
LISTG = ["Rosenbrock", "Sphere", "Rastigrin", "StyblinskiTang"]
LISTH = ["Sphere", "StyblinskiTang"]


def fRosenbrock(x):
    return 100 * np.sum(np.square(x[1:] - np.square(x[:-1])), axis=0) + np.sum(np.square(1 - x[:-1]), axis=0)


def gRosenbrock(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    g = np.zeros(x.shape)
    g[1:-1] = 200 * (xm - xm_m1**2) - 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm)
    g[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    g[-1] = 200 * (x[-1] - x[-2]**2)
    return g.reshape(-1, 1)


def fSphere(x):
    return np.sum(np.square(x))


def gSphere(x):
    return 2 * x.reshape(-1, 1)


def hSphere(x):
    return 2 * np.eye(x.shape[0])


def fStyblinskiTang(x):
    return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)


def gStyblinskiTang(x):
    return 2 * x**3 - 16 * x + 2.5


def hStyblinskiTang(x):
    return 6 * np.diag(x.ravel()**2) - 16


def fRastigrin(x):
    return 10 * x.shape[0] + np.sum(np.square(x) - 10 * np.cos(2 * PI * x))


def gRastigrin(x):
    return 2 * x.reshape(-1, 1) + 10 * 2 * np.pi * np.sin(2 * PI * x.reshape(-1, 1))
