import numpy as np
from . import constraints
from . import ndfunctions as ndf


PI = np.pi

LIST = ["SphereWithLinCons", "Norm1SphereWithSphereCons", "StyblinskiTangWithPosCons"]


def getinfos(name, n):
    assert name in LIST, "This constrained function is not implemented"
    infos = {
        "SphereWithLinCons": {
            "cons": constraints.Constraints(),
            "xopt": np.array([[-1 / n]] * n),
            "fopt": 1 / n,
        },
        "Norm1SphereWithSphereCons": {
            "cons": constraints.Constraints(),
            "xopt": np.array([[-np.sqrt(3)]] * n),
            "fopt": -np.sqrt(3) * n,
        },
        "StyblinskiTangWithPosCons": {
            "cons": constraints.Constraints(),
            "xopt": np.eye(n, 1, 0) * 2.7468 - (np.ones((n, 1)) - np.eye(n, 1, 0)) * 2.90353,
            "fopt": -25.02946 - 39.16617 * (n - 1),
        }
    }
    infos["SphereWithLinCons"]["cons"].addineqcons(np.array([[1] * n]), -np.array([[1]] * n))
    infos["Norm1SphereWithSphereCons"]["cons"].addcons(lambda x: np.sum(np.square(x)) - 3 * x.shape[0], lambda x: 2 * x, lambda x: 2 * np.eye(x.shape[0]))
    infos["StyblinskiTangWithPosCons"]["cons"].addcons(lambda x: -x[0], lambda x: np.eye(x.shape[0], 1, 0), lambda x: np.zeros((x.shape[0], x.shape[0])))
    return infos[name]


def fSphereWithLinCons(x):
    return ndf.fSphere(x)


def gSphereWithLinCons(x):
    return ndf.gSphere(x)


def hSphereWithLinCons(x):
    return ndf.hSphere(x)


def fNorm1SphereWithSphereCons(x):
    return np.sum(x)


def gNorm1SphereWithSphereCons(x):
    return np.ones((x.shape[0], 1))


def hNorm1SphereWithSphereCons(x):
    return np.zeros((x.shape[0], x.shape[0]))


def fStyblinskiTangWithPosCons(x):
    return ndf.fStyblinskiTang(x)


def gStyblinskiTangWithPosCons(x):
    return ndf.gStyblinskiTang(x)


def hStyblinskiTangWithPosCons(x):
    return ndf.hStyblinskiTang(x)
