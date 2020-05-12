import numpy as np
from . import constraints
from . import ndfunctions as ndf


PI = np.pi

LIST = ["SphereWithLinCons", "Norm1SphereWithSphereCons"]


def getinfos(name, n):
    assert name in LIST, "This constrained function is not implemented"
    infos = {
        "SphereWithLinCons": {
            "cons": constraints.Constraints(),
            "xopt": np.array([[1]] * n),
            "fopt": n,
        },
        "Norm1SphereWithSphereCons": {
            "cons": constraints.Constraints(),
            "xopt": np.array([[-np.sqrt(3)]] * n),
            "fopt": -np.sqrt(3) * n,
        },
    }
    infos["SphereWithLinCons"]["cons"].addineqcons(np.array([[1] * n]), np.array([[-1]] * n))
    infos["Norm1SphereWithSphereCons"]["cons"].addcons(lambda x: np.sum(np.square(x)) - 3 * x.shape[0], lambda x: 2 * x, lambda x: 2 * np.eye(x.shape[0]))
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
    return x


def hNorm1SphereWithSphereCons(x):
    return np.eye(x.shape[0])
