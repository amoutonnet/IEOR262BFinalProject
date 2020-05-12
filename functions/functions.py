from . import ndfunctions as ndf
from . import ddfunctions as ddf
from . import consfunctions as cndf
from . import constraints as cons
import numpy as np


def optinfo(name, x, fx):
    n = x.shape[0]
    if name in cndf.LIST:
        info = cndf.getinfos(name, n)
        return (np.linalg.norm(info["xopt"] - x), abs(info["fopt"] - fx))
    else:
        if n == 2:
            if name not in ["Holder", "CrossInTray"]:
                return (np.linalg.norm(ddf.INFOS[name]["xopt"] - x), abs(ddf.INFOS[name]["fopt"] - fx))
            else:
                return (min([np.linalg.norm(xopt - x) for xopt in ddf.INFOS[name]["xopts"]]), abs(ddf.INFOS[name]["fopt"] - fx))
        elif n > 2:
            if name != "StyblinskiTang":
                return (np.linalg.norm(np.array([ndf.INFOS[name]["xopti"]] * n) - x), abs(ndf.INFOS[name]["fopt"] - fx))
            else:
                return (np.linalg.norm(np.array([ndf.INFOS[name]["xopti"]] * n) - x), abs(ndf.INFOS[name]["foptovern"] * n - fx))
        else:
            raise NotImplementedError


def getcons(name, n):
    if name in cndf.LIST:
        info = cndf.getinfos(name, n)
        return info["cons"]
    else:
        return cons.Constraints()


def fdec(func):
    def wrapper(*args):
        x = args[1]
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        x = x.reshape(-1, 1)
        assert len(x.shape) == 2 and x.shape[1] == 1, 'input must be a vector'
        return func(args[0], x)
    return wrapper


@fdec
def function(name, x):
    if name in cndf.LIST:
        fx = 0
        _locals = locals()
        exec("fx = cndf.f%s(x.ravel())" % name, globals(), _locals)
        return _locals["fx"]
    if x.shape[0] == 2:
        assert name in ddf.LISTF, "This function is not defined"
        fx = 0
        _locals = locals()
        exec("fx = ddf.f%s(x[0][0], x[1][0])" % name, globals(), _locals)
        return _locals["fx"]
    elif x.shape[0] > 2:
        assert name in ndf.LISTF, "This function is not defined"
        fx = 0
        _locals = locals()
        exec("fx = ndf.f%s(x.ravel())" % name, globals(), _locals)
        return _locals["fx"]
    else:
        raise NotImplementedError


@fdec
def gradient(name, x):
    if name in cndf.LIST:
        gx = 0
        _locals = locals()
        exec("gx = cndf.g%s(x.ravel())" % name, globals(), _locals)
        return _locals["gx"].reshape(-1, 1)
    if x.shape[0] == 2:
        assert name in ddf.LISTG, "The gradient for this function is not defined"
        gx = 0
        _locals = locals()
        exec("gx = ddf.g%s(x[0][0], x[1][0])" % name, globals(), _locals)
        return _locals["gx"].reshape(-1, 1)
    elif x.shape[0] > 2:
        assert name in ndf.LISTG, "The gradient for this function is not defined"
        gx = 0
        _locals = locals()
        exec("gx = ndf.g%s(x.ravel())" % name, globals(), _locals)
        return _locals["gx"]
    else:
        raise NotImplementedError


@fdec
def hessian(name, x):
    if name in cndf.LIST:
        hx = 0
        _locals = locals()
        exec("hx = cndf.h%s(x.ravel())" % name, globals(), _locals)
        return _locals["hx"]
    if x.shape[0] == 2:
        assert name in ddf.LISTH, "The hessian for this function is not defined"
        hx = 0
        _locals = locals()
        exec("hx = ddf.h%s(x[0][0], x[1][0])" % name, globals(), _locals)
        return _locals["hx"]
    elif x.shape[0] > 2:
        assert name in ndf.LISTH, "The hessian for this function is not defined"
        hx = 0
        _locals = locals()
        exec("hx = ndf.h%s(x.ravel())" % name, globals(), _locals)
        return _locals["hx"]
    else:
        raise NotImplementedError
