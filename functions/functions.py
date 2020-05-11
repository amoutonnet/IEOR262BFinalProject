from . import ndfunctions as ndf
from . import ddfunctions as ddf
import numpy as np


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
    if x.shape[0] == 2:
        assert name in ddf.LISTG, "The gradient for this function is not defined"
        gx = 0
        _locals = locals()
        exec("gx = ddf.g%s(x[0][0], x[1][0])" % name, globals(), _locals)
        return _locals["gx"]
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
