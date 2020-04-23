import numpy as np

DELTA = 1e-8


class Constraint():
    def __init__(self, f):
        self.f = f

    def evaluate(self, x):
        return self.f(x)
        
    def test(self, x):
        raise NotImplementedError


class EqConstraint(Constraint):
    def __init__(self, f):
        super().__init__(f)

    def test(self, x):
        fx = self.f(x)
        if isinstance(fx, np.ndarray):
            return all(np.abs(fx) <= DELTA)
        else:
            return abs(fx) <= DELTA


class IneqConstraint(Constraint):
    def __init__(self, f):
        super().__init__(f)

    def test(self, x):
        fx = self.f(x)
        if isinstance(fx, np.ndarray):
            return all(fx <= 0)
        else:
            return fx <= 0


class AffEqConstraint(EqConstraint):
    def __init__(self, A, b):
        super().__init__(lambda x: np.dot(A, x) - b)


class AffIneqConstraint(IneqConstraint):
    def __init__(self, A, b):
        super().__init__(lambda x: np.dot(A, x) - b)


class PosConstraint(IneqConstraint):
    def __init__(self):
        super().__init__(lambda x: -x)
