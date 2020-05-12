import numpy as np

DELTA = 1e-8


class Constraint():
    def __init__(self, f, g, h):
        self.f = f
        self.g = g
        self.h = h

    def test(self, x):
        return self.f(x) <= 0

    def logbarrier(self, x, theta):
        return theta * np.log(-self.f(x))

    def gradlogbarrier(self, x, theta):
        return -theta * self.g(x) / self.f(x)

    def hesslogbarrier(self, x, theta):
        return +theta * (np.dot(self.g(x).T, self.g(x)) / np.square(self.f(x)) - self.h(x) / self.f(x))


class Constraints():
    def __init__(self):
        self.constraints = []

    def test(self, x):
        if self.constraints:
            return all(list(map(lambda c: c.test(x), self.constraints)))
        else:
            return True

    def evaluate(self, x):
        return list(map(lambda c: c.f(x), self.constraints))

    def logbarrier(self, x, theta):
        if self.constraints:
            return np.sum(np.array(list(map(lambda c: c.logbarrier(x, theta), self.constraints))), axis=0)
        else:
            return 0

    def gradlogbarrier(self, x, theta):
        if self.constraints:
            return np.sum(np.array(list(map(lambda c: c.gradlogbarrier(x, theta), self.constraints))), axis=0)
        else:
            return np.zeros(x.shape)

    def hesslogbarrier(self, x, theta):
        if self.constraints:
            return np.sum(np.array(list(map(lambda c: c.hesslogbarrier(x, theta), self.constraints))), axis=0)
        else:
            return np.zeros((x.shape[0], x.shape[0]))

    def addineqcons(self, A, b):
        assert len(A.shape) == 2 and len(b.shape) == 2, "Constraints must be matrices"
        for row, rval in zip(A, b):
            self.addcons(lambda x: np.dot(row, x) - rval, lambda x: row.reshape(-1, 1), lambda x: np.zeros((x.shape[0], x.shape[0])))

    def addposcons(self, n):
        self.addineqcons(-np.eye(n), np.zeros((n, 1)))

    def addcons(self, f, g, h):
        self.constraints += [Constraint(f, g, h)]

    def __len__(self):
        return len(self.constraints)
