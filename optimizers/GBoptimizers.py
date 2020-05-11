import numpy as np
import sys
import copy
from . import base


class GradientBasedOptimizer(base.Optimizer):
    def __init__(self, dim, function, gradient, hessian, constraints, name, max_iter, ftol, gtol, xtol):
        super().__init__(
            dim,
            function,
            constraints,
            name,
            max_iter,
            ftol,
            xtol
        )
        self.gradient = gradient
        self.hessian = hessian
        self.gtol = gtol

    def stop_criteria(self):
        if self.it >= self.max_iter:
            return True, "maximum number of iterations reached"
        if self.xdiff < self.xtol:
            return True, "x_tol reached"
        if self.fdiff < self.ftol:
            return True, "f_tol reached"
        if np.linalg.norm(self.gradient(self.x_next)) < self.gtol:
            return True, "g_tol reached"
        return False, None


class NewtonLineSearchOptimizer(GradientBasedOptimizer):
    def __init__(
        self,
        dim,
        function,
        gradient,
        hessian,
        constraints,
        max_iter=1000,
        ftol=0,
        gtol=0,
        xtol=0,
        epsilon=1e-3
    ):
        super().__init__(
            dim,
            function,
            gradient,
            hessian,
            constraints,
            "Newton Line Search",
            max_iter,
            ftol,
            gtol,
            xtol
        )
        self.epsilon = epsilon

    def bisection_search_algorithm(self, x, d):
        alpha_l, alpha_u = 0, 1
        def grad_h(a): return np.dot(self.gradient(x + a * d).T, d)
        while 1:
            alpha_u = alpha_u * 1.5 if self.test_constraints(x + alpha_u * d) else alpha_u * 0.90
            if grad_h(alpha_u) > 0:
                break
        while 1:
            alpha = (alpha_l + alpha_u) / 2
            if abs(grad_h(alpha)) < self.epsilon:
                break
            elif grad_h(alpha) > 0:
                alpha_u = alpha
            else:
                alpha_l = alpha
        return alpha

    def step(self, x, fx):
        if np.linalg.norm(self.gradient(x)) != 0:
            d = np.dot(-np.linalg.inv(self.hessian(x)), self.gradient(x))
            x = x + self.bisection_search_algorithm(x, d) * d
        return x, self.function(x)

    def stop_criteria(self):
        stop, reason = super().stop_criteria()
        if not stop:
            if not self.test_constraints(self.x):
                return True, "constraints violated"
            return False, None
        else:
            return stop, reason


class NewtonLogBarrierOptimizer(GradientBasedOptimizer):
    def __init__(
        self,
        dim,
        function,
        gradient,
        hessian,
        constraints,
        max_iter=1000,
        ftol=0,
        gtol=0,
        xtol=0,
        mu=1.1,
        theta0=100,
        epsilon=1e-8
    ):
        super().__init__(
            dim,
            function,
            gradient,
            hessian,
            constraints,
            "Newton Log Barrier",
            max_iter,
            ftol,
            gtol,
            xtol
        )
        assert self.constraints, "%s method need constraints to work" % self.name
        self.barrierfunc = lambda x, theta: function(x) + constraints.logbarrier(x, theta)
        self.barriergrad = lambda x, theta: gradient(x) + constraints.gradlogbarrier(x, theta)
        self.barrierhess = lambda x, theta: hessian(x) + constraints.hesslogbarrier(x, theta)
        self.mu = mu
        self.epsilon = epsilon
        self.theta = theta0

    def step(self, x, fx):
        d = np.dot(-np.linalg.inv(self.barrierhess(x, self.theta)), self.barriergrad(x, self.theta))
        x = x + d
        self.theta /= self.mu
        return x, self.function(x)

    def stop_criteria(self):
        stop, reason = super().stop_criteria()
        if not stop:
            if not self.test_constraints(self.x):
                return True, "constraints violated"
            if len(self.constraints) * self.theta < self.epsilon:
                return True, "barrier reached"
            return False, None
        else:
            return stop, reason
