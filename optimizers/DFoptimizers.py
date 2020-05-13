import numpy as np
import sys
import copy
from . import base


class MADSOptimizer(base.Optimizer):
    def __init__(
        self,
        dim,
        function,
        constraints,
        getoptinfo,
        max_iter=1000,
        ftol=0,
        xtol=0,
        lambd_min=0,
        mu_min=0,
        epsilon=0,
        mu=1,
        lambd=1,
        use_minibasis=False,
    ):
        super().__init__(
            dim,
            function,
            constraints,
            getoptinfo,
            "MADS",
            max_iter,
            ftol,
            xtol,
        )
        # We verify the characteristics of the hyperparameters
        assert(mu <= lambd)
        self.mu, self.lambd = mu, lambd
        self.mu_min, self.lambd_min = mu_min, lambd_min
        self.epsilon = epsilon
        self.generated_poll_dirs = {}
        self.use_minibasis = use_minibasis
        self.D = self.generate_pss()
        self.success = False

    def generate_pss(self):
        basis = np.eye(self.dim)
        if self.use_minibasis:
            return np.concatenate((basis[..., np.newaxis], -np.sum(basis, axis=1)[np.newaxis, :, np.newaxis]), axis=0)
        else:
            return np.concatenate((basis[..., np.newaxis], -basis[..., np.newaxis]), axis=0)

    def search(self, x, fx):
        for d in self.D:
            tempx = x + self.mu * d
            tempfx = self.function(tempx)
            if self.test_constraints(tempx) and fx >= tempfx:
                return tempx, tempfx, True
        return x, fx, False

    def generate_poll_direction(self, l):
        if l in self.generated_poll_dirs:
            ihat = np.where(abs(self.generated_poll_dirs[l]) == 2**l)[0]
            return self.generated_poll_dirs[l], ihat
        else:
            self.generated_poll_dirs[l] = np.random.randint(-2**l + 1, 2**l, size=(self.dim, 1))
            ihat = np.random.randint(0, self.dim)
            self.generated_poll_dirs[l][ihat, 0] = (2 * np.random.randint(2) - 1) * 2**l
            return self.generated_poll_dirs[l], ihat

    def generate_poll_basis(self):
        l = int(-np.log(self.mu) / np.log(4))
        b, ihat = self.generate_poll_direction(l)
        L = np.diag((2 * np.random.randint(2, size=self.dim) - 1) * 2**l) + np.tril(np.random.randint(-2**l + 1, 2**l, size=(self.dim, self.dim)), -1)
        B = np.zeros(L.shape)
        perm = np.concatenate((np.arange(0, ihat, 1), np.arange(ihat + 1, self.dim, 1)))
        np.random.shuffle(perm)
        for i in range(self.dim - 1):
            for j in range(self.dim - 1):
                B[perm[i], j] = L[i, j]
        for i in range(self.dim):
            B[i, self.dim - 1] = b[i]
        np.random.shuffle(np.transpose(B))
        if self.use_minibasis:
            self.lambd = self.dim * np.sqrt(self.mu)
            return np.concatenate((B[..., np.newaxis], -np.sum(B, axis=1)[np.newaxis, :, np.newaxis]), axis=0)
        else:
            self.lambd = np.sqrt(self.mu)
            return np.concatenate((B[..., np.newaxis], -B[..., np.newaxis]), axis=0)

    def poll(self, x, fx):
        for d in self.generate_poll_basis():
            tempx = x + self.mu * d
            tempfx = self.function(tempx)
            if self.test_constraints(tempx) and fx >= tempfx:
                return tempx, tempfx, True
        return x, fx, False

    def update_mu(self, success):
        if success:
            if self.mu <= 1 / 4:
                self.mu *= 4
        else:
            self.mu /= 4

    def stop_criteria(self):
        if self.it >= self.max_iter:
            return True, "maximum number of iterations reached"
        if self.success and self.xdiff < self.xtol:
            return True, "x_tol reached"
        if self.success and self.fdiff < self.ftol:
            return True, "f_tol reached"
        if self.lambd < self.epsilon:
            return True, "lambda reached epsilon"
        if self.mu < self.epsilon:
            return True, "mu reached epsilon"
        if self.lambd < self.lambd_min:
            return True, "lambda reached its minimal value"
        if self.mu < self.mu_min:
            return True, "mu reached its minimal value"
        if not self.test_constraints(self.x_next):
            return True, "constraints violated"
        return False, None

    def step(self, x, fx):
        self.success = False
        x, fx, self.success = self.search(x, fx)
        if not self.success:
            x, fx, self.success = self.poll(x, fx)
        self.update_mu(self.success)
        return x, fx


class CMAESOptimizer(base.Optimizer):
    """
    CMAES code inspired from this matlab version by N.Hansen, Inria: http://cma.gforge.inria.fr/purecmaes.m
    MSR and Adaptative augmented Lagrangian from by this paper: Atamna et al, 2016 Augmented Lagrangian Constraint Handling for CMA-ES—Case of a Single Linear Constraint
    Functions: lagrangian (for constrained problems), jth_est_value (for MSR) generate_offsprings, select_offsprings, update_x_mean, update_params, stop_criteria, step
    """

    def __init__(
        self,
        dim,
        function,
        constraints,
        getoptinfo,
        max_iter=float('inf'),
        ftol=0,
        xtol=0,
        learning_rate=1e-1,
        lambd=None,
        stop_eigenvalue=1e7
    ):
        super().__init__(
            dim,
            function,
            constraints,
            getoptinfo,
            "CMAES",
            max_iter,
            ftol,
            xtol,
        )
        assert len(self.constraints) <= 1, 'This algorithm can handle only up to one constraint'
        """ User defined parameters """
        self.sigma = learning_rate                                              # Initial learning rate
        self.lambd = 4 + int(3 * np.log(self.dim)) if lambd is None else lambd  # Population size (offsprings)
        self.constrained_problem = bool(len(self.constraints))                  # Is there is a constraint
        # self.stop_eigenvalue = stop_eigenvalue                                # if max(D) > min(D) * stop_eigenvalue, stop optimization
        self.init_strategy_params()
        self.init_dynamic_params()

    def init_strategy_params(self):
        """ Strategy parameters settings: Selection """
        self.mu_temp = self.lambd / 2                                               # number of offsprings, smaller => faster convergence, bigger => avoid local optima
        self.mu = int(self.mu_temp)                                                 # number of parents
        self.weights = np.log(self.mu_temp + 0.5) - np.log(np.arange(self.mu) + 1)  # Raw weights for recombination
        self.weights = self.weights / np.sum(self.weights)                          # normalized weights for recombination
        self.mu_eff = 1 / np.sum(self.weights**2)                                   # variance effectiveness of sum(w_i*x_i)

        """ Strategy parameters settings: Adaptation """
        self.cc = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)                    # time constant for cumulation for C
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mu_eff)                                                       # learning rate  for rank-one update for C
        self.cm = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.dim + 2)**2 + self.mu_eff))  # learning rate for rank-mu update for C
        self.cs = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)                                              # time constant for cumulation for sigma
        self.ds = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1) + self.cs                     # damping for sigma, usually close to 1
        if self.constrained_problem:
            self.k1 = 3                                                                                         # constant param for first condition on omega update
            self.k2 = 5                                                                                         # constant param for second condition on omega update

    def init_dynamic_params(self):
        self.B = np.eye(self.dim)                                                           # Coordinate system to go from C to D: defines rotation
        self.D = np.eye(self.dim)                                                           # diagonal D of covariance matric C: defines scaling
        self.C = np.linalg.multi_dot([self.B, self.D, self.D.T, self.B.T])                  # covariance matrix
        self.invsqrtC = np.linalg.multi_dot([self.B, np.linalg.inv(self.D), self.B.T])      # C^-1/2
        self.eigeneval = 0                                                                  # track updates of B and D
        self.ps = np.zeros((self.dim, 1))                                                   # evolution path for learning rate sigma for CMAES
        self.pc = np.zeros((self.dim, 1))                                                   # evolution path for C
        self.chi_N = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))  # expectation of ||N(0, I)||
        if self.constrained_problem:
            self.chi = 2**(1 / self.dim)                                                    # Parameter
            self.gamma = 5                                                                  # Lagrange factor
            self.omega = 1                                                                  # penalty factor of the augmented Lagrangian

    def lagrangian(self, x):
        """
        Computes the augmented lagrangian of function, constraint using current values of gamma and omega at x
        """
        cons_val = self.constraints.evaluate(x)[0]
        if np.all(self.gamma + self.omega * cons_val) >= 0:
            return np.squeeze(self.function(x) + self.gamma * cons_val + self.omega * cons_val**2 / 2)
        else:
            return - self.gamma**2 / 2 / self.omega

    def generate_offsprings(self):
        """
        Generates lambd offsprings~N(x_mean, sigma**2 *C)
        """
        yk = np.random.randn(self.dim, self.lambd)
        yk = np.linalg.multi_dot([self.B, self.D, yk])
        xk = self.x_mean + self.sigma * yk  # Offsprings
        return xk, yk

    def select_offsprings(self, xk, yk):
        """
        # Evaluate offsprings and select best individuals
        """
        # print(xk)
        if self.constrained_problem:
            fk = np.apply_along_axis(self.lagrangian, axis=0, arr=xk)  # evaluating each offspring on lagrangian value basis
        else:
            fk = np.apply_along_axis(self.function, axis=0, arr=xk)  # evaluating each offspring on objective function value basis
        # print(fk)
        idx = np.argsort(fk)  # indices of each xk by descending value of f(xk)
        selection_x = xk[:, idx[:self.mu]]  # best individuals
        selection_y = yk[:, idx[:self.mu]]  # best individuals before centered on x_mean and sigma**2 reduced
        # if self.MSR:
        #     return selection_x, selection_y, fk
        return selection_x, selection_y

    def update_x_mean(self, best_individuals):
        self.x_mean = np.real(np.dot(self.weights, best_individuals.T).reshape(self.dim, -1))

    def update_params(self, best_individuals_N0C, fk=None):
        """
        Updates all dynamic params ps, pc, C, B, D, sigma, eigeneval, classic CMAES or MSR
        """
        # Cumulation paths
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * np.dot(self.invsqrtC, self.x_mean - self.x_old) / self.sigma
        # Heaviside function: boolean to prevent a too steep update of pc, especially for small sigma
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * (self.it + 1))) < (1.4 + 2 / (self.dim + 1)) * self.chi_N
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * (self.x_mean - self.x_old) / self.sigma

        # Covariance matrix adaptation
        delta_hsig = (1 - hsig) * self.cc * (2 - self.cc) <= 1  # to correct the rank1 update in the case where hsig = 0
        rank1_update = np.dot(self.pc, self.pc.T) + delta_hsig * self.C
        rankmu_update = np.linalg.multi_dot([best_individuals_N0C, np.diag(self.weights), best_individuals_N0C.T])
        self.C = (1 - self.c1 - self.cm) * self.C + self.c1 * rank1_update + self.cm * rankmu_update

        # Step-size sigma update
        self.sigma *= np.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chi_N - 1))

        if self.constrained_problem:
            cons_val = self.constraints.evaluate(self.x_mean)[0]
            cons_val_old = self.constraints.evaluate(self.x_old)[0]
            # Compute condition 1 and 2 for omega with omega_t and gamma_t, not gamma_t+1
            condition_1 = self.omega * cons_val**2 < self.k1 * abs(self.lagrangian(self.x_mean) - self.lagrangian(self.x_old)) / self.dim
            condition_2 = self.k2 * abs(cons_val - cons_val_old) < abs(cons_val_old)
            # Update Lagrange factor
            self.gamma = max(0, self.gamma + self.omega * cons_val)
            # Update penalty factor
            if condition_1 or condition_2:
                self.omega *= self.chi**(1 / 4)
            else:
                self.omega /= self.chi

        # Update B and D from C
        if self.it - self.eigeneval > 1 / (self.c1 + self.cm) / self.dim / 10:
            self.eigeneval = self.it
            self.C = np.triu(self.C) + np.triu(self.C, 1).T  # enforce symetry
            d, self.B = np.linalg.eig(self.C)
            d = np.abs(np.real(d))
            self.D = np.sqrt(np.diag(d))
            self.invsqrtC = np.linalg.multi_dot([self.B, np.linalg.pinv(self.D), self.B.T])
        else:
            if self.verbose:
                print('No update of B and D at iteration %i' % self.it)

    def stop_criteria(self):
        if self.it >= self.max_iter:
            return True, "maximum number of iterations reached"
        if self.xdiff < self.xtol:
            return True, "x_tol reached"
        if self.fdiff < self.ftol:
            return True, "f_tol reached"
        if self.constrained_problem:
            if not self.test_constraints(self.x_next) and self.verbose:
                print("constraints violated")
        return False, None

    def step(self, x, fx):
        self.x_mean = x
        self.x_old = copy.deepcopy(self.x_mean)
        xk, yk, = self.generate_offsprings()
        best_individuals, best_individuals_N0C = self.select_offsprings(xk, yk)
        self.update_x_mean(best_individuals)
        self.update_params(best_individuals_N0C)
        return self.x_mean, self.function(self.x_mean)
