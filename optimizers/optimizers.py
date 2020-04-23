import numpy as np
import sys
import matplotlib.pyplot as plt
import copy
np.set_printoptions(threshold=float('inf'), linewidth=1000, suppress=True, precision=2)


def print_row(*strings, width=[11, 10, 15, 15, 25], header=False):
    to_print = '|'
    for i, s in enumerate(strings[:-1]):
        to_print += s.center(width[i])
        to_print += ' | '
    to_print += strings[-1].center(width[-1]) + '|'
    if header:
        print('-' * len(to_print))
    print(to_print)
    if header:
        print('-' * len(to_print))


class Optimizer():

    def __init__(self, dim, function, constraints):
        self.dim = dim
        self.function = function
        self.constraints = constraints
        self.it = 1

    def step(self, x, fx):
        raise NotImplementedError

    def stop_criteria(self):
        raise NotImplementedError

    def test_constraints(self, x):  # Returns False if a constraint is violated, True otherwise
        for cons in self.constraints:
            if not cons.test(x):
                return False
        return True

    def optimize(self, x0, max_iter=float('inf'), ftol=0, xtol=0, plot=False, verbose=False):
        x = copy.deepcopy(x0)
        fx = self.function(x)
        self.it = 1
        finish = False
        self.track = list()
        print('\n%s\n' % ('Optimization Starting'.center(100, '-')))
        print_row('Iteration', 'f(x)', '||xk-xk-1||', '|f(xk)-f(xk-1)|', 'x', header=True)
        while not finish:
            x_next, fx_next = self.step(x, fx)
            xdiff = np.linalg.norm(x_next - x)
            fdiff = abs(fx_next - fx)
            if self.it >= max_iter or xdiff < xtol or fdiff < ftol or self.stop_criteria():
                finish = True
            x, fx = x_next, fx_next
            self.track.append((x, fx, xdiff, fdiff))

            if verbose:
                print_row('%d' % self.it, '%3.3f' % fx, '%3.3f' % xdiff, '%3.3f' % fdiff, '%s' % str(x.reshape(-1)))
            self.it += 1

        print('\n\n%s\n' % ('Final Results After {:4d} Iterations'.format(self.it).center(100, '-')))
        print_row('f(x)', '%3.3f' % fx, 'x', '%s' % str(x.reshape(-1)), width=[10, 10, 10, 25], header=True)
        if plot:
            self.plot_results()

    def plot_results(self):
        xs, fxs, xdiffs, fdiffs = map(np.array, zip(*self.track))
        xs = np.sum(np.abs(np.squeeze(xs))**2, axis=-1)**(1. / 2)
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True)
        ax[0, 0].plot(xs)
        ax[0, 0].set_ylabel(r'$||x||$')
        ax[0, 1].plot(fxs)
        ax[0, 1].set_ylabel(r'$f(x)$')
        ax[1, 0].plot(xdiffs)
        ax[1, 0].set_ylabel(r'$||x_k - x_{k-1}||$')
        ax[1, 0].set_xlabel('steps')
        ax[1, 1].plot(fdiffs)
        ax[1, 1].set_ylabel(r'$|f(x_k) - f(x_{k-1})|$')
        ax[1, 1].set_xlabel('steps')
        fig.suptitle('Evolution of the Optimization')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


class MADSOptimizer(Optimizer):
    def __init__(
        self,
        dim,
        function,
        constraints,
        delta_m=1,
        delta_p=1,
        D=None,
        G=None,
        tau=4,
        w_minus=-1,
        w_plus=1,
        use_minibasis=True,
    ):
        super().__init__(dim, function, constraints)
        # We verify the characteristics of the hyperparameters
        assert(delta_m <= delta_p)
        self.delta_m, self.delta_p = delta_m, delta_p
        self.tau, self.w_minus, self.w_plus = tau, w_minus, w_plus
        self.generated_poll_dirs = {}
        self.use_minibasis = use_minibasis
        if D is None:
            self.D = self.generate_pss()
        else:
            self.D = D
        if G is None:
            self.G = np.eye(self.dim)
        else:
            self.G = G

    def generate_pss(self):
        basis = np.eye(self.dim)
        if self.use_minibasis:
            return np.concatenate((basis[..., np.newaxis], -np.sum(basis, axis=1)[np.newaxis, :, np.newaxis]), axis=0)
        else:
            return np.concatenate((basis[..., np.newaxis], -basis[..., np.newaxis]), axis=0)

    def search(self, x, fx):
        for d in self.D:
            tempx = x + self.delta_m * d
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
        l = int(-np.log(self.delta_m) / np.log(4))
        b, ihat = self.generate_poll_direction(l)
        L = np.diag((2 * np.random.randint(2, size=self.dim) - 1) * 2**l) + np.tril(np.random.randint(-2**l + 1, 2**l, size=(self.dim, self.dim)), -1)
        B = np.zeros_like(L)
        perm = np.concatenate((np.arange(0, ihat, 1), np.arange(ihat + 1, self.dim, 1)))
        np.random.shuffle(perm)
        for i in range(self.dim - 1):
            for j in range(self.dim - 1):
                B[perm[i], j] = L[i, j]
        for i in range(self.dim):
            B[i, self.dim - 1] = b[i]
        np.random.shuffle(np.transpose(B))
        if self.use_minibasis:
            self.delta_p = self.dim * np.sqrt(self.delta_m)
            return np.concatenate((B[..., np.newaxis], -np.sum(B, axis=1)[np.newaxis, :, np.newaxis]), axis=0)
        else:
            self.delta_p = np.sqrt(self.delta_m)
            return np.concatenate((B[..., np.newaxis], -B[..., np.newaxis]), axis=0)

    def poll(self, x, fx):
        for d in self.generate_poll_basis():
            tempx = x + self.delta_m * d
            tempfx = self.function(tempx)
            if self.test_constraints(tempx) and fx >= tempfx:
                return tempx, tempfx, True
        return x, fx, False

    def update_delta_m(self, success):
        if success:
            if self.delta_m <= 1 / self.tau:
                self.delta_m *= self.tau
        else:
            self.delta_m /= self.tau
    
    def stop_criteria(self):   
        return False # True if optimization must stop because of optimizer condition, not defined here

    def step(self, x, fx):
        success = False
        while not success and self.delta_m > 1 / 4**30:
            x, fx, success = self.search(x, fx)
            if not success:
                x, fx, success = self.poll(x, fx)
            self.update_delta_m(success)
        return x, fx


class CMAESOptimizer(Optimizer):
    """
    CMAES code inspired from this matlab version by N.Hansen, Inria: http://cma.gforge.inria.fr/purecmaes.m
    MSR and Adaptative augmented Lagrangian from by this paper: Atamna et al, 2016 Augmented Lagrangian Constraint Handling for CMA-ES—Case of a Single Linear Constraint
    Functions: lagrangian (for constrained problems), generate_offsprings, select_offsprings, update_x_mean, update_params, stop_criteria, step
    """
    def __init__(self, dim, function, constraints, learning_rate, lambd=None, MSR = False, constrained_problem=False, stop_eigenvalue=1e7):
        super().__init__(dim, function, constraints)
        assert len(self.constraints) <= 1, 'This algorithm can handle only up to one constraint'
        assert len(self.constraints[0].evaluate(np.zeros(self.dim))) == 1, 'This algorithm can handle only up to one constraint'
        """ User defined parameters """
        self.sigma = learning_rate                                              # Initial learning rate
        if lambd != None:
            self.lambd = lambd
        else:
            self.lambd = 4 + int(3 * np.log(self.dim))                          # Recommended value for the number of offsprings lambda
        self.MSR = MSR                                                          # Is Mean Success Rule ste-size activated
        self.constrained_problem = constrained_problem                          # Is there a constraint
        self.stop_eigenvalue = stop_eigenvalue                                  # if max(D) > min(D) * stop_eigenvalue, stop optimization
        self.init_strategy_params()                     
        self.init_dynamic_params()                      

    def init_strategy_params(self):
        """ Strategy parameters settings: Selection """
        self.mu = self.lambd / 2                                                # number of offsprings, smaller => faster convergence, bigger => avoid local optima
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(self.mu) +1)    # Raw weights for recombination
        self.weights = self.weights / np.sum(self.weights)                      # normalized weights for recombination
        self.mu = int(self.mu)                                                  # number of parents
        self.mu_eff = 1 / np.sum(self.weights**2)                               # variance effectiveness of sum(w_i*x_i)

        """ Strategy parameters settings: Adaptation """
        self.cc = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)                    # time constant for cumulation for C
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mu_eff)                                                       # learning rate  for rank-one update for C
        self.cm = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.dim+2)**2 + self.mu_eff))   # learning rate for rank-mu update for C
        if self.MSR:
            self.cs = 0.3                                                                                       # time constant for cumulation for sigma
            self.ds = 2 - 2 / self.dim                                                                          # damping for sigma
            self.compared_offspring = int(0.3 * self.lambd) - 1                                                 # success if the offspring perfoms better than the 30th best offspring
            if self.constrained_problem:
                self.k1 = 3                                                                                     # constant param for first condition on omega update
                self.k2 = 5                                                                                     # constant param for second condition on omega update
        else:
            self.cs = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)                                          # time constant for cumulation for sigma
            self.ds = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1) + self.cs                 # damping for sigma, usually close to 1

    def init_dynamic_params(self):
        self.B = np.eye(self.dim)                                                       # Coordinate system to go from C to D: defines rotation
        self.D = np.eye(self.dim)                                                       # diagonal D of covariance matric C: defines scaling
        self.C = np.linalg.multi_dot([self.B, self.D, self.D.T, self.B.T])              # covariance matrix
        self.invsqrtC = np.linalg.multi_dot([self.B, np.linalg.inv(self.D), self.B.T])  # C^-1/2
        self.eigeneval = 0                                                              # track updates of B and D
        self.ps = np.zeros((self.dim, 1))                                               # evolution path fo learning rate sigma
        self.pc = np.zeros((self.dim, 1))                                               # evolution path for C
        self.chi_N = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))      # expectation of ||N(0, I||
        if self.MSR and self.constrained_problem:
            self.chi = 2**(1 / self.dim)                                                # 
            self.gamma = 5                                                              # Lagrange factor
            self.omega = 1                                                              # penalty factor of the augmented Lagrangian

    def lagrangian(self, x):
        """
        Computes the augmented lagrangian of function, constraint using current values of gamma and omega at x
        """
        cons_val = self.constraints[0].evaluate(x)
        if np.all(self.gamma + self.omega * cons_val) >= 0:
            return np.squeeze(self.function(x) + self.gamma * cons_val + self.omega * cons_val**2)
        else:
            return - self.gamma**2 / 2 / self.omega 

    def generate_offsprings(self):
        """
        Generates lambd offsprings~N(x_mean, sigma**2 *C)
        """
        yk = np.random.randn(self.dim, self.lambd)
        yk = np.linalg.multi_dot([self.B, self.D, yk])
        xk = self.x_mean.reshape(self.dim, -1) + self.sigma * yk  # Offsprings
        return xk, yk

    def select_offsprings(self, xk, yk):
        """
        # Evaluate offsprings and select best individuals
        """
        if self.MSR and self.constrained_problem:
            fk = np.apply_along_axis(self.lagrangian, axis = 0, arr=xk) # evaluating each offspring on lagrangian value basis
        else:
            fk = np.apply_along_axis(self.function, axis=0, arr=xk)  # evaluating each offspring on objective function value basis
        idx = np.argsort(fk)  # indices of each xk by descending value of f(xk)
        selection_x = xk[:, idx[:self.mu]]  # best individuals
        selection_y = yk[:, idx[:self.mu]]  # best individuals before centered on x_mean and sigma**2 reduced
        if self.MSR:
            return selection_x, selection_y, fk, fk[idx[self.compared_offspring]]
        return selection_x, selection_y

    def update_x_mean(self, best_individuals):
        self.x_mean = np.dot(best_individuals, self.weights)

    def update_params(self, best_individuals_N0C, fk=None, jth_offspring=None):
        """
        Updates all dynamic params ps, pc, C, B, D, sigma, eigeneval, classic CMAES or MSR
        """

        if self.MSR:
            # Cumulation path for covariance matrix adaptation
            self.pc = (1 - self.cc) * self.pc + np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) *  (self.x_mean - self.x_old) / self.sigma

            # Covariance matrix adaptation
            rank1_update = np.dot(self.pc, self.pc.T)
            rankmu_update = np.linalg.multi_dot([best_individuals_N0C, np.diag(self.weights), best_individuals_N0C.T])
            self.C = (1 - self.c1 - self.cm) * self.C + self.c1 * rank1_update + self.cm * rankmu_update

            # Step-size sigma update
            K_succ = np.count_nonzero(fk < jth_offspring)
            success_measure = 2 * K_succ / self.lambd - 1
            self.ps  = (1 - self.cs) * self.ps + self.cs * success_measure
            self.sigma *= np.exp(self.ps/self.ds)

            if self.constrained_problem:
                cons_val = self.constraints[0].evaluate(self.x_mean)
                cons_val_old = self.constraints[0].evaluate(self.x_old)
                # Update Lagrange factor
                self.gamma = max(0, self.gamma + self.omega * cons_val)
                # Update penalty factor
                condition_1 = self.omega * cons_val**2 < self.k1 * abs(self.lagrangian(self.x_mean) - self.lagrangian(self.x_old)) / self.dim
                condition_2 = self.k2 * abs(cons_val - cons_val_old) < abs(cons_val_old)
                if condition_1 or condition_2:
                    self.omega *= self.chi**(1/4)
                else:
                    self.omega /= self.chi

        else:
            # Cumulation paths
            self.ps  = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * np.dot(self.invsqrtC, self.x_mean - self.x_old) / self.sigma
            # Heaviside function: boolean to prevent a too steep update of pc, especially for small sigma
            hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * (self.it + 1))) < (1.4 + 2 / (self.dim + 1)) * self.chi_N
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) *  (self.x_mean - self.x_old) / self.sigma

            # Covariance matrix adaptation
            delta_hsig = (1 - hsig) * self.cc * (2 - self.cc) <= 1 # to correct the rank1 update in the case where hsig = 0
            rank1_update = np.dot(self.pc, self.pc.T) + delta_hsig * self.C
            rankmu_update = np.linalg.multi_dot([best_individuals_N0C, np.diag(self.weights), best_individuals_N0C.T])
            self.C = (1 - self.c1 - self.cm) * self.C + self.c1 * rank1_update + self.cm * rankmu_update

            # Step-size sigma update
            self.sigma *= np.exp((self.cs/self.ds) * (np.linalg.norm(self.ps)/self.chi_N - 1))

        # Update B and D from C
        if self.it - self.eigeneval > 1 / (self.c1 + self.cm) / self.dim / 10:
            self.eigeneval = self.it
            self.C = np.triu(self.C) + np.triu(self.C, 1).T       # enforce symetry
            d, self.B = np.linalg.eig(self.C)
            self.D = np.sqrt(np.diag(d))
            self.invsqrtC = np.linalg.multi_dot([self.B, np.linalg.inv(self.D), self.B.T])
        else: 
            print('No update of B and D at iteration %i' % self.it)

    def stop_criteria(self):
        if max(np.diag(self.D)) > self.stop_eigenvalue * min(np.diag(self.D)):
            print("Optimization stopped because of Covariance matrix eigenvalues stopping criteria")
            return True
        return False

    def step(self, x, fx):
        self.x_mean = x
        self.x_old = copy.deepcopy(self.x_mean)
        xk, yk, = self.generate_offsprings()
        # generates mu best individuals and related non centered and non sigma**2 reduced points, function values unsorted for all xk and offspring for sigma update
        if self.MSR:
            best_individuals, best_individuals_N0C, fk, jth_offspring = self.select_offsprings(xk, yk)
            self.update_x_mean(best_individuals)
            self.update_params(best_individuals_N0C, fk, jth_offspring)
        else:
            best_individuals, best_individuals_N0C = self.select_offsprings(xk, yk)
            self.update_x_mean(best_individuals)
            self.update_params(best_individuals_N0C)
        if self.stop_criteria():
            return self.x_old, self.function(self.x_old)
        return self.x_mean, self.function(self.x_mean)

