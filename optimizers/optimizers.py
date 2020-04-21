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

    def step(self, x, fx):
        raise NotImplementedError

    def test_constraints(self, x):
        for cons in self.constraints:
            if not cons.test(x):
                return False
        return True

    def optimize(self, x0, max_iter=float('inf'), ftol=0, xtol=0, plot=False, verbose=False):
        x = copy.deepcopy(x0)
        fx = self.function(x)
        it = 1
        finish = False
        self.track = list()
        print('\n%s\n' % ('Optimization Starting'.center(100, '-')))
        print_row('Iteration', 'f(x)', '||xk-xk-1||', '|f(xk)-f(xk-1)|', 'x', header=True)
        while not finish:
            x_next, fx_next = self.step(x, fx)
            xdiff = np.linalg.norm(x_next - x)
            fdiff = abs(fx_next - fx)
            if it > max_iter or xdiff < xtol or fdiff < ftol:
                finish = True
            x, fx = x_next, fx_next
            self.track.append((x, fx, xdiff, fdiff))

            if verbose:
                print_row('%d' % it, '%3.3f' % fx, '%3.3f' % xdiff, '%3.3f' % fdiff, '%s' % str(x.reshape(-1)))
            it += 1

        print('\n\n%s\n' % ('Final Results After {:4d} Iterations'.format(it).center(100, '-')))
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

    def step(self, x, fx):
        success = False
        while not success and self.delta_m > 1 / 4**30:
            x, fx, success = self.search(x, fx)
            if not success:
                x, fx, success = self.poll(x, fx)
            self.update_delta_m(success)
        return x, fx


class CMAESOptimizer(Optimizer):
    def __init__(self, dim, function, constraints):
        super().__init__(dim, function, constraints)

    def step(self, x, fx):
        # TOCOMPLETE
        raise NotImplementedError
