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
        print('f(x)={:.3f}'.format(fx))
        print('x={:s}'.format(str(x.reshape(-1))))
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
        alpha=0.1,
        beta_1=0.8,
        beta_2=0.9,
        gamma=1.1,
        forcing_function=lambda x: x * x,
    ):
        super().__init__(dim, function, constraints)
        # We verify the characteristics of the hyperparameters
        assert(
            alpha > 0 and
            beta_1 > 0 and
            beta_2 >= beta_1 and
            1 > beta_2 and
            gamma > 1
        )
        self.alpha, self.beta_1, self.beta_2, self.gamma = alpha, beta_1, beta_2, gamma
        self.forcing_function = forcing_function
        self.pss = self.get_norms(self.generate_pss())

    def generate_pss(self):
        return np.concatenate((np.eye(self.dim)[..., np.newaxis], -np.eye(self.dim)[..., np.newaxis]), axis=0)

    def get_norms(self, pss):
        to_return = []
        for i in pss:
            to_return.append((i, np.linalg.norm(i)))
        return to_return

    def search(self, x, fx):
        return x, fx, False

    def poll(self, x, fx):
        for d, normd in self.pss:
            tempx = x + self.alpha * d
            tempfx = self.function(tempx)
            if self.test_constraints(tempx) and fx - tempfx > self.forcing_function(self.alpha * normd):
                return tempx, tempfx, True
        return x, fx, False

    def step(self, x, fx):
        success = False
        while not success:
            x, fx, success = self.search(x, fx)
            if not success:
                x, fx, success = self.poll(x, fx)
            if success:
                self.alpha *= np.random.uniform(1, self.gamma)
            else:
                self.alpha *= np.random.uniform(self.beta_1, self.beta_2)
        return x, fx


class CMAESOptimizer(Optimizer):
    def __init__(self, dim, function, constraints):
        super().__init__(dim, function, constraints)

    def step(self, x, fx):
        # TOCOMPLETE
        raise NotImplementedError
