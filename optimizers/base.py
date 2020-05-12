import numpy as np
import sys
import copy
np.set_printoptions(threshold=float('inf'), linewidth=1000, suppress=True, precision=2)


def print_row(*strings, width=[11, 20, 20, 20, 50], header=False):
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

    def __init__(self, dim, function, constraints, name, max_iter, ftol, xtol):
        self.dim = dim
        self.function = function
        self.constraints = constraints
        self.it = 1
        self.max_iter = max_iter
        self.ftol = ftol
        self.xtol = xtol
        self.name = name

    def step(self, x, fx):
        raise NotImplementedError

    def stop_criteria(self):
        raise NotImplementedError

    def test_constraints(self, x):  # Returns False if a constraint is violated, True otherwise
        return self.constraints.test(x)

    def optimize(self, x0, verbose=False):
        self.x = copy.deepcopy(x0)
        self.fx = self.function(self.x)
        self.it = 1
        self.track = [(self.x, self.fx, None, None)]
        self.verbose = verbose
        print('\n%s\n' % ('  {} Optimization Starting  '.format(self.name.title()).center(135, '•')))
        if self.verbose:
            print_row('Iteration', 'f(x)', '||xk-xk-1||', '|f(xk)-f(xk-1)|', 'x', header=True)
            print_row('0', '%3.3f' % self.fx, 'nan', 'nan', '%s' % str(self.x.reshape(-1)))
        while True:
            self.x_next, self.fx_next = self.step(self.x, self.fx)
            self.xdiff = np.linalg.norm(self.x_next - self.x)
            self.fdiff = abs(self.fx_next - self.fx)
            self.track.append((self.x_next, self.fx_next, self.xdiff, self.fdiff))

            finish, reason = self.stop_criteria()
            if self.verbose and not (reason is not None and "constraints violated" in reason):
                print_row('%d' % self.it, '%3.3f' % self.fx_next, '%3.3f' % self.xdiff, '%3.3f' % self.fdiff, '%s' % str(self.x_next.reshape(-1)))
            if finish:
                if not "constraints violated" in reason:
                    self.x, self.fx = self.x_next, self.fx_next
                break
            self.x, self.fx = self.x_next, self.fx_next
            self.it += 1

        reason = reason.upper().center(len(reason) + 4, ' ').center(135, '>')
        reason = reason[:int(len(reason) / 2)] + reason[int(len(reason) / 2):].replace('>', '<')

        print('%s\n' % reason)

        print('\n%s\n' % ('Final Results After {:4d} Iterations'.format(self.it).center(135, '-')))
        print_row('f(x)', '%3.3f' % self.fx, 'x', '%s' % str(self.x.reshape(-1)), width=[10, 20, 10, 50], header=True)
        return self.track
