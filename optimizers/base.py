import numpy as np
import sys
import copy
import time
np.set_printoptions(threshold=float('inf'), linewidth=1000, suppress=True, precision=5)


def print_row(*strings, width=[11, 16, 16, 16, 16, 16, 40], header=False):
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

    def __init__(self, dim, function, constraints, getoptinfo, name, max_iter, ftol, xtol):
        self.dim = dim
        self.function = function
        self.constraints = constraints
        self.getoptinfo = getoptinfo
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
        info = self.getoptinfo(self.x, self.fx)
        self.track = [(self.x, self.fx, None, None, None, info[0], info[1])]
        self.verbose = verbose
        if self.verbose:
            print('\n%s\n' % ('  {} Optimization Starting  '.format(self.name).center(135, '•')))
            print_row('Iteration', '||xk-x*||', '|f(xk)-f(x*)|', '||xk-xk-1||', '|f(xk)-f(xk-1)|', 'f(x)', 'x', header=True)
            print_row('0', '%3.5f' % info[0], '%3.5f' % info[1], 'nan', 'nan', '%3.5f' % self.fx, '%s' % str(self.x.reshape(-1)))
        while True:
            start = time.time()
            self.x_next, self.fx_next = self.step(self.x, self.fx)
            end = time.time()
            self.xdiff = np.linalg.norm(self.x_next - self.x)
            self.fdiff = abs(self.fx_next - self.fx)
            info = self.getoptinfo(self.x_next, self.fx_next)

            finish, reason = self.stop_criteria()
            if self.verbose and not (reason is not None and "constraints violated" in reason):
                print_row('%d' % self.it, '%3.5f' % info[0], '%3.5f' % info[1], '%3.5f' % self.xdiff, '%3.5f' % self.fdiff, '%3.5f' % self.fx_next, '%s' % str(self.x_next.reshape(-1)))
            if finish:
                if not "constraints violated" in reason:
                    self.track.append((self.x_next, self.fx_next, end - start, self.xdiff, self.fdiff, info[0], info[1]))
                    self.x, self.fx = self.x_next, self.fx_next
                break
            else:
                self.track.append((self.x_next, self.fx_next, end - start, self.xdiff, self.fdiff, info[0], info[1]))
            self.x, self.fx = self.x_next, self.fx_next
            self.it += 1

        if self.verbose:
            reason = reason.upper().center(len(reason) + 4, ' ').center(151, '>')
            reason = reason[:int(len(reason) / 2)] + reason[int(len(reason) / 2):].replace('>', '<')

            print('%s\n' % reason)

            print('\n%s\n' % ('Final Results After {:4d} Iterations'.format(self.it).center(151, '-')))
            print_row('||xk-x*||', '|f(xk)-f(x*)|', 'f(x)', 'x', width=[16, 16, 16, 40], header=True)
            print_row('%3.5f' % self.track[-1][-2], '%3.5f' % self.track[-1][-1], '%3.5f' % self.track[-1][1], '%s' % str(self.track[-1][0].reshape(-1)), width=[16, 16, 16, 40])
            print('-' * 99)
        return self.track
