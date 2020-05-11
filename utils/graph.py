import numpy as np
import matplotlib.pyplot as plt


def plot_track(track, opt):
    xs, fxs, xdiffs, fdiffs = map(np.array, zip(*track))
    xs = np.sum(np.abs(np.squeeze(xs))**2, axis=-1)**(1. / 2)
    plt.rc('grid', linestyle="--", color='black', alpha=0.5)
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True)
    ax[0, 0].plot(xs, linewidth=1, alpha=0.9)
    ax[0, 0].set_ylabel(r'$||x||$')
    ax[0, 0].grid()
    ax[0, 1].plot(fxs, linewidth=1, alpha=0.9)
    ax[0, 1].set_ylabel(r'$f(x)$')
    ax[0, 1].grid()
    ax[1, 0].plot(xdiffs, linewidth=1, alpha=0.9)
    ax[1, 0].set_ylabel(r'$||x_k - x_{k-1}||$')
    ax[1, 0].set_xlabel('steps')
    ax[1, 0].grid()
    ax[1, 1].plot(fdiffs, linewidth=1, alpha=0.9)
    ax[1, 1].set_ylabel(r'$|f(x_k) - f(x_{k-1})|$')
    ax[1, 1].set_xlabel('steps')
    ax[1, 1].grid()
    fig.suptitle('Evolution of the Optimization with {:s}'.format(opt))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
