import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_track(track, opt, name):
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
    # ax[0, 1].set_yscale('log')
    ax[1, 0].plot(xdiffs, linewidth=1, alpha=0.9)
    ax[1, 0].set_ylabel(r'$||x_k - x_{k-1}||$')
    ax[1, 0].set_xlabel('steps')
    ax[1, 0].grid()
    ax[1, 1].plot(fdiffs, linewidth=1, alpha=0.9)
    ax[1, 1].set_ylabel(r'$|f(x_k) - f(x_{k-1})|$')
    ax[1, 1].set_xlabel('steps')
    ax[1, 1].grid()
    fig.suptitle('Evolution of the Optimization of function {:s} with {:s}'.format(name, opt))
    fig.tight_layout(rect=[0, 0, 1, 0.95])

def plot_box(data, func_name):
    plt.rc('grid', linestyle="--", color='black', alpha=0.5)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    sns.boxplot(x=data['Optimizer'], y=data['Timestamp'], showmeans=True, ax=ax[0])
    sns.stripplot(x=data['Optimizer'], y=data['Timestamp'], color="grey", jitter=0.2, size=2.5, ax=ax[0])
    ax[0].set_ylabel('Time (s)')
    sns.boxplot(x=data['Optimizer'], y=data['NbIter'], showmeans=True, ax=ax[1])
    sns.stripplot(x=data['Optimizer'], y=data['NbIter'], data=data, color="grey", jitter=0.2, size=2.5, ax=ax[1])
    ax[1].set_ylabel('Iterations per optimization')
    fig.suptitle("Time and Iterations per optimization and per Optimizer for function {:s}".format(func_name))
    fig.legend()
