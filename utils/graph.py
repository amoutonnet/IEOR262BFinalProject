import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_track(track, opt, name):
    xs, fxs, timeperiter, _, _, xoptdiffs, foptdiffs = map(np.array, zip(*track))
    xs = np.sum(np.abs(np.squeeze(xs))**2, axis=-1)**(1. / 2)
    plt.rc('grid', linestyle="--", color='black', alpha=0.5)
    fig, ax = plt.subplots(nrows=1, ncols=5, sharex=True)
    ax[0].plot(xs, linewidth=1, alpha=0.9)
    ax[0].set_ylabel(r'$||x||$')
    ax[0].grid()
    ax[1].plot(fxs, linewidth=1, alpha=0.9)
    ax[1].set_ylabel(r'$f(x)$')
    ax[1].grid()
    ax[2].plot(timeperiter, linewidth=1, alpha=0.9)
    ax[2].set_ylabel('Time per Iteration (s)')
    ax[2].grid()
    # ax[0, 1].set_yscale('log')
    ax[3].plot(xoptdiffs, linewidth=1, alpha=0.9)
    ax[3].set_ylabel(r'$||x_k - x^*||$')
    ax[3].set_xlabel('steps')
    ax[3].grid()
    ax[4].plot(foptdiffs, linewidth=1, alpha=0.9)
    ax[4].set_ylabel(r'$|f(x_k) - f(x^*)|$')
    ax[4].set_xlabel('steps')
    ax[4].grid()
    fig.suptitle('Evolution of the Optimization of function {:s} with {:s}'.format(name, opt))
    # fig.tight_layout(rect=[0, 0, 1, 0.95])


def plot_box(data, func_names, n_iter):

    fig, axes = plt.subplots(2, 2, sharex=True)

    infos = {
        "Timestamp": (axes[0, 0], "Time (s)", "Total Time per Optimization"),
        "NbIter": (axes[0, 1], "Iterations per Optimization", "Total Iterations per Optimization"),
        "Finalxoptdiff": (axes[1, 0], r"$\|x_{final}-x^*\|$", r"Final $\|x_k-x^*\|$"),
        "Finalfoptdiff": (axes[1, 1], r"$|f(x_{final})-f(x^*)|$", r"Final $|f(x_k)-f(x^*)|$"),
    }

    for col in data.columns:
        if col != 'Optimizer':
            ax = infos[col][0]
            sns.boxplot(x=data['Optimizer'], y=data[col], showmeans=True, ax=ax)
            sns.stripplot(x=data['Optimizer'], y=data[col], color="grey", jitter=0.2, size=2.5, ax=ax)
            ax.set_ylabel(infos[col][1])
            if col in ["Finalxoptdiff", "Finalfoptdiff"]:
                ax.set_yscale("log")
            ax.set_title(infos[col][2])

    # namefunctions = ', '.join(func_names) + " functions" if len(func_names) > 1 else "%s function" % func_names[0]

    fig.suptitle("Performances of Optimizers over all functions\n(Means and Confidence Intervals processed over %d Optimizations)" % (n_iter))
