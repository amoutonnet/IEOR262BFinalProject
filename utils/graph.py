import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_track(track, opt, name):
    xs, fxs, timeperiter, _, _, xoptdiffs, foptdiffs = map(np.array, zip(*track))
    xs = np.sum(np.abs(np.squeeze(xs))**2, axis=-1)**(1. / 2)
    plt.rc('grid', linestyle="--", color='black', alpha=0.5)
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True)
    ax[0, 0].plot(xs, linewidth=1, alpha=0.9)
    ax[0, 0].set_ylabel(r'$||x||$')
    ax[0, 0].grid()
    ax[0, 1].plot(fxs, linewidth=1, alpha=0.9)
    ax[0, 1].set_ylabel(r'$f(x)$')
    ax[0, 1].grid()
    # ax[0, 2]
    ax[1, 0].plot(xoptdiffs, linewidth=1, alpha=0.9)
    ax[1, 0].set_ylabel(r'$||x_k - x^*||$')
    ax[1, 0].set_xlabel('steps')
    ax[1, 0].set_yscale('log')
    ax[1, 0].grid()
    ax[1, 1].plot(foptdiffs, linewidth=1, alpha=0.9)
    ax[1, 1].set_ylabel(r'$|f(x_k) - f(x^*)|$')
    ax[1, 1].set_xlabel('steps')
    ax[1, 1].set_yscale('log')
    ax[1, 1].grid()
    ax[1, 2].plot(timeperiter, linewidth=1, alpha=0.9)
    ax[1, 2].set_ylabel('Time per Iteration (s)')
    ax[1, 2].grid()
    fig.suptitle('Evolution of the Optimization of function {:s} with {:s}'.format(name, opt))
    # fig.tight_layout(rect=[0, 0, 1, 0.95])


def plot_box(data, func_names, n_iter, cv_threshold=1e-3):
    plt.style.use("seaborn-darkgrid")
    fig, axes = plt.subplots(nrows=2, ncols=2)

    infos = {
        "Timestamp": (axes[0, 0], "Time (s)", "Total Time per Optimization that converged"),
        "NbIter": (axes[0, 1], "Iterations per Optimization", "Total Iterations per Optimization that converged"),
        "Finalxoptdiff": (axes[1, 0], r"$\|x_{final}-x^*\|$", r"Final $\|x_k-x^*\|$"),
        "Finalfoptdiff": (axes[1, 1], r"$|f(x_{final})-f(x^*)|$", r"Final $|f(x_k)-f(x^*)|$"),
    }

    boxprops = {'edgecolor': 'k', 'facecolor': 'w'}
    for col in data.columns:
        if col != 'Optimizer' and col != 'Function':
            ax = infos[col][0]
            if col in ["Finalxoptdiff", "Finalfoptdiff"]:
                sns.boxplot(x=data['Optimizer'], y=data[col], ax=ax, boxprops=boxprops)
                sns.stripplot(x=data['Optimizer'], y=data[col], jitter=0.2, size=2.5, ax=ax, hue=data['Function'], palette="Set2")
                ax.axhline(y=cv_threshold, color='r', linestyle='--', label='Convergence threshold', linewidth=0.7)
                ax.set_yscale("log")
            else:
                data_cv = data.loc[(data["Finalfoptdiff"] <= cv_threshold) & (data["Finalxoptdiff"] <= cv_threshold)]  # keep only optimizations that converged for stripplot
                sns.boxplot(x=data['Optimizer'], y=data_cv[col], ax=ax, boxprops=boxprops)
                sns.stripplot(x=data_cv['Optimizer'], y=data_cv[col], jitter=0.2, size=2.5, ax=ax, hue=data['Function'], palette="Set2")
            ax.set_ylabel(infos[col][1])
            ax.set_title(infos[col][2])
            if col != "Finalfoptdiff":
                ax.legend_.remove()  # removing subplots legends
    handles, labels = infos["Finalfoptdiff"][0].get_legend_handles_labels()  # Savign labels for whole figure legend
    infos["Finalfoptdiff"][0].legend_.remove()  # removing subplot legend

    # Whole figure legend
    fig.legend(handles, labels, ncol=len(func_names) + 1, loc='upper center', bbox_to_anchor=(0.5, 0.95), prop={'size': 9})

    fig.suptitle("Performances of Optimizers processed over %d Optimizations" % (n_iter))

def print_table(data, opts_list, func_names, n_iter, cv_matrix, print_setup="{:.1f}%", table_title="Table", latex=False):
    width = 15
    add_width_first_col = 10
    nb_col = (len(func_names) + 1)
    total_line_width = nb_col * (width + 1) + add_width_first_col + 1
    for name in func_names: # adjusting total line width for title
        if len(name) + 2 > width:
            total_line_width += len(name) + 2 - width
    print("-" * (total_line_width))
    print("|" + table_title.center(total_line_width - 2) + "|")
    print("-" * (total_line_width))
    first_line = "|" + "Optimizer".center(width + add_width_first_col) + "|"
    for name in func_names:
        first_line += name.center(max(width, len(name) + 2)) + "|"
    print(first_line)
    print("-" * (total_line_width))
    for opt in opts_list:
        line = "|" + opt.center(width + add_width_first_col) + "|"
        for name in func_names:
            if cv_matrix.loc[opt, name]:
                line += print_setup.format(data.loc[opt, name]).center(max(width, len(name) + 2)) + "|"
            else:
                if "%" in print_setup:
                    line += print_setup.format(0).center(max(width, len(name) + 2)) + "|"
                else:
                    line += "/".center(max(width, len(name) + 2)) + "|"
        print(line)
    print("-" * (total_line_width) + "\n")

    if latex:
        print(table_title)
        first_line = "Optimizer"
        for name in func_names:
            first_line += " & " + name
        print(first_line + " \\\\")
        for opt in opts_list:
            line = opt
            for name in func_names:
                if cv_matrix.loc[opt, name]:
                    line += " & " + print_setup.format(data.loc[opt, name])
                else:
                    if "%" in print_setup:
                        line += " & " + print_setup.format(0)
                    else:
                        line += " & /"
            print(line.replace("%", "\%") + " \\\\")

def print_statistics(data, opts_list, func_names, n_iter, cv_threshold, latex=False):
    data_cv = data.loc[(data["Finalfoptdiff"] <= cv_threshold) & (data["Finalxoptdiff"] <= cv_threshold)]
    cv_stats = pd.pivot_table(data_cv, values='Finalfoptdiff', index='Optimizer', columns='Function', aggfunc=len, fill_value=0)
    cv_stats = cv_stats.multiply(100/n_iter)
    time_stats = pd.pivot_table(data_cv, values=['Timestamp', 'NbIter'], index='Optimizer', columns='Function', aggfunc=np.mean, fill_value=0)
    
    cv_matrix = pd.DataFrame(np.zeros((len(opts_list), len(func_names)), dtype=bool), columns=func_names, index=opts_list)
    for opt in opts_list:
        for name in func_names:
            if opt in cv_stats.index and name in cv_stats.columns:
                cv_matrix.loc[opt, name] = cv_stats.loc[opt, name] > 0

    print_table(cv_stats,
        opts_list,
        func_names,
        n_iter,
        cv_matrix,
        print_setup="{:.0f}%",
        table_title="Percentage of  Optimization that converged",
        latex=latex
        )
    print_table(time_stats['Timestamp'],
        opts_list,
        func_names,
        n_iter,
        cv_matrix,
        print_setup="{:.3f}s",
        table_title="Average duration of optimizations that converged (Time in seconds)",
        latex=latex
        )
    print_table(time_stats['NbIter'],
        opts_list,
        func_names,
        n_iter,
        cv_matrix,
        print_setup="{:.0f}",
        table_title="Average number of iterations per optimization that converged",
        latex=latex
        )