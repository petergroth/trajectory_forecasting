import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
import seaborn as sns

if __name__ == '__main__':
    ticksize = 30
    titlesize = 30

    path = f"results/waymo/waymo_results.csv"
    df = pd.read_csv(path)
    # Compute average errors
    processed_results = df.groupby(["misc.model_type", "regressor.min_dist", "misc.regressor_type"])[["test_ade_loss",
                                                         "test_fde_loss",
                                                         "test_vel_loss",
                                                         "test_collision/trajectory",
                                                         "test_mean_nllh",
                                                         "test_ade_ttp_loss",
                                                         "test_fde_ttp_loss",
                                                         "test_ttp_mean_nllh"
                                                         ]].agg(["mean", "std"])



    # Define model names
    model_types = ["Constant baseline", "MPGNN", "RMPGNN", "RMPGNN (RG)", "RMPGNN (RG, UA)"]
    vals = processed_results.values[[6, 4, 5, 2, 3]]
    columns = ["ade_mean", "ade_std", "fde_mean", "fde_std", "or_mean", "or_std", "ade_ttp_mean",
               "ade_ttp_std", "fde_ttp_mean", "fde_ttp_std"]
    df_clean = pd.DataFrame(vals[:, [0, 1, 2, 3, 6, 7, 10, 11, 12, 13]], index=model_types,
                            columns=columns)

    # Plot mean values
    col = ["ade", "fde", "or", "ade_ttp", "fde_ttp"]
    lim = [[0, 5], [0, 20], [0, 0.025], [0, 17], [0, 70]]

    plt.style.use('seaborn')
    plt.rcParams["axes.labelsize"] = titlesize
    plt.rcParams["axes.titlesize"] = titlesize
    plt.rcParams["xtick.labelsize"] = ticksize
    plt.rcParams["ytick.labelsize"] = ticksize
    plt.rcParams["legend.fontsize"] = ticksize
    rc("text", usetex=True)
    rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"]})

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[1, :2])
    axs = [ax1, ax2]
    titles = ["Mean ADE", "Mean FDE"]
    for i, val in enumerate(col[:2]):
        ax = axs[i]
        sns.barplot(x='index', y=f"{val}_mean", yerr=df_clean[f"{val}_std"].values,
                    ax=ax, palette=sns.color_palette("muted"), capsize=.3,
                    data=df_clean.reset_index(), hue='index', dodge=False, errcolor='k')
        ax.set_ylabel(None)
        ax.set_xlabel(None)
        rc("text", usetex=True)
        rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"]})

        ax.set_xticks([])
        if i:
            ax.set_yticks(np.arange(0, 21, 4))
        # plt.savefig(
        #     f"../../thesis/graphics/waymo/results_{col[i]}_full.pdf"
        # )
        if not i:
            ax.legend(loc='upper left', framealpha=0.8, facecolor='w', frameon=True,
                        bbox_to_anchor=(1, 0.4) )
        else:
            ax.legend([])
        ax.set_title(titles[i])

    plt.savefig(
        f"../../thesis/graphics/waymo/results_adefde_full.pdf"
    )
    #
    # plt.show()