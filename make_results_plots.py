import json
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns

# matplotlib.rcParams['text.usetex'] = True

colors = ["orange", "magenta", "blue", "green"]
order = ["Ours", "MDP", "MeanR", "TIS"]

# lmbda_target = 1e-2
# alpha_target = 1e-4
psi_target = "EfficientKernelPCI"
r_tol, a_tol = 1e-3, 0
n_min = 200
horizon = 3.0

method_name_map = {
    "PCIMethod": "Ours",
    "MeanRBenchmark": "MeanR",
    "MDPTabularDirectBenchmark": "MDP",
    "TimeIndependentSamplingEfficientBenchmark": "TIS",
    # "TennenholtzBenchmark": "TMS20",
}

inputs = [
    ("easy", "toy_easy_results", 0.2, 1e-4, 1e-4),
    ("hard", "toy_hard_results", 0.2, 1e-2, 1e-4),
    ("optim", "toy_optim_results", 0.2, 1e-4, 1e-4),
    ("easy", "toy_mdp_easy_results", 0, 1e-4, 1e-4),
    ("hard", "toy_mdp_hard_results", 0, 1e-2, 1e-2),
    ("optim", "toy_mdp_optim_results", 0, 1e-4, 1e-2),
]


def main():
    for pi_name, file_prefix, eps_noise, alpha_target, lmbda_target in inputs:
        if eps_noise == 0:
            title_args = (pi_name, "0")
            path_args = (pi_name, "mdp")
        else:
            title_args = (pi_name, "%.1f" % eps_noise)
            path_args = (pi_name, "pomdp")
        json_path = "results_toy/%s.json" % file_prefix
        pv_save_path = "results_toy/plots/%s_%s_pv.pdf" % path_args
        mse_save_path = "results_toy/plots/%s_%s_mse.pdf" % path_args
        make_results_plot(json_path, pv_save_path, mse_save_path,
                          alpha_target, lmbda_target)


def not_close(a, b):
    return not np.isclose(a, b, rtol=r_tol, atol=a_tol)


def make_results_plot(json_path, pv_save_path, mse_save_path,
                      alpha_target, lmbda_target):
    with open(json_path) as f:
        results = json.load(f)["results"]
    target_pv = horizon * np.mean([row_["target_pv"] for row_ in results])

    pv_rows = []
    mse_rows = []
    n_vals = sorted(set([row_["n"] for row_ in results]))
    for n in n_vals:
        if n < n_min:
            continue
        output_row = {
            "Method": "Target PV",
            "n": n,
            "SE": 0,
            "estimate": target_pv,
        }
        pv_rows.append(output_row)

    for row in results:
        if row["n"] < n_min:
            continue
        if row["record_kind"] == "psi-method":
            if row["psi_method"] != psi_target:
                continue
            alpha = float(row["nuisance_placeholders"]["alpha"])
            lmbda = float(row["nuisance_placeholders"]["lmbda"])
            if not_close(alpha, alpha_target) or not_close(lmbda, lmbda_target):
                continue
        if row["method"] not in method_name_map:
            continue
        pv_estimate = horizon * row["pv_estimate"]
        output_row = {
            "Method": method_name_map[row["method"]],
            "n": row["n"],
            "SE": (pv_estimate - target_pv) ** 2,
            "estimate": pv_estimate,
        }
        pv_rows.append(output_row)
        mse_rows.append(output_row)


    # PV Estimate plot
    fig, ax = plt.subplots(figsize=(10, 4))
    hue_order = ["Target PV"] + order
    pv_df = pd.DataFrame(pv_rows)
    palette = ["black"] + colors
    dashes = [""] + sns._core.unique_dashes(len(hue_order))[1:]
    plot = sns.lineplot(x="n", y="estimate", hue="Method", palette=palette,
                        style="Method", data=pv_df, ci="sd", dashes=dashes,
                        hue_order=hue_order, style_order=hue_order)
    # title = "$\\pi_e$: %s, $\\epsilon_{\\textup{noise}}$: $%s$" % title_args
    # title = "Policy: %s, Noise: %s" % title_args
    # ax.set_title(title, fontsize=16)
    ax.set_xlabel("Training Set Size", fontsize=24)
    ax.set_ylabel("PV Estimate", fontsize=24)
    ax.set_ylim([-10, 10])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)

    ax.set_xscale("log")
    # ax.xaxis.set_major_formatter(ScalarFormatter())
    # ax.set_yscale("log")
    # ax.yaxis.set_major_formatter(ScalarFormatter())

    plot.legend(prop={'size': 14}, loc="lower left",
                bbox_to_anchor=(0.01, 0.01),
                framealpha=1.0)
    fig.savefig(pv_save_path, bbox_inches="tight")

    # MSE plot
    fig, ax = plt.subplots(figsize=(10, 4))
    hue_order = order
    mse_df = pd.DataFrame(mse_rows)
    palette = colors
    dashes = sns._core.unique_dashes(len(hue_order)+1)[1:]
    plot = sns.lineplot(x="n", y="SE", hue="Method", palette=palette,
                        style="Method", data=mse_df, ci=95, dashes=dashes,
                        hue_order=hue_order, style_order=hue_order,
                        legend=False)
    # title = "$\\pi_e$: %s, $\\epsilon_{\\textup{noise}}$: $%s$" % title_args
    # title = "Policy: %s, Noise: %s" % title_args
    # ax.set_title(title, fontsize=16)
    ax.set_xlabel("Training Set Size", fontsize=24)
    ax.set_ylabel("MSE", fontsize=24)

    ax.set_xscale("log")
    # ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_yscale("log")
    # ax.yaxis.set_major_formatter(ScalarFormatter())
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)

    # plot.legend(prop={'size': 12}, loc="upper right",
    #             bbox_to_anchor=(0.99, 0.99),
    #             framealpha=1.0)
    fig.savefig(mse_save_path, bbox_inches="tight")



if __name__ == "__main__":
    main()
