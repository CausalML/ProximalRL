import json
import os
from collections import defaultdict
from multiprocessing import Queue, Process

import numpy as np

from experiment_setups.toy_setups import noisy_easy_setup, noisy_hard_setup, \
    noisy_optim_setup, mdp_easy_setup, mdp_hard_setup, mdp_optim_setup
from nuisance_estimation.abstract_nuisance import CrossFitNuisance
from utils.hyperparameter_optimization import iterate_placeholder_values, \
    fill_placeholders, fill_global_values

setup_list = [noisy_easy_setup, noisy_hard_setup, noisy_optim_setup,
              mdp_easy_setup, mdp_hard_setup, mdp_optim_setup]
# setup_list = [mdp_easy_setup, mdp_hard_setup, mdp_optim_setup]
# setup_list = [mdp_optim_setup]
save_dir = "results_toy"


def main():
    for setup in setup_list:
        run_experiment(setup)


def run_experiment(setup):
    print("")
    print("STARTING EXPERIMENT:", setup["setup_name"])
    print("")
    results = []

    n_range = sorted(setup["n_range"], reverse=True)
    num_procs = setup["num_procs"]
    num_reps = setup["num_reps"]
    num_jobs = len(n_range) * num_reps

    if num_procs == 1:
        # run jobs sequentially
        for n in n_range:
            for rep_i in range(setup["num_reps"]):
                results.extend(do_job(setup, n, rep_i))
    else:
        # run jobs in separate processes using queue'd system
        jobs_queue = Queue()
        results_queue = Queue()

        for n in n_range:
            for rep_i in range(setup["num_reps"]):
                jobs_queue.put((setup, n, rep_i))

        procs = []
        for i in range(num_procs):
            p = Process(target=run_jobs_loop, args=(jobs_queue, results_queue))
            procs.append(p)
            jobs_queue.put("STOP")
            p.start()

        num_done = 0
        while num_done < num_jobs:
            results.extend(results_queue.get())
            num_done += 1
        for p in procs:
            p.join()

    # build aggregate results
    aggregate_results = build_aggregate_results(results)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "%s_results.json" % setup["setup_name"])
    with open(save_path, "w") as f:
        output = {"results": results, "setup": setup,
                  "aggregate_results": aggregate_results}
        json.dump(output, f, default=lambda c_: c_.__name__,
                  indent=2, sort_keys=True)


def run_jobs_loop(jobs_queue, results_queue):
    for job_args in iter(jobs_queue.get, "STOP"):
        results = do_job(*job_args)
        results_queue.put(results)


def do_job(setup, n, rep_i):
    results = []
    print("setting up scenario for %s setup (n=%d, rep=%d)"
          % (setup["setup_name"], n, rep_i))

    # set up environment, and sample data
    horizon = setup["horizon"]
    pci_reducer = setup["pci_reducer"]["class"](**setup["pci_reducer"]["args"])
    env = setup["environment"]["class"](pci_reducer=pci_reducer,
                                        **setup["environment"]["args"])
    pi_e = setup["target_policy"]["class"](**setup["target_policy"]["args"])
    pci_dataset = env.sample_pci_dataset(horizon=horizon, pi_e=pi_e, n=n)

    gamma = setup["gamma"]
    num_a = env.get_num_a()
    verbose = setup["verbose"]

    # estimate policy value using rollout with pi_e
    if verbose:
        print("")
        print("running oracle rollout estimation")
    target_pv = env.estimate_policy_value_oracle(horizon, pi_e, gamma,
                                                 setup["num_test"])
    if verbose:
        print("oracle policy value estimate:", target_pv)

    if verbose:
        mean_r_array = np.array(
            [pci_dataset.get_r_t(t).mean() for t in range(horizon)])
        gamma_array = np.array([gamma ** i for i in range(horizon)])
        mean_r = (mean_r_array * gamma_array).sum() / gamma_array.sum()
        print("observational mean reward:", mean_r)
        print("")

    # iterate over nuisance methods
    for nuisance_method_template in setup["nuisance_methods"]:
        placeholder_options = nuisance_method_template["placeholder_options"]
        for placeholder_values in iterate_placeholder_values(
                placeholder_options):

            nuisance_method = fill_global_values(
                nuisance_method_template, setup)
            nuisance_method = fill_placeholders(
                nuisance_method, placeholder_values)

            if verbose:
                print("running %s nuisance estimation under %s setup"
                      " (n=%d, rep=%d)" % (nuisance_method["name"],
                                           setup["setup_name"], n, rep_i))
                if placeholder_options:
                    print("using placeholder values for %s method: %r"
                          % (nuisance_method["name"], placeholder_values))

            # set up and fit nuisance method
            num_folds = nuisance_method["num_folds"]
            if num_folds > 1:
                pci_dataset.make_folds(num_folds)
                nuisances = CrossFitNuisance(
                    horizon=horizon, gamma=gamma, num_a=num_a,
                    embed_z=env.embed_z, embed_w=env.embed_w,
                    embed_x=env.embed_x, embed_a=env.embed_a,
                    zxa_sq_dist=env.zxa_sq_dist, wxa_sq_dist=env.wxa_sq_dist,
                    num_folds=num_folds,
                    base_nuisance_class=nuisance_method["class"],
                    base_nuisance_args=nuisance_method["args"])
            else:
                nuisances = nuisance_method["class"](
                    horizon=horizon, gamma=gamma, num_a=num_a,
                    embed_z=env.embed_z, embed_w=env.embed_w,
                    embed_x=env.embed_x, embed_a=env.embed_a,
                    zxa_sq_dist=env.zxa_sq_dist, wxa_sq_dist=env.wxa_sq_dist,
                    num_folds=num_folds, **nuisance_method["args"])
            nuisances.fit(pci_dataset)

            # iterate over psi methods
            if verbose:
                print("psi method results:")
                print("")
            for psi_method in setup["psi_methods"]:
                psi = psi_method["class"](nuisances=nuisances, gamma=gamma,
                                          num_a=num_a, horizon=horizon,
                                          **psi_method["args"])
                pv_estimate = psi.estimate_policy_value(pci_dataset)

                row = {
                    "record_kind": "psi-method",
                    "n": n,
                    "pv_estimate": pv_estimate,
                    "method": "PCIMethod",
                    "target_pv": target_pv,
                    "square_error": (pv_estimate - target_pv) ** 2,
                    "nuisance_method": nuisance_method["name"],
                    "nuisance_placeholders": placeholder_values,
                    "psi_method": psi_method["name"],
                    "rep": rep_i,
                }
                results.append(row)
                if verbose:
                    print(json.dumps(row, sort_keys=True, indent=2))
                    print("")

    # iterate over benchmark methods
    if verbose:
        print("benchmark results:")
        print("")
    for benchmark in setup["benchmark_methods"]:
        method = benchmark["class"](num_a=num_a, gamma=gamma, horizon=horizon,
                                    **benchmark["args"])
        method.fit(pci_dataset)
        pv_estimate = method.estimate(pci_dataset, pi_e)
        row = {
            "record_kind": "benchmark",
            "n": n,
            "pv_estimate": pv_estimate,
            "method": benchmark["name"],
            "target_pv": target_pv,
            "square_error": (pv_estimate - target_pv) ** 2,
            "rep": rep_i,
        }
        results.append(row)
        if verbose:
            print(json.dumps(row, sort_keys=True, indent=2))
            print("")

    return results


def build_aggregate_results(results):
    results_list_collection = defaultdict(list)
    target_pv_list = []

    # put together lists of results for each method
    for row in results:
        if row["record_kind"] == "psi-method":
            # make psi method key
            method_key = row["nuisance_method"] + "__" + row["psi_method"]
            options = row["nuisance_placeholders"]
            if options:
                options_key = "__".join(["%s=%r" % (k, v)
                                         for k, v in sorted(options.items())])
                method_key = method_key + "__" + options_key

        elif row["record_kind"] == "benchmark":
            method_key = row["method"]
        else:
            raise ValueError("Invalid Record Kind: %s" % row["record_kind"])

        key = (row["n"], method_key)
        results_list_collection[key].append(row["pv_estimate"])
        target_pv_list.append(row["target_pv"])

    # compute aggregate statistics
    aggregate_results = {}
    target_pv = float(np.mean(target_pv_list))
    for key, results_list in sorted(results_list_collection.items()):
        n, method_key = key
        results_key = "%05d___%s" % (n, method_key)
        error_array = np.array(results_list) - target_pv
        mse = float((error_array ** 2).mean())
        bias_mean = float(error_array.mean())
        bias_std = float(error_array.std())
        aggregate_results[results_key] = {
            "mse": mse,
            "bias": bias_mean,
            "bias_std": bias_std,
            "pred_pv_mean": float(np.mean(results_list)),
            "target_pv": target_pv,
        }
    return aggregate_results


if __name__ == "__main__":
    main()