from benchmarks.mdp_tabular_direct_benchmark import MDPTabularDirectBenchmark
from benchmarks.mean_r_benchmark import MeanRBenchmark
from benchmarks.tennenholtz_benchmark import TennenholtzBenchmark
from benchmarks.time_independent_sampling_efficient_benchmark import \
    TimeIndependentSamplingEfficientBenchmark
from environments.pci_reducer import CurrPrevObsPCIReducer
from environments.toy_environment import ToyEnvironment, \
    ToyEvaluationPolicyEasy, ToyEvaluationPolicyHard, ToyEvaluationPolicyOptim
from methods.direct_pci_method import DirectPCIMethod
from methods.efficient_pci_method import EfficientPCIMethod
from methods.is_pci_method import ImportanceSamplingPCIMethod
from nuisance_estimation.discrete_single_kernel_nuisance_estimation import \
    DiscreteSingleKNuisanceEstimation
from utils.hyperparameter_optimization import HyperparameterPlaceholder
from utils.kernels import TripleMedianKernel
from utils.neural_nets import TabularNet

psi_methods = [
    {
        "class": EfficientPCIMethod,
        "name": "EfficientKernelPCI",
        "args": {},
    },
    {
        "class": ImportanceSamplingPCIMethod,
        "name": "ImportanceSamplingKernelPCI",
        "args": {},
    },
    {
        "class": DirectPCIMethod,
        "name": "DirectKernelPCI",
        "args": {},
    },
]


nuisance_methods = [
    {
        "class": DiscreteSingleKNuisanceEstimation,
        "name": "SingleKernelNuisance",
        "placeholder_options": {"alpha": [1e-2, 1e-4, 1e-6, 1e-8],
                                "lmbda": [1e0, 1e-2, 1e-4, 1e-6]},
        "num_folds": 5,
        "args": {
            "q_net_class": TabularNet,
            "q_net_args": {},
            "g_kernel_class": TripleMedianKernel,
            "g_kernel_args": {},
            "h_net_class": TabularNet,
            "h_net_args": {},
            "f_kernel_class": TripleMedianKernel,
            "f_kernel_args": {},
            "q_alpha": HyperparameterPlaceholder("alpha"),
            "h_alpha": HyperparameterPlaceholder("alpha"),
            "q_lmbda": HyperparameterPlaceholder("lmbda"),
            "h_lmbda": HyperparameterPlaceholder("lmbda"),
            "num_rep": 2,
        },
    },
]


benchmark_methods = [
    {
        "class": MDPTabularDirectBenchmark,
        "name": "MDPTabularDirectBenchmark",
        "args": {},
    },
    {
        "class": MeanRBenchmark,
        "name": "MeanRBenchmark",
        "args": {},
    },
    {
        "class": TennenholtzBenchmark,
        "name": "TennenholtzBenchmark",
        "args": {},
    },
    # {
    #     "class": TimeIndependentSamplingBenchmark,
    #     "name": "TimeIndependentSamplingBenchmark",
    #     "args": {},
    # },
    {
        "class": TimeIndependentSamplingEfficientBenchmark,
        "name": "TimeIndependentSamplingEfficientBenchmark",
        "args": {},
    },
]


n_range = [10000, 5000, 2000, 1000, 500, 200, 100]
num_test = 10000
# num_test = 1000
horizon = 3
gamma = 1.0
num_reps = 100
num_procs = 1

easy_policy = {"class": ToyEvaluationPolicyEasy, "args": {}}
hard_policy = {"class": ToyEvaluationPolicyHard, "args": {}}
optim_policy = {"class": ToyEvaluationPolicyOptim, "args": {}}

noisy_env = {"class": ToyEnvironment, "args": {"eps_noise": 0.2}}
mdp_env = {"class": ToyEnvironment, "args": {"eps_noise": 0}}

general_setup = {
    "pci_reducer": {
        "class": CurrPrevObsPCIReducer,
        "args": {},
    },
    "n_range": n_range,
    "num_test": num_test,
    "horizon": horizon,
    "gamma": gamma,
    "verbose": False,
    "num_reps": num_reps,
    "num_procs": num_procs,
    "psi_methods": psi_methods,
    "nuisance_methods": nuisance_methods,
    "benchmark_methods": benchmark_methods,
}

noisy_easy_setup = {"setup_name": "toy_easy", "target_policy": easy_policy,
                    "environment": noisy_env, **general_setup}
noisy_hard_setup = {"setup_name": "toy_hard", "target_policy": hard_policy,
                    "environment": noisy_env, **general_setup}
noisy_optim_setup = {"setup_name": "toy_optim", "target_policy": optim_policy,
                     "environment": noisy_env, **general_setup}

mdp_easy_setup = {"setup_name": "toy_mdp_easy", "target_policy": easy_policy,
                  "environment": mdp_env, **general_setup}
mdp_hard_setup = {"setup_name": "toy_mdp_hard", "target_policy": hard_policy,
                  "environment": mdp_env, **general_setup}
mdp_optim_setup = {"setup_name": "toy_mdp_optim", "target_policy": optim_policy,
                   "environment": mdp_env, **general_setup}
