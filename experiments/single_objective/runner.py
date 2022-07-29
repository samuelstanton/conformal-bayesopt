import random
import numpy as np
import torch
import time
import math
import os

from scipy.stats import norm

from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qUpperConfidenceBound,
)
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.objective import IdentityMCObjective
from botorch import fit_gpytorch_model
from botorch.sampling.samplers import IIDNormalSampler

import sys
sys.path.append("../../conformalbo/")
from helpers import assess_coverage
from acquisitions import (
    qConformalExpectedImprovement,
    qConformalNoisyExpectedImprovement,
    qConformalUpperConfidenceBound,
    qConformalKnowledgeGradient,
)
from ratio_estimation import RatioEstimator
from utils import (
    generate_initial_data,
    initialize_model,
    parse,
    optimize_acqf_and_get_observation,
    update_random_observations,
    get_problem,
    initialize_noise_se,
)
sys.path.append("../../")
from lambo.utils import DataSplit, update_splits
from conformalbo.trbo import TurboState, update_state, get_tr_bounds


def main(
    seed: int = 0,
    dim: int = 10,
    method: str = "exact",
    batch_size: int = 3,
    n_batch: int = 50,
    tgt_grid_res: int = 64,
    temp: float = 1e-2,
    mc_samples: int = 256,
    num_init: int = 10,
    noise_se: float = 0.1,
    dtype: str = "double",
    verbose: bool = True,
    output: str = None,
    problem: str = None,
    min_alpha: float = 0.05,
    max_grid_refinements: int = 4,
    sgld_steps: int = 100,
    sgld_temperature: float = 1e-3,
    sgld_lr: int = 1e-3,
    rand_orthant: bool = False,
):
    dtype = torch.double if dtype == "double" else torch.float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    bb_fn = get_problem(problem, dim)
    bb_fn = bb_fn.to(device, dtype)
    # we manually optimize in [0,1]^d
    bounds = torch.zeros_like(bb_fn.bounds)
    bounds[1] += 1.
    print(f"function: {problem}, x bounds: {bounds}")

    # keys = ["cucb", "cei", "cnei", "ucb", "ei", "nei", "rnd"]
    keys = ["tr_ei", "cei", "ei", "rnd"]
    best_actual = {k: [] for k in keys}
    acq_max = {k: [] for k in keys}
    iid_coverage = {k: [] for k in keys}
    query_coverage = {k: [] for k in keys}
    global_states = {k: TurboState(
        batch_size=batch_size,
        dim=dim,
    ) for k in keys}

    # initialize noise se
    problem_noise_se = initialize_noise_se(bb_fn, noise_se, device=device, dtype=dtype)

    # call helper functions to generate initial training data and initialize model
    (
        all_inputs,
        all_targets,
        best_actual_obj,
    ) = generate_initial_data(num_init, bb_fn, problem_noise_se, device, dtype,
            rand_orthant=rand_orthant,
    )

    data_dict = {}
    for k in keys:
        best_actual[k].append(best_actual_obj)
        data_dict[k] = (all_inputs, all_targets)

    optimize_acqf_kwargs = {
        "bounds": bounds,
        "BATCH_SIZE": batch_size,
        "fn": bb_fn,
        "noise_se": problem_noise_se,
    }
    
    # the implicit amount of noise changes by iteration
    current_noise_se = noise_se

    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(1, n_batch + 1):
        print("\nStarting iteration: ", iteration, "mem allocated: ",
                torch.cuda.memory_reserved() / 1024**3)
        t0 = time.time()
        for k in keys:
            torch.cuda.empty_cache()
            # os.system("nvidia-smi")

            if k == "rnd":
                # update random
                best_actual[k] = update_random_observations(
                    batch_size, best_actual[k], bb_fn.bounds, bb_fn, dim=bounds.shape[1],
                    noise_se=problem_noise_se,
                )
                continue

            # get the data
            all_inputs, all_targets = data_dict[k]
            perm = np.random.permutation(np.arange(all_inputs.size(0)))
            num_total = all_inputs.size(0)
            num_test = math.ceil(0.2 * num_total)
            num_train = num_total - num_test
            # print(f"train: {num_train}, test: {num_test}")

            test_inputs, test_targets = all_inputs[perm][:num_test], all_targets[perm][:num_test]
            train_inputs, train_targets = all_inputs[perm][num_test:], all_targets[perm][num_test:]
            
            current_noise_se = (problem_noise_se / train_targets.std(0)).clamp(min=0.01).item()
            print("noise se is now: ", current_noise_se)

            # prepare new model, transform
            mll, model, trans = initialize_model(
                train_inputs,
                train_targets,
                method=method,
                train_yvar=current_noise_se**2,
            )
            model.requires_grad_(True)
            fit_gpytorch_model(mll)
            model.requires_grad_(False)

            # transform test targets
            trans.eval()
            test_targets = trans(test_targets)[0]

            alpha = max(1.0 / math.sqrt(num_train), min_alpha)
            conformal_kwargs = dict(
                alpha=alpha,
                grid_res=tgt_grid_res,
                max_grid_refinements=max_grid_refinements,
                ratio_estimator=None,
            )
            conformal_opt_kwargs = dict(
                temperature=sgld_temperature,
                lr=sgld_lr,
                sgld_steps=sgld_steps,
            )

            torch.cuda.empty_cache()

            # now assess coverage on an IID heldout set
            conformal_kwargs['temp'] = 1e-6  # set temp to low value when evaluating coverage
            iid_coverage[k].append(
                assess_coverage(model, test_inputs, test_targets, **conformal_kwargs)
            )
            last_cvrg = iid_coverage[k][-1]
            print(
                f"IID coverage ({k}): credible {last_cvrg[0]:0.4f}, conformal {last_cvrg[1]:0.4f}, "
                f"target level: {1 - alpha:0.4f}"
            )
            model.train()
            torch.cuda.empty_cache()

            # don't fix noise, scale is wrong
            mll, model, trans = initialize_model(
                all_inputs,
                all_targets,
                method=method,
                train_yvar=current_noise_se**2,
            )
            model.requires_grad_(True)
            fit_gpytorch_model(mll)
            model.requires_grad_(False)
            trans.eval()

            if k == 'tr_ei':
                optimize_acqf_kwargs["bounds"] = get_tr_bounds(all_inputs, all_targets, model, global_states[k])
            else:
                optimize_acqf_kwargs["bounds"] = bounds

            # now prepare the acquisition
            batch_range = (0, -3) if k[0] == "c" else (0, -2)
            sampler = IIDNormalSampler(num_samples=mc_samples, batch_range=batch_range)
            base_kwargs = dict(
                model=model,
                sampler=sampler,
            )

            # prepare density ratio estimator
            rx_estimator = RatioEstimator(all_inputs.size(-1), device, dtype)
            rx_estimator.dataset._update_splits(
                DataSplit(all_inputs.cpu(), torch.zeros(all_inputs.size(0), 1))
            )

            alpha = max(1.0 / math.sqrt(num_total), min_alpha)
            conformal_kwargs['alpha'] = alpha
            conformal_kwargs['temp'] = temp
            conformal_kwargs['max_grid_refinements'] = 0
            conformal_kwargs['ratio_estimator'] = rx_estimator

            if k == "ei" or k == 'tr_ei':
                acqf = qExpectedImprovement(
                    **base_kwargs,
                    best_f=trans(all_targets)[0].max(),
                )
            elif k == "nei":
                acqf = qNoisyExpectedImprovement(
                    **base_kwargs,
                    X_baseline=all_inputs,
                    prune_baseline=True,
                )
            elif k == "ucb":
                acqf = qUpperConfidenceBound(
                    **base_kwargs,
                    beta=norm.ppf(1. - alpha / 2.),
                )
            elif k == "kg":
                acqf = qKnowledgeGradient(
                    **base_kwargs,
                    # current_value=trans(all_targets)[0].max(),
                    num_fantasies=None,
                    objective=IdentityMCObjective(),
                    inner_sampler=IIDNormalSampler(mc_samples),
                )
            elif k == "cei":
                acqf = qConformalExpectedImprovement(
                    **conformal_kwargs,
                    **base_kwargs,
                    best_f=trans(all_targets)[0].max(),
                )
            elif k == "cnei":
                acqf = qConformalNoisyExpectedImprovement(
                    **conformal_kwargs,
                    **base_kwargs,
                    X_baseline=all_inputs,
                    prune_baseline=True,
                )
            elif k == "cucb":
                acqf = qConformalUpperConfidenceBound(
                    **conformal_kwargs,
                    optimistic=True,
                    **base_kwargs,
                    beta=norm.ppf(1. - alpha / 2.),
                )
            elif k == "ckg":
                acqf = qConformalKnowledgeGradient(
                    **conformal_kwargs,
                    **base_kwargs,
                    # current_value=trans(all_targets)[0].max(),
                    num_fantasies=None,
                    objective=IdentityMCObjective(),
                    inner_sampler=IIDNormalSampler(mc_samples, batch_range=(0, -3)),
                )

            # optimize acquisition
            if k[0] == "c":
                optimize_acqf_kwargs = {**optimize_acqf_kwargs, **conformal_opt_kwargs}

            new_x, new_y, new_f, new_a, all_x, all_y = optimize_acqf_and_get_observation(
                acqf, **optimize_acqf_kwargs
            )

            if k == 'tr_ei':
                global_states[k] = update_state(global_states[k], new_y)

            # evaluate coverage on query candidates
            sample_idxs = np.random.permutation(all_x.shape[0])[:math.ceil(num_test / batch_size)]
            test_inputs = torch.cat(
                [new_x, all_x[sample_idxs].flatten(0, -2)]
            )
            test_targets = trans(torch.cat(
                [new_y, all_y[sample_idxs].flatten(0, -2)]
            ))[0]
            conformal_kwargs['temp'] = 1e-6  # set temp to low value when evaluating coverage
            conformal_kwargs['max_grid_refinements'] = max_grid_refinements
            query_coverage[k].append(
                assess_coverage(model, test_inputs, test_targets, **conformal_kwargs)
            )
            last_cvrg = query_coverage[k][-1]
            print(
                f"query coverage ({k}): credible {last_cvrg[0]:0.4f}, conformal {last_cvrg[1]:0.4f}, "
                f"target level: {1 - alpha:0.4f}"
            )

            # free up GPU memory
            del acqf
            model.train()
            del model
            torch.cuda.empty_cache()
        
            best_actual[k].append(
                max(best_actual[k][-1], new_f.max().item())
            )
            acq_max[k].append(new_a.item())

            # update dataset
            all_inputs = torch.cat([all_inputs, new_x])
            all_targets = torch.cat([all_targets, new_y])
            data_dict[k] = (all_inputs, all_targets)

        t1 = time.time()

        # best = {key: val[-1] for key, val in best_actual.items()}
        if verbose:
            print(f"\nBatch: {iteration:>2}, time: {t1-t0:>4.2f}, alpha: {alpha:0.4f}")
            # print("max acq val:")
            # [print(f"{key}: {val[-1]:0.4f}") for key, val in acq_max.items() if len(val)]
            print("best score so far:")
            [print(f"{key}: {val[-1]:0.4f}") for key, val in best_actual.items()]

    output_dict = {
        "best_achieved": best_actual,
        "acq_max_val": acq_max,
        "coverage": iid_coverage,
        "query_coverage": query_coverage,
        "inputs": {k: data_dict[k][0] for k in keys},
    }
    return output_dict


if __name__ == "__main__":
    args = parse()
    output_dict = main(**vars(args))
    torch.save({"pars": vars(args), "results": output_dict}, args.output)
