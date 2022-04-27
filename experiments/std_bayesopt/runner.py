import random
import numpy as np
import torch
import time
import math
import os

from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qUpperConfidenceBound,
)
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch import fit_gpytorch_model
from botorch.sampling.samplers import IIDNormalSampler

from utils import (
    generate_initial_data,
    initialize_model,
    parse,
    optimize_acqf_and_get_observation,
    update_random_observations,
    get_problem,
)
from helpers import assess_coverage
from acquisitions import *


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
):
    dtype = torch.double if dtype == "double" else torch.float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    bb_fn = get_problem(problem, dim)
    bb_fn = bb_fn.to(device, dtype)
    # we manually optimize in [0,1]^d
    bounds = torch.zeros_like(bb_fn.bounds)
    bounds[1] += 1.0

    keys = ["cei", "cnei", "ckg", "rnd", "ei", "nei", "kg"]
    best_observed = {k: [] for k in keys}
    coverage = {k: [] for k in keys}

    train_yvar = torch.tensor(noise_se**2, device=device, dtype=dtype)

    # call helper functions to generate initial training data and initialize model
    (
        train_x_ei,
        train_obj_ei,
        best_observed_value_ei,
    ) = generate_initial_data(num_init, bb_fn, noise_se, device, dtype)
    heldout_x, heldout_obj, _ = generate_initial_data(
        10 * num_init, bb_fn, noise_se, device, dtype
    )

    alpha = max(1.0 / math.sqrt(train_x_ei.size(-2)), min_alpha)

    mll_model_dict = {}
    data_dict = {}
    for k in keys:
        mll_and_model = initialize_model(
            train_x_ei,
            train_obj_ei,
            train_yvar,
            method=method,
            alpha=alpha,
            tgt_grid_res=tgt_grid_res,
            max_grid_refinements=max_grid_refinements,
        )
        mll_model_dict[k] = mll_and_model
        best_observed[k].append(best_observed_value_ei)
        data_dict[k] = (train_x_ei, train_obj_ei)

    optimize_acqf_kwargs = {
        "bounds": bounds,
        "BATCH_SIZE": batch_size,
        "fn": bb_fn,
        "noise_se": noise_se,
    }

    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(1, n_batch + 1):
        print("Starting iteration: ", iteration, "mem allocated: ",
                torch.cuda.memory_reserved() / 1024**3)
        t0 = time.time()
        for k in keys:
            torch.cuda.empty_cache()
            os.system("nvidia-smi")

            if k == "rnd":
                # update random
                best_observed[k] = update_random_observations(
                    batch_size, best_observed[k], bb_fn.bounds, bb_fn, dim=bounds.shape[1]
                )
                continue

            # fit the model
            mll, model, trans = mll_model_dict[k]
            inputs, objective = data_dict[k]
            trans.eval()
            t_objective = trans(objective)[0]
            model.requires_grad_(True)
            fit_gpytorch_model(mll)
            model.requires_grad_(False)

            rx_estimator = None
            conformal_kwargs = dict(
                alpha=alpha,
                grid_res=tgt_grid_res,
                max_grid_refinements=max_grid_refinements,
                ratio_estimator=rx_estimator
            )
            
            torch.cuda.empty_cache()

            # now assess coverage on the heldout set
            # TODO: update the heldout sets
            conformal_kwargs['temp'] = 1e-6  # set temp to low value when evaluating coverage
            coverage[k].append(
                assess_coverage(model, heldout_x, trans(heldout_obj)[0], **conformal_kwargs)
            )
            print(coverage[k][-1], k)
            model.train()
            torch.cuda.empty_cache()

            # now prepare the acquisition
            conformal_kwargs['temp'] = temp
            # TODO: check to see if we want to move to QMC eventually
            iid_sampler = IIDNormalSampler(num_samples=mc_samples)
            if k == "ei":
                acqf = qExpectedImprovement(
                    model=model,
                    best_f=(t_objective).max(),
                    sampler=iid_sampler,
                )
            elif k == "nei":
                acqf = qNoisyExpectedImprovement(
                    model=model,
                    X_baseline=inputs,
                    sampler=iid_sampler,
                )
            elif k == "ucb":
                acqf = qUpperConfidenceBound(
                    model=model,
                    beta=0.1,
                )
            elif k == "kg":
                acqf = qKnowledgeGradient(
                    model=model,
                    current_value=t_objective.max(),
                    num_fantasies=None,
                    sampler=iid_sampler,
                )
            elif k == "cei":
                acqf = qExpectedImprovement(
                    model=model,
                    best_f=(t_objective).max(),
                    sampler=iid_sampler,
                )
                acqf = conformalize_acq_fn(acqf, **conformal_kwargs)
            elif k == "cnei":
                acqf = qNoisyExpectedImprovement(
                    model=model,
                    X_baseline=inputs,
                    sampler=iid_sampler,
                )
                acqf = conformalize_acq_fn(acqf, **conformal_kwargs)
            elif k == "cucb":
                acqf = qUpperConfidenceBound(
                    model=model,
                    beta=0.1,
                )
                acqf = conformalize_acq_fn(acqf, **conformal_kwargs)
            elif k == "ckg":
                acqf = qKnowledgeGradient(
                    model=model,
                    current_value=t_objective.max(),
                    num_fantasies=64,
                    # sampler=iid_sampler,
                )
                acqf = conformalize_acq_fn(acqf, **conformal_kwargs)

            # optimize acquisition
            new_x, new_obj = optimize_acqf_and_get_observation(
                acqf, **optimize_acqf_kwargs
            )
            del acqf
            model.train()
            del model
            torch.cuda.empty_cache()
        
            inputs = torch.cat([inputs, new_x])
            objective = torch.cat([objective, new_obj])

            best_observed[k].append(objective.max().item())
            # prepare new model
            alpha = max(1.0 / math.sqrt(inputs.size(-2)), min_alpha)
            mll, model, trans = initialize_model(
                inputs,
                objective,
                method=method,
                alpha=alpha,
                tgt_grid_res=tgt_grid_res,
            )
            mll_model_dict[k] = (mll, model, trans)
            data_dict[k] = inputs, objective
            print(torch.cuda.memory_reserved() / 1024**3)

        t1 = time.time()

        best = {key: val[-1] for key, val in best_observed.items()}
        if verbose:
            print(f"\nBatch {iteration:>2}, time = {t1-t0:>4.2f}, best values:")
            [print(f"{key}: {val:0.2f}") for key, val in best.items()]

    output_dict = {
        "best_achieved": best_observed,
        "coverage": coverage,
        "inputs": {k: data_dict[k][0] for k in keys},
    }
    return output_dict


if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True):
        args = parse()
        output_dict = main(**vars(args))
    torch.save({"pars": vars(args), "results": output_dict}, args.output)
