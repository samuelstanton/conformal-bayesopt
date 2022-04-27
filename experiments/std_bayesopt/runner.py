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
from botorch.sampling.samplers import IIDNormalSampler, SobolQMCNormalSampler

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

from lambo.utils import DataSplit, update_splits


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
    bounds[1] += 1.
    print(f"function: {problem}, x bounds: {bounds}")

    keys = ["cei", "cnei", "cucb", "rnd", "ei", "nei", "ucb"]
    best_observed = {k: [] for k in keys}
    coverage = {k: [] for k in keys}

    train_yvar = torch.tensor(noise_se**2, device=device, dtype=dtype)

    # call helper functions to generate initial training data and initialize model
    (
        train_x_ei,
        train_obj_ei,
        best_observed_value_ei,
    ) = generate_initial_data(num_init, bb_fn, noise_se, device, dtype)
    new_split = DataSplit(
        train_x_ei.cpu().numpy(), train_obj_ei.cpu().numpy()
    )
    train_split, val_split, test_split = update_splits(
        train_split=DataSplit(),
        val_split=DataSplit(),
        test_split=DataSplit(),
        new_split=new_split,
        holdout_ratio=0.2
    )
    train_inputs = torch.tensor(train_split[0], device=device)
    train_targets = torch.tensor(train_split[1], device=device)

    # heldout_x, heldout_obj, _ = generate_initial_data(
    #     10 * num_init, bb_fn, noise_se, device, dtype
    # )

    # mll_model_dict = {}
    data_dict = {}
    for k in keys:
        mll_and_model = initialize_model(
            train_inputs,
            train_targets,
            train_yvar,
            method=method,
        )
        # mll_model_dict[k] = mll_and_model
        best_observed[k].append(best_observed_value_ei)
        data_dict[k] = (train_split, val_split, test_split)

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

            # get the data
            train_split, val_split, test_split = data_dict[k]
            # print(f'{train_split[0].shape[0]} train, {val_split[0].shape[0]} val, {test_split[0].shape[0]} test')
            train_inputs = torch.tensor(train_split[0], device=device)
            train_targets = torch.tensor(train_split[1], device=device)

            # prepare new model
            mll, model, trans = initialize_model(
                train_inputs,
                train_targets,
                method=method,
            )
            # mll_model_dict[k] = (mll, model, trans)

            model.requires_grad_(True)
            fit_gpytorch_model(mll)
            model.requires_grad_(False)

            alpha = max(1.0 / math.sqrt(train_inputs.size(-2)), min_alpha)
            rx_estimator = None
            conformal_kwargs = dict(
                alpha=alpha,
                grid_res=tgt_grid_res,
                max_grid_refinements=max_grid_refinements,
                ratio_estimator=rx_estimator
            )
            
            torch.cuda.empty_cache()

            # now assess coverage on the heldout set
            conformal_kwargs['temp'] = 1e-6  # set temp to low value when evaluating coverage
            val_inputs = torch.tensor(
                np.concatenate([val_split[0], test_split[0]]), device=device
            )
            val_targets = torch.tensor(
                np.concatenate([val_split[1], test_split[1]]), device=device
            )
            trans.eval()
            val_targets = trans(val_targets)[0]
            # import pdb; pdb.set_trace()
            coverage[k].append(
                assess_coverage(model, val_inputs, val_targets, **conformal_kwargs)
            )
            last_cvrg = coverage[k][-1]
            print(
                f"{k}: cred. coverage {last_cvrg[0]:0.4f}, conf. coverage {last_cvrg[1]:0.4f}, "
                f"target coverage: {1 - alpha:0.4f}"
            )
            model.train()
            torch.cuda.empty_cache()

            # now prepare the acquisition
            conformal_kwargs['temp'] = temp
            # TODO: check to see if we want to move to QMC eventually
            # iid_sampler = IIDNormalSampler(num_samples=mc_samples)
            qmc_sampler = SobolQMCNormalSampler(num_samples=mc_samples)
            if k == "ei":
                acqf = qExpectedImprovement(
                    model=model,
                    best_f=(train_targets).max(),
                    sampler=qmc_sampler,
                )
            elif k == "nei":
                acqf = qNoisyExpectedImprovement(
                    model=model,
                    X_baseline=train_inputs,
                    sampler=qmc_sampler,
                    prune_baseline=True,
                )
            elif k == "ucb":
                acqf = qUpperConfidenceBound(
                    model=model,
                    beta=0.1,
                )
            elif k == "kg":
                acqf = qKnowledgeGradient(
                    model=model,
                    current_value=train_targets.max(),
                    num_fantasies=None,
                    sampler=qmc_sampler,
                )
            elif k == "cei":
                acqf = qExpectedImprovement(
                    model=model,
                    best_f=(train_targets).max(),
                    sampler=qmc_sampler,
                )
                acqf = conformalize_acq_fn(acqf, **conformal_kwargs)
            elif k == "cnei":
                acqf = qNoisyExpectedImprovement(
                    model=model,
                    X_baseline=train_inputs,
                    sampler=qmc_sampler,
                    prune_baseline=True,
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
                    current_value=train_targets.max(),
                    num_fantasies=None,
                    sampler=qmc_sampler,
                )
                acqf = conformalize_acq_fn(acqf, **conformal_kwargs)

            # optimize acquisition
            new_x, observed_obj, exact_obj = optimize_acqf_and_get_observation(
                acqf, **optimize_acqf_kwargs
            )
            del acqf
            model.train()
            del model
            torch.cuda.empty_cache()
        
            # inputs = torch.cat([inputs, new_x])
            # objective = torch.cat([objective, new_obj])

            # best_observed[k].append(objective.max().item())
            # # prepare new model
            # alpha = max(1.0 / math.sqrt(inputs.size(-2)), min_alpha)
            # mll, model, trans = initialize_model(
            #     inputs,
            #     objective,
            #     method=method,
            #     alpha=alpha,
            #     tgt_grid_res=tgt_grid_res,
            # )
            # mll_model_dict[k] = (mll, model, trans)
            # data_dict[k] = inputs, objective
            print(torch.cuda.memory_reserved() / 1024**3)
            best_observed[k].append(
                max(best_observed[k][-1], exact_obj.max().item())
            )

            # update splits
            new_split = DataSplit(
                new_x.cpu(),
                observed_obj.cpu(),
            )
            train_split, val_split, test_split = update_splits(
                train_split, val_split, test_split, new_split, holdout_ratio=0.2
            )
            data_dict[k] = (train_split, val_split, test_split)

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
    args = parse()
    output_dict = main(**vars(args))
    torch.save({"pars": vars(args), "results": output_dict}, args.output)
