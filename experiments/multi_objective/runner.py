import random
import numpy as np
import torch
import time
import math
import os

from botorch.acquisition.multi_objective import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch import fit_gpytorch_model
from botorch.sampling.samplers import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

import sys
sys.path.append("../../conformalbo/")
from helpers import assess_coverage
from mobo_acquisitions import (
    qConformalNoisyExpectedHypervolumeImprovement,
    qConformalExpectedHypervolumeImprovement,
)
from utils import (
    generate_initial_data,
    initialize_model,
    parse,
    optimize_acqf_and_get_observation,
    update_random_observations,
    get_problem,
)
sys.path.append("../../")
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
    problem: str = "branincurrin",
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

    keys = ["cehvi", "cnehvi", "ehvi", "nehvi", "rnd"]
    hv_dict = {k: [] for k in keys}
    coverage = {k: [] for k in keys}

    # TODO check this
    train_yvar = torch.tensor(noise_se**2, device=device, dtype=dtype)

    # call helper functions to generate initial training data and initialize model
    (
        all_inputs,
        all_targets,
        _,
    ) = generate_initial_data(num_init, bb_fn, noise_se, device, dtype)

    # initial hypervolumes
    bd = DominatedPartitioning(ref_point=bb_fn.ref_point, Y=all_targets)
    volume = bd.compute_hypervolume().item()
    
    data_dict = {}
    for k in keys:
        hv_dict[k].append(volume)
        data_dict[k] = (all_inputs, all_targets)

    optimize_acqf_kwargs = {
        "bounds": bounds,
        "BATCH_SIZE": batch_size,
        "fn": bb_fn,
        "noise_se": noise_se,
    }

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
                _, all_targets = data_dict[k]
                _, next_targets, _ = generate_initial_data(batch_size, bb_fn, noise_se, device, dtype)
                all_targets = torch.cat((all_targets, next_targets))
                
                # update hypervolumes
                bd = DominatedPartitioning(ref_point=bb_fn.ref_point, Y=all_targets)
                volume = bd.compute_hypervolume().item()
                hv_dict[k].append(volume)
                continue

            # get the data
            all_inputs, all_targets = data_dict[k]
            perm = np.random.permutation(np.arange(all_inputs.size(0)))
            num_test = math.ceil(0.2 * all_inputs.size(0))
            num_train = all_inputs.size(0) - num_test
            # print(f"train: {num_train}, test: {num_test}")

            test_inputs, test_targets = all_inputs[perm][:num_test], all_targets[perm][:num_test]
            train_inputs, train_targets = all_inputs[perm][num_test:], all_targets[perm][num_test:]

            # prepare new model, transform
            mll, model, trans = initialize_model(
                train_inputs,
                train_targets,
                method=method,
            )
            model.requires_grad_(True)
            # TODO: support torch fits for mtgps
            fit_gpytorch_model(mll)
            model.requires_grad_(False)

            # transform test targets
            trans.eval()
            test_targets = trans(test_targets)[0]

            alpha = max(1.0 / math.sqrt(num_train), min_alpha)
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
            coverage[k].append(
                assess_coverage(model, test_inputs, test_targets, **conformal_kwargs)
            )
            last_cvrg = coverage[k][-1]
            print(
                f"{k}: cred. coverage {last_cvrg[0]:0.4f}, conf. coverage {last_cvrg[1]:0.4f}, "
                f"target coverage: {1 - alpha:0.4f}"
            )
            model.train()
            torch.cuda.empty_cache()

            mll, model, trans = initialize_model(
                all_inputs,
                all_targets,
                method=method,
            )
            model.requires_grad_(True)
            fit_gpytorch_model(mll)
            model.requires_grad_(False)
            trans.eval()

            # now prepare the acquisition
            qmc_sampler = SobolQMCNormalSampler(num_samples=mc_samples)
            base_kwargs = dict(
                model=model,
                sampler=qmc_sampler,
            )
            conformal_kwargs['alpha'] = max(1.0 / math.sqrt(all_inputs.size(0)), min_alpha)
            conformal_kwargs['temp'] = temp
            conformal_kwargs['max_grid_refinements'] = 0

            if k == "ehvi":
                with torch.no_grad():
                    pred = model.posterior(all_inputs).mean
                partitioning = FastNondominatedPartitioning(
                    ref_point=bb_fn.ref_point,
                    Y=pred,
                )
                acqf = qExpectedHypervolumeImprovement(
                    ref_point=bb_fn.ref_point,
                    partitioning=partitioning,
                    **base_kwargs,
                )
            elif k == "nehvi":
                acqf = qNoisyExpectedHypervolumeImprovement(
                    ref_point=bb_fn.ref_point,
                    X_baseline=all_inputs,
                    prune_baseline=True,
                    **base_kwargs,
                )
            elif k == "cehvi":
                with torch.no_grad():
                    pred = model.posterior(all_inputs).mean
                partitioning = FastNondominatedPartitioning(
                    ref_point=bb_fn.ref_point,
                    Y=pred,
                )
                acqf = qConformalExpectedHypervolumeImprovement(
                    ref_point=bb_fn.ref_point,
                    partitioning=partitioning,
                    **base_kwargs,
                    **conformal_kwargs,
                )
            elif k == "cnehvi":
                acqf = qConformalNoisyExpectedHypervolumeImprovement(
                    ref_point=bb_fn.ref_point,
                    X_baseline=all_inputs,
                    prune_baseline=True,
                    **base_kwargs,
                    **conformal_kwargs,
                )

            # optimize acquisition
            new_x, observed_obj, exact_obj = optimize_acqf_and_get_observation(
                acqf, **optimize_acqf_kwargs
            )
            del acqf
            model.train()
            del model
            torch.cuda.empty_cache()
        
            # update dataset
            all_inputs = torch.cat([all_inputs, new_x])
            all_targets = torch.cat([all_targets, observed_obj])
            data_dict[k] = (all_inputs, all_targets)

            # update hypervolumes
            bd = DominatedPartitioning(ref_point=bb_fn.ref_point, Y=all_targets)
            volume = bd.compute_hypervolume().item()
            hv_dict[k].append(volume)

        t1 = time.time()

        best = {key: val[-1] for key, val in hv_dict.items()}
        if verbose:
            print(f"\nBatch: {iteration:>2}, time: {t1-t0:>4.2f}, alpha: {alpha:0.4f}, best values:")
            [print(f"{key}: {val:0.4f}") for key, val in best.items()]

    output_dict = {
        "best_achieved": hv_dict,
        "coverage": coverage,
        "inputs": {k: data_dict[k][0] for k in keys},
    }
    return output_dict


if __name__ == "__main__":
    args = parse()
    output_dict = main(**vars(args))
    torch.save({"pars": vars(args), "results": output_dict}, args.output)
