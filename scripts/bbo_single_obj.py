import random
import numpy as np
import torch
import time
import math
import hydra
import wandb
import warnings
import logging

from omegaconf import OmegaConf

from scipy.stats import norm

from sklearn.model_selection import train_test_split

from upcycle.scripting import startup
from upcycle.logging.analysis import flatten_config

from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qUpperConfidenceBound,
)
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.objective import IdentityMCObjective
from botorch import fit_gpytorch_model
from botorch.sampling.samplers import IIDNormalSampler
from botorch.models import transforms
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning

from conformalbo.helpers import assess_coverage
from conformalbo.acquisitions import (
    ConformalAcquisition,
    qConformalExpectedImprovement,
    qConformalNoisyExpectedImprovement,
    qConformalUpperConfidenceBound,
    qConformalKnowledgeGradient,
)
from conformalbo.ratio_estimation import RatioEstimator
from conformalbo.trbo import TurboState, update_state, get_tr_bounds
from conformalbo.utils import (
    display_metrics,
    generate_initial_data,
    initialize_model,
    parse,
    optimize_acqf_and_get_observation,
    # set_beta,
    update_random_observations,
    initialize_noise_se,
    DataSplit,
    update_splits,
    fit_surrogate,
    fit_and_transform,
    set_alpha,
    evaluate_surrogate,
)


@hydra.main(config_path='../hydra_config', config_name='bbo_single_obj')
def main(cfg):
    # setup
    random.seed(None)  # make sure random seed resets between multirun jobs for random job-name generation
    log_cfg = flatten_config(OmegaConf.to_container(cfg, resolve=True), sep='/')
    wandb.init(project=cfg.project_name, config=log_cfg, mode=cfg.wandb_mode,
               group=cfg.exp_name)
    cfg['job_name'] = wandb.run.name
    cfg, _ = startup(cfg)  # random seed is fixed here

    dtype = torch.double if cfg.dtype == "double" else torch.float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ret_val = bbo_single_obj(cfg, dtype, device)
    except Exception as err:
        ret_val = float('NaN')
        logging.exception(err)

    wandb.finish()  # necessary to log Hydra multirun output to different jobs
    return ret_val


def get_opt_metrics(bb_fn, baseline_X):
    metrics = {}
    baseline_inputs = baseline_X * (bb_fn.bounds[1] - bb_fn.bounds[0]) + bb_fn.bounds[0]
    baseline_f = bb_fn(baseline_inputs).view(*baseline_inputs.shape[:-1], -1)
    if baseline_f.shape[-1] == 1:
        metrics['f_best'] = baseline_f.max().item()
    else:
        box_decomp = FastNondominatedPartitioning(bb_fn.ref_point, baseline_f)
        metrics['f_hypervol'] = box_decomp.compute_hypervolume().item()
    return metrics


def construct_acq_fn(cfg, bb_fn, surrogate, baseline_X, baseline_Y, outcome_transform, ratio_estimator):
    params = dict(model=surrogate)
    if "X_baseline" in cfg.acq_fn.params:
        params['X_baseline'] = baseline_X
    if 'best_f' in cfg.acq_fn.params:
        params['best_f'] = baseline_Y.max()
    if 'beta' in cfg.acq_fn.params:
        beta = norm.ppf(1 - cfg.conformal_params.alpha / 2.).item()
        params['beta'] = beta
    if 'ref_point' in cfg.acq_fn.params:
        params['ref_point'] = outcome_transform(bb_fn.ref_point)[0].view(-1)
    if 'partitioning' in cfg.acq_fn.params:
        ref_point = outcome_transform(bb_fn.ref_point)[0].view(-1)
        params['partitioning'] = FastNondominatedPartitioning(ref_point, baseline_Y)
    if 'Conformal' in cfg.acq_fn.obj._target_:
        params.update(cfg.conformal_params)
        params['ratio_estimator'] = ratio_estimator
    return hydra.utils.instantiate(cfg.acq_fn.obj, **params)


def bbo_single_obj(cfg, dtype, device):
    
    # set up black-box obj. fn.
    bb_fn = hydra.utils.instantiate(cfg.task)
    bb_fn = bb_fn.to(device, dtype)
    # we manually optimize in [0,1]^d
    opt_bounds = torch.zeros_like(bb_fn.bounds)
    opt_bounds[1] += 1.
    task_name = cfg.task._target_.split('.')[-1]
    print(f"obj. function: {task_name}, x bounds: {opt_bounds}")

    # initialize noise se
    problem_noise_se = initialize_noise_se(bb_fn, cfg.noise_se, device=device, dtype=dtype)
    (
        all_X,
        all_outcomes,
        best_actual_obj,
    ) = generate_initial_data(cfg.num_init, bb_fn, problem_noise_se, device, dtype,
            rand_orthant=cfg.rand_orthant,
    )

    optimize_acqf_kwargs = {
        "bounds": opt_bounds,
        "BATCH_SIZE": cfg.q_batch_size,
        "fn": bb_fn,
        "noise_se": problem_noise_se,
    }
    opt_global_state = TurboState(
        batch_size=cfg.q_batch_size,
        dim=all_X.shape[-1]
    )

    # the implicit amount of noise changes by iteration
    current_noise_se = problem_noise_se

    perf_metrics = dict(num_baseline=cfg.num_init)
    perf_metrics.update(get_opt_metrics(bb_fn, all_X))
    display_metrics(perf_metrics)
    wandb.log(perf_metrics, step=0)

    for q_batch_idx in range(1, cfg.num_q_batches + 1):
        print(
            "\nStarting iteration: ", q_batch_idx, "mem allocated: ",
            torch.cuda.memory_reserved() / 1024**3
        )

        train_X, holdout_X, train_outcomes, holdout_outcomes = train_test_split(
            all_X, all_outcomes, test_size=cfg.holdout_frac
        )

        # transform data for inference
        # input_transform = transforms.input.Normalize(train_inputs.shape[-1])
        outcome_transform = transforms.outcome.Standardize(train_outcomes.shape[-1])
        # train_X, holdout_X = fit_and_transform(input_transform, train_inputs, holdout_inputs)
        (train_Y, _), (holdout_Y, _) = fit_and_transform(outcome_transform, train_outcomes, holdout_outcomes)
        surrogate = fit_surrogate(train_X, train_Y)
        set_alpha(cfg, train_X.shape[-2])
        holdout_metrics = evaluate_surrogate(cfg, surrogate, holdout_X, holdout_Y, dr_estimator=None, log_prefix='holdout')
        perf_metrics.update(holdout_metrics)
        
        # possible memory leak
        surrogate.train()
        del surrogate
        torch.cuda.empty_cache()

        # train_X = fit_and_transform(input_transform, all_inputs)
        all_Y, _ = fit_and_transform(outcome_transform, all_outcomes)
        surrogate = fit_surrogate(all_X, all_Y)
        set_alpha(cfg, train_X.shape[-2])
        # set_beta(cfg)

        dr_estimator = RatioEstimator(all_X.size(-1), device, dtype)
        dr_estimator.dataset._update_splits(
            DataSplit(all_X.cpu(), torch.zeros(all_X.size(0), 1))
        )

        acq_fn = construct_acq_fn(cfg, bb_fn, surrogate, train_X, train_Y, outcome_transform, dr_estimator)
        # acq_name = cfg.acq_fn._target_.split('.')[-1]
        # if 'Conformal' in acq_name:
        #     acq_fn = hydra.utils.instantiate(
        #         cfg.acq_fn, model=surrogate, ratio_estimator=dr_estimator, **cfg.conformal_params
        #     )
        # else:
        #     acq_fn = hydra.utils.instantiate(
        #         cfg.acq_fn, model=surrogate
        #     )

        if cfg.use_turbo_bounds:
            opt_bounds = get_tr_bounds(all_X, all_outcomes, surrogate, opt_global_state)
        optimize_acqf_kwargs["bounds"] = opt_bounds

        # optimize acquisition
        if isinstance(acq_fn, ConformalAcquisition):
            opt_acqf_kwargs = {**optimize_acqf_kwargs, **cfg.opt_params}
        else:
            opt_acqf_kwargs = optimize_acqf_kwargs
        query_X, query_outcomes, query_f, query_a, cand_X, cand_outcomes = optimize_acqf_and_get_observation(
                acq_fn, **opt_acqf_kwargs
            )

        # update TuRBO trust region
        if cfg.use_turbo_bounds:
            opt_global_state = update_state(opt_global_state, query_outcomes)
        # update dataset
        all_X = torch.cat([all_X, query_X])
        all_outcomes = torch.cat([all_outcomes, query_outcomes])

        opt_metrics = dict(
            acq_val=query_a.item(),
            num_baseline=perf_metrics['num_baseline'] + cfg.q_batch_size
        )
        opt_metrics.update(get_opt_metrics(bb_fn, all_X))
        perf_metrics.update(opt_metrics)

        cand_Y, _ = outcome_transform(cand_outcomes)
        query_metrics = evaluate_surrogate(
            cfg, surrogate, cand_X.flatten(0, -2), cand_Y.flatten(0, -2), dr_estimator, log_prefix='query'
        )
        perf_metrics.update(query_metrics)

        display_metrics(opt_metrics)

        # possible memory leak
        surrogate.train()
        del surrogate
        torch.cuda.empty_cache()

        wandb.log(perf_metrics, step=q_batch_idx)


# def main(
#     seed: int = 0,
#     dim: int = 10,
#     method: str = "exact",
#     batch_size: int = 3,
#     n_batch: int = 50,
#     tgt_grid_res: int = 64,
#     temp: float = 1e-2,
#     mc_samples: int = 256,
#     num_init: int = 10,
#     noise_se: float = 0.1,
#     dtype: str = "double",
#     verbose: bool = True,
#     output: str = None,
#     problem: str = None,
#     min_alpha: float = 0.05,
#     max_grid_refinements: int = 4,
#     sgld_steps: int = 100,
#     sgld_temperature: float = 1e-3,
#     sgld_lr: int = 1e-3,
#     rand_orthant: bool = False,
# ):



    # keys = ["cucb", "cei", "cnei", "ucb", "ei", "nei", "rnd"]
    # keys = ["tr_ucb", "ucb", "rnd"]
    # best_actual = {k: [] for k in keys}
    # acq_max = {k: [] for k in keys}
    # iid_coverage = {k: [] for k in keys}
    # query_coverage = {k: [] for k in keys}
    # global_states = {k: TurboState(
    #     batch_size=batch_size,
    #     dim=dim,
    # ) for k in keys}

    # data_dict = {}
    # for k in keys:
        # best_actual[k].append(best_actual_obj)
        # data_dict[k] = (all_inputs, all_outcomes)
    


    # run N_BATCH rounds of BayesOpt after the initial random batch
    # for iteration in range(1, n_batch + 1):

        # t0 = time.time()
        # for k in keys:
            # torch.cuda.empty_cache()
            # os.system("nvidia-smi")

            # TODO
            # if k == "rnd":
            #     # update random
            #     best_actual[k] = update_random_observations(
            #         batch_size, best_actual[k], bb_fn.bounds, bb_fn, dim=bounds.shape[1],
            #         noise_se=problem_noise_se,
            #     )
            #     continue

            # get the data
            # all_inputs, all_outcomes = data_dict[k]
            # perm = np.random.permutation(np.arange(all_inputs.size(0)))
            # num_total = all_inputs.size(0)
            # num_test = math.ceil(0.2 * num_total)
            # num_train = num_total - num_test
            # print(f"train: {num_train}, test: {num_test}")

            # test_inputs, test_targets = all_inputs[perm][:num_test], all_outcomes[perm][:num_test]
            # train_inputs, train_targets = all_inputs[perm][num_test:], all_outcomes[perm][num_test:]
            
            # current_noise_se = (problem_noise_se / train_targets.std(0)).clamp(min=0.01).item()
            # print("noise se is now: ", current_noise_se)

            # prepare new model, transform
            # mll, model, trans = initialize_model(
                # train_inputs,
                # train_targets,
                # method=method,
                # train_yvar=current_noise_se**2,
            # )
            # model.requires_grad_(True)
            # fit_gpytorch_model(mll)
            # model.requires_grad_(False)

            # transform test targets
            # trans.eval()
            # test_targets = trans(test_targets)[0]

            # alpha = max(1.0 / math.sqrt(num_train), min_alpha)
            # conformal_kwargs = dict(
                # alpha=alpha,
                # grid_res=tgt_grid_res,
                # max_grid_refinements=max_grid_refinements,
                # ratio_estimator=None,
            # )
            # conformal_opt_kwargs = dict(
                # temperature=sgld_temperature,
                # lr=sgld_lr,
                # sgld_steps=sgld_steps,
            # )

            # torch.cuda.empty_cache()

            # now assess coverage on an IID heldout set
            # conformal_kwargs['temp'] = 1e-6  # set temp to low value when evaluating coverage
            # iid_coverage[k].append(
                # assess_coverage(model, test_inputs, test_targets, **conformal_kwargs)
            # )
            # last_cvrg = iid_coverage[k][-1]
            # print(
                # f"IID coverage ({k}): credible {last_cvrg[0]:0.4f}, conformal {last_cvrg[1]:0.4f}, "
                # f"target level: {1 - alpha:0.4f}"
            # )
            # model.train()
            # torch.cuda.empty_cache()

            # don't fix noise, scale is wrong
            # mll, model, trans = initialize_model(
                # all_inputs,
                # all_outcomes,
                # method=method,
                # train_yvar=current_noise_se**2,
            # )
            # model.requires_grad_(True)
            # fit_gpytorch_model(mll)
            # model.requires_grad_(False)
            # trans.eval()

            # if 'tr_' in k:
                # optimize_acqf_kwargs["bounds"] = get_tr_bounds(all_inputs, all_outcomes, model, global_states[k])
            # else:
                # optimize_acqf_kwargs["bounds"] = bounds

            # now prepare the acquisition
            # batch_range = (0, -3) if k[0] == "c" else (0, -2)
            # sampler = IIDNormalSampler(num_samples=mc_samples, batch_range=batch_range)
            # base_kwargs = dict(
                # model=model,
                # sampler=sampler,
            # )

            # prepare density ratio estimator
            # dr_estimator = RatioEstimator(all_inputs.size(-1), device, dtype)
            # dr_estimator.dataset._update_splits(
            #     DataSplit(all_inputs.cpu(), torch.zeros(all_inputs.size(0), 1))
            # )

            # alpha = max(1.0 / math.sqrt(num_total), min_alpha)
            # conformal_kwargs['alpha'] = alpha
            # conformal_kwargs['temp'] = temp
            # conformal_kwargs['max_grid_refinements'] = 0
            # conformal_kwargs['ratio_estimator'] = dr_estimator

            # TODO
            # if k in ["ei", "tr_ei"]:
            #     acqf = qExpectedImprovement(
            #         **base_kwargs,
            #         best_f=trans(all_outcomes)[0].max(),
            #     )
            # elif k in ["nei", "tr_nei"]:
            #     acqf = qNoisyExpectedImprovement(
            #         **base_kwargs,
            #         X_baseline=all_inputs,
            #         prune_baseline=True,
            #     )
            # elif k in ["ucb", "tr_ucb"]:
            #     acqf = qUpperConfidenceBound(
            #         **base_kwargs,
            #         beta=norm.ppf(1. - alpha / 2.),
            #     )
            # elif k == "kg":
            #     acqf = qKnowledgeGradient(
            #         **base_kwargs,
            #         # current_value=trans(all_outcomes)[0].max(),
            #         num_fantasies=None,
            #         objective=IdentityMCObjective(),
            #         inner_sampler=IIDNormalSampler(mc_samples),
            #     )
            # elif k == "cei":
            #     acqf = qConformalExpectedImprovement(
            #         **conformal_kwargs,
            #         **base_kwargs,
            #         best_f=trans(all_outcomes)[0].max(),
            #     )
            # elif k == "cnei":
            #     acqf = qConformalNoisyExpectedImprovement(
            #         **conformal_kwargs,
            #         **base_kwargs,
            #         X_baseline=all_inputs,
            #         prune_baseline=True,
            #     )
            # elif k == "cucb":
            #     acqf = qConformalUpperConfidenceBound(
            #         **conformal_kwargs,
            #         optimistic=True,
            #         **base_kwargs,
            #         beta=norm.ppf(1. - alpha / 2.),
            #     )
            # elif k == "ckg":
            #     acqf = qConformalKnowledgeGradient(
            #         **conformal_kwargs,
            #         **base_kwargs,
            #         # current_value=trans(all_outcomes)[0].max(),
            #         num_fantasies=None,
            #         objective=IdentityMCObjective(),
            #         inner_sampler=IIDNormalSampler(mc_samples, batch_range=(0, -3)),
            #     )

            # optimize acquisition
            # if k[0] == "c":
            #     optimize_acqf_kwargs = {**optimize_acqf_kwargs, **conformal_opt_kwargs}

            # new_x, new_y, new_f, new_a, all_x, all_y = optimize_acqf_and_get_observation(
            #     acqf, **optimize_acqf_kwargs
            # )

            # if 'tr_' in k:
            #     global_states[k] = update_state(global_states[k], new_y)

            # # evaluate coverage on query candidates
            # sample_idxs = np.random.permutation(all_x.shape[0])[:math.ceil(num_test / batch_size)]
            # test_inputs = torch.cat(
            #     [new_x, all_x[sample_idxs].flatten(0, -2)]
            # )
            # test_targets = trans(torch.cat(
            #     [new_y, all_y[sample_idxs].flatten(0, -2)]
            # ))[0]
            # conformal_kwargs['temp'] = 1e-6  # set temp to low value when evaluating coverage
            # conformal_kwargs['max_grid_refinements'] = max_grid_refinements
            # query_coverage[k].append(
            #     assess_coverage(model, test_inputs, test_targets, **conformal_kwargs)
            # )
            # last_cvrg = query_coverage[k][-1]
            # print(
            #     f"query coverage ({k}): credible {last_cvrg[0]:0.4f}, conformal {last_cvrg[1]:0.4f}, "
            #     f"target level: {1 - alpha:0.4f}"
            # )

    #         # free up GPU memory
    #         del acqf
    #         model.train()
    #         del model
    #         torch.cuda.empty_cache()
        
    #         best_actual[k].append(
    #             max(best_actual[k][-1], new_f.max().item())
    #         )
    #         acq_max[k].append(new_a.item())

    #         # update dataset
    #         all_inputs = torch.cat([all_inputs, new_x])
    #         all_outcomes = torch.cat([all_outcomes, new_y])
    #         data_dict[k] = (all_inputs, all_outcomes)

    #     t1 = time.time()

    #     # best = {key: val[-1] for key, val in best_actual.items()}
    #     if verbose:
    #         print(f"\nBatch: {iteration:>2}, time: {t1-t0:>4.2f}, alpha: {alpha:0.4f}")
    #         # print("max acq val:")
    #         # [print(f"{key}: {val[-1]:0.4f}") for key, val in acq_max.items() if len(val)]
    #         print("best score so far:")
    #         [print(f"{key}: {val[-1]:0.4f}") for key, val in best_actual.items()]

    # output_dict = {
    #     "best_achieved": best_actual,
    #     "acq_max_val": acq_max,
    #     "coverage": iid_coverage,
    #     "query_coverage": query_coverage,
    #     "inputs": {k: data_dict[k][0] for k in keys},
    # }
    # return output_dict


# if __name__ == "__main__":
#     args = parse()
#     output_dict = main(**vars(args))
#     torch.save({"pars": vars(args), "results": output_dict}, args.output)

if __name__ == '__main__':
    main()
