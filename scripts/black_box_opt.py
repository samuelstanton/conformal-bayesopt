import random
import torch
import hydra
import wandb
import warnings
import logging

from omegaconf import OmegaConf

from scipy.stats import norm

from sklearn.model_selection import train_test_split

from upcycle.scripting import startup
from upcycle.logging.analysis import flatten_config

from botorch.models import transforms

from conformalbo.acquisitions import ConformalAcquisition
from conformalbo.trbo import TurboState, update_state, get_tr_bounds
from conformalbo.utils import (
    display_metrics,
    generate_initial_data,
    optimize_acqf_and_get_observation,
    initialize_noise_se,
    DataSplit,
    fit_surrogate,
    fit_and_transform,
    set_alpha,
    evaluate_surrogate,
    construct_acq_fn,
    get_opt_metrics,
)


@hydra.main(config_path='../hydra_config', config_name='black_box_opt')
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
            ret_val = black_box_opt(cfg, dtype, device)
    except Exception as err:
        ret_val = float('NaN')
        logging.exception(err)

    wandb.finish()  # necessary to log Hydra multirun output to different jobs
    return ret_val


def black_box_opt(cfg, dtype, device):
    
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
        "NUM_RESTARTS": cfg.num_opt_restarts,
        "BATCH_SIZE": cfg.q_batch_size,
        "fn": bb_fn,
        "noise_se": problem_noise_se,
    }
    opt_global_state = TurboState(
        batch_size=cfg.q_batch_size,
        dim=all_X.shape[-1]
    )

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
        outcome_transform = transforms.outcome.Standardize(train_outcomes.shape[-1])
        (train_Y, _), (holdout_Y, _) = fit_and_transform(outcome_transform, train_outcomes, holdout_outcomes)
        surrogate = fit_surrogate(train_X, train_Y)
        set_alpha(cfg, train_X.shape[-2], cfg.alpha_power)
        holdout_metrics = evaluate_surrogate(cfg, surrogate, holdout_X, holdout_Y, dr_estimator=None, log_prefix='holdout')
        perf_metrics.update(holdout_metrics)
        
        # possible memory leak
        surrogate.train()
        del surrogate
        torch.cuda.empty_cache()

        all_Y, _ = fit_and_transform(outcome_transform, all_outcomes)
        surrogate = fit_surrogate(all_X, all_Y)
        set_alpha(cfg, all_X.shape[-2], cfg.alpha_power)

        dr_estimator = hydra.utils.instantiate(cfg.dr_estimator, in_size=all_X.size(-1), device=device, dtype=dtype)
        dr_estimator.set_neg_samples(all_X)
        # dr_estimator.dataset._update_splits(
        #     DataSplit(all_X.cpu(), torch.zeros(all_X.size(0), 1))
        # )

        acq_fn = construct_acq_fn(cfg, bb_fn, surrogate, train_X, train_Y, outcome_transform, dr_estimator)

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
        print(f"query X:\n{query_X}")
        print(f"query f:\n{query_f}")

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

    f_best = perf_metrics.get('f_best', None)
    f_hypervol = perf_metrics.get('f_hypervol', None)
    ret_val = f_hypervol if f_best is None else f_best
    return ret_val


if __name__ == '__main__':
    main()
