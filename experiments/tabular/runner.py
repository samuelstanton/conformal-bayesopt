from multiprocessing.sharedctypes import Value
import hydra
import wandb
import random
import torch
import numpy as np
import logging
import pandas as pd
from pathlib import Path, PurePath
import os
import math
import warnings
import copy

from omegaconf import OmegaConf

from scipy.stats import norm, spearmanr

from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.models import transforms, SingleTaskGP
from botorch import fit_gpytorch_model
from botorch.sampling import IIDNormalSampler

from upcycle.scripting import startup
from upcycle.logging.analysis import flatten_config

# TODO proper packaging
import sys
sys.path.append("../../conformalbo/")
from helpers import assess_coverage
from acquisitions import qConformalUpperConfidenceBound
from ratio_estimation import RatioEstimator


@hydra.main(config_path='./hydra_config', config_name='tabular_search')
def main(cfg):
    # setup
    random.seed(None)  # make sure random seed resets between multirun jobs for random job-name generation
    log_cfg = flatten_config(OmegaConf.to_container(cfg, resolve=True), sep='/')
    # log_config = {'/'.join(('config', key)): val for key, val in log_config.items()}
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
            ret_val = tabular_search(cfg, dtype, device)
    except Exception as err:
        ret_val = float('NaN')
        logging.exception(err)

    wandb.finish()  # necessary to log Hydra multirun output to different jobs
    return ret_val
    

def get_src_df(cfg):
    src_path = PurePath(cfg.task.src_path)
    if src_path.is_absolute():
        abs_src_path = src_path
    # resolve relative src path in config
    else:
        abs_script_path = os.path.realpath(__file__)
        print(f"script path: {abs_script_path}")
        project_root = Path(abs_script_path).parents[2]
        abs_src_path = project_root / src_path
   
    print(f"reading source data from {abs_src_path}")
    src_df = pd.read_csv(abs_src_path)
    src_df = src_df.sample(cfg.num_total_rows)

    assert all([k in src_df for k in cfg.task.obj_cols])
    # if the sign is 1 the objective will be maximized
    for o_col, o_sign in zip(cfg.task.obj_cols, cfg.task.obj_signs):
        src_df.loc[:, o_col] = o_sign * src_df[o_col]

    drop_cols = [k for k in cfg.task.exclude_feature_cols if k in src_df.columns]
    drop_cols += [k for k in cfg.task.obj_cols if k not in drop_cols]

    return src_df, drop_cols


def baseline_candidate_split(cfg, src_df):
    if len(cfg.task.obj_cols) > 1:
        raise NotImplementedError

    sorted_df = src_df.sort_values(cfg.task.obj_cols[0])

    assert cfg.num_baseline < cfg.num_total_rows
    baseline_df = sorted_df.iloc[:cfg.num_baseline]
    candidate_df = sorted_df.iloc[cfg.num_baseline:]

    return baseline_df, candidate_df


def df_to_input_outcome(df, outcome_cols, drop_cols, device, dtype):
    tsr_kwargs = dict(device=device, dtype=dtype)
    inputs = torch.tensor(df.drop(drop_cols, axis=1).values, **tsr_kwargs)
    outcomes = torch.tensor(df.loc[:, outcome_cols].values, **tsr_kwargs)
    return inputs, outcomes


def fit_and_transform(transform, train_arr, holdout_arr=None):
    transform.train()
    train_result = transform(train_arr)
    transform.eval()
    if holdout_arr is None:
        return train_result
    return train_result, transform(holdout_arr)


def fit_surrogate(train_X, train_Y):
    surrogate = SingleTaskGP(train_X=train_X, train_Y=train_Y)
    surrogate_mll = ExactMarginalLogLikelihood(surrogate.likelihood, surrogate)
    surrogate.train()
    surrogate.requires_grad_(True)
    fit_gpytorch_model(surrogate_mll)
    surrogate.requires_grad_(False)
    surrogate.eval()
    return surrogate


def fit_dr_estimator(cfg, dr_estimator, baseline_X, candidate_X):
    num_baseline = baseline_X.shape[-2]
    num_candidate = candidate_X.shape[-2]
    emp_ratio = torch.tensor(
        num_baseline / num_candidate, device=dr_estimator.device
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=emp_ratio)
    dr_estimator.requires_grad_(True)
    optimizer = torch.optim.Adam(
        dr_estimator.parameters(), lr=cfg.dr_estimator.lr, weight_decay=cfg.dr_estimator.weight_decay
    )
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.dr_estimator.num_grad_steps)
    baseline_Z = torch.zeros(num_baseline, 1, device=dr_estimator.device)
    candidate_Z = torch.ones(num_candidate, 1, device=dr_estimator.device)
    cat_X = torch.cat([baseline_X, candidate_X])
    cat_Z = torch.cat([baseline_Z, candidate_Z])

    num_val = int(0.2 * cat_X.shape[-2])
    rand_perm = np.random.permutation(cat_X.shape[-2])

    val_X = cat_X[rand_perm[:num_val]]
    val_Z = cat_Z[rand_perm[:num_val]]
    train_X = cat_X[rand_perm[num_val:]]
    train_Z = cat_Z[rand_perm[num_val:]]

    best_val_loss = float("inf")
    for _ in range(cfg.dr_estimator.num_grad_steps):
        with torch.no_grad():
            val_logits = dr_estimator.classifier(val_X)
            val_loss = loss_fn(val_logits, val_Z)

        dr_estimator.zero_grad()
        aug_X = train_X + cfg.dr_estimator.noise_aug_scale * torch.randn_like(train_X)
        train_logits = dr_estimator.classifier(aug_X)
        train_loss = loss_fn(train_logits, train_Z)

        if val_loss < best_val_loss:
            best_val_loss = val_loss.item()
            best_weights = copy.deepcopy(dr_estimator.state_dict())
            ckpt_train_loss = train_loss.item()

        train_loss.backward()
        optimizer.step()
        lr_sched.step()
    
    dr_estimator.load_state_dict(best_weights)
    dr_estimator.requires_grad_(False)
    dr_estimator.update_target_network()

    tgt_network = dr_estimator._target_network
    tgt_logits = tgt_network(val_X)
    tgt_loss = loss_fn(tgt_logits, val_Z).item()

    metrics = dict(
        dre_train_loss=ckpt_train_loss,
        dre_val_loss=best_val_loss,
        dre_tgt_loss=tgt_loss,
        dre_last_train_loss=train_loss.item()
    )
    display_metrics(metrics)
    return metrics


def set_alpha(cfg, num_train):
    alpha = 1 / math.sqrt(num_train)
    cfg.conformal_params['alpha'] = alpha
    return alpha


def evaluate_surrogate(cfg, surrogate, holdout_X, holdout_Y, dr_estimator=None, log_prefix=''):
    eval_metrics = {}
    # p(y | x, D)
    y_post = surrogate.posterior(holdout_X, observation_noise=True)
    # NLL, RMSE, and Spearman's Rho
    eval_metrics['nll'] = -1 * torch.distributions.Normal(y_post.mean, y_post.variance.sqrt()).log_prob(holdout_Y).mean().item()
    eval_metrics["rmse"] = (y_post.mean - holdout_Y).pow(2).mean().sqrt().item()
    try:
        s_rho = np.stack([
            spearmanr(
                holdout_Y[:, idx], y_post.mean[:, idx]
            ).correlation for idx in range(holdout_Y.shape[-1])
        ]).mean().item()
    except Exception:
        s_rho = float('NaN')
    eval_metrics["s_rho"] = s_rho
    # credible and conformal coverage
    eval_metrics["exp_cvrg"] = 1 - cfg.conformal_params.alpha
    eval_metrics["cred_cvrg"], eval_metrics["conf_cvrg"] = assess_coverage(
        surrogate, holdout_X, holdout_Y, ratio_estimator=dr_estimator, **cfg.conformal_params
    )
    # add log prefix
    if len(log_prefix) > 0:
        eval_metrics = {'_'.join([log_prefix, key]): val for key, val in eval_metrics.items()}
    display_metrics(eval_metrics)
    return eval_metrics


def display_metrics(metrics):
    df = pd.DataFrame((metrics,)).round(4)
    print(df.to_markdown())


def acq_fn_factory(cfg, surrogate, dr_estimator=None):
    batch_range = (0, -3) if cfg.acq_fn == "cucb" else (0, -2)
    num_samples = 1 if cfg.acq_fn == "cucb" else cfg.num_f_samples
    common_kwargs = dict(
        model=surrogate,
        sampler=IIDNormalSampler(num_samples, batch_range=batch_range),
        beta = norm.ppf(1. - cfg.conformal_params.alpha / 2.),
    )
    if cfg.acq_fn == "ucb":
        acq_fn = qUpperConfidenceBound(**common_kwargs)
    elif cfg.acq_fn == "cucb":
        acq_fn = qConformalUpperConfidenceBound(
            **cfg.conformal_params, optimistic=True, **common_kwargs, ratio_estimator=dr_estimator)
    else:
        raise NotImplementedError
    return acq_fn


def tabular_search(cfg, dtype, device):
    if cfg.q_batch_size > 1:
        raise NotImplementedError
    if len(cfg.task.obj_cols) > 1:
        raise NotImplementedError

    # get source data
    src_df, drop_cols = get_src_df(cfg)
    # best_src_outcome = src_df[cfg.task.obj_cols].max(0).values.item()
    # split source into baseline, candidates
    baseline_df, candidate_df = baseline_candidate_split(cfg, src_df)
    
    num_steps = min(cfg.max_num_steps, candidate_df.shape[-2])
    perf_metrics = dict(cum_regret=0)
    for step_idx in range(1, num_steps + 1):
        print(f"\n#### T = {step_idx} ####")
        # first evaluate holdout metrics
        holdout_df = baseline_df.sample(frac=cfg.holdout_frac)
        train_df = baseline_df.drop(holdout_df.index)
        train_inputs, train_outcomes = df_to_input_outcome(
            train_df, cfg.task.obj_cols, drop_cols, device, dtype
        )
        holdout_inputs, holdout_outcomes = df_to_input_outcome(
            holdout_df, cfg.task.obj_cols, drop_cols, device, dtype
        )
        # transform data for inference
        input_transform = transforms.input.Normalize(train_inputs.shape[-1])
        outcome_transform = transforms.outcome.Standardize(train_outcomes.shape[-1])
        train_X, holdout_X = fit_and_transform(input_transform, train_inputs, holdout_inputs)
        (train_Y, _), (holdout_Y, _) = fit_and_transform(outcome_transform, train_outcomes, holdout_outcomes)
        surrogate = fit_surrogate(train_X, train_Y)
        set_alpha(cfg, train_X.shape[-2])
        holdout_metrics = evaluate_surrogate(cfg, surrogate, holdout_X, holdout_Y, dr_estimator=None, log_prefix='holdout')
        wandb.log(holdout_metrics, step=step_idx)
        
        # possible memory leak
        surrogate.train()
        del surrogate
        torch.cuda.empty_cache()

        # now prepare to rank candidates
        baseline_inputs, baseline_outcomes = df_to_input_outcome(
            baseline_df, cfg.task.obj_cols, drop_cols, device, dtype
        )
        candidate_df = candidate_df.sample(frac=1.)  # shuffle candidates
        candidate_inputs, candidate_outcomes = df_to_input_outcome(
            candidate_df, cfg.task.obj_cols, drop_cols, device, dtype
        )
        baseline_X, candidate_X = fit_and_transform(input_transform, baseline_inputs, candidate_inputs)
        (baseline_Y, _), (candidate_Y, _) = fit_and_transform(outcome_transform, baseline_outcomes, candidate_outcomes)
        surrogate = fit_surrogate(baseline_X, baseline_Y)
        dr_estimator = RatioEstimator(train_X.shape[-1], device, dtype, cfg.dr_estimator.ema_weight)
        dre_metrics = fit_dr_estimator(cfg, dr_estimator, baseline_X, candidate_X)
        wandb.log(dre_metrics, step=step_idx)

        # select, evaluate query
        set_alpha(cfg, baseline_X.shape[-2])
        acq_fn = acq_fn_factory(cfg, surrogate, dr_estimator)
        candidate_scores = acq_fn(candidate_X.unsqueeze(-2))

        sorted_idx = candidate_scores.argsort(0, descending=True)
        best_idx = sorted_idx[0].item()
        # evaluate on subset of candidates same size as holdout set
        num_holdout = holdout_X.shape[-2]
        query_X = candidate_X[sorted_idx[:num_holdout]]
        query_Y = candidate_Y[sorted_idx[:num_holdout]]
        query_metrics = evaluate_surrogate(cfg, surrogate, query_X, query_Y, dr_estimator, log_prefix='query')
        query_df = candidate_df.iloc[best_idx]
        wandb.log(query_metrics, step=step_idx)

        # possible memory leak
        del acq_fn
        surrogate.train()
        del surrogate
        torch.cuda.empty_cache()

        # log performance
        best_cand_outcome = candidate_df[cfg.task.obj_cols].max(0).values.item()
        perf_metrics["query_outcome"] = query_df[cfg.task.obj_cols].values.item()
        perf_metrics["num_baseline"] = baseline_df.shape[-2]
        perf_metrics["num_candidate"] = candidate_df.shape[-2]
        perf_metrics["instant_regret"] = best_cand_outcome - perf_metrics["query_outcome"]
        perf_metrics["cum_regret"] += perf_metrics["instant_regret"]
        display_metrics(perf_metrics)
        wandb.log(perf_metrics, step=step_idx)

        # add query to baselines
        baseline_df = baseline_df.append(query_df)
        candidate_df = candidate_df.iloc[:best_idx].append(
            candidate_df.iloc[best_idx + 1:]
        )

    return perf_metrics["cum_regret"]


if __name__ == '__main__':
    main()
