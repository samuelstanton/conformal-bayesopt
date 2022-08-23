import math
import torch

from torch import Tensor

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
)
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.logging import logger
from botorch.optim.initializers import (
    gen_batch_initial_conditions,
    gen_one_shot_kg_initial_conditions,
)

from conformalbo.optim import SGLD


def optimize_acqf_sgld(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: Optional[int] = None,
    options: Optional[Dict[str, Union[bool, float, int, str]]] = None,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    nonlinear_inequality_constraints: Optional[List[Callable]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    batch_initial_conditions: Optional[Tensor] = None,
    return_best_only: bool = True,
    sequential: bool = False,
    warmup_steps: int = 32,
    sgld_steps: int = 256,
    lr: float = 1e-3,
    temperature: float = 5e-2,
    **kwargs: Any,
) -> Tuple[Tensor, Tensor]:
    r"""Generate a set of candidates via multi-start optimization.

    @NOTE: This is a modified version of `botorch.optim.optimize_acqf`.

    Intended SGLD stationary distribution is $p(x) \propto \exp\{ - a(x) \}$

    Args:
        acq_function: An AcquisitionFunction.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        q: The number of candidates.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of samples for initialization. This is required
            if `batch_initial_conditions` is not specified.
        options: Options for candidate generation.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`
        nonlinear_inequality_constraints: A list of callables with that represent
            non-linear inequality constraints of the form `callable(x) >= 0`. Each
            callable is expected to take a `(num_restarts) x q x d`-dim tensor as an
            input and return a `(num_restarts) x q`-dim tensor with the constraint
            values. The constraints will later be passed to SLSQP. You need to pass in
            `batch_initial_conditions` in this case. Using non-linear inequality
            constraints also requires that `batch_limit` is set to 1, which will be
            done automatically if not specified in `options`.
        fixed_features: A map `{feature_index: value}` for features that
            should be fixed to a particular value during generation.
        post_processing_func: A function that post-processes an optimization
            result appropriately (i.e., according to `round-trip`
            transformations).
        batch_initial_conditions: A tensor to specify the initial conditions. Set
            this if you do not want to use default initialization strategy.
        return_best_only: If False, outputs the solutions corresponding to all
            random restart initializations of the optimization.
        sequential: If False, uses joint optimization, otherwise uses sequential
            optimization.
        kwargs: Additonal keyword arguments.

    Returns:
        A two-element tuple containing

        - a `(num_restarts) x q x d`-dim tensor of generated candidates.
        - a tensor of associated acquisition values. If `sequential=False`,
            this is a `(num_restarts)`-dim tensor of joint acquisition values
            (with explicit restart dimension if `return_best_only=False`). If
            `sequential=True`, this is a `q`-dim tensor of expected acquisition
            values conditional on having observed canidates `0,1,...,i-1`.

    Example:
        >>> # generate `q=2` candidates jointly using 20 random restarts
        >>> # and 512 raw samples
        >>> candidates, acq_value = optimize_acqf(qEI, bounds, 2, 20, 512)

        >>> generate `q=3` candidates sequentially using 15 random restarts
        >>> # and 256 raw samples
        >>> qEI = qExpectedImprovement(model, best_f=0.2)
        >>> bounds = torch.tensor([[0.], [1.]])
        >>> candidates, acq_value_list = optimize_acqf(
        >>>     qEI, bounds, 3, 15, 256, sequential=True
        >>> )
    """
    print(sgld_steps, temperature, lr)

    if sequential and q > 1:
        if not return_best_only:
            raise NotImplementedError(
                "`return_best_only=False` only supported for joint optimization."
            )
        if isinstance(acq_function, OneShotAcquisitionFunction):
            raise NotImplementedError(
                "sequential optimization currently not supported for one-shot "
                "acquisition functions. Must have `sequential=False`."
            )
        candidate_list, acq_value_list = [], []
        base_X_pending = acq_function.X_pending
        for i in range(q):
            candidate, acq_value = optimize_acqf_sgld(
                acq_function=acq_function,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options or {},
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                nonlinear_inequality_constraints=nonlinear_inequality_constraints,
                fixed_features=fixed_features,
                post_processing_func=post_processing_func,
                batch_initial_conditions=None,
                return_best_only=return_best_only,
                sequential=False,
            )
            candidate_list.append(candidate)
            acq_value_list.append(acq_value)
            candidates = torch.cat(candidate_list, dim=-2)
            acq_function.set_X_pending(
                torch.cat([base_X_pending, candidates], dim=-2)
                if base_X_pending is not None
                else candidates
            )
            logger.info(f"Generated sequential candidate {i+1} of {q}")
        # Reset acq_func to previous X_pending state
        acq_function.set_X_pending(base_X_pending)
        return candidates, torch.stack(acq_value_list, dim=-1)

    options = options or {}

    # Handle the trivial case when all features are fixed
    if fixed_features is not None and len(fixed_features) == bounds.shape[-1]:
        X = torch.tensor(
            [fixed_features[i] for i in range(bounds.shape[-1])],
            device=bounds.device,
            dtype=bounds.dtype,
        )
        X = X.expand(q, *X.shape)
        with torch.no_grad():
            acq_value = acq_function(X)
        return X, acq_value

    if batch_initial_conditions is None:
        if nonlinear_inequality_constraints:
            raise NotImplementedError(
                "`batch_initial_conditions` must be given if there are non-linear "
                "inequality constraints."
            )
        if raw_samples is None:
            raise ValueError(
                "Must specify `raw_samples` when `batch_initial_conditions` is `None`."
            )

        ic_gen = (
            gen_one_shot_kg_initial_conditions
            if isinstance(acq_function, qKnowledgeGradient)
            else gen_batch_initial_conditions
        )
        batch_initial_conditions = ic_gen(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            fixed_features=fixed_features,
            options=options,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
        )

    batch_limit: int = options.get(
        "batch_limit", num_restarts if not nonlinear_inequality_constraints else 1
    )
    batched_ics = batch_initial_conditions.detach().clone().split(batch_limit)

    # rescale temp so noise has consistent norm
    scaled_temp = temperature / math.sqrt(bounds.size(-1))
    sgld_optimizers = [
        SGLD([ic_batch], lr=lr, momentum=.1, temperature=scaled_temp) for ic_batch in batched_ics
    ]
    # cyclic learning rate so we actually find the peaks of the modes
    sgld_lr_scheds = [
        torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=sgld_steps // 4, eta_min=lr / 16.
        ) for opt in sgld_optimizers
    ]
    # collect SGLD iterates, bootstrap ratio estimator
    sgld_iterates = [
        [] for _ in batched_ics
    ]
    dr_estimates = [acq_function.ratio_estimator(batch_initial_conditions), None]
    for _s in range(sgld_steps):
        for batch_idx, (ic_batch, batch_optimizer) in enumerate(zip(batched_ics, sgld_optimizers)):
            # once SGLD chain has warmed up start training density ratio estimator
            if _s >= warmup_steps:
                sgld_iterates[batch_idx].append(ic_batch.detach().clone())
                if callable(options.get('callback')):
                    options.get('callback')(ic_batch.detach().clone().cpu())

            if _s < (sgld_steps - 1):
                ic_batch.requires_grad_(True)
                batched_acq_values_ = acq_function(ic_batch)
                batch_optimizer.zero_grad()
                neg_log_density = -batched_acq_values_.sum()
                # lb_violation = torch.max(
                #     2 * (torch.sigmoid((bounds[0] - ic_batch) / 1.) - 0.5),
                #     torch.zeros_like(ic_batch),
                # ).sum()
                # ub_violation = torch.max(
                #     2 * (torch.sigmoid((ic_batch - bounds[1]) / 1.) - 0.5),
                #     torch.zeros_like(ic_batch),
                # ).sum()
                lb_violation = torch.sigmoid((bounds[0] - ic_batch) / 1e-2).mean()
                ub_violation = torch.sigmoid((ic_batch - bounds[1]) / 1e-2).mean()
                loss = neg_log_density + 1e2 * (lb_violation + ub_violation)
                loss.backward()
                batch_optimizer.step()
                sgld_lr_scheds[batch_idx].step()

            # dr_estimates[1] = acq_function.ratio_estimator(batch_initial_conditions)
            # dr_est_diff = torch.norm(dr_estimates[0] - dr_estimates[1]) / torch.norm(dr_estimates[1] + 1e-6)
            # print(f"density ratio estimate rel diff: {dr_est_diff.item():0.4f}")
            # dr_estimates[0] = dr_estimates[1].clone()

    batched_iterates = [torch.stack(iterates) for iterates in sgld_iterates]
    batch_candidates = torch.cat(batched_iterates, dim=1)

    # final evaluation
    with torch.no_grad():
        flat_cands = batch_candidates.flatten(0, -3)
        flat_acq_vals = torch.cat([
            acq_function(cand_batch) for cand_batch in flat_cands.split(batch_limit)
        ])

    # check bounds
    try:
        # in bounds elementwise?
        in_bounds = (flat_cands >= bounds[0]).prod(-1) * (flat_cands <= bounds[1]).prod(-1)
        # in bounds batchwise?
        in_bounds = in_bounds.prod(-1).bool()
        batch_candidates = flat_cands[in_bounds]
        batch_acq_values = flat_acq_vals[in_bounds]
    # if none feasible, use initial solutions
    except:
        print('all SGLD iterates out of bounds, reverting to initial conditions')
        batch_candidates = batch_initial_conditions.flatten(0, -3)
        batch_acq_values = torch.cat([
            acq_function(cand_batch) for cand_batch in batch_candidates.split(batch_limit)
        ])

    if post_processing_func is not None:
        batch_candidates = post_processing_func(batch_candidates)

    if return_best_only:
        best = torch.argmax(batch_acq_values, dim=0)
        batch_candidates = batch_candidates[best]
        batch_acq_values = batch_acq_values[best]

    if isinstance(acq_function, OneShotAcquisitionFunction):
        if not kwargs.get("return_full_tree", False):
            batch_candidates = acq_function.extract_candidates(X_full=batch_candidates)

    return batch_candidates, batch_acq_values


def optimize_acqf_sgld_list(
    acq_function_list: List[AcquisitionFunction],
    bounds: Tensor,
    num_restarts: int,
    raw_samples: Optional[int] = None,
    options: Optional[Dict[str, Union[bool, float, int, str]]] = None,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    return_best_only: bool = True,
) -> Tuple[Tensor, Tensor]:
    r"""Generate a list of candidates from a list of acquisition functions.
    The acquisition functions are optimized in sequence, with previous candidates
    set as `X_pending`. This is also known as sequential greedy optimization.
    Args:
        acq_function_list: A list of acquisition functions.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        num_restarts:  Number of starting points for multistart acquisition
            function optimization.
        raw_samples: Number of samples for initialization. This is required
            if `batch_initial_conditions` is not specified.
        options: Options for candidate generation.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`
        fixed_features: A map `{feature_index: value}` for features that
            should be fixed to a particular value during generation.
        post_processing_func: A function that post-processes an optimization
            result appropriately (i.e., according to `round-trip`
            transformations).
    Returns:
        A two-element tuple containing
        - a `q x d`-dim tensor of generated candidates.
        - a `q`-dim tensor of expected acquisition values, where the value at
            index `i` is the acquisition value conditional on having observed
            all candidates except candidate `i`.
    """
    if not acq_function_list:
        raise ValueError("acq_function_list must be non-empty.")
    candidate_list, acq_value_list = [], []
    candidates = torch.tensor([], device=bounds.device, dtype=bounds.dtype)
    base_X_pending = acq_function_list[0].X_pending
    for acq_function in acq_function_list:
        if candidate_list:
            acq_function.set_X_pending(
                torch.cat([base_X_pending, candidates], dim=-2)
                if base_X_pending is not None
                else candidates
            )
        options = {} if options is None else options
        if hasattr(acq_function.ratio_estimator, 'optimize_callback'):
            options['callback'] = acq_function.ratio_estimator.optimize_callback

        candidate, acq_value = optimize_acqf_sgld(
            acq_function=acq_function,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            fixed_features=fixed_features,
            post_processing_func=post_processing_func,
            return_best_only=True,
            sequential=False,
        )
        candidate_list.append(candidate)
        acq_value_list.append(acq_value)
        candidates = torch.cat(candidate_list, dim=-2)
    return candidates, torch.stack(acq_value_list, dim=-1)