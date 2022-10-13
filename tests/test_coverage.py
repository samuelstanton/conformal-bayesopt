import torch
from conformalbo.helpers import conf_mask_to_bounds


def _create_test_case(cvrg_level, q_batch_shape, grid_size, q_batch_size, target_dim):
    # draw random grid points
    target_grid = torch.randn(*q_batch_shape, grid_size, q_batch_size, target_dim)

    # randomly select upper, lower bounds
    idx_shape = (*q_batch_shape, 1, q_batch_size, target_dim)
    sample_1 = torch.gather(
        target_grid,
        index=torch.randint(grid_size // 2, grid_size, idx_shape),
        dim=-3
    ).squeeze(-3)
    sample_2 = torch.gather(
        target_grid,
        index=torch.randint(0, grid_size // 2, idx_shape),
        dim=-3
    ).squeeze(-3)
    real_lb = torch.min(sample_1, sample_2) - 1e-6
    real_ub = torch.max(sample_1, sample_2) + 1e-6

    # randomly construct targets at specified coverage level
    in_bounds_tgts = real_lb + torch.rand_like(real_lb) * (real_ub - real_lb)
    rand_mask = (torch.rand_like(real_lb) < 0.5).float()
    out_bounds_tgts = rand_mask * (real_lb - 1.) + (1 - rand_mask) * (real_ub + 1.)
    rand_mask = (torch.rand_like(real_lb) < cvrg_level).float()
    real_tgts = rand_mask * in_bounds_tgts + (1 - rand_mask) * out_bounds_tgts

    # mask grid elements not in bounds
    conf_pred_mask = (
            (target_grid >= real_lb.unsqueeze(-3)) * (target_grid <= real_ub.unsqueeze(-3))
    ).float()
    return target_grid, conf_pred_mask, real_tgts, real_lb, real_ub


def test_mask_to_bounds_1d():
    # test params
    q_batch_shape = [256]
    grid_size = 256
    q_batch_size = 1
    target_dim = 1
    cvrg_level = 0.8

    target_grid, conf_pred_mask, real_tgts, real_lb, real_ub = _create_test_case(
        cvrg_level, q_batch_shape, grid_size, q_batch_size, target_dim
    )

    # estimate bounds, coverage
    est_lb, est_ub = conf_mask_to_bounds(target_grid, conf_pred_mask)
    est_cvrg = (
            (real_tgts >= est_lb) * (real_tgts <= est_ub)
    ).float().mean(dim=[0, -2])

    # actual coverage
    act_cvrg = (
            (real_tgts >= real_lb) * (real_tgts <= real_ub)
    ).float().mean(dim=[0, -2])

    assert torch.allclose(act_cvrg, est_cvrg)


def test_mask_to_bounds_2d():
    # test params
    q_batch_shape = [256]
    grid_size = 256
    q_batch_size = 1
    target_dim = 2
    cvrg_level = 0.8

    target_grid, conf_pred_mask, real_tgts, real_lb, real_ub = _create_test_case(
        cvrg_level, q_batch_shape, grid_size, q_batch_size, target_dim
    )

    # estimate bounds, coverage
    est_lb, est_ub = conf_mask_to_bounds(target_grid, conf_pred_mask)
    est_cvrg = (
            (real_tgts >= est_lb) * (real_tgts <= est_ub)
    ).float().mean(dim=[0, -2])

    # actual coverage
    act_cvrg = (
            (real_tgts >= real_lb) * (real_tgts <= real_ub)
    ).float().mean(dim=[0, -2])

    assert torch.allclose(act_cvrg, est_cvrg)
