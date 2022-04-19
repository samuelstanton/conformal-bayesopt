import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn

from botorch.acquisition import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.posteriors import Posterior
from botorch.sampling import MCSampler
from botorch.models import SingleTaskGP

from gpytorch.lazy import DiagLazyTensor

import torchsort

# from acquisition import qExpectedImprovement, qNoisyExpectedImprovement

def conformal_gp_regression(gp, test_inputs, target_grid, alpha, temp=1e-6,
                            ratio_estimator=None, **kwargs):
    """
    Full conformal Bayes for exact GP regression.
    Args:
        gp (gpytorch.models.GP)
        inputs (torch.Tensor): (num_q_batches, q_batch_size, input_dim)
        target_grid (torch.Tensor): (num_grid_pts, target_dim)
        alpha (float)
        ratio_estimator (torch.nn.Module)
    Returns:
        conf_pred_mask (torch.Tensor): (num_q_batches, q_batch_size, num_grid_pts, target_dim)
    """

    # make sure caches are populated
    gp.eval()
    gp.standard()
    gp.posterior(test_inputs.detach())

    gp.conf_pred_mask = None  # without this line the deepcopy in `gp.condition_on_observations` fails
    # code smell reminder:
    gp.conformal()  # fix for BoTorch acq fn batch shape checks

    # retraining: condition the GP at every target grid point for every test input
    expanded_inputs = test_inputs.unsqueeze(-3).expand(
        *[-1]*(test_inputs.ndim-2), target_grid.shape[0], -1, -1
    )
    expanded_targets = target_grid.expand(*test_inputs.shape[:-1], -1, -1)
    # the q batch and grid size are flipped
    expanded_targets = expanded_targets.transpose(-2, -3)

    updated_gps = gp.condition_on_observations(expanded_inputs, expanded_targets)
    
    # get ready to compute the conformal scores
    train_inputs = updated_gps.train_inputs[0]
    train_labels = updated_gps.prediction_strategy.train_labels
    train_labels = train_labels.unsqueeze(-1)  # (num_test, grid_size, num_train + 1, target_dim)
    # lik_train_train_covar = updated_gps.prediction_strategy.lik_train_train_covar
    prior_mean = updated_gps.prediction_strategy.train_prior_dist.mean.unsqueeze(-1)
    prior_covar = updated_gps.prediction_strategy.train_prior_dist.lazy_covariance_matrix
    noise = updated_gps.likelihood.noise
    
    # compute conformal scores (posterior predictive log-likelihood)
    updated_gps.standard()
    posterior = updated_gps.posterior(train_inputs, observation_noise=True)
    pred_dist = torch.distributions.Normal(posterior.mean, posterior.variance.sqrt())
    o_conf_scores = pred_dist.log_prob(train_labels)
    conf_scores = o_conf_scores.squeeze(-1)

    num_total = conf_scores.size(-1)
    original_shape = conf_scores.shape
    ranks_by_score = torchsort.soft_rank(
        conf_scores.flatten(0, -2),
        regularization="l2",
        regularization_strength=0.1,
    ).view(*original_shape)

    if ratio_estimator is None:
        imp_weights = 1. / num_total
    else:
        with torch.no_grad():
            imp_weights = ratio_estimator(train_inputs)
            imp_weights /= imp_weights.sum(dim=-1, keepdim=True)

    # TODO following line assumes q_batch_size = 1
    rank_mask = 1 - torch.sigmoid(
        (ranks_by_score - ranks_by_score[..., num_total - 1:num_total]) / temp
    )
    cum_weights = (rank_mask * imp_weights).sum(-1)
    conf_pred_mask = torch.sigmoid(
        (cum_weights - alpha) / temp
    )
    return conf_pred_mask, updated_gps


# TODO: write a sub-class for these
class qConformalExpectedImprovement(qExpectedImprovement):
    def forward(self, X):
        """
        :param X: (*batch_shape, q, d)
        :return: (*batch_shape)
        """
        unconformalized_acqf = super().forward(X)  # batch x grid

        res = torch.trapezoid(
            y=self.model.conf_pred_mask * unconformalized_acqf,
            x=self.model.conf_tgt_grid,
            dim=-1
        )
        return res


class qConformalNoisyExpectedImprovement(qNoisyExpectedImprovement):
    def forward(self, X):
        unconformalized_acqf = super().forward(X) # batch x grid x q
        res = torch.trapezoid(
            y=self.model.conf_pred_mask * unconformalized_acqf,
            x=self.model.conf_tgt_grid,
            dim=-1
        )
        return res


def generate_target_grid(bounds, resolution):
    target_dim = bounds.shape[1]
    grid_coords = [np.linspace(bounds[0,i], bounds[1,i], resolution) for i in range(target_dim)]
    target_grid = np.stack(np.meshgrid(*grid_coords), axis=-1)
    target_grid = torch.tensor(target_grid).view(-1, target_dim)
    return target_grid.float()


class ConformalPosterior(Posterior):
    def __init__(self, X, gp, target_bounds, alpha, tgt_grid_res, ratio_estimator=None):
        self.gp = gp
        self.X = X
        self.target_bounds = target_bounds
        self.tgt_grid_res = tgt_grid_res
        self.alpha = alpha
        self.ratio_estimator = ratio_estimator
        
    @property
    def device(self):
        return self.X.shape
    
    @property
    def dtype(self):
        return self.X.shape
    
    @property
    def event_shape(self):
        return self.X.shape[:-2] + torch.Size([1])
    
    def rsample(self, sample_shape=(), base_samples=None):
        target_grid = generate_target_grid(self.target_bounds, self.tgt_grid_res)
        target_grid = target_grid.to(self.X) 
        # for later on in the evaluation
        self.gp.conf_tgt_grid = target_grid.squeeze(-1)
        self.gp.conf_pred_mask, conditioned_gps = conformal_gp_regression(
            self.gp, self.X, target_grid, self.alpha, ratio_estimator=self.ratio_estimator
        )

        conditioned_gps.standard()
        reshaped_x = self.X[:, None].expand(-1, self.tgt_grid_res, -1, -1)
        posteriors = conditioned_gps.posterior(reshaped_x)
        out = posteriors.rsample(sample_shape, base_samples)
        return out
    

class PassSampler(MCSampler):
    def __init__(self, num_samples):
        super().__init__(batch_range=(0, -2))
        self._sample_shape = torch.Size([num_samples])
        self.collapse_batch_dims = True
        
    def _construct_base_samples(self, posterior, shape):
        pass


class ConformalSingleTaskGP(SingleTaskGP):
    def __init__(self, conformal_bounds, alpha, tgt_grid_res, ratio_estimator=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conformal_bounds = conformal_bounds
        self.alpha = alpha
        self.is_conformal = False
        self.tgt_grid_res = tgt_grid_res
        self.ratio_estimator = ratio_estimator
    
    def conformal(self):
        self.is_conformal = True
    
    def standard(self):
        self.is_conformal = False
        
    def posterior(self, X, observation_noise=False, posterior_transform=None):
        if self.is_conformal:
            posterior = ConformalPosterior(
                X, self, self.conformal_bounds, alpha=self.alpha, tgt_grid_res=self.tgt_grid_res,
                ratio_estimator=self.ratio_estimator
            )
            if hasattr(self, "outcome_transform"):
                posterior = self.outcome_transform.untransform_posterior(posterior)
            return posterior
        else:
            return super().posterior(
                X=X, observation_noise=observation_noise, posterior_transform=posterior_transform
            )
    
    @property
    def batch_shape(self):
        if self.is_conformal:
            try:
                return self.conf_pred_mask.shape
            except:
                pass
        return self.train_inputs[0].shape[:-2]        

