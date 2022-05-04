from botorch.acquisition.multi_objective import (
   qExpectedHypervolumeImprovement,
   qNoisyExpectedHypervolumeImprovement,
)

from acquisitions import ConformalAcquisition

class qConformalExpectedHypervolumeImprovement(ConformalAcquisition, qExpectedHypervolumeImprovement):
    def __init__(self, alpha, temp, grid_res, max_grid_refinements, ratio_estimator,
                 optimistic=False, grid_sampler=None, *args, **kwargs):
        qExpectedHypervolumeImprovement.__init__(self, *args, **kwargs)
        ConformalAcquisition.__init__(
            self, alpha, temp, grid_res, max_grid_refinements, ratio_estimator,
            optimistic, grid_sampler
        )

    def _nonconformal_fwd(self, X, conditioned_model):
        old_model = self.model
        self.model = conditioned_model
        res = qExpectedHypervolumeImprovement(self, X)
        self.model = old_model
        return res

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X):
        return ConformalAcquisition.forward(self, X)


class qConformalNoisyExpectedHypervolumeImprovement(ConformalAcquisition, qNoisyExpectedHypervolumeImprovement):
    def __init__(self, alpha, temp, grid_res, max_grid_refinements, ratio_estimator,
                 optimistic=False, grid_sampler=None, *args, **kwargs):
        qNoisyExpectedHypervolumeImprovement.__init__(self, *args, **kwargs)
        ConformalAcquisition.__init__(
            self, alpha, temp, grid_res, max_grid_refinements, ratio_estimator,
            optimistic, grid_sampler
        )

    def _nonconformal_fwd(self, X, conditioned_model):
        old_model = self.model
        self.model = conditioned_model
        res = qNoisyExpectedHypervolumeImprovement(self, X)
        self.model = old_model
        return res

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X):
        return ConformalAcquisition.forward(self, X)
