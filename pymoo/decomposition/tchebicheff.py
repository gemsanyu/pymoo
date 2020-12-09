import autograd.numpy as anp

from pymoo.model.decomposition import Decomposition


class Tchebicheff(Decomposition):

    def __init__(self,
                 eps=0.0,
                 _type="auto",
                 variant=None,
                 **kwargs):
        """
        Parameters
        variant :   None / "original" -> the original one
                    'augmented' augmented weighted Tchebicheff (Steuer, 1986)
                    'modified' modified weighted Tchebicheff (Steuer and Choo, 1983)

        """
        super().__init__(eps, _type, **kwargs)
        self.variant = variant
        if self.variant is None:
            self.variant = "original"


    def _do_original(self, F, weights, **kwargs):
        v = anp.abs(F - self.utopian_point) * weights
        tchebi = v.max(axis=1)
        return tchebi

    def _do_augmented(self, F, weights, rho=0.1, **kwargs):
        F_normalized = anp.abs(F - self.utopian_point)
        v = F_normalized*weights + rho*anp.sum(F_normalized)
        tchebi = v.max(axis=1)
        return tchebi

    def _do_modified(self, F, weights, rho=0.1, **kwargs):
        F_normalized = anp.abs(F - self.utopian_point)
        v = F_normalized*weights + rho*anp.sum(F_normalized)
        tchebi = v.max(axis=1)
        return tchebi

    def _do(self, F, weights, rho=0.1, **kwargs):
        if self.variant is "augmented":
            return self._do_augmented(F, weights, rho, **kwargs)
        elif self.variant is "modified":
            return self._do_modified(F, weights, rho, **kwargs)
        else:
            return self._do_original(F, weights, **kwargs)
