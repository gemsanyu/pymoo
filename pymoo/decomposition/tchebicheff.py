import autograd.numpy as anp

from pymoo.model.decomposition import Decomposition


class Tchebicheff(Decomposition):

    def _do(self, F, weights, **kwargs):
        v = anp.abs(F - self.utopian_point) * weights
        tchebi = v.max(axis=1)
        return tchebi

class Tchebicheff2(Decomposition):

    def _do(self, F, weights, normalize=False, **kwargs):
        w = weights
        w[weights==0] = 10**6
        f = F
        if normalize:
            if self.nadir_point is None:
                raise("Tchebicheff2::_do: if normalize then nadir point must be provided")
            f = (f-self.utopian_point)/(self.nadir_point-self.utopian_point)

        v = anp.abs(f - self.utopian_point)/w
        tchebi = v.max(axis=1)
        return tchebi
