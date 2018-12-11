import numpy as np
from scipy.stats import norm
from models.option import OptionRight

class OptionCalcs:
    def intrinsic(self, right, F, K):
        if right is OptionRight.CALL:
            return max(0.0, F - K)
        if right is OptionRight.PUT:
            return max(0.0, K - F)
        if right is OptionRight.STRADDLE:
            return max(K - F, F - K)
        raise Exception(f"Unexpected option right {right}")

class BlackScholes:
    def __init__(self, F, K, right, sigma, T):
        self.F = F
        self.K = K
        self.right = right
        self.sigma = sigma
        self.T = T
        self.d1 = 1.0 / (sigma * np.sqrt(T)) * (np.log(F / K) + 0.5 * sigma * sigma * T)
        self.d2 = self.d1 - sigma * np.sqrt(T)
        self.N1 = norm.cdf(self.d1)
        self.N2 = norm.cdf(self.d2)


    def undiscounted_value(self):
        if self.right is OptionRight.CALL:
            return self.N1 * self.F - self.N2 * self.K
        if self.right is OptionRight.PUT:
            return (1.0 - self.N2) * self.K - (1. - self.N1) * self.F
        if self.right is OptionRight.STRADDLE:
            return (1.0 - 2.0 * self.N2) * self.K - (1. - 2.0 * self.N1) * self.F
        raise Exception(f"Unexpected option right {self.right}")

    def intrinsic(self):
        return OptionCalcs().intrinsic(self.right, self.F, self.K)

    def dup(self, F=None, K=None, right=None, sigma=None, T=None):
        return BlackScholes(
            F or self.F,
            K or self.K,
            right or self.right,
            sigma or self.sigma,
            T or self.T
        )


    def __repr__(self):
        return f"BlackScholes: {str(self.__dict__)}"
