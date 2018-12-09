import numpy as np
from scipy.stats import norm
from models.option import OptionRight


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
        if self.right is OptionRight.CALL:
            return max(0.0, self.F - self.K)
        if self.right is OptionRight.PUT:
            return max(0.0, self.K - self.F)
        if self.right is OptionRight.STRADDLE:
            return max(self.K - self.F, self.F - self.K)
        raise Exception(f"Unexpected option right {self.right}")

    def __repr__(self):
        return f"BlackScholes: {str(self.__dict__)}"
