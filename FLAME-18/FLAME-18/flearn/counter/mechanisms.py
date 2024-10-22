from math import exp


class LDPMechanism:
    """Base class implementing parameter computations for a generic Local Randomizer.
    For now we only support randomizers satisfying pure differential privacy.
    """

    def __init__(self, eps0=1, name='Generic'):
        """Parameters:
        eps0 (float): Privacy parameter
        name (str): Randomizer's name
        """
        self.eps0 = eps0
        self.name = name

    def get_name(self):
        return self.name

    def set_eps0(self, eps0):
        self.eps0 = eps0

    def get_eps0(self):
        return self.eps0

    def get_gamma(self):
        """Returns upper and lower bounds for gamma, the blanket probability of the randomizer.
        This function implements a generic bound which holds for any pure DP local randomizer.
        """
        return exp(-self.get_eps0()), 1

    def get_max_l(self, eps):
        """Returns the maximum value of the privacy amplification random variable.
        This function implements a generic bound which hold for any pure DP local randomizer.
        """
        _, gamma_ub = self.get_gamma()
        eps0 = self.get_eps0()
        return gamma_ub * exp(eps0) * (1-exp(eps-2*eps0))

    def get_range_l(self, eps):
        """Returns the range of the privacy amplification random variable.
        This function implements a generic bound which hold for any pure DP local randomizer.
        """
        _, gamma_ub = self.get_gamma()
        eps0 = self.get_eps0()
        return gamma_ub * (exp(eps)+1) * (exp(eps0)-exp(-eps0))

    def get_var_l(self, eps):
        """Returns the variance of the privacy amplification random variable.
        This function implements a generic bound which hold for any pure DP local randomizer.
        """
        gamma_lb, gamma_ub = self.get_gamma()
        eps0 = self.get_eps0()
        return gamma_ub * (exp(eps0) * (exp(2*eps)+1) - 2 * gamma_lb * exp(eps-2*eps0))


class LaplaceMechanism(LDPMechanism):
    """Class implementing parameter computation for a Laplace mechanism with inputs in [0,1].
    Bounds below are specialized exact calculations for this mechanism.
    """
    def __init__(self, eps0=1, name='Laplace'):
        super(LaplaceMechanism, self).__init__(eps0=eps0, name=name)

    def get_gamma(self):
        gamma = exp(-self.get_eps0()/2)
        return gamma, gamma

    def get_max_l(self, eps):
        eps0 = self.get_eps0()
        return exp(eps0/2) * (1-exp(eps-eps0))

    def get_range_l(self, eps):
        eps0 = self.get_eps0()
        return (exp(eps)+1) * (exp(eps0/2)-exp(-eps0/2))

    def get_var_l(self, eps):
        eps0 = self.get_eps0()
        return (exp(2*eps)+1)/3 * (2*exp(eps0/2)+exp(-eps0)) - 2 * exp(eps) * (2*exp(-eps0/2) - exp(-eps0))


class RRMechanism(LDPMechanism):
    """Class implementing parameter computation for a k-ary randomized response mechanism
    Bounds below are specialized exact calculations for this mechanism.
    """

    def __init__(self, eps0=1, k=2, name='RR'):
        super(RRMechanism, self).__init__(eps0=eps0, name=name)
        self.k = k

    def get_name(self, with_k=True):
        name = self.name
        if with_k:
            name += '-{}'.format(self.get_k())
        return name

    def set_k(self, k):
        self.k = k

    def get_k(self):
        return self.k

    def get_gamma(self):
        k = self.get_k()
        eps0 = self.get_eps0()
        gamma = k/(exp(eps0) + k - 1)
        return gamma, gamma

    def get_max_l(self, eps):
        k = self.get_k()
        gamma, _ = self.get_gamma()
        return gamma * (1 - exp(eps)) + (1-gamma) * k

    def get_range_l(self, eps):
        k = self.get_k()
        gamma, _ = self.get_gamma()
        return (1-gamma) * k * (exp(eps)+1)

    def get_var_l(self, eps):
        k = self.get_k()
        gamma, _ = self.get_gamma()
        return gamma * (2-gamma) * (exp(eps)-1)**2 + (1-gamma)**2 * k * (exp(2*eps) + 1)
