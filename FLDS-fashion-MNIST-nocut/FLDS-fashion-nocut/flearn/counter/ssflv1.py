from .baseCounter import baseCounter
import math

class Counter1(baseCounter):
    '''
    SS-FL-V1
    '''
    def __init__(self, dim, m, e_l, bound='bennett', mechanism='laplace', k=2):
        super(Counter1, self).__init__(dim=dim, m=m, e_l=e_l, bound=bound, mechanism=mechanism, k=2)
        self.split_d = dim
        self.compose_d = dim
        self.e_ld = self.split(e_l, self.split_d)
        self.d_cd = self.decompose_d(self.compose_d)
        assert self.check_el(n=self.m, delta=self.d_cd), "ERROR: eps_ld is too large"
        self.e_cd = self.amplify(self.e_ld, self.m, self.d_cd)
        self.e_c = self.compose_e(self.e_cd, self. d_cd, self.compose_d)

    def print_details(self):
        print()