from .baseCounter import baseCounter
import math

class Counter2(baseCounter):
    '''
    SS-FL-V2
    '''
    def __init__(self, rate, m_p, dim, m, e_l, bound='bennett', mechanism='laplace', k=2):
        super(Counter2, self).__init__(dim=dim, m=m, e_l=e_l, bound=bound, mechanism=mechanism, k=2)
        self.rate = rate
        self.m_p = m_p
        self.split_d = int(dim/rate)
        self.compose_d = 2*self.split_d
        self.e_ld = self.split(e_l, self.split_d)
        self.d_cd = self.decompose_d(self.compose_d)
        self.d_ck = self.d_cd * self.rate
        assert self.check_el(n=self.m_p, delta=self.d_ck), "ERROR: eps_ld is too large"
        self.e_ck = self.amplify(self.e_ld, self.m_p, self.d_ck)
        self.e_cd = self.amplify_sup(self.e_ck, self.rate)
        self.e_c = self.compose_e(self.e_cd, self.d_cd, self.compose_d)

    def no_sub_amplification(self):
        print("If the amplification of subsampling is not counted...")
        self.e_cd = self.amplify(self.e_ld, self.m_p, self.d_cd)
        self.e_c = self.compose_e(self.e_cd, self.d_cd, self.compose_d)
        print("{}-LDP on vector-level".format(self.e_l))
        print("{}-LDP on dimension-level".format(self.e_ld))
        print("({}, {})-DP on dimension-level".format(self.e_cd, self.d_cd))
        print("({}, {})-DP on vector-level".format(self.e_c, self.d_c))