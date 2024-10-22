from flearn.counter.mechanisms import *
from flearn.counter.amplification_bounds import *
import math
from sympy import *

class baseCounter(object):
    def __init__(self, dim, m, e_l, bound='bennett', mechanism='laplace', k=2):
        self.dim = dim
        self.m = m
        self.e_l = e_l
        self.d_c = 0.01/math.pow(self.m, 1.1)
        if mechanism == 'laplace':
            self.mechanism = LaplaceMechanism()
        elif mechanism == 'krr':
            self.mechanism = RRMechanism(k=k)
        else:
            raise ValueError('Please choose mechanism with laplace or krr')
        
        if bound == 'hoeffding':
            self.bound = Hoeffding(self.mechanism)
        elif bound == 'bennett':
            self.bound = BennettExact(self.mechanism)
        else:
            raise ValueError('Please choose bound with hoeffding or bennett')

    def check_el(self, n, delta):
        '''
        check eps_l
        '''
        if self.e_ld < math.log(n/math.log(1/delta))/2:
            return True
        else:
            return False


    def split(self, e_l, split_d):
        '''
        get splitted local eps for each dimension
        '''
        return e_l/split_d

    def decompose_d(self, compose_d):
        '''
        get central delta for each dimension before composition
        - for SS-FL-V1, shuffling
        - for SS-FL-V2, shuffling + subsampling
        '''
        return self.d_c / (compose_d + 1)

    def amplify(self, e_lk, m_p, d_ck):
        '''
        get amplified central (eps, delta) for each dimension
        - e_lk: local privacy budget for each dimension under the local DP
        - m_p: # of messages
        - d_ck: target delta for each dimension under the central DP
        '''
        return self.bound.get_eps(e_lk, m_p, d_ck)

    def amplify_sup(self, eps, rate):
        return math.log(1 + (math.exp(eps) - 1)*1.0/rate )

    def compose_e(self, e_cd, d_cd, T):
        '''
        get composed central eps for the vector
        '''
        sequential = T*e_cd
        advanced = math.sqrt(2*T*math.log(1/d_cd))*e_cd + T*e_cd*(math.exp(e_cd)-1)
        if advanced < sequential:
            return advanced
        else:
            print("sequantial composition is working...")
            return sequential

    def print(self):
        '''
        print
        '''
        print("{}-LDP on vector-level".format(self.e_l))
        print("{}-LDP on dimension-level".format(self.e_ld))
        print("({}, {})-DP on dimension-level".format(self.e_cd, self.d_cd))
        print("({}, {})-DP on vector-level".format(self.e_c, self.d_c))