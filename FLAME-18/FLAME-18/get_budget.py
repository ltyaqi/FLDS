import sys
from flearn.counter.ssflv1 import *
from flearn.counter.ssflv2 import *
from flearn.counter.baseCounter import *
import math
from sympy import *
import numpy as np

'''
instructions of setting epsilon in Figure 5
'''

e_l = 78.5
c1 = Counter1(dim=7850, m=1000, e_l=e_l)
c2 = Counter2(rate=50, m_p=int(1000/3), dim=7850, m=1000, e_l=e_l)

c1.print()
c2.print()
c2.no_sub_amplification()