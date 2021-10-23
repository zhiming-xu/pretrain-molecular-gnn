from torch import nn
import math
import numpy as np


def shifted_softplus(x):
    return nn.functional.softplus(x) - math.log(2.)


def softplus_inverse(x):
    return x + np.log(-np.expm1(-x))