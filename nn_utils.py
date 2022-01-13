#!/usr/bin/env python
# -*- coding: utf-8 -*-
from torch import nn
import math
import numpy as np


def shifted_softplus(x):
    return nn.functional.softplus(x) - math.log(2.)


def softplus_inverse(x):
    return x + np.log(-np.expm1(-x))


class CyclicKLWeight:
    def __init__(self, cycle_epoch, high=1.) -> None:
        self.cycle_epoch = cycle_epoch
        self.high = high
        self.iter = 0

    def step(self):
        self.iter += 1
        cur = self.iter % self.cycle_epoch
        if cur <= self.cycle_epoch // 2:
            return self.high / self.cycle_epoch * 2 * cur
        else:
            return self.high