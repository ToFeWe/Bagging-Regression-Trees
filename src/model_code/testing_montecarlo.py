# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 16:14:08 2018

@author: gl1rk
"""

from montecarlosimulation import MonteCarloSimulation

a = MonteCarloSimulation(n_test_train=[1,1]).calc_mse_all_ratios(min_ratio=0.9, n_ratios=2)