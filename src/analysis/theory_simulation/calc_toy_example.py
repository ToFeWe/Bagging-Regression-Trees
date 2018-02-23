# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 18:52:10 2017

@author: gl1rk
"""

import numpy as np
from scipy.stats import norm
import pickle
import scipy.integrate as integrate

#number of gridpoints
c_gridpoints = 100
c_range = np.linspace(-5, 5, num=c_gridpoints)

#initalize the array that will be used to save the values
bagged_var = np.ones(c_gridpoints) * np.nan
nonbagged_var = np.ones(c_gridpoints) * np.nan
bagged_bias = np.ones(c_gridpoints) * np.nan

# Save results to dict
simulation_results = {}

# Save the c_range
simulation_results['c_range'] = c_range

#Calculate the
for index, c in enumerate(c_range):
    bagged_var[index] = (
            integrate.quad(lambda y: norm.cdf(c-y) ** 2 * norm.pdf(y), -np.inf,np.inf)[0]
            -
            integrate.quad(lambda y: norm.cdf(c-y) * norm.pdf(y), -np.inf,np.inf)[0]**2
                        )
    nonbagged_var[index] = norm.cdf(c) * (1 - norm.cdf(c))

    bagged_bias[index] = (
                          integrate.quad(lambda y: norm.cdf(c-y) * norm.pdf(y), -np.inf,np.inf)[0] -
                          norm.cdf(c)
                          ) ** 2

simulation_results['bagged_var'] = bagged_var
simulation_results['bagged_bias'] = bagged_bias
simulation_results['nonbagged_var'] = nonbagged_var



with open('../../../bld/out/analysis/TheoryPartSimulation/toy_example/output_simulation_asy.p',
          'wb') as f:
    pickle.dump(simulation_results, f)
