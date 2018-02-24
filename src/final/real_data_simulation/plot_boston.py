# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:29:29 2017

@author: Tobias Werner

"""

import numpy as np
import pickle
import matplotlib.pyplot as plt


#Settings
plt.style.use(['seaborn-white'])


path_to_file = '../../../../bld/out/analysis/RealData/output_boston.p'
with open(path_to_file,'rb') as f:
        output_boston = pickle.load(f)

# Init values from dict
ratiorange = output_boston['ratiorange']
n_ratios = ratiorange.shape[0]
mse_plot_bagged = np.ones(n_ratios) * output_boston['mse_bagging']
# For ratio=1 subagging is the same as fitting a single tree
mse_plot_tree = np.ones(n_ratios) * output_boston['mse_subagging'][-1]
mse_plot_subagging = output_boston['mse_subagging']

plt.figure(figsize=(10, 10))

plt.plot(ratiorange,mse_plot_subagging, color='tab:red',
        label = '$MSE \: Subagging$')
plt.plot(ratiorange,mse_plot_bagged, color='tab:blue', label = '$MSE \: Bagging$')
plt.plot(ratiorange,mse_plot_tree, color='tab:olive',  label = '$MSE \: Tree$')
plt.xlabel('$a$')
plt.ylim(ymin=0)
plt.title(('$MSE \: for \: Boston \: Housing \:Data$'))




plt.legend(bbox_to_anchor=(0.13, -0.13), ncol=3,loc='lower left',frameon=True, fontsize=15)
plt.tight_layout(pad=0.4, w_pad=1, h_pad=2.5)

plt.savefig('../../../../bld/out/figures/RealData/'+
            '/plot_boston.pdf', bbox_inches='tight')
