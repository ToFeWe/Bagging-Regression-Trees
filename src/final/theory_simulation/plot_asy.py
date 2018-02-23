# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 17:42:26 2017

@author: Tobias Werner
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

path_to_file = '../../../../bld/out/analysis/TheoryPartSimulation/toy_example/output_simulation_asy.p'
with open(path_to_file,'rb') as f:
        output_simulation = pickle.load(f)


c_range = output_simulation['c_range']

# drop c_range from dictonary
output_simulation.pop('c_range')

plt.style.use(['seaborn-white'])
fig, ax = plt.subplots(figsize=(10, 3), ncols=3)

# Variance Subplot
ax[0].plot(c_range,output_simulation['bagged_var'], color='tab:blue')
ax[0].plot(c_range,output_simulation['nonbagged_var'], color='tab:olive')
ax[0].xaxis.set_ticks(np.arange(-4,4+1,2))
ax[0].set_title('$Variance$')
ax[0].set_xlabel('$c$')

# Bias Subplot

ax[1].plot(c_range, output_simulation['bagged_bias'],
           label = r'$\hat{\theta}_{n;B}(x_{n}(c))$', 
           color='tab:blue'
)
ax[1].plot(c_range,np.zeros(c_range.size), 
           label = r'$\hat{\theta}_{n}(x_{n}(c))$', 
           color='tab:olive'
)
ax[1].xaxis.set_ticks(np.arange(-4,4+1,2))
ax[1].set_title('$Bias^{2}$')
ax[1].set_xlabel('$c$')
ax[1].legend(bbox_to_anchor=(-0.25, -0.6),ncol=2 ,loc='lower left',frameon=True, fontsize=15)



# AMSE Subplot
ax[2].plot(c_range,
  			 np.add(output_simulation['bagged_bias'],
             		 output_simulation['bagged_var']), 
           color='tab:blue'
)
ax[2].plot(c_range,output_simulation['nonbagged_var'], color='tab:olive')
ax[2].xaxis.set_ticks(np.arange(-4,4+1,2))
ax[2].set_title('$AMSE$')
ax[2].set_xlabel('$c$')

fig.tight_layout(pad=0.4, w_pad=1, h_pad=2.5)

fig.savefig('../../../../bld/out/figures/TheoryPartSimulation/'+
            'toy_example/plot_asy.pdf', bbox_inches='tight')


