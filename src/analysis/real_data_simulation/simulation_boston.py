# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:07:27 2018

@author: Tobias Werner
"""

import sys
sys.path.append('../../BaggingSimulation')

from draw_bootstrap import BaggingTree
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pickle


try:
    from joblib import Parallel, delayed
except ImportError:
    sys.exit("""You need joblib to run the simulations!
                Install it from https://pythonhosted.org/joblib/
                or run:> pip install joblib.""")
# Speed up
import pstats
import cProfile


####Parameter####
n_repeat = 100  # Number of iterations for computing expectations
ratiotest = 0.1
ratiotrain = 1 - ratiotest

ratiobootstrap = 1
n_ratios = 30  # Number of iterations for the ratio list
B_iterations_ = 50
min_split_tree = 2    # As we want the lowest Bias Tree


random_seed_split = 13014
random_seed_fit = 30031

boston_full = load_boston()
boston_X = boston_full['data']
boston_y = boston_full['target']


def split_fit_predict_bagging(X, y, rnd_i, bootstrap=True, ratio=1):
    '''
    Split into test and training set. Note that the Random Seed is changed
    for each iteration but still deterministic overall

    '''
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ratiotest, random_state=random_seed_split + rnd_i)
    bagging_instance = BaggingTree(
        random_seed=random_seed_fit,
        ratio=ratio,
        bootstrap=bootstrap,
        B_iterations=B_iterations_,
        min_split_tree=min_split_tree
        )
    fitted_model = bagging_instance.fit(X_train, y_train)
    prediction = fitted_model.predict(X_test)

    y_mse = np.mean((y_test - prediction) ** 2)
    return y_mse


def simulate_bagging_parallel(X, y):
    ''' Runs the simulation for bagging in parallel.
    '''
    # Note that you can change n_jobs to 1 in case your system suffers too much
    mse_not_expected = Parallel(
        n_jobs=-2)(delayed(split_fit_predict_bagging)(X, y, i) for i in range(n_repeat))
    mse_sim = np.mean(mse_not_expected)
    return mse_sim

def simulate_subagging_parallel(X, y, n_ratios_):
    ''' Runs the simulation for subagging in parallel.
    '''
    # Note that you can change n_jobs to 1 in case your system suffers too much
    mse_subagging = np.ones(n_ratios) * np.nan
    ratiorange = np.linspace(0.1, 1, n_ratios)
    for idx, a in enumerate(ratiorange):
        mse_not_expected = (
                Parallel(n_jobs=-2)(delayed(split_fit_predict_bagging)(
                        X, y, i, bootstrap=False, ratio=a) for i in range(n_repeat))
        )
        mse_subagging[idx] = np.mean(mse_not_expected)
    return mse_subagging

if __name__ == '__main__':

    mse_bagging = simulate_bagging_parallel(boston_X, boston_y)
    mse_subagging = simulate_subagging_parallel(boston_X, boston_y, n_ratios_=n_ratios)
    print('MSE for Bagging: ', mse_bagging)
    print('MSE for the different subsampling ratios: \n', mse_subagging)
    simulated_data = {}
    simulated_data['ratiorange'] = np.linspace(0.1, 1, n_ratios)
    simulated_data['mse_bagging'] = mse_bagging
    simulated_data['mse_subagging'] = mse_subagging

    with open('../../../bld/out/analysis/RealData/output_boston.p',
          'wb') as f:
        pickle.dump(simulated_data, f)




#    prof_file = "Profile.prof"
#    cProfile.runctx(
#            "simulate_bagging_parallel()",
#            globals(),
#            locals(),
#            prof_file
#            )
#    s = pstats.Stats(prof_file)
#    s.strip_dirs().sort_stats("time").print_stats(10)
