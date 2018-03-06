from baggingtree import BaggingTree
from datasimulation import DataSimulation
import numpy as np
from sklearn.tree import DecisionTreeRegressor
X, y = DataSimulation().friedman_1_model()

b = BaggingTree(ratio=1.0,b_iterations=1, min_split_tree=2, random_seed=12, bootstrap=True).fit(X,y).predict(np.ones((50,10)))


a = DecisionTreeRegressor(min_samples_split=99999).fit(X,y).fit(X,y)