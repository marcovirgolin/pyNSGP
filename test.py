# Libraries
import numpy as np 
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from copy import deepcopy

# Internal imports
from pynsgp.Nodes.BaseNode import Node
from pynsgp.Nodes.SymbolicRegressionNodes import *
from pynsgp.Fitness.FitnessFunction import SymbolicRegressionFitness
from pynsgp.Evolution.Evolution import pyNSGP

from pynsgp.SKLearnInterface import pyNSGPEstimator as NSGP

np.random.seed(42)

# Load regression dataset 
X, y = sklearn.datasets.load_boston( return_X_y=True )

# Take a dataset split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5, random_state=42 )

# Prepare NSGP settings
nsgp = NSGP(pop_size=512, max_generations=50, verbose=True, max_tree_size=50, 
	crossover_rate=0.8, mutation_rate=0.1, op_mutation_rate=0.1, min_depth=2, initialization_max_tree_height=6, 
	tournament_size=2, use_linear_scaling=True, use_erc=True, use_interpretability_model=True,
	functions = [ AddNode(), SubNode(), MulNode(), DivNode(), LogNode(), SinNode(), CosNode() ])

# Fit like any sklearn estimator
nsgp.fit(X_train,y_train)

# Obtain the front of non-dominated solutions (according to the training set)
front = nsgp.get_front()
print('len front:',len(front))
for solution in front:
	print(solution.GetHumanExpression())

# You can also use sympy to simplify the formulas :) (if you use PowNode, replace ^ to ** before use)
'''
from sympy import simplify
for solution in front:
	simplified = simplify(solution.GetHumanExpression())
	print(simplified)
'''
	
# You can use cross-validation, hyper-parameter tuning, anything a sklearn estimator can normally do
from sklearn.model_selection import cross_validate

cv_result = cross_validate(nsgp, X, y, scoring='neg_mean_squared_error', cv=5)

print (cv_result)
