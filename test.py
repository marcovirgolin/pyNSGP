# Libraries
import numpy as np 
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from copy import deepcopy

from sympy import simplify

# Internal imports
from pynsgp.Nodes.BaseNode import Node
from pynsgp.Nodes.SymbolicRegressionNodes import *
from pynsgp.Fitness.FitnessFunction import SymbolicRegressionFitness
from pynsgp.Evolution.Evolution import pyNSGP

from pynsgp.SKLearnInterface import pyNSGPEstimator as NSGP

np.random.seed(42)

# Load regression dataset 
X, y = sklearn.datasets.load_boston( return_X_y=True )

#X = scale(X)
#y = scale(y)

# Take a dataset split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5, random_state=42 )

nsgp = NSGP(pop_size=100, max_generations=10, verbose=True, max_tree_size=32, 
	crossover_rate=0.9, mutation_rate=0.0, op_mutation_rate=1.0, min_depth=2, initialization_max_tree_height=6, 
	tournament_size=2, use_linear_scaling=True, use_erc=True, use_interpretability_model=True,
	functions = [ AddNode(), SubNode(), MulNode(), DivNode(), SinNode(), CosNode() ])

nsgp.fit(X_train,y_train)

front = nsgp.get_front()
population = nsgp.get_population()
print('len population:',len(population))
print('len front:',len(front))
for solution in front:
	simplified = simplify(solution.GetHumanExpression())
	print(solution.GetSubtree(), '$', solution.GetHumanExpression(), '$', simplified)

quit()

from sklearn.model_selection import cross_validate

cv_result = cross_validate(nsgp, X, y, scoring='neg_mean_squared_error', cv=5)

print (cv_result)

quit()







# Libraries
import numpy as np 
import sklearn.datasets
from sklearn.model_selection import train_test_split
from copy import deepcopy

# Internal imports
from pynsgp.Nodes.BaseNode import Node
from pynsgp.Nodes.SymbolicRegressionNodes import *
from pynsgp.Fitness.FitnessFunction import SymbolicRegressionFitness
from pynsgp.Evolution.Evolution import pyNSGP

np.random.seed(42)

# Load regression dataset 
X, y = sklearn.datasets.load_boston( return_X_y=True )
# Take a dataset split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5, random_state=42 )
# Set fitness function
fitness_function = SymbolicRegressionFitness( X_train, y_train )

# Set functions and terminals
functions = [ AddNode(), SubNode(), MulNode(), AnalyticQuotientNode() ]	# chosen function nodes	
terminals = [ EphemeralRandomConstantNode() ]	# use one ephemeral random constant node
for i in range(X.shape[1]):
	terminals.append(FeatureNode(i))	# add a feature node for each feature

# Run GP
nsgp = pyNSGP(fitness_function, functions, terminals, linear_scaling=True, erc=True, 
	pop_size=500, max_generations=100, crossover_rate=0.5, 
	mutation_rate=0.5, tournament_size=2, initialization_max_tree_height=6, max_tree_size=100)	# other parameters are optional
nsgp.Run()

# Print results
# Show the final front

final_front = nsgp.latest_front
print ('obj0, obj1, rank, crowddist, expr')
for p in final_front:
	print (np.round(p.objectives[0],3), p.objectives[1], p.rank, p.crowding_distance, p.GetSubtree())

'''
final_evolved_function = fitness_function.elite
nodes_final_evolved_function = final_evolved_function.GetSubtree()
print ('Function found (',len(nodes_final_evolved_function),'nodes ):\n\t', nodes_final_evolved_function) # this is in Polish notation
# Print results for training set
print ('Training\n\tMSE:', np.round(final_evolved_function.fitness,3), 
	'\n\tRsquared:', np.round(1.0 - final_evolved_function.fitness / np.var(y_train),3))
# Re-evaluate the evolved function on the test set
test_prediction = final_evolved_function.GetOutput( X_test )
test_mse = np.mean(np.square( y_test - test_prediction ))
print ('Test:\n\tMSE:', np.round( test_mse, 3), 
	'\n\tRsquared:', np.round(1.0 - test_mse / np.var(y_test),3))
'''
