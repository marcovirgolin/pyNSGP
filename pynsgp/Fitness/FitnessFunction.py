import numpy as np
from copy import deepcopy

class SymbolicRegressionFitness:

	def __init__( self, X_train, y_train, use_linear_scaling=True, use_interpretability_model=False ):
		self.X_train = X_train
		self.y_train = y_train
		self.y_train_var = np.var(y_train)
		self.use_linear_scaling = use_linear_scaling
		self.use_interpretability_model = use_interpretability_model
		self.elite = None
		self.elite_scaling_a = 0.0
		self.elite_scaling_b = 1.0
		self.evaluations = 0



	def Evaluate( self, individual ):

		self.evaluations = self.evaluations + 1
		individual.objectives = []

		obj1 = self.EvaluateMeanSquaredError(individual)
		# set in scale with the rest
		obj1 /= self.y_train_var * 100 
		individual.objectives.append( obj1 )

		if self.use_interpretability_model:
			obj2 = self.EvaluatePHIsModel(individual)
		else:
			obj2 = self.EvaluateNumberOfNodes(individual)	
		individual.objectives.append( obj2 )

		if not self.elite or individual.objectives[0] < self.elite.objectives[0]:
			del self.elite
			self.elite = deepcopy(individual)



	def EvaluateMeanSquaredError(self, individual):
		output = individual.GetOutput( self.X_train )

		a = 0.0
		b = 1.0

		if self.use_linear_scaling:
			b = np.cov(self.y_train, output)[0,1] / (np.var(output) + 1e-10)
			a = np.mean(self.y_train) - b*np.mean(output)
			individual.ls_a = a
			individual.ls_b = b

		scaled_output = a + b*output

		fit_error = np.mean( np.square( self.y_train - scaled_output ) )

		if np.isnan(fit_error):
			fit_error = np.inf
		
		return fit_error


	def EvaluateNumberOfNodes(self, individual):
		result = len(individual.GetSubtree())
		return result


	def EvaluatePHIsModel(self, individual):
		subtree = individual.GetSubtree()
		n_nodes = len(subtree)
		n_ops = 0
		n_naops = 0
		n_vars = 0
		dimensions = set()
		n_constants = 0
		for n in subtree:
			if n.arity > 0:
				n_ops += 1
				if n.is_not_arithmetic:
					n_naops += 1
			else:
				str_repr = str(n)
				if str_repr[0] == 'x':
					n_vars += 1
					idx = int(str_repr[1:len(str_repr)])
					dimensions.add(idx)
				else:
					n_constants += 1
		n_nacomp = individual.Count_n_nacomp()
		n_dim = len(dimensions)

		'''
		print('-------------------')
		print(subtree)
		print('nodes:',n_nodes)
		print('dimensions', n_dim)
		print('variables', n_vars)
		print('constants', n_constants)
		print('ops', n_ops)
		print('naops', n_naops)
		print('nacomp', n_nacomp)
		print('------------------')
		'''

		result = self._ComputeInterpretabilityScore( n_dim, n_vars, 
			n_constants, n_nodes, n_ops, n_naops, n_nacomp )
		result = -1 * result

		return result
		

	def _ComputeInterpretabilityScore(self, n_dim, n_vars, n_const, n_nodes, n_ops, na_ops, na_comp):
		# correctness weighted by confidence:
		features = [n_nodes, n_ops, na_ops, na_comp]
		coeffs = [-0.00195041, -0.00502375, -0.03351907, -0.04472121]
		result = np.sum(np.multiply(features, coeffs)) * 100
		return result	
