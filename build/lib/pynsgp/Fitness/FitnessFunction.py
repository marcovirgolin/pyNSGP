import numpy as np
from copy import deepcopy

class SymbolicRegressionFitness:

	def __init__( self, X_train, y_train, use_linear_scaling=True ):
		self.X_train = X_train
		self.y_train = y_train
		self.use_linear_scaling = use_linear_scaling
		self.elite = None
		self.elite_scaling_a = 0.0
		self.elite_scaling_b = 1.0
		self.evaluations = 0

	def Evaluate( self, individual ):

		self.evaluations = self.evaluations + 1

		output = individual.GetOutput( self.X_train )

		a = 0.0
		b = 1.0

		if self.use_linear_scaling:
			b = np.cov(self.y_train, output)[0,1] / (np.var(output) + 1e-10)
			a = np.mean(self.y_train) - b*np.mean(output)
			individal.ls_a = a
			individal.ls_b = b

		scaled_output = a + b*output

		fit_error = np.mean( np.square( self.y_train - scaled_output ) )
		if np.isnan(fit_error):
			fit_error = np.inf
		
		individual.objectives = []
		individual.objectives.append( fit_error )
		individual.objectives.append( len(individual.GetSubtree()) )

		if not self.elite or individual.fitness < self.elite.fitness:
			del self.elite
			self.elite = deepcopy(individual)