import numpy as np

from pynsgp.Nodes.BaseNode import Node

class AddNode(Node):
	
	def __init__(self):
		super(AddNode,self).__init__()
		self.arity = 2

	def __repr__(self):
		return '+'

	def _GetHumanExpressionSpecificNode( self, args ):
		return '( ' + args[0] + ' + ' + args[1] + ' )'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		X1 = self._children[1].GetOutput( X )
		return X0 + X1

class SubNode(Node):
	def __init__(self):
		super(SubNode,self).__init__()
		self.arity = 2

	def __repr__(self):
		return '-'

	def _GetHumanExpressionSpecificNode( self, args ):
		return '( ' + args[0] + ' - ' + args[1] + ' )'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		X1 = self._children[1].GetOutput( X )
		return X0 - X1

class MulNode(Node):
	def __init__(self):
		super(MulNode,self).__init__()
		self.arity = 2

	def __repr__(self):
		return '*'

	def _GetHumanExpressionSpecificNode( self, args ):
		return '( ' + args[0] + ' * ' + args[1] + ' )'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		X1 = self._children[1].GetOutput( X )
		return np.multiply(X0 , X1)
	
class DivNode(Node):
	def __init__(self):
		super(DivNode,self).__init__()
		self.arity = 2

	def __repr__(self):
		return '/'

	def _GetHumanExpressionSpecificNode( self, args ):
		return '( ' + args[0] + ' / ' + args[1] + ' )'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		X1 = self._children[1].GetOutput( X )
		return np.multiply( np.sign(X1), X0) / ( 1e-6 + np.abs(X1) )

class AnalyticQuotientNode(Node):
	def __init__(self):
		super(AnalyticQuotientNode,self).__init__()
		self.arity = 2
		self.is_not_arithmetic = True

	def __repr__(self):
		return 'aq'

	def _GetHumanExpressionSpecificNode( self, args ):
		return '( ' + args[0] + ' / sqrt( 1 + ' + args[1] + '**2 ) )'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		X1 = self._children[1].GetOutput( X )
		return X0 / np.sqrt( 1 + np.square(X1) )

class PowNode(Node):

	def __init__(self):
		super(PowNode,self).__init__()
		self.arity = 2
		self.is_not_arithmetic = True

	def __repr__(self):
		return '^'

	def _GetHumanExpressionSpecificNode( self, args ):
		return '( '+args[0]+'**( ' + args[0] + ' ))'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		X1 = self._children[1].GetOutput( X )
		return np.power(X0, X1)

	
class ExpNode(Node):
	def __init__(self):
		super(ExpNode,self).__init__()
		self.arity = 1
		self.is_not_arithmetic = True

	def __repr__(self):
		return 'exp'

	def _GetHumanExpressionSpecificNode( self, args ):
		return 'exp( ' + args[0] + ' )'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		return np.exp(X0)


class LogNode(Node):
	def __init__(self):
		super(LogNode,self).__init__()
		self.arity = 1
		self.is_not_arithmetic = True

	def __repr__(self):
		return 'log'

	def _GetHumanExpressionSpecificNode( self, args ):
		return 'log( ' + args[0] + ' )'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		return np.log( np.abs(X0) + 1e-6 )


class SinNode(Node):
	def __init__(self):
		super(SinNode,self).__init__()
		self.arity = 1
		self.is_not_arithmetic = True

	def __repr__(self):
		return 'sin'

	def _GetHumanExpressionSpecificNode( self, args ):
		return 'sin( ' + args[0] + ' )'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		return np.sin(X0)

class CosNode(Node):
	def __init__(self):
		super(CosNode,self).__init__()
		self.arity = 1
		self.is_not_arithmetic = True

	def __repr__(self):
		return 'cos'

	def _GetHumanExpressionSpecificNode( self, args ):
		return 'cos( ' + args[0] + ' )'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		return np.cos(X0)


class FeatureNode(Node):
	def __init__(self, id):
		super(FeatureNode,self).__init__()
		self.id = id

	def __repr__(self):
		return 'x'+str(self.id)

	def _GetHumanExpressionSpecificNode( self, args ):
		return 'x'+str(self.id)

	def GetOutput(self, X):
		return X[:,self.id]

	
class EphemeralRandomConstantNode(Node):
	def __init__(self):
		super(EphemeralRandomConstantNode,self).__init__()
		self.c = np.nan

	def __Instantiate(self):
		self.c = np.round( np.random.random() * 10 - 5, 3 )

	def __repr__(self):
		if np.isnan(self.c):
			self.__Instantiate()
		return str(self.c)

	def _GetHumanExpressionSpecificNode( self, args ):
		if np.isnan(self.c):
			self.__Instantiate()
		return str(self.c)

	def GetOutput(self,X):
		if np.isnan(self.c):
			self.__Instantiate()
		return np.array([self.c] * X.shape[0])
