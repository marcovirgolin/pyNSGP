import numpy as np


class Node:	# Base class with general functionalities

	def __init__(self):
		self.objectives = []
		self.parent = None
		self.arity = 0	# arity is the number of expected inputs
		self.rank = 0
		self.crowding_distance = 0
		self._children = []
		self.ls_a = 0.0
		self.ls_b = 1.0

	def Dominates(self, other):
		better_somewhere = False 
		for i in range(len(self.objectives)):
			if self.objectives[i] > other.objectives[i]:
				return False
			if self.objectives[i] < other.objectives[i]:
				better_somewhere = True 

		return better_somewhere

	def GetSubtree( self ):
		result = []
		self.__GetSubtreeRecursive(result)
		return result

	def AppendChild( self, N ):
		self._children.append(N)
		N.parent = self

	def DetachChild( self, N ):
		assert(N in self._children)
		for i, c in enumerate(self._children):
			if c == N:
				self._children.pop(i)
				N.parent = None
				break
		return i

	def InsertChildAtPosition( self, i, N ):
		self._children.insert( i, N )
		N.parent = self

	def GetOutput( self, X ):
		return None

	def GetDepth(self):
		n = self
		d = 0
		while (n.parent):
			d = d+1
			n = n.parent
		return d

	def GetHeight(self):
		subtree = self.GetSubtree()
		leaves = [x for x in subtree if x.arity == 0]
		max_h = 0
		for l in leaves:
			d = l.GetDepth()
			if d > max_h:
				max_h = d
		return max_h


	def __GetSubtreeRecursive( self, result ):
		result.append(self)
		for c in self._children:
			c.__GetSubtreeRecursive( result )
		return result

