from copy import deepcopy
import numpy as np
from numpy.random import randint
from numpy.random import random


def GenerateRandomTree(functions, terminals, max_height, curr_height=0, method='grow', min_depth=2):

	if curr_height == max_height:
		idx = randint(len(terminals))
		n = deepcopy( terminals[idx] )
	else:
		if method == 'grow' and curr_height >= min_depth:
			term_n_funs = terminals + functions
			idx = randint( len(term_n_funs) )
			n = deepcopy( term_n_funs[idx] )
		elif method == 'full' or (method == 'grow' and curr_height < min_depth):
			idx = randint( len(functions) )
			n = deepcopy( functions[idx] )
		else:
			raise ValueError('Unrecognized tree generation method')

		for i in range(n.arity):
			c = GenerateRandomTree( functions, terminals, max_height, curr_height=curr_height + 1, method=method )
			n.AppendChild( c )
	return n

def OnePointMutation( individual, functions, terminals ):
	
	arity_functions = {}
	for f in functions:
		arity = f.arity
		if arity not in arity_functions:
			arity_functions[arity] = [f]
		else:
			arity_functions[arity].append(f)

	nodes = individual.GetSubtree()
	prob = 1.0/len(nodes)

	for i in range(len(nodes)):
		if random() < prob:
			arity = nodes[i].arity 
			if arity == 0:
				idx = randint( len(terminals) )
				n = deepcopy( terminals[idx] )
			else:
				idx = randint(len(arity_functions[arity]))
				n = deepcopy(arity_functions[arity][idx])
			
			# update link to children
			for child in nodes[i]._children:
				n.AppendChild(child)
				
			# update link to parent node
			p = nodes[i].parent
			if p:
				idx = p.DetachChild( nodes[i] )
				p.InsertChildAtPosition(idx, n)
			else:
				nodes[i] = n
				individual = n
			

	return individual




def SubtreeMutation( individual, functions, terminals, max_height=4 ):

	mutation_branch = GenerateRandomTree( functions, terminals, max_height )
	
	nodes = individual.GetSubtree()

	#nodes = __GetCandidateNodesAtUniformRandomDepth( nodes )

	to_replace = nodes[randint(len(nodes))]

	if not to_replace.parent:
		del individual
		return mutation_branch


	p = to_replace.parent
	idx = p.DetachChild(to_replace)
	p.InsertChildAtPosition(idx, mutation_branch)

	return individual


def SubtreeCrossover( individual, donor ):
	
	# this version of crossover returns 1 child

	nodes1 = individual.GetSubtree()
	nodes2 = donor.GetSubtree()	# no need to deep copy all nodes of parent2

	#nodes1 = __GetCandidateNodesAtUniformRandomDepth( nodes1 )
	#nodes2 = __GetCandidateNodesAtUniformRandomDepth( nodes2 )

	to_swap1 = nodes1[ randint(len(nodes1)) ]
	to_swap2 = deepcopy( nodes2[ randint(len(nodes2)) ] )	# we deep copy now, only the sutbree from parent2

	p1 = to_swap1.parent

	if not p1:
		return to_swap2

	idx = p1.DetachChild(to_swap1)
	p1.InsertChildAtPosition(idx, to_swap2)

	return individual


def __GetCandidateNodesAtUniformRandomDepth( nodes ):

	depths = np.unique( [x.GetDepth() for x in nodes] )
	chosen_depth = depths[randint(len(depths))]
	candidates = [x for x in nodes if x.GetDepth() == chosen_depth]

	return candidates
