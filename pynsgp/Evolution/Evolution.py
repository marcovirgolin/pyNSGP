import numpy as np
from numpy.random import random, randint
import time
from copy import deepcopy

from pynsgp.Variation import Variation
from pynsgp.Selection import Selection


class pyNSGP:

	def __init__(
		self,
		fitness_function,
		functions,
		terminals,
		pop_size=500,
		crossover_rate=0.9,
		mutation_rate=0.1,
		op_mutation_rate=1.0,
		max_evaluations=-1,
		max_generations=-1,
		max_time=-1,
		initialization_max_tree_height=4,
		min_depth=2,
		max_tree_size=100,
		tournament_size=4,
		verbose=False
		):

		self.pop_size = pop_size
		self.fitness_function = fitness_function
		self.functions = functions
		self.terminals = terminals
		self.crossover_rate = crossover_rate
		self.mutation_rate = mutation_rate
		self.op_mutation_rate = op_mutation_rate

		self.max_evaluations = max_evaluations
		self.max_generations = max_generations
		self.max_time = max_time

		self.initialization_max_tree_height = initialization_max_tree_height
		self.min_depth = min_depth
		self.max_tree_size = max_tree_size
		self.tournament_size = tournament_size

		self.generations = 0

		self.verbose = verbose


	def __ShouldTerminate(self):
		must_terminate = False
		elapsed_time = time.time() - self.start_time
		if self.max_evaluations > 0 and self.fitness_function.evaluations >= self.max_evaluations:
			must_terminate = True
		elif self.max_generations > 0 and self.generations >= self.max_generations:
			must_terminate = True
		elif self.max_time > 0 and elapsed_time >= self.max_time:
			must_terminate = True

		if must_terminate and self.verbose:
			print('Terminating at\n\t', 
				self.generations, 'generations\n\t', self.fitness_function.evaluations, 'evaluations\n\t', np.round(elapsed_time,2), 'seconds')

		return must_terminate


	def Run(self):

		self.start_time = time.time()

		self.population = []

		# rampled half-n-half
		curr_max_depth = self.min_depth
		init_depth_interval = self.pop_size / (self.initialization_max_tree_height - 1) / 2
		next_depth_interval = init_depth_interval

		for i in range( int(self.pop_size/2) ):
			if i >= next_depth_interval:
				next_depth_interval += init_depth_interval
				curr_max_depth += 1

			g = Variation.GenerateRandomTree( self.functions, self.terminals, curr_max_depth, curr_height=0, method='grow', min_depth=self.min_depth )
			self.fitness_function.Evaluate( g )
			self.population.append( g )
			
			f = Variation.GenerateRandomTree( self.functions, self.terminals, curr_max_depth, curr_height=0, method='full', min_depth=self.min_depth ) 
			self.fitness_function.Evaluate( f )
			self.population.append( f )


		while not self.__ShouldTerminate():

			selected = Selection.TournamentSelect( self.population, self.pop_size, tournament_size=self.tournament_size )

			O = []
			for i in range( self.pop_size ):
				o = deepcopy(selected[i])
				if ( random() < self.crossover_rate ):
					o = Variation.SubtreeCrossover( o, selected[ randint( self.pop_size ) ] )
				if ( random() < self.mutation_rate ):
					o = Variation.SubtreeMutation( o, self.functions, self.terminals, max_height=self.initialization_max_tree_height )
				if ( random() < self.op_mutation_rate ):
					o = Variation.OnePointMutation( o, self.functions, self.terminals )

				if (len(o.GetSubtree()) > self.max_tree_size) or (o.GetHeight() < self.min_depth):
					del o
					o = deepcopy( selected[i] )
				else:
					self.fitness_function.Evaluate(o)

				O.append(o)


			PO = self.population+O
			
			new_population = []
			fronts = self.FastNonDominatedSorting(PO)
			self.latest_front = deepcopy(fronts[0])

			curr_front_idx = 0
			while curr_front_idx < len(fronts) and len(fronts[curr_front_idx]) + len(new_population) <= self.pop_size:
				self.ComputeCrowdingDistances( fronts[curr_front_idx] )
				for p in fronts[curr_front_idx]:
					new_population.append(p)
				curr_front_idx += 1

			if len(new_population) < self.pop_size:
				# fill in remaining
				self.ComputeCrowdingDistances( fronts[curr_front_idx] )
				fronts[curr_front_idx].sort(key=lambda x: x.crowding_distance, reverse=True) 

				while len(fronts[curr_front_idx]) > 0 and len(new_population) < self.pop_size:
					new_population.append( fronts[curr_front_idx][0] )	# pop first because they were sorted in desc order
					fronts[curr_front_idx].pop(0)

				# clean up leftovers
				while len(fronts[curr_front_idx]) > 0:
					del fronts[curr_front_idx][0]

			self.population = new_population

			self.generations = self.generations + 1

			if self.verbose:
				print ('g:',self.generations,'elite obj1:', np.round(self.fitness_function.elite.objectives[0],3), ', size:', len(self.fitness_function.elite.GetSubtree()))


	def FastNonDominatedSorting(self, population):
		rank_counter = 0
		nondominated_fronts = []
		dominated_individuals = {}
		domination_counts = {}
		current_front = []

		for i in range( len(population) ):
			p = population[i]

			dominated_individuals[p] = []
			domination_counts[p] = 0

			for j in range( len(population) ):
				if i == j:
					continue
				q = population[j]

				if p.Dominates(q):
					dominated_individuals[p].append(q)
				elif q.Dominates(p):
					domination_counts[p] += 1

			if domination_counts[p] == 0:
				p.rank = rank_counter
				current_front.append(p)

		while len(current_front) > 0:
			next_front = []
			for p in current_front:
				for q in dominated_individuals[p]:
					domination_counts[q] -= 1
					if domination_counts[q] == 0:
						q.rank = rank_counter + 1
						next_front.append(q)
			nondominated_fronts.append(current_front)
			rank_counter += 1
			current_front = next_front

		return nondominated_fronts


	def ComputeCrowdingDistances(self, front):
		number_of_objs = len(front[0].objectives)
		front_size = len(front)

		for p in front:
			p.crowding_distance = 0

		for i in range(number_of_objs):
			front.sort(key=lambda x: x.objectives[i], reverse=False)

			front[0].crowding_distance = front[-1].crowding_distance = np.inf

			min_obj = front[0].objectives[i]
			max_obj = front[-1].objectives[i]

			if min_obj == max_obj:
				continue

			for j in range(1, front_size - 1):

				if np.isinf(front[j].crowding_distance):
					# if extrema from previous sorting
					continue

				prev_obj = front[j-1].objectives[i]
				next_obj = front[j+1].objectives[i]

				front[j].crowding_distance += (next_obj - prev_obj)/(max_obj - min_obj)
