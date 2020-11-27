# Package for generating measurement space D of Z dimensions with cardinalities of N1xN2x...Nz.
# Can be used to generate space, measurement values, class assignment functions, and class assignments.
from random import randint, uniform
from typing import Tuple, List
from probability import genCmlProbs, genProbs
from config import CARDINALITY_RANGE_LIMIT

def genSpace(Z: int, N=CARDINALITY_RANGE_LIMIT) -> Tuple[int]:
# Generates tuple of Z values representing a Z dimensional measurespace.
# Each dimension has cardinality of values from 0 to Nz-1, with Nz
# being randomly generated between 2 and N or the default limit in config. 
	dimen = [randint(2, N) for i in range(Z)]
	return tuple(dimen)

def genSamples(Z: int, cmlProbs: Tuple[float])-> Tuple[int]:
# Generates linear array of Z measures based on distribution in cmlProbs
# Elements range from 0 to len(cmlProbs). For efficiency, uses binary search.
	samples = []
	for i in range(Z):
		x = uniform(0,1)
		samples.append(binCompare(cmlProbs,x))
	return tuple(samples)

def binCompare(arr: Tuple[float], val: float)-> int:
	## Binary search for comparing val x in array arr. Finds first value greater than x. 
    start = 0
    end = len(arr) - 1
  
    index = -1
    while start <= end: 
        mid = (start + end) // 2
  
        # Move to right side if target is 
        # greater. 
        if arr[mid] < val: 
            start = mid + 1 
  
        # Move left side. 
        else: 
            index = mid 
            end = mid - 1 
  
    return index

class MeasurementGenerator:
	def __init__(self, dimen: Tuple[int]):
		self.dimen = dimen

		# Caclulates the size of space
		self.range = 1
		for N in self.dimen:
			self.range *= N

		# Generates probabilities for later measurements
		self.cmlProbs = genCmlProbs(genProbs(self.range))

	def genMeas(self, Z: int)-> Tuple[int]:
		# Generates measures given sample size Z. 
		return genSamples(Z, self.cmlProbs)

	def updateProbs(self, probs: Tuple[float]):
		# Updates probability distribution with new cumulative probability distribution

		# Checks we're not passing anything that's not cumulative. 
		sums = sum(probs)
		assert sums <= 1.00000005 and sums >= .99999995, "Probs must be a cumulative probability distribution."

		self.cmlProbs = genCmlProbs(probs)

class ClassAssign:
	def __init__(self, dimen: Tuple[int], K: int):
	## Dimen is representation of measurement space M
	## K is amount of classes to be assigned

		self.dimen = dimen
		self.probs = genProbs(K) # Creating value for testing purposes. 
		self.cmlProbs = genCmlProbs(self.probs)
		self.range = K

	# Amount of possible measures is cardinality of M as represented by dimen
		self.mSpaceSize = 1
		for N in dimen:
			self.mSpaceSize *= N

		# Creates values and then fills. 
		self.tags = None
		self.genTags()

	def genTags(self):
	# Creates tag assignments. 
		tags = []

		for i in range(self.mSpaceSize):
			# Generates tag for each measure possibility
			prb = uniform(0,1)
			j = 0
			while prb > self.cmlProbs[j]:
				j += 1
			tags.append(j)

		self.tags = tuple(tags)

	def regenTags(self, probs:List[float]):
	# Regenerates tags given probability set.
		self.cmlProbs = genCmlProbs(probs)
		self.genTags()

	def assign(self, measures: Tuple[int]):
	## Returns class asignment for each element in measures
		results = tuple(self.tags[i] for i in measures)
		return results

def genCCP(K: int, dimen: Tuple[int])-> List[List[float]]:  
# Generates a randomly alotted class conditional probability array.
# Array is of size K, each index represents a corresponding class conditional probability list
# and each list is of size of measure space represented by dimen.

	# Generates M space length.
	N = 1
	for n in dimen:
		N *= n

	# Generates K normalized probability tables; one for each class. 
	classProbs = [list(genProbs(N)) for _ in range(K)]

	return classProbs

def genGain(K: int, identity=False)-> List[List[int]]: 
### Generates either a penalty matrix or identity matrix determined by 'identity'= False or True. Respectively.
### identity matrix has value 1 for all diagonal entries except and zero for all else
### Penalty matrix assigns -1 to all values except diagonals, which have zero entries. 
	eGain = [[0 for n in range(K)] for m in range(K)]

	# Checks if we want an identity matrix
	if identity:
		maximal = 1
		minimal = 0

	else: 
		maximal = 0
		minimal = -1

	# fills matrix
	for i in range(K):
		for j in range(K):
			eGain[i][j] = minimal

	# Makes diagonals max
	for k in range(K):
		eGain[k][k] = maximal

	return eGain


def main():
#### Unit tests

####### Tests for sampling function.
	# Tests for distributions to be within range
	for i in range(1,1000):
		sampSize = randint(1,1000)
		spaceRang = randint(3,1000)
		a = genProbs(spaceRang)
		b = genCmlProbs(a)
		c = genSamples(sampSize, b)
		for j in c:
			assert j<=(spaceRang-1) and j>=0

	# Tests for suitable randomness so same samples aren't being generated.
	for _ in range(1000):

		sampSize = randint(2,1000)
		spaceRang = randint(3,1000) # Provides sufficient space for variety

		a = genProbs(spaceRang)
		b = genCmlProbs(a)

		c = genSamples(sampSize, b)
		z = genSamples(sampSize, b)

		assert c != z, c

#####Tests genMeas method
	# Checks for measurement space to be of proper length
	Z = randint(2,1000)
	dimen = genSpace(6)
	size = randint(2,1000)
	generator = MeasurementGenerator(dimen)
	x = generator.genMeas(size)
	assert len(x) == size 

	# Checks for values to actually exist in measurement space
	for _ in range(1,1000):
		maxValue = 1
		for i in dimen:
			maxValue *= i
		for j in x:
			assert j <= maxValue-1, maxValue-j

###### Tests ClassAssign class
	# Verifies no class is assigned outside class values
	for _ in range(1000):
		dimen = genSpace(5)
		maxValue = 1
		for i in dimen:
			maxValue *= i
		tagSize = randint(1,100)
		tagAssign = ClassAssign(dimen, tagSize)

		for j in tagAssign.tags:
			assert j <= tagSize-1, (j, tagSize-1)

		# Verfies assignment funciton is of proper length.
		assert len(tagAssign.tags) == maxValue

if __name__ == "__main__":
	main()
