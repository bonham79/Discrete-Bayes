# Tools for expressing evaluation metrics on testing. 
from typing import Tuple,List

def genConMatrix(real: Tuple[int], assigned: Tuple[int],  K: int)-> List[List[int]]: ####CONVERT INTO LINEAR?
	# generates a KxK confusion matrix of counts with rows indexing real class values
	# and columns indexing assigned values. Returned as list for memory convenience
	# During iterative testing. 

	matrix = [[0 for n in range(K)] for m in range(K)]

	for i in range(len(real)):
		# Iterates along measure sequences and adds count for 
		# row designated in real sequence and column from assigned.
		matrix[real[i]][assigned[i]] += 1

	return matrix

def normConMatrix(matrix: List[List[int]])-> List[List[float]]:
	# Normalizes values in confusion matrix.
	normMatrix = [[None for n in matrix[0]] for m in matrix]

	total = sum([sum(counts) for counts in matrix])

	for i in range(len(matrix)):
		for j in range(len(matrix[i])):
			normMatrix[i][j] = matrix[i][j]/total
	return normMatrix

def calcExpGain(conMatrix: List[List[float]], eGain: Tuple[Tuple[float]])-> float:
	# Calculates expected gain given a normalized confusion matrix conMatrix and 
	# matrix of expected gain values eGain. Both must be of same size. 
	K = len(conMatrix)
	vals = [conMatrix[real][assigned] * eGain[real][assigned] for real in range(K) for assigned in range(K)]
	return sum(vals)

def main():
#### units tests

### Makes sure confusion matrix is valid for toy examples.
	results = (1,2,2)
	realVal1 = (2,1,2)
	realVal2 = (1,2,2)

	test1 = genConMatrix(realVal1, results, 3)
	test2 = genConMatrix(realVal2, results, 3)

	ideal1 = [[0,0,0],[0,0,1],[0,1,1]]
	ideal2 = [[0,0,0],[0,1,0],[0,0,2]]

	assert test1 == ideal1
	assert test2 == ideal2

### Tests for normConMatrix to be properly cumulative
	matrix = [[3,4,3,2],[3,2,3,4],[0,3,2,3],[8,3,2,4]]
	total = sum([sum(counts) for counts in normConMatrix(matrix)])
	assert total > .9999999999999995 and total < 1.0000000000005

### Tests for normConMatrix to generate probabilities as expected
	matrix = [[4,3,1],[2,2,2],[3,2,1]]
	assert normConMatrix(matrix) == [[.2,.15,.05],[.1,.1,.1],[.15,.1,.05]],normConMatrix(matrix)

### Tests for calcExpGain
	matrix = [[.3,.3], [.04,.36]]
	egain = ((1,0),(0,2))
	expected = calcExpGain(matrix, egain)
	assert expected == 1.02, expected

if __name__ == "__main__":
	main()