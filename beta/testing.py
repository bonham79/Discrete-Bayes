# Functions for testing accuracy of classifier. 
import random
from typing import Tuple, List
from bayes import BayesClassifier, calcClassProb
from evaluation import genConMatrix, normConMatrix, calcExpGain

def test(classifier: BayesClassifier, measures: Tuple[int], tags: Tuple[int], V=0)->float:
	## Training protocol for testing measurements. Shuffles measures and tags while retaining
	# pairings and then classifies according to Bayes classifier. Returns expected gain
	# of testing. If V is provided, performs V-fold testing on data of V partitions
	
	# Shuffles tags and measures
	K = classifier.range
	e = classifier.eGain
	measures, tags = shuffle(measures,tags)

	# Sees if we are v-folding or just testing over a random measure. 
	if V:
		normMatrix = vFold(measures, tags, V, classifier)
	else:
		results = classifier.assign(measures)
		matrix = genConMatrix(tags, results, K)
		normMatrix = normConMatrix(matrix)
	return calcExpGain(normMatrix, e)

def shuffle(measures: Tuple[int], classes: Tuple[int]) -> Tuple[Tuple[int]]:
	# Shuffles values while preserving index pairings. (classes[i] is always paired measures[i])
	pairs = list(zip(measures, classes))
	random.shuffle(pairs)
	newMeasures, newClasses = zip(*pairs)
	return tuple(newMeasures), tuple(newClasses)

def partition(meas: Tuple[int], V:int)->List[List[int]]:
	# Partitions meas into equal size buckets for V-fold testing.
	# Buckets preserve order. 
	S = len(meas)
	parts = [[] for v in range(V)]
	assert V <= S, "Can't partition array smaller than V."

	## Iterates through parts and increases each list one at a time while until their total size = S. 
	for d in range(S):
		fold = d % V
		parts[fold].append([None])

	## Assigns values to buckets while preserving order.
	## Fold tracks which partition, section tracks which entry in partition, item tracks with measurement needs to be added.
	fold = 0
	item = 0
	while fold < V:
		section = 0
		while section < len(parts[fold]):
			parts[fold][section] = meas[item]
			section += 1 
			item += 1
		fold += 1
	return parts

def vFold(meas: Tuple[int], tags: Tuple[int], V: int, classifier: BayesClassifier)-> List[List[float]]:
	# Performs a round of V-fold validation tests on measurements 'meas' and respective real classes 'tags'
	# Performs V test using classifier. Returns a normalized confusion matrix of tests.
	results = []

	measFold = partition(meas, V)
	tagsFold = partition(tags, V)

	for v in range(V):
		# Creates folds
		# Assigns testing and training 
		trainTags = [tag for i in range(V) if i != v for tag in tagsFold[i]]
		testMeas = measFold[v]

		# Updates with new probability values
		trainProb = calcClassProb(trainTags, classifier.range)
		classifier.priorUpdate(trainProb)

		results.append(classifier.assign(testMeas)) # Unfolds tuple
	results = tuple(i for tpl in results for i in tpl)
	matrix = genConMatrix(tags, results, classifier.range)
	return results#normConMatrix(matrix)

def main():
#### Testing Shuffle
	# Checks that shuffle does not return same values. 
	sampleSize = random.randint(1,1000)

	# Samples unique value pairs. (Makes easier to track. Misassignment of identical values doesn't alter data.)
	meas = tuple(random.sample(range(1,10*sampleSize),sampleSize))
	tags = tuple(random.sample(range(1,10*sampleSize),sampleSize)) 

	measSh, tagsSh = shuffle(meas,tags)
	assert (measSh,tagsSh) != (meas,tags)
	
	# Checks that original pairs are present
	for i in range(sampleSize):

		# Finds where the measurement was shuffled to
		indexSh = measSh.index(meas[i])

		# Checks that this new index pairs with original tag value.
		assert tags[i] == tagsSh[indexSh]

### Testing vFold
	# Generates trivial matrix to make sure all parts are working.
	# This should have perfect accuracy. 
	meas = (0,1,2,2,1,2,2,1,1,0,2,2)
	tags = (2,1,0,0,1,0,0,1,1,2,0,0)
	cp = (.5,.2,.3)
	ccp = [[0,0,1],[0,1,0],[1, 0,0]]
	tagger = BayesClassifier(cp, ccp)

	test = vFold(meas, tags, 5, tagger)
	print(test)

### Testing partition
### Partition should retain order
	for i in range(3,1000):
		l = [_ for _ in range(random.randint(3,1000))]
		k = partition(l, 3)
		m = tuple(j for i in k for j in i)
		assert tuple(l) == m, m




if __name__ == "__main__":
	main()