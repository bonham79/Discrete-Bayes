# Functions for calculating and utilizing discrete bayes classifier. 
from typing import Tuple, List
from space import genGain

def calcClassProb(tags: Tuple[int], K: int)-> Tuple[float]:
	# Creates tuple of floats expressing the prior probability of class corresponding to index.
	# Bases calculations off probabilities in tags and knowledge of K class labels.
	# Values are smoothed by delta to avoid 0 probability of class. 
	probs = [0.0 for _ in range(K)]
	for tag in tags:
		# Adds normalized count for each occurence of a class label.
		probs[tag] += 1/len(tags)
	return tuple(probs)

def calcPosterior(priors: Tuple[float], conds: List[List[float]])->Tuple[Tuple[float]]:
	# Calculates the posterior probabilities given likelihood and priors
	# For all measures. Output is tuple MxK, with M measures and K classes. 
	# Note: actually calculates numerator P(d|c) * P(c) since 
	# real probability can be calculated by normalizing over sum of all
	# P(c|d) for all c in K for measure d.
	K = len(priors)
	M = len(conds[0])
	posteriors = []

	# Loops over measures and multiplies corresponding values
	for measure in range(M):
		posteriors.append(tuple(priors[c] * conds[c][measure] for c in range(K)))

	return tuple(posteriors)

def normPosterior(posteriors: Tuple[Tuple[float]])->Tuple[Tuple[float]]:
	# Normalizes all posterior values over sum of classes conditional probabilities, providing real posterior.
	M = len(posteriors)
	K = len(posteriors[0])
	normPosterior = []
	for d in range(M):
		normed = tuple(posteriors[d][c] / sum(posteriors[d]) for c in range(K))
		normPosterior.append(normed)
	return tuple(normPosterior)


def calcClassConditionalProb(measures: Tuple[int], tags: Tuple[int], dimen: Tuple[int], K: int, delta: float=0.0)->List[List[float]]:
	# Generates class conditional probabilities given a training set of linearized measures and class tags values.
	# dimen represents the cardinalities of each dimension in the measurement space and K is number
	# of possible class values. Unseen values can be additively smoothed by delta. 

	# Calculates size of linearized measurement space
	M = 1
	for N in dimen:
		M *= N

	# Preallocating to force memory error if one occurs. Additively smoothes 0 values.
	ccp = [[delta for d in range(M)] for k in range(K)]

	# Counts each instance of measure given a class. 
	for i in range(len(measures)):
		ccp[tags[i]][measures[i]] += 1

	# Normalizes probabilities for each class conditional set.
	for j in range(K):
		total = sum(ccp[j])
		ccp[j] = [ccp[j][k]/total for k in range(M)]

	return ccp

def calcMeasProb(priors: Tuple[float], conds: List[List[float]])-> Tuple[float]:
	# Given priors and class conditional probabilities, calculates probability distribution of measures.
	K = len(priors)
	M = len(conds[0])
	newProbs = []

	# Just for safety check. Checks that the lengths are corresponding. 
	assert M == len(conds[-1]) and len(conds) == K

	# P(d) = E P(d|c)*P(c) for all c. 
	for measure in range(M):
		prob = sum([conds[c][measure] * priors[c] for c in range(K)])
		newProbs.append(prob)

	return tuple(newProbs)

class BayesClassifier:
	# Class of objects conditioned on given class probabilities and class conditional probabilities
	# Computes likely class designation for a set of values. prior is class Probabilities,
	# cond is K tables of class conditional probabilities of measures. egain is economic gain matrix.
	# If no matrix is present, generates identity matrix. 
	def __init__(self, prior: Tuple[float], cond: List[List[float]], eGain: Tuple[Tuple[float]] = None):
		self.prior = prior
		self.cond = cond
		self.range = len(cond)
		self.spaceSize = len(cond[0])
		if eGain:
			self.eGain = eGain
		else:
			self.eGain = genGain(self.range, identity=True)

	def eGainUpdate(self, eGainNew: Tuple[Tuple[float]]):
		self.eGain = eGainNew

	def priorUpdate(self, priorNew: Tuple[float]):
		self.prior = priorNew

	def condUpdate(self, condNew: List[List[float]]):
		self.cond = condNew

	def assign(self, test: Tuple[int])-> Tuple[int]:
		# Assigns bayesian estimated set of measures. Classes are chosen on optimal expected gain ffor measure.
		results = []
		K = self.range
		posterior = calcPosterior(self.prior, self.cond)
		for measure in test:
			# Precalculates posterior likliehoods. (P(d) is ignored due to being cancelled out as denominator in inequality.)
			
			# Begins with assumption class 0 is optimal.
			bestVal = sum([self.eGain[true][0] * posterior[measure][true] for true in range(K)])
			bestTag = 0

			for assigned in range(1,K):
				# Calculates expected gain for each class beyond 0. Classes with higher gain replace previously chosen class. 
				val = sum([self.eGain[true][assigned] * posterior[measure][true] for true in range(K)])
				if val > bestVal:
					bestTag = assigned
					bestVal = val
			results.append(bestTag)
		return tuple(results)

	def optimize(self, delta: float, meas: Tuple[int], tags: Tuple[int]):
		# Updates all conditional probability values so they show bias to real
		# values of the associated measurements.
		for i in range(len(meas)):
			self.cond[tags[i]][meas[i]] += delta

		# Normalizes
		for j in range(self.range):
			total = sum(self.cond[j])
			self.cond[j] = [self.cond[j][k]/total for k in range(self.spaceSize)]

def main():
	from matrix_tools import linear
### Unit tests.

	### Test for calcClassProb
	labels = (0,1,2,2,1)
	realVal = (.2,.4,.4)

	prb = calcClassProb(labels, 3)
	assert prb == realVal

### Tests for bayes classifier.
	# # data taken from http://www.cs.rpi.edu/academics/courses/fall03/ai/misc/naive-example.pdf
	testData = ((0,1,1), (1,0,1), (1,1,0), (1,0,0), (1,0,0), (0,1,0))
	testDataLin = tuple(linear(i,(2,2,2)) for i in testData)
	classProbs = (.5, .5)
	priorCCP0 = [2/5, .43*.44*.44, .43*.56*.56, 1/5, 1/5, .57*.44*.44, 1/5, 1/5]
	priorCCP1 = [2/5, 1/5, .56*.31*.43, .56*.31*.57, .44*.69*.43, 1/5, .44*.31*.43, 1/5]
	priorCCP0Norm = [i/sum(priorCCP0) for i in priorCCP0]
	priorCCP1Norm = [i/sum(priorCCP1) for i in priorCCP1]
	bayes = BayesClassifier(classProbs, [priorCCP0Norm,priorCCP1Norm])

	results = bayes.assign(testDataLin)

	assert(results[0] == 0), results

	# Another toy problem
	testData = (2,1,3,3,0,2,2,2,3)
	priorCCP0 = [1/4*.5, .25*.5, .75*.5, .75*.5]
	priorCCP1 = [.5*.75, .5*.25, .5*.75, .5*.25]

	bayes.condUpdate([priorCCP0, priorCCP1])
	results = bayes.assign(testData)
	assert(results[2] == 0)

	# Another toy problem https://stattrek.com/probability/bayes-theorem.aspx
	classProbs = (5/365, 360/365)
	condProbs = [[.9, .1], [.1,.9]]
	classifier = BayesClassifier(classProbs, condProbs)
	results = classifier.assign((0,))
	assert results[0] == 1 

	# Another toy problem https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading11.pdf
	priors = (.4,.4,.2)
	conds = [[.5,.5], [.6,.4], [.9,.1]]
	classifier = BayesClassifier(priors, conds)
	results = classifier.assign((0,1,1)) #Printing out values in funciton should yield .2,.24,.18

	# Another toy problem.
	priors = (.6,.4)
	conds = [[.12/.6,.18/.6,.3/.6],[.2/.4,.16/.4,.04/.4]]
	gain = ((1,0),(0,2))
	classifier = BayesClassifier(priors, conds, eGain=gain)
	assert classifier.assign((0,1,2)) == (1,1,0)

### Unit tests for optimize method.
	priors = (.25,.25,.25,.25)
	conds = [[.5,.5], [.5,.5], [.5,.5], [.5,.5]]
	classifier = BayesClassifier(priors, conds)
	classifier.optimize(.1, (0,1), (1,0)) # Should update (1,0) and (0,1) by delta

# Tests conditional probability calculator
	meas = (0,1,1)
	tags = (0,1,2)
	assert calcClassConditionalProb(meas, tags, (2,), 3, delta=1)[0][0] == 2/3

## Tests calculation of measures
	### Tests for biasMeasures. 
	# Let's give some conditionals that bias one measure
	conds = ((.9, .1), (.9,.1))
	# And feed non biased priors
	priors = (.5,.5)
	# Should provide measurements that favor the first element
	newProbs = calcMeasProb(priors, conds)
	print(newProbs)

	# Let's give some conditionals that biases one measure
	conds = ((.1, .8, .1), (1/3,1/3,1/3))
	# And feed biased priors
	priors = (.7,.3)
	# Should provide measurements that favor the second element as first class is most prevelant. 
	newProbs = calcMeasProb(priors, conds)
	print(newProbs)

	# Toy problem from MIT https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading11.pdf
	priors = (.4,.4,.2)
	conds = (.5,.5), (.6,.4), (.9,.1) # Heads, tails
	newProbs = calcMeasProb(priors, conds) 
	assert newProbs[0] == .62, newProbs

	## Tests for calcPosterior. Same as above
	posts = calcPosterior(priors, conds)
	print("Posteriors are: {}. Should ideally be {}".format(posts[0], ".2, .24,.18")) # should be around .2, .24, .18

	## tests for normPosterior
	posts = normPosterior(posts)
	for d in posts:
		sums = sum(d)
		assert sums >= .9999, sums
		assert sums <= 1.0000005, sums

if __name__ == "__main__":
	main()