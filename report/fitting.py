## Updates probabilities to generate ideal measurement space for classifier.
from typing import List, Tuple
from bayes import BayesClassifier, calcMeasProb, calcPosterior, normPosterior
from space import MeasurementGenerator, genSamples
from probability import genCmlProbs

def biasCCP(classifier: BayesClassifier, delta: float)-> List[List[float]]:
	# Destrucctively Updates CCP values in classifier. Classes are generated from classifier
	# for all measurements d in M space. Each d|c is increased by delta towards classifier's selection
	# and is normalized over class space. Returns conditional matrix. Not used in function but preserved
	# for legacy update. 

	# Classifies for all d in measurement space M
	M = classifier.spaceSize
	K = classifier.range
	bayesValues = classifier.assign(range(M))
	conds = classifier.cond # Note, this alters the original classifier. Had to be done for speed and memory efficiency.

	# Updates conditionals
	for measure in range(M):
		tag = bayesValues[measure]
		conds[tag][measure] += delta

	# Normalizes conditionals
	for val in range(K):
		sums = sum(conds[val])
		conds[val] = [prb/sums for prb in conds[val]]

	return conds

def genBiasTags(measures: Tuple[int], posteriors: Tuple[Tuple[float]])->Tuple[int]:
	# Assigns measures by posterior conditional probability distribution p(c|d)
	# Note, we can't use the classAssign function as it uses same class probabilities
	# for all measures - and thus produces overlap.
	Z = len(measures)
	tags = [] 

	# For each measure
	for d in range(Z):

		# We find its set of class conditionals
		conds = posteriors[measures[d]]

		# We generate a cumulative probability distribution
		cmlConds = genCmlProbs(conds)

		# generate a sample
		tag = genSamples(1, cmlConds)

		# Adds sample to tags (since outputs a tuple we need to index)
		tags.append(tag[0])

	return tuple(tags)

def biasMeasGenerator(generator: MeasurementGenerator, priors: Tuple[float], conds: List[List[float]]):
	# Updates measures with bias towards those indicated by class conditional probabilites. 
	# No output. Destructively alters generator. 
	newProbs = calcMeasProb(priors, conds)
	generator.updateProbs(newProbs)
	return 

def fitData(classifier: BayesClassifier, generator: MeasurementGenerator, Z: int)-> Tuple[BayesClassifier, Tuple[int], Tuple[int]]:
	# Uses classifier values to generate new measure and tag generators that are biased towards the classifier.
	# Then generates Z new measure, tag pairs.
	# Returns measure-tag pairs for another round of testing
	conds = classifier.cond
	priors = classifier.prior
	posts = calcPosterior(priors, conds)
	postsNorm = normPosterior(posts)

	biasMeasGenerator(generator, priors, conds)

	measures = generator.genMeas(Z)

	tags = genBiasTags(measures, postsNorm)

	return measures, tags

def main():

	priors = (.6,.4)
	conds = [[.12/.6,.18/.6,.3/.6],[.2/.4,.16/.4,.04/.4]]
	gain = ((1,0),(0,2))

	### Tests for biasMeasGenerator
	generator = MeasurementGenerator((2,2))
	# print(generator.cmlProbs) ## For peeking. 
	prev = generator.cmlProbs[1] - generator.cmlProbs[0]

	# Let's give some conditionals that biases one measure
	conds = ((.1, .7, .1,.1), (.25,.25,.25,.25))
	# And feed biased priors
	priors = (.7,.3)

	biasMeasGenerator(generator, priors, conds)
	# print(generator.cmlProbs)
	now = generator.cmlProbs[1] - generator.cmlProbs[0] 
	# Should show bias towards second value now. 
	assert now > prev


	generator = MeasurementGenerator((2,2))
	# print(generator.cmlProbs)
	# Records previous probability for second value. 
	prev1 = generator.cmlProbs[1] - generator.cmlProbs[0]
	# Records previous probability for final value. 
	prev2 = generator.cmlProbs[-1] - generator.cmlProbs[-2]

	# Let's give some conditionals that biases one measure
	conds = ((0.0, .5, 0.0,.5), (.05,.4,.05,.5))
	# And feed biased priors
	priors = (.7,.3)

	biasMeasGenerator(generator, priors, conds)
	# print(generator.cmlProbs)
	now1 = generator.cmlProbs[1] - generator.cmlProbs[0] 
	now2 = generator.cmlProbs[-1] - generator.cmlProbs[-2] 

	# Should show bias towards second value now. May vary due to the original conditional probability being random. 
	assert now1 > prev1, (prev1, now1)
	assert now2 > prev2, (prev2, now2)

if __name__ == "__main__":
	main()