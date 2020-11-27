## Updates probabilities to generate ideal measurement space for classifier.
from typing import List, Tuple
from bayes import BayesClassifier, calcMeasProb, calcPosterior, normPosterior
from space import MeasurementGenerator, genSamples
from probability import genCmlProbs

def genBiasTags(measures: Tuple[int], posteriors: Tuple[Tuple[float]])->Tuple[int]:
	# Assigns measures by posterior conditional probability distribution p(c|d)
	# Note, we can't use the classAssign function as it uses same class probabilities
	# for all measures - and thus produces overlap. NOTE: possibly fix this aspect for greater modularity.
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

def fitData(classifier: BayesClassifier, generator: MeasurementGenerator, Z: int, delta: float)-> Tuple[BayesClassifier, Tuple[int], Tuple[int]]:
	# Takes delta and uses to update conditional probability 
	# Uses new value to update distribution of measurement space
	# Then generates Z new measures
	# Then generates new tags in line with conditional probabbility distribution of class per tag
	# Returns a classifier and measure-tag pairs for another round of testing
	conds = classifier.cond
	priors = classifier.prior
	posts = calcPosterior(priors, conds)
	postsNorm = normPosterior(posts)

	biasMeasGenerator(generator, priors, conds)

	measures = generator.genMeas(Z)

	tags = genBiasTags(measures, postsNorm)

	return classifier, measures, tags

def main():
	from bayes import calcClassProb

	priors = (.6,.4)
	conds = [[.12/.6,.18/.6,.3/.6],[.2/.4,.16/.4,.04/.4]]
	gain = ((1,0),(0,2))

	### Tests for biasCCP
	classifier = BayesClassifier(priors, conds, eGain=gain) 
	# Classification should be 1,1,0. The update should alter, 0|1, 1|1, 2|0. 
	newCCP = biasCCP(classifier,.05)
	# print(newCCP) # Uncomment to see if ccp comforms to predictiosn. Should bias 2|0 and raise 0 and 1 |1. 

	### Tests for biasMeasGenerator
	generator = MeasurementGenerator((2,2))
	# print(generator.cmlProbs)
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
	prev1 = generator.cmlProbs[1] - generator.cmlProbs[0]
	prev2 = generator.cmlProbs[-1] - generator.cmlProbs[-2]

	# Let's give some conditionals that biases one measure
	conds = ((.1, .5, .1,.3), (.25,.25,.25,.25))
	# And feed biased priors
	priors = (.7,.3)

	biasMeasGenerator(generator, priors, conds)
	# print(generator.cmlProbs)
	now1 = generator.cmlProbs[1] - generator.cmlProbs[0] 
	now2 = generator.cmlProbs[-1] - generator.cmlProbs[-2] 

	# Should show bias towards second value now. 
	assert now1 > prev1, (prev1, now1)
	assert now2 > prev2, (prev2, now2)

if __name__ == "__main__":
	main()