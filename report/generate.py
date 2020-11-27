#### For generating experimental values for midterm.  
#### Command line inputs:
## samples-integer of how many samples to generate for testing. 
## dimen-a tuple expressing the cardinality of each dimension of measurement space M
## classes-integer expressing the number of class types. Numbered 0 to classes-1.
## seed - integer for random seed generator.

### Outputs:
## data.txt - file containing comma separated values of measurements and and class values
#  Each row is pair of measurement-tuple and class with class values being last value on right. 
## eGain.txt - csv file of economic gain matrix generated from training. Each row represents
#  a true-assigned gain matrix row, with columns delineated by commas. 
## prior.txt - csv file of prior probability values generated from measures after v-fold testing.
#  each column indexes a class numbered 0...classes-1. 
## conditional.csv - csv file of generated class conditional probabilities. Each row indexes
#  a class from 0...classes-1. Each comma separated element refers to linearized representation
#  of measurement from mSpace. (example: (0,0,0) becomes 0 and is first element. (0,0,1) becomes 1
#  and is second element. 
## Command line outputs: expected gain from vFold training. If optimization is specified, outputs
#  improved expected gain as well for each iteration of updating.

### Optional Inputs:
##  --vfold (-v). Specifies how many folds in V-fold testing to partion. (Default=10)
##  --optimize (-o). Specifies iterative optimization factor of class conditional probability
#   values. If not specified or left 0, assumes no iteration. 10 iterations are conducted. 
##  --iteration (-t). Number of iterations for iterative update of conditional values. 
##  --identity (-i). Specifies creation of identity matrix for economic gain matrix. 
#	if not specified, generates a 'penalty' matrix assigning -1 to all incorrect classifications
#   and 0 for correct classfication.
import argparse
import reader
import config
import sys
from testing import test
from random import seed
from space import MeasurementGenerator, ClassAssign, genCCP, genGain
from bayes import BayesClassifier, calcClassProb
from fitting import fitData

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("samples", help="Number of measurement samples to generate.", type=int)
	parser.add_argument("dimen", help="Measurement space.", type=str)
	parser.add_argument("classes", help="Number of classes.", type=int)		
	parser.add_argument("seed", help="Random seed for experiement duplication.", type=int)	
	parser.add_argument( "--vfolds", "-v",default=10, help="Number of v-folds to partition testing data for v-folds testing. Default is 10.", type=int)
	parser.add_argument("--optimization", "-o", default=0.0, help="Specify if iterative improvement of class conditional probability values should be taken.", type=float)
	parser.add_argument("--iteration", "-t", default=10, help="Number of iterations for conditional update.", type=int)
	parser.add_argument("--identity", "-i", action="store_true", default=False, help="Specify if economic gain matrix should be identity.")
	args = parser.parse_args()

	# Checks that our given limits are even feasible memory wise

	# Prompts for reader friendliness
	print("Generating testing data for seed {}".format(args.seed))

	# Sets seed
	seed(args.seed)

	# Assigns values
	dimen = eval(args.dimen)
	# Calculates size of domain
	M = 1
	for N in dimen:
		M *= N
	K = args.classes
	V = args.vfolds
	Z = args.samples
	print("Dimensions of Measurement Space: {}".format(dimen))
	print("Number of Samples: {}".format(Z))
	print("Classes: {}".format(K))

	# Checks that this is even possible to calculate.
	if config.computeLimit(M, K):
		print("Possible measurements exceed memory capabilities.")
		sys.exit()

	print("Generating {0}x{0} Gain Matrix. Identity Matrix: {1}".format(K, args.identity))

	gain = genGain(K, identity=args.identity)
	print("{}x{} Economic Gain Matrix Generated".format(len(gain), len(gain[0])))

	# Generates measures
	print("Generating {} Measure-Value pairs.".format(Z))
	print("Generating measures.")
	generator = MeasurementGenerator(dimen)
	measures = generator.genMeas(Z)

	assigner = ClassAssign(dimen, K)
	tags = assigner.assign(measures)
	print("{} measures and {} values generated.".format(len(measures), len(tags)))

	## Generates classifier. 
	print("Generating class conditional probabilities for {} classes and {} possible measures.".format(K, M))

	conditionals = genCCP(K, dimen)
	print("Class conditional probabilities generated for {} classes and {} possible measures".format(len(conditionals), len(conditionals[0])))

	classifier = BayesClassifier(None, conditionals, eGain = gain) # No priors given since vFold always assigns.

	print("Testing classifier. V-fold factor: {}".format(V))
	expGain = test(classifier, measures, tags, V=V)

	print("The expected gain for the given data is: {}".format(expGain))

	#### Here we will work on updating
	if args.optimization:
		print("Fitting data for improved performance. Improvement factor {} used over {} iterations.".format(args.optimization, args.iteration))
		gains = []
		# Going to set priors generated from this measurement set as permanent priors. 
		priors = calcClassProb(tags, K)
		classifier.priorUpdate(priors)
		for i in range(args.iteration):
			# print(priors)
			classifier.optimize(args.optimization, measures, tags)

			measures, tags = fitData(classifier, generator, Z)

			expGain = test(classifier, measures, tags, V=V)
			gains.append(expGain)
			print("Expected Gain from iteration {} is {}".format(i+1, expGain))
		print("The expected gain for fitted data after {} iterations is: {}".format(args.iteration, gains[-1]))

	# Writes all data to files
	print("Writing to file.")
	reader.writeData(measures, tags, dimen)
	reader.writePriors(classifier.prior)
	reader.writeGain(gain)
	reader.writeCCP(classifier.cond)
	print("Done.")

if __name__ == "__main__":
	main()
