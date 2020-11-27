### Reads files containing measures, eGain, prior probabilities, and class conditional probabilities
### Performs either v-fold testing or regular linear testing and calculates expected gain for measures. 
####Inputs
### data = csv file of rows of tuple-int pair of measurements values and their real tags.
### gain = csv file of rows of tuples for each row in economic gain matrix. Assumes matrix is identity if not listed.
### priors = csv file of rows of floats representing prior probability of classes 
### conditionals = csv file of rows of lists of floats, each corresponding to the class conditional probability of measures given class tags
### dimen = tuple entered in command line with each ith element expressing the cardinality of Ith dimension of measurement space
### V = int for amount of v-folds to partition data into for V-fold testing. Assumes no V-fold testing if absent and tags whole data set as testing data.
#### Outputs
###  gain = float value of expected gain from testing.
import argparse
import reader
from testing import test
from bayes import BayesClassifier

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("data", help="File of measure class pairs to test.", type=str)
	parser.add_argument("dimen", help="Tuple representing measure space.", type=str)
	parser.add_argument("priors", help="File designating prior probabilities of classes.", type=str)
	parser.add_argument("conditionals", help="File designating class conditional probabilities.", type=str)
	parser.add_argument("--eGain", "-e", help="Economic gain matrix for data. If not provided assumes identity matrix.", type=str)
	parser.add_argument("--vFolds", "-v", help="Number of v-fold partitions for testing. If not provided, assumes all data is for testing.", type=int)
	args = parser.parse_args()

	# Reading data
	dimen = eval(args.dimen)
	measures, tags = reader.readData(args.data, dimen)
	priors = reader.readPriors(args.priors)
	conds = reader.readCCP(args.conditionals)
	e = False
	if args.eGain:
		e = reader.readGain(args.eGain)

	classifier = BayesClassifier(priors, conds, eGain=e)

	expGain = test(classifier, measures, tags, V=args.vFolds)

	print("The expected gain for the data is: {}".format(expGain))

if __name__ == "__main__":
	main()
