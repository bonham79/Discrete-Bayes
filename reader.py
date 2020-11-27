# Modules for reading and writing data generated from testing.
import csv
from typing import Tuple, List
from matrix_tools import linear, delinear

def readData(file, dimen: Tuple[int])-> Tuple[Tuple[int]]:
	# Reads in csv file 'file' for data values.
	# Assumes data is grouped as csv row 'measure,tag'
	# Returns tuple of measures and tags with measures as linear indexes.
	# Example: (1,1,3),5 for measure (1,1,3) of class 5. 
	measures = []
	tags = []
	with open(file) as data:
		csvReader = csv.reader(data, delimiter=',')
		for row in csvReader:
			# convert to real values
			measures.append(eval(",".join(row[:-1])))
			tags.append(int(row[-1]))

			# Checks for measures to remain accurate length.
			assert len(measures[-1]) == len(measures[0]), "Measures aren't uniform lengths"

	# Linearizes indexes
	measuresLin = tuple(linear(tpl, dimen) for tpl in measures)
	return measuresLin, tuple(tags)

def writeData(measures: Tuple[int], tags: Tuple[int], dimen: Tuple[int]):
	# Writes measurement tags pairs to file. Unique to make life
	# easier for measurement writing. 
	assert len(measures) == len(tags)

	m = len(measures)

	with open("data.txt", mode="w") as data:
		for i in range(m):
			meas = delinear(measures[i],dimen)
			data.write(",".join([str(meas),str(tags[i])]))
			data.write("\n")

def readGain(file)->Tuple[Tuple[float]]:
	# Reads in Economic Gain matrix on assumption that each row is the 'ith' row of matrix
	# representing true class assignment i. Assumes matrix values are separated by commas
	# Example:
	# 1,1,3
	# 2,3,2
	# 3,2,1
	# Will be read in as a three by three matrix with row one being (1,1,3)

	eGain = []
	with open(file) as data:
		csvReader = csv.reader(data)
		for row in csvReader:
			eGain.append(eval(",".join(row)))

	# Quick check for generated matrix to be square. 
	for i in eGain:
		assert len(i) == len(eGain), "Not square matrix"

	eGainTpl = tuple(tuple(i) for i in eGain)
	return eGainTpl

def writeGain(e: Tuple[Tuple[int]]):
	# Writes eGain matrix to file.
	# Each row is row of eGain matrices. 
	with open("eGain.txt", mode="w") as data:
		for i in e:
			data.write(",".join([str(j) for j in i]))
			data.write("\n")

def readPriors(file)-> Tuple[float]:
	# Reads in prior class probabilities from file. Assumes each row is 
	# prior probability. 
	priors = []
	with open(file) as data:
		csvReader = csv.reader(data)
		for row in csvReader:
			priors.append(eval(row[0]))
	sums = sum(priors)
	# Checks priors are appropriate. 
	assert sums <= 1.0005 and sums >= .99995
	return(tuple(priors))

def writePriors(priors: Tuple[float]):
	# For writing priors to file
	with open("priors.txt", mode="w") as data:
		for i in priors:
			data.write(str(i))
			data.write("\n")


def readCCP(file)->List[List[float]]:
	# Reads in class conditional probabilities
	# Assumes each row is a listing of probabilities for data given
	# class indexed by row place. entries are delineated by commas. 
	ccps = []
	with open(file) as data:
		csvReader = csv.reader(data)
		for row in csvReader:
			ccps.append(list(eval(",".join(row))))

	# Makes sure our values are appropriate. 
	for i in range(1,len(ccps)):
		sums = sum(ccps[i])
		assert len(ccps[i]) == len(ccps[i-1])
		assert sums <= 1.0005 and sums >= .99995

	return ccps

def writeCCP(conds: List[List[float]]):
	# Writes class conditional probabilities to file.
	with open("conds.txt", mode="w") as data:
		for i in conds:
			ccp = [str(j) for j in i]
			data.write(",".join(ccp))
			data.write("\n")

def main():
	## Measure reading tests
	measures, tags = readData("test.txt", (4,4,4))
	writeData(measures,tags, (4,4,4))
	measures2, tags2 = readData("data.txt", (4,4,4))
	assert measures == measures2
	assert tags == tags2

	## Egain reader tests
	e = ((5,3,2),(3,2,3),(5,5,5))
	writeGain(e)
	f = readGain("eGain.txt")
	assert f == e

	## Priors reader test
	priors = (.5,.3,.2)
	writePriors(priors)
	priors2 = readPriors("priors.txt")
	assert priors == priors2

	## Conds reader tests
	conds = [[.3,.05,.45,.2],[.3,.3,.3,.1],[.9, .025, .025, .05]]
	writeCCP(conds)
	conds2 = readCCP("conds.txt")
	assert conds == conds2

if __name__ == "__main__":
	main()
