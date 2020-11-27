# Probability functions for generating data sets.
from random import uniform
from typing import Tuple

def genProbs(Z: int) -> Tuple[float]:
#Generates array of normalized probabilities of length Z. 

# Generates probability scores.
	probs = [uniform(0,1) for _ in range(Z)]

#Normalizes probability values. 
	total = sum(probs)
	for i in range(Z):
		probs[i] = probs[i]/total
	
	return tuple(probs)

def genCmlProbs(probs: Tuple[float]) -> Tuple[float]:
#Converts a probability distribution into a cumulative probability distribution.
	probs_cml = [_ for _ in probs]
	for i in range(1, len(probs_cml)):
		probs_cml[i] +=  probs_cml[i-1]
	return tuple(probs_cml)

def main():
	from random import randint

	# Checks prob distribution is normalized for 100000 tries. 
	for _ in range(100000):
		x = randint(1,1000)
		assert (sum(genProbs(x)) <= 1.0000000005) or (sum(genProbs(x)) >= .999999999995)

	# tests that cumulative probability sums to 1 for 100000 tries
	for _ in range(100000):
		x = genProbs(randint(1,1000))
		y = genCmlProbs(x)
		assert ((y[-1] <= 1.0000005) or (y[-1] >= .99999999995))

if __name__ == "__main__":
    main()