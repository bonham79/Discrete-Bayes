# Tools for transforming M dimensional tuples into a linear address in an
# array. 'linear' transforms into a integer address in a linear array
# and 'delinear' reverts integer to original M dimensional tuple.

from typing import Tuple

def linear(index: Tuple[int], dimen: Tuple[int]) -> int:
# Takes tuple and performs row-order linear conversion for N-dimensional matrix
# N = len(dimen). dimen is set of ranges for each dimension in matrix.
# index is the original tuple in N space. 
	size = len(index)
	linIndex = index[0]
	for i in range(1,size):
		linIndex *= dimen[i]
		linIndex += index[i]
	return linIndex

def delinear(linIndex: int, dimen: Tuple[int]) -> Tuple[int]:
# Takes value from linearly-converted array and returns original tuple of N
# dimension matrix location. dimen is original ranges of N dimension array.
# linIndex is the linear index in the linearly-converted array.
	N = len(dimen)
	value = linIndex
	index = [_ for _ in dimen]

	# Each index is just the current valus's remainder after division
	# by respective dimension's range.
	for i in range(N):
		index[N-i-1] = value % dimen[N-i-1]
		value = (value - index[N-i-1])//dimen[N-i-1]

	return tuple(index)

def main():
# Runs 100000 conversions from tuple to linear index to tuple. Asserts product
# of packing and unpacking creates no change. 
	import random

	for _ in range(100000):
		dimen = tuple([random.randint(1,10000) for _ in range(random.randint(1,100))])
		tup = tuple([random.randint(0,dimen[i]-1) for i in range(len(dimen))])
		assert tup == delinear(linear(tup,dimen),dimen)

if __name__ == "__main__":
    main()