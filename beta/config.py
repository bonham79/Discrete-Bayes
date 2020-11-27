import os
import sys

CARDINALITY_RANGE_LIMIT = 20
CLASS_RANGE_LIMIT = 5
SAMPLE_SIZE_LIMIT = CARDINALITY_RANGE_LIMIT * 10
DIMENSION_RANGE_LIMIT = 6
ECONOMIC_GAIN_MAX = 10
ECONOMIC_GAIN_MIN = -10
V_FOLD_FACTOR = 10

def computeLimit(M:int, K:int)-> bool:
	# Predicts whether measurement space and class sizes are possible with memory limitations of computer. 
	floatSize = sys.getsizeof(0.0)# Yields the size for floats on this machine
	listSize = sys.getsizeof([0.0]) # Yields size of size of list on implementiation.
	doubleListSize = sys.getsizeof([0.0,0.0]) # Yields size of list with extra element to see increase form pointer.
	pointerSize = doubleListSize - listSize
	# what happens when we have M float, K lists, and M*K pointers
	sizeOfCCP = floatSize*M + listSize*K + M*K*pointerSize

	# Gets physical memory size on machine.
	mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')  # e.g. 4015976448

	# Determines if physical memory permits calculations. (Memory is scaled by an arbitrary factor.)
	return mem_bytes * .5 <= sizeOfCCP

