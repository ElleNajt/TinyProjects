'''
Code for finding sum ambiguous pairs.

That is, we will work in the structure ZZp, which are the integers with the binary operations:

- Usual Addition
- Multiplication is done mod p.

This structure underlies what happens when using hash functions to do
isolation lemma disambiguation type stuff.

We will say that a pair of two distinct subsets of {0,1,...,p-1} are (p) sum ambiguous
if, for all a in Z_p, multiplying each set by a (mod p) results in sets that
sum to the same values in Z.
'''

import itertools
import numpy as np
from sympy import nextprime

class ZZp:
    def __init__(self, p):
        return
    
def mult(a, A,p ):
    #Returns a*b mod pfor each b in A, mo
    output = []
    for x in A:
        output.append(a * x % p)
    return output


def search(p,k):
    #Searches for two sets of size k in ZZp that are sum ambiguous
    prime = nextprime( p**6)
    numbers = list(range(p))
    hashes = list(range(prime))
    successes = []
    for A in itertools.combinations_with_replacement(numbers, k):
        for B in itertools.combinations_with_replacement(numbers, k):
            if A != B:
                was_disambiguated = False
                for a in hashes:
                    aA = mult(a,A, prime )
                    aB = mult(a,B, prime)
                    if np.sum(aA) != np.sum(aB):
                        #print(A, aA, B, aB)
                        #print(a, np.sum(aA), np.sum(aB))
                        was_disambiguated = True
                if was_disambiguated == False:
                    successes.append( [A,B])
    return successes

p = 8
k = 3
successes = search(p,k)
print(successes)
##p = 11, k = 3 gives plenty - until we increased the range of the hash to p**2
                