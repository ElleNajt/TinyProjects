# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 15:49:56 2020

@author: lnajt
"""
#import pyximport; pyximport.install()

import fib
import primes

def uncompiled_fib(n):
    """Print the Fibonacci series up to n."""
    a, b = 0, 1
    while b < n:
        print(b, end=' ')
        a, b = b, a + b

    print()
    
#number = 1000000000000000000000000000000000
#fib.fib(number)
#uncompiled_fib(number)

def uncompiled_primes(nb_primes):
    p = []
    if nb_primes > 1000:
        nb_primes = 1000

    len_p = 0  # The current number of elements in p.
    n = 2
    while len_p < nb_primes:
        # Is n prime?
        for i in p[:len_p]:
            if n % i == 0:
                break

        # If no break occurred in the loop, we have a prime.
        else:
            p.append(n)
            len_p += 1
        n += 1

    # Let's return the result in a python list:
    result_as_list  = [prime for prime in p[:len_p]]
    return result_as_list
    
print(primes.primes(1000))
print(uncompiled_primes(1000))