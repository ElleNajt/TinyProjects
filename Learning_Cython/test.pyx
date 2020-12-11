# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 06:44:22 2020

@author: lnajt
"""

from __future__ import print_function

def fib(n):
    """Print the Fibonacci series up to n."""
    a, b = 0, 1
    while b < n:
        print(b, end=' ')
        a, b = b, a + b

    print()