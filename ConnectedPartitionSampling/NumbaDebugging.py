# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 12:10:56 2020

@author: lnajt
"""


from numba import njit
from numba.typed import Dict

test = Dict() 
for x in [1,2,3,4,5]:
    test[x] = x
    
    
print(test)
test[6] = 10
print(test)

@njit
def make_dict():
    test = Dict() 
    for x in [1,2,3,4,5]:
        test[x] = x
        
        
    print(test)
    test[6] = 10
    print(test)
    
make_dict()

test = Dict() 
for x in [1,2,3,4,5]:
    test[x] = 1
    
for x in test.values():
    print(x)
    
    
    

import numba as nb
from numba.typed import Dict
    
params_default = nb.typed.Dict.empty(
    key_type=nb.typeof('1'),
    value_type=nb.typeof('1')
)
    
#@nb.njit
def remove_values(dictionary, input_set):
    #removes values of the dictionary from the input set
    print(dictionary, input_set)
    for value in dictionary.values():
        print(input_set)
        print(value)
        if value in input_set:
            input_set.remove(value)
    print(input_set)
    return input_set
    
test_set = set([1,2,3,4,5,6])
test_dictionary = Dict() 
for x in [1,2,3,4,5,6]:
    test_dictionary[x] = 1
    

test_set = remove_values(test_dictionary, test_set)
print(test_set)

test_set = set([1,2,3,4,5,6])
input_set = test_set
dictionary = test_dictionary

print(dictionary, input_set)
for value in dictionary.values():
    print(input_set)
    print(value)
    if value in input_set:
        input_set.remove(value)
print(input_set)