# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 12:50:18 2020

@author: lnajt
"""

import numba as nb

from numba.typed import Dict


params_default = nb.typed.Dict.empty(
    key_type=nb.typeof('1'),
    value_type=nb.typeof('1')
)


#@nb.njit
def remove_values(my_dictionary, my_input_set):
    #removes values of the my_input_set from the input set
    print(my_dictionary, my_input_set)
    for this_value in my_dictionary.values():
        print(my_input_set)
        print(this_value)
        if this_value in my_input_set:
            my_input_set.remove(this_value)
    print(my_input_set)
    return my_input_set
    
test_set = set([1,2,3,4,5,6])
test_dictionary = Dict() 
for x in [1,2,3,4,5,6]:
    test_dictionary[x] = 1
    
print(test_set)
test_set = remove_values(test_dictionary, test_set)
print(test_set)