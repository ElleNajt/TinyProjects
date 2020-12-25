
import numba as nb
from numba.typed import Dict

params_default = nb.typed.Dict.empty(
    key_type=nb.typeof('1'),
    value_type=nb.typeof('1')
)

  
@nb.njit
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
    
print(test_dictionary)

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