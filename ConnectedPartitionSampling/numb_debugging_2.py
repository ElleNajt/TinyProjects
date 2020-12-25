from numba import njit
from numba.typed import List

@njit 
def is_empty(my_list):
    if len(my_list) == 0:
        print("empty")
    else:
        print("not empty")
    
non_empty_list = List()
non_empty_list.append(1)
is_empty(non_empty_list)    

empty_list_typed = List()
empty_list_typed.append(1)
empty_list_typed.remove(1)
is_empty(empty_list_typed)

empty_list = List()
print(len(List())) # This correctly prints 0
is_empty(empty_list)

