import numpy as np
import itertools
from random import choice

# These are the defaults initialised with learning rate and regualarisation 
# constant. One can change as required.
default_a_min = -10
default_a_max = 2
default_b_min = -10
default_b_max = 2
default_c_min = 0
default_c_max = 7
default_d_min = 0
default_d_max = 6
default_e_min = 0
default_e_max = 6



# This method accepts two tuples in the format of (x1=start for learning rate, 
# y1=start for regularization paramter) and (x2=end for learning rate, y2=end 
# for regualrisation parameter) along with an optional precision paramter.
def check_default_bounds(range_min, range_max):
    a1, b1, c1, d1, e1 = range_min
    a2, b2, c2, d2, e2 = range_max
    
    is_a1_outofbounds = a1 < default_a_min or a1 > default_a_max
    is_a2_outofbounds = a2 < default_a_min or a2 > default_a_max
    is_b1_outofbounds = b1 < default_b_min or b1 > default_b_max
    is_b2_outofbounds = b2 < default_b_min or b2 > default_b_max
    is_c1_outofbounds = c1 < default_c_min or c1 > default_c_max
    is_c2_outofbounds = c2 < default_c_min or c2 > default_c_max
    is_d1_outofbounds = d1 < default_d_min or d1 > default_d_max
    is_d2_outofbounds = d2 < default_d_min or d2 > default_d_max
    is_e1_outofbounds = e1 < default_e_min or e1 > default_e_max
    is_e2_outofbounds = e2 < default_e_min or e2 > default_e_max
    
    return (
        is_a1_outofbounds or
        is_a2_outofbounds or
        is_b1_outofbounds or
        is_b2_outofbounds or
        is_c1_outofbounds or
        is_c2_outofbounds or
        is_d1_outofbounds or
        is_d2_outofbounds or
        is_e1_outofbounds or
        is_e2_outofbounds
    )


def grid_search(range_min, range_max, n_points = 20):
    import numpy as np
    
    a1, b1, c1, d1, e1 = range_min
    a2, b2, c2, d2, e2 = range_max
    
    if(check_default_bounds(range_min, range_max) == True):
        print("Invalid bounds for the given grid")
        return
    
    learning_rate_arr = np.logspace(a1, a2, num=n_points)
    regularisation_constant_arr = np.logspace(b1, b2, num=n_points)
    iterations_arr = np.logspace(c1, c2, num=n_points)
    batch_size_arr = np.logspace(d1, d2, num=n_points)
    hidden_neurons_arr = np.logspace(e1, e2, num=n_points)
    
    grid = list(
        itertools.product(
            learning_rate_arr,
            regularisation_constant_arr,
            iterations_arr,
            batch_size_arr,
            hidden_neurons_arr
        )
    )
    
    random_tuple = choice(grid)
    random_tuple = [np.float32(z) for z in random_tuple]
    
    return random_tuple


# Method to get the first initialised random
def _get_first_random_grid_search():
    n_points = 20
    
    random1 = grid_search(
        (default_x1, default_y1),
        (default_x2, default_y2),
        n_points
    )
    return random1


if __name__ == '__main__':
    random1 = _get_first_random_grid_search()
    
    print(random1)
