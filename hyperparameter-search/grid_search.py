import numpy as np
import itertools
from random import choice

# These are the defaults initialised with learning rate and regualarisation constant. One can change as required.
default_x1 = -1
default_y1 = -1
default_x2 = -4
default_y2 = -3

# This method accepts two tuples in the format of (x1=start for learning rate, y1=start for regualrization paramter)
# and (x2=end for learning rate, y2=end for regualrisation parameter) along with an
# optional precision paramter.


def check_default_bounds(x1y1, x2y2):
    x1, y1 = x1y1
    x2, y2 = x2y2
    
    is_x1_outofbounds = x1 > default_x1 or x2 < default_x2
    is_y1_outofbounds = y1 > default_y1 or y2 < default_y2
    return is_x1_outofbounds or is_y1_outofbounds


def grid_search(x1y1, x2y2, precision=3):
    x1, y1 = x1y1
    x2, y2 = x2y2
    
    if(check_default_bounds((x1, y1), (x2, y2)) == True):
        print("Invalid bounds for the given grid")
        return
    
    learning_rate_arr = np.logspace(x1, x2, num=20)
    regularisation_constant = np.logspace(y1, y2, num=20)
    grid = list(itertools.product(learning_rate_arr, regularisation_constant))
    random_tuple = choice(grid)
    random_ln = round(random_tuple[0], precision)
    random_reg = round(random_tuple[1], precision)
    random = (random_ln, random_reg)
    return random


# Method to get the first initialised random
def get_first_random_grid_search():
    random1 = grid_search((default_x1, default_y1), (default_x2, default_y2))
    return random1

random1 = get_first_random_grid_search()

print(random1)
