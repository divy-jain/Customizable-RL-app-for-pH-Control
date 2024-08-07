import itertools
import numpy as np
import pandas as pd


def convert_input_data_training(df: pd.DataFrame, n, m, h, c_key, t_key):
    assert h >= m
    assert h >= 1

    inputs = [t_key, c_key]
    cols, names = [], []
    # Input sequence (t-n, ... t-1)
    for i in range(n, -1, -1):
        cols.append(df.shift(i))
        if i == 0:
            names += [f'{x}(t)' for x in inputs]
        else:
            names += [('%s(t-%d)' % (x, i)) for x in inputs]

    # Input sequence (u_t, u_t+1 ..., u_t+m)
    for i in range(1, m + 1):
        cols.append(df[c_key].shift(-i))
        names += ['%s(t+%d)' % (c_key, i)]

    target_index = len(names)

    # Output sequence (t+1, ... t+h)
    for i in range(1, h + 1):
        cols.append(df[t_key].shift(-i))
        names += ['%s(t+%d)' % (t_key, i)]

    # Put it all together
    data = pd.concat(cols, axis=1)
    data.columns = names
    # Drop rows with NaN values
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    data = data.applymap(float)

    return data.iloc[:, :target_index].copy(), data.iloc[:, target_index:].copy()


def compute_all_possible_strategies(m, action_space: list, du_max=None):
    strategies = np.array(list(itertools.product(action_space, repeat=m + 1)), dtype=float)

    if du_max is not None:
        # Filter out strategies that dont match the constraint
        diff = np.abs(np.diff(strategies, axis=1))
        strategies = strategies[np.max(diff, axis=1) <= du_max]

    return strategies


def normalize(x, min_x, max_x):
    return (x - min_x) / (max_x - min_x)


def denormalize(y, min_x, max_x):
    return y * (max_x - min_x) + min_x


#The compute_all_possible_strategies function generates all possible sequences of control actions (strategies) of length m+1, using the given action_space and filtering out strategies that don't match the constraint du_max if provided. 
# 1. Let's assume m = 2, action_space = [0, 1], and du_max = 1.
# 2. The function first creates all possible combinations of control actions of length m+1 (in this case, 3) using itertools.product. This results in the following 2D array:

# strategies = [[0, 0, 0],
#               [0, 0, 1],
#               [0, 1, 0],
#               [0, 1, 1],
#               [1, 0, 0],
#               [1, 0, 1],
#               [1, 1, 0],
#               [1, 1, 1]]
# 3. If du_max is provided, the function filters out strategies that don't match the constraint. The constraint checks whether the absolute difference between consecutive actions in each strategy is less than or equal to du_max. In our example, du_max = 1, so we calculate the differences:
# The np.diff function calculates the differences between consecutive elements of an array along a given axis. In this case, the axis is set to 1, which means the differences are calculated between consecutive elements in each row of the strategies array.

# diff = [[0, 0],
#         [0, 1],
#         [1, 1],
#         [1, 0],
#         [1, 1],
#         [1, 0],
#         [0, 1],
#         [0, 0]]
# 4. We then find the maximum difference in each strategy:

# max_diff = [0, 1, 1, 1, 1, 1, 1, 0]
# We filter out the strategies where the maximum difference is less than or equal to du_max:

# filtered_strategies = [[0, 0, 0],
#                        [0, 0, 1],
#                        [0, 1, 0],
#                        [0, 1, 1],
#                        [1, 0, 0],
#                        [1, 0, 1],
#                        [1, 1, 0],
#                        [1, 1, 1]]
# In this case, all strategies have a maximum difference less than or equal to du_max. If du_max were smaller, some strategies would be filtered out.

# 5. Finally, the function returns the filtered strategies:

# return filtered_strategies

# In summary, the function computes all possible control action sequences (strategies) of length m+1, using the given action space, and filters out strategies that don't match the constraint du_max if provided.



#######################################
# itertools.product is a function in the Python itertools module that computes the Cartesian product of input iterables. In simple terms, it generates all possible combinations of elements from the input iterables, where each combination has one element from each input iterable.

# For example, if you have two lists A = [0, 1] and B = ['x', 'y'], the Cartesian product of A and B would be [(0, 'x'), (0, 'y'), (1, 'x'), (1, 'y')].

# itertools.product can be used with any number of input iterables and can also generate combinations of a specified length (repeat parameter). Here's an example:

# python
# Copy code
# import itertools

# A = [0, 1]
# B = ['x', 'y']

# # Compute the Cartesian product of A and B
# result = list(itertools.product(A, B))
# print(result)  # Output: [(0, 'x'), (0, 'y'), (1, 'x'), (1, 'y')]

# # Compute the Cartesian product of A and B, with combinations of length 3
# result_repeat = list(itertools.product(A, B, repeat=3))
# print(result_repeat)
# # Output: [(0, 'x', 0), (0, 'x', 1), (0, 'y', 0), (0, 'y', 1), (1, 'x', 0), (1, 'x', 1), (1, 'y', 0), (1, 'y', 1)]
# In the context of the compute_all_possible_strategies function, itertools.product is used to generate all possible sequences of control actions of length m+1 using the given action_space.





