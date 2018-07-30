#请和perceptron.py在同一目录下使用

import numpy as np
from perceptron import perceptron

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
targets = [[0], [1], [1], [0]]
add_ons = [[[1], [1], [1], [1]], [[1], [1], [1], [0]], [[1], [1], [0], [1]], [[1], [1], [0], [0]], [[1], [0], [1], [1]], [[1], [0], [1], [0]], [[1], [0], [0], [1]], [[1], [0], [0], [0]], [[0], [1], [1], [1]], [[0], [1], [1], [0]], [[0], [1], [0], [1]], [[0], [1], [0], [0]], [[0], [0], [1], [1]], [[0], [0], [1], [0]], [[0], [0], [0], [1]], [[0], [0], [0], [0]]]

num_of_failure = 0
good_iter_nums = []
good_add_ons = []

for add_on in add_ons:
    percp = perceptron(inputs, targets, add_on)
    iter_in_prac = percp.train()
    pre = percp.predict(inputs)
    if np.sum(pre == targets) == 4:
        good_iter_nums.append(iter_in_prac)
        good_add_ons.append(add_on)
    else:
        num_of_failure = num_of_failure + 1
        print(add_on)
        
print(num_of_failure)

print('max: ')
print(max(good_iter_nums))
print(good_add_ons[good_iter_nums.index(max(good_iter_nums))])

print('min: ')
print(min(good_iter_nums))
print(good_add_ons[good_iter_nums.index(min(good_iter_nums))])
