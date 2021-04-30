import numpy as np

a = np.array([3,4,5])
b = np.array([1,2,3])
c = np.array([2,3,4])
print((c<a).all() and (c>b).all())