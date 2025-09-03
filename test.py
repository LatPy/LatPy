import latpy
import numpy as np

B = np.array([[1, 2, 5], [3, 4, 6], [7, 8, 11]])

C = latpy.LatPy(B)
print(C)
print(C.compute_gso())
print(C.volume())
print(C.sl())
print(C.hf())
print(C.rhf())
