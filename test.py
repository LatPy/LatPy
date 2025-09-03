import latpy
import numpy as np

B = np.array([[1, 2], [3, 4]])
print(latpy.compute_gso(B))
print(latpy.volume(B))
print(latpy.sl(B))
print(latpy.pot(B))

C = latpy.LatPy(B)
print(C)
print(C.compute_gso())
print(C.volume())
print(C.sl())
