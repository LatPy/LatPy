import latpy
import numpy as np

B = np.array([
    [1, 2, 5], 
    [3, 4, 6], 
    #[7, 8, 11]
], dtype=np.int64)

C = latpy.LatPy(B)
print(C)
print(C.compute_gso())
print(C.volume())
print(C.sl())
print(C.pot())
print(C.hf())
print(C.rhf())
print(C.gh())
print(C.od())
print(C.lagrange())
print(C.size(0.55))
print(C.size(0.55).compute_gso())
