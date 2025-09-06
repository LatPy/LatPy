import latpy
import numpy as np

B = np.random.randint(100, 999, size=(50, 50), dtype=np.int64)

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
print(C.size(0.5))
print(C.size(0.5).is_size())
print(C.lll(eta=0.5))
print(C.lll(eta=0.5)[0].is_lll(0.99))
print(C.deep_lll(eta=0.5))
print(C.deep_lll(eta=0.5, gamma=50)[0].is_lll(0.99))
print(C.l2(eta=0.5))
print(C.l2(eta=0.5)[0].is_lll(0.99))
