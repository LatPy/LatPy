import latpy
import numpy as np

latpy.lib.helloPrint()

B = np.array([[1, 2], [3, 4]])
print(latpy.compute_gso(B))
print(latpy.volume(B))
print(latpy.sl(B))
print(latpy.pot(B))
