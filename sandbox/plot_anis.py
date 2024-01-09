# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# %%
soa = np.array([[0, 0, 0, -0.318662, -0.947867, 0.001013], [0, 0, 0, 0.947849, -0.318663, -0.005861],
                [0, 0, 0, -0.005879, 9.071099, -0.999982]])

# soa = np.array([[0, 0, 0, -0.318662, 0.947849, -0.005879], [0, 0, 0, -0.947867, -0.318663, 9.071099],
#                 [0, 0, 0, 0.001013, -0.005861, -0.999982]])

X, Y, Z, U, V, W = zip(*soa)

# %%
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(X, Y, Z, U, V, W)
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
plt.show()
# %%
