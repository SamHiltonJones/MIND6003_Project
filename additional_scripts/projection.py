import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

array = np.load('arrays/significant_clusters1.npy')
filtered_array = array[~np.isinf(array).any(axis=1)]
import numpy as np
import matplotlib.pyplot as plt

def rotate_points(x, y, z, theta_x, theta_y):
    """ Rotate the points in the point cloud around the X and Y axes """
    cos_theta, sin_theta = np.cos(theta_x), np.sin(theta_x)
    y_new = y * cos_theta - z * sin_theta
    z_new = y * sin_theta + z * cos_theta
    y, z = y_new, z_new

    cos_theta, sin_theta = np.cos(theta_y), np.sin(theta_y)
    x_new = x * cos_theta + z * sin_theta
    z = z * cos_theta - x * sin_theta
    x = x_new

    return x, y, z

array = np.load('arrays/significant_clusters1.npy')
filtered_array = array[~np.isinf(array).any(axis=1)]

num_samples = int(1.0 * len(filtered_array))
random_indices = np.random.choice(filtered_array.shape[0], num_samples, replace=False)
sampled_array = filtered_array[random_indices]

x = sampled_array[:, 0]
y = sampled_array[:, 1]
z = sampled_array[:, 2]

theta_x = np.pi / 6  
theta_y = np.pi / 4 

x_rotated, y_rotated, z_rotated = rotate_points(x, y, z, theta_x, theta_y)

projected_x = x_rotated
projected_y = z_rotated

plt.figure(figsize=(10, 6))
plt.scatter(projected_x, projected_y, c='r', marker='o')
plt.title("2D Projection of 3D Point Cloud after Rotation")
plt.xlabel("Projected X (from rotated XY)")
plt.ylabel("Projected Y (from rotated Z)")
plt.grid(True)
plt.show()
