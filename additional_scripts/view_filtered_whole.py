import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

array1 = np.load('install/office_robot_pkg/share/office_robot_pkg/point_cloud/filtered_pcds/filtered_point_cloud.npy')
array2 = np.load('arrays/whole_map_point_cloud.npy')

filtered_array1 = array1[~np.isin(array1, [np.inf, -np.inf]).any(axis=1)]
filtered_array2 = array2[~np.isin(array2, [np.inf, -np.inf]).any(axis=1)]

filtered_array2 = filtered_array2[~np.isinf(filtered_array2).any(axis=1) & 
                        (filtered_array2[:, 0] > -6) & (filtered_array2[:, 0] < 5) & 
                        (filtered_array2[:, 1] > -5) & (filtered_array2[:, 1] < 5)]

num_samples1 = int(1.0 * len(filtered_array1)) 
num_samples2 = int(1.0 * len(filtered_array2)) 

sampled_indices1 = np.random.choice(filtered_array1.shape[0], num_samples1, replace=False)
sampled_indices2 = np.random.choice(filtered_array2.shape[0], num_samples2, replace=False)

sampled_array1 = filtered_array1[sampled_indices1]
sampled_array2 = filtered_array2[sampled_indices2]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(sampled_array1[:, 0], sampled_array1[:, 1], sampled_array1[:, 2], c='r', marker='o', label='Filtered Points', s = 0.5)
ax.scatter(sampled_array2[:, 0], sampled_array2[:, 1], sampled_array2[:, 2], c='b', marker='^', label='Whole Map Points', s= 0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# plt.title('Comparison of Two Point Clouds')
# plt.legend()
plt.show()
