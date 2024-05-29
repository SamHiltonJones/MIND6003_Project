import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

array = np.load('arrays/updated_whole_map_point_cloud.npy')

filtered_array = array[~np.isinf(array).any(axis=1) & 
                        (array[:, 0] > -6) & (array[:, 0] < 5) & 
                        (array[:, 1] > -5) & (array[:, 1] < 5)]

# Sample a subset of the filtered data
num_samples = int(1.0 * len(filtered_array))
random_indices = np.random.choice(filtered_array.shape[0], num_samples, replace=False)
sampled_array = filtered_array[random_indices]

# print("Sampled Array Shape:", sampled_array.shape)
# print("Data Type:", sampled_array.dtype)

# print("First 10 elements of sampled data:\n", sampled_array[:10])

# print("Statistics of sampled data:")
# print("  Min:", np.min(sampled_array, axis=0))
# print("  Max:", np.max(sampled_array, axis=0))
# print("  Mean:", np.mean(sampled_array, axis=0))
# print("  Std Deviation:", np.std(sampled_array, axis=0))

x = sampled_array[:, 0]
y = sampled_array[:, 1]
z = sampled_array[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='b', marker='o', s=1.0)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# plt.title('Randomly Sampled Point Cloud Visualization')
plt.show()
