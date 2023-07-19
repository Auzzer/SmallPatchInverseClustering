import numpy as np
import itertools
from sklearn import svm, cluster, manifold
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
# Part 1: generate pseudo points
# Step 1.1: Split each feature uniformly into intervals (cubes)
def split_into_intervals(data, num_intervals):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    min_intervals = []; max_intervals = []
    for i in range(data.shape[1]):  # Iterate over each feature
        interval_temp = np.linspace(min_vals[i], max_vals[i], num_intervals + 1)
        min_intervals.append(interval_temp[:-1])
        max_intervals.append(interval_temp[1:])

    return min_intervals, max_intervals

# Step 1.2 Judge whether the cubes are empty or not based on all the possible combinations
def is_cube_empty(cube_min, cube_max, dataset):
    """
    Check if the n-dimensional cube defined by its minimum and maximum values is empty or not.
    """
    for point in dataset:
        inside_cube = all(cube_min[i] <= point[i] <= cube_max[i] for i in range(len(cube_min)))
        if inside_cube:
            return False  # Cube is not empty if at least one data point lies inside

    return True  # If no data points are inside the cube, it is empty

def generate_all_situations(num_elements, num_choices):
    all_situations = itertools.product(range(num_choices), repeat=num_elements)
    return all_situations

def generate_pseudo_points(data, num_interval):
    min_intervals, max_intervals = split_into_intervals(data, num_interval)
    empty_cubes_centers = []

    all_situations = generate_all_situations(data.shape[1], num_interval)
    for situation in all_situations:
        cube_min = [min_intervals[i][situation[i]] for i in range(data.shape[1])]
        cube_max = [max_intervals[i][situation[i]] for i in range(data.shape[1])]
        empty = is_cube_empty(cube_min, cube_max, data)
        if empty == True:
            empty_cubes_centers.append((np.array(cube_max)+np.array(cube_min))/2)
    return empty_cubes_centers



# Step 1.3: Create the new dataset with formal data points (label 1) and pseudo points (label 0)
def create_new_dataset(data, num_intervals):
    pseudo_points = generate_pseudo_points(data, num_intervals)

    if not pseudo_points:  # No pseudo points were generated
        return data, np.ones(data.shape[0])  # Return the original data with all labels set to 1

    new_data = np.vstack([data, pseudo_points])
    new_labels = np.hstack([np.ones(data.shape[0]), np.zeros(len(pseudo_points))])
    return new_data, new_labels

# Part 2: Sparse Region Identification
def Model4Sparse(new_data, new_labels, cost, gamma):
    # svm model
    clf = svm.SVC(gamma=gamma, C=cost)
    l0Num = np.count_nonzero(new_labels==0); l1Num = np.count_nonzero(new_labels==1)
    weight_arr = np.concatenate((np.repeat(l0Num/(l0Num+l1Num), l0Num), np.repeat(l1Num/(l0Num+l1Num), l1Num)), axis=0)
    clf.fit(new_data, new_labels, sample_weight=weight_arr)
    return clf

class Patch:
    def __init__(self, center, points, label):
        self.center = center
        self.points = points
        self.label = label

def create_patches_from_data(patch_labels, centers, data):
    patches = []
    for patch_num in range(centers.shape[0]):
        points_in_patch = data[patch_labels == patch_num]
        patches.append(Patch(centers[patch_num], points_in_patch, patch_num))
    return patches


def find_m_closest_pairs(patch1, patch2, m):
    distances = distance.cdist(patch1.points, patch2.points)
    sorted_indices = np.argsort(distances, axis=None)
    closest_indices = np.unravel_index(sorted_indices[:m], distances.shape)
    closest_pairs1 = patch1.points[closest_indices[0]]
    closest_pairs2 = patch2.points[closest_indices[1]]
    midpoints = (closest_pairs1 + closest_pairs2) / 2
    return closest_pairs1, closest_pairs2, midpoints

def aggregate_patches(patches, k, m, model):
    centers = np.array([patch.center for patch in patches])
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(centers)
    unrecovered_patches = set(patches)
    aggregated_patches = []

    while unrecovered_patches:
        current_patch = unrecovered_patches.pop()
        aggregated_patches.append(current_patch)
        distances, indices = nbrs.kneighbors(current_patch.center.reshape(1, -1), n_neighbors=k)
        nearest_patches = [patches[i] for i in indices[0]]

        while True:
            for patch in nearest_patches:
                if patch in unrecovered_patches:
                    _, _, midpoints = find_m_closest_pairs(current_patch, patch, m)
                    if sum(model.predict(mid_point.reshape((1, -1))) == 1 for mid_point in midpoints) > m / 2:
                        print("Aggregating patch")  # New print statement
                        unrecovered_patches.remove(patch)
                        aggregated_patches[-1].points = np.concatenate([aggregated_patches[-1].points, patch.points])
                        aggregated_patches[-1].center = np.mean(aggregated_patches[-1].points, axis=0)
            nearest_patches_in_unrecovered = [patch for patch in nearest_patches if patch in unrecovered_patches]
            print(f"nearest_patches_in_unrecovered: {nearest_patches_in_unrecovered}")  # New print statement
            if not nearest_patches_in_unrecovered:
                break
            else:
                distances, indices = nbrs.kneighbors(current_patch.center.reshape(1, -1), n_neighbors=k)
                nearest_patches = [patches[i] for i in indices[0]]

    return aggregated_patches



def Clustering(data, num_intervals, patch_num, cost, gamma, m, k, patch_label=None):
    """

    :param data: original dataset
    :param patch_num: the number of merged patches
    :param m: m-closest points pair
    :param k: k-nearest patch center
    :param patch_label: aggregate the data
    :return:
    step1: By k-means++, get the patches and their centers, the number of patches is determined by patch_num,
    and initialize all the patches' states as "unrecover"

    step2: iterate until all the patches are "recovered" (iter1):
        random pick up an unlabelled patch
        iterate until all the "unrecovered" patches are determined whether they should be aggregated to this picked patch:(iter2)
            find k-nearest patches of the selected patch(the measure is distance between patches' centers),
            then judge the midpoints of m-closest points pair lie in the sparse region
            if all the k-nearest patches are not be aggregated: jump to next iteration in iter1
            if not all the k-nearest patches are not be aggregated:
                1. aggregate them with present label,
                2. find the farthest patch (the measure is the distance between patches' centers)
                jump to the next iteration in iter2


    """
    n, p = data.shape
    new_data, new_labels = create_new_dataset(data, num_intervals)
    model = Model4Sparse(new_data, new_labels, cost, gamma)
    kmeans = cluster.KMeans(n_clusters=patch_num, init='k-means++').fit(data)
    patch_labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    patches = create_patches_from_data(patch_labels, centers, data)
    recover_label = aggregate_patches(patches, k, m, model)
    return recover_label



##### test tool
# a testing sample
import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

# Generate a circular dataset
n_samples = 500
data, labels = make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=42)

# Visualize the circular dataset
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
plt.title('Circular Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Define the number of intervals (cubes) for each feature
num_intervals = 20

# Apply the algorithm to create the new dataset with formal and pseudo points
new_data, new_labels = create_new_dataset(data, num_intervals)
print(new_labels.shape)

# Visualize the new dataset
plt.scatter(new_data[:, 0], new_data[:, 1], c=new_labels, cmap='viridis')
plt.title('New Dataset with Formal and Pseudo Points')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

kmeans = cluster.KMeans(n_clusters=15, init='k-means++').fit(data)
patch_label = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(data[:, 0], data[:, 1], c=patch_label, cmap='viridis')
plt.title('Result of patches\' label')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
