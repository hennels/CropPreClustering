import argparse
import open3d as o3d
from numba import njit, typed
import numpy as np
from scipy.optimize import linear_sum_assignment

@njit()
def iou_matrix(labels1, labels2):
    assert labels1.shape[0] == labels2.shape[0]
    all_sets = typed.List()
    member_sets1 = {}
    member_sets2 = {}
    for i in range(labels1.shape[0]):
        v = labels1[i]
        if v not in member_sets1:
            new_set = set()
            new_set.add(i)
            j = len(all_sets)
            member_sets1[v] = j
            all_sets.append(new_set)
        else:
            all_sets[member_sets1[v]].add(i)
        v = labels2[i]
        if v not in member_sets2:
            new_set = set()
            new_set.add(i)
            j = len(all_sets)
            member_sets2[v] = j
            all_sets.append(new_set)
        else:
            all_sets[member_sets2[v]].add(i)
    out = np.empty((len(member_sets1), len(member_sets2)), dtype=np.float64)
    for i, v1 in enumerate(member_sets1.values()):
        set1 = all_sets[v1]
        size_v1 = len(set1)
        for j, v2 in enumerate(member_sets2.values()):
            set2 = all_sets[v2]
            size_v2 = len(set2)
            intersection = len(set1.intersection(set2))
            out[i, j] = intersection/(size_v1 + size_v2 - intersection)
    return out

parser = argparse.ArgumentParser(description='Compare clusterings of a point cloud.')
parser.add_argument('cloud1', help='Path to point cloud1.')
parser.add_argument('cloud2', help='Path to point cloud2.')
args = parser.parse_args()

# load clouds
cloud1 = o3d.io.read_point_cloud(args.cloud1)
cloud2 = o3d.io.read_point_cloud(args.cloud2)

colors1 = np.asarray(cloud1.colors)
colors2 = np.asarray(cloud2.colors)

print("Getting labels")
u1, labels1 = np.unique(colors1, return_inverse=True, axis=0)
u2, labels2 = np.unique(colors2, return_inverse=True, axis=0)
print("Number of clusters:", u1.shape[0], u2.shape[0])

print("Making matrix")
dmat = iou_matrix(labels1, labels2)

print("Finding matching")
row_ind, col_ind = linear_sum_assignment(dmat, maximize=True)
ious = dmat[row_ind, col_ind].copy()

print("Mean:", ious.mean())
print("Median:", np.median(ious))
