import argparse
import open3d as o3d
import numpy as np
from croppreclustering.paperalgs import nRAIN

parser = argparse.ArgumentParser(description='Perform non-random RAIN on a point cloud.')
parser.add_argument('cloud', help='Path to point cloud.')
parser.add_argument('d', type=float, help='Neighborhood distance threshold.')
parser.add_argument('--min_size', type=int, default=1, help='Only show clusters larger or equal to this size.')
parser.add_argument('--leaf_size', type=int, default=16, help='Number of points in kdtree leaf nodes.')
args = parser.parse_args()

pcd = o3d.io.read_point_cloud(args.cloud)
X = np.asarray(pcd.points)

classes = nRAIN(X, args.d, leafsize=args.leaf_size, minimum_size=args.min_size)
n = classes.max()
print("nRAIN: {}".format(n))
colors = np.random.rand(np.int64(n)+1, 3)
colors[0, :] = 0.0
pcd.colors = o3d.utility.Vector3dVector(colors[classes, :])
o3d.visualization.draw_geometries([pcd])