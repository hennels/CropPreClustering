import argparse
import open3d as o3d
import numpy as np
from croppreclustering.paperalgs import GDquickshift

parser = argparse.ArgumentParser(description='Perform Ground Density Quickshift on a point cloud.')
parser.add_argument('cloud', help='Path to point cloud.')
parser.add_argument('k', type=int, help='Kth neighbor distance (density parameter). Must be integer > 1.')
parser.add_argument('d', type=float, help='Neighborhood distance threshold.')
parser.add_argument('--min_size', type=int, default=1, help='Only show clusters larger or equal to this size.')
parser.add_argument('--leaf_size', type=int, default=16, help='Number of points in kdtree leaf nodes.')
parser.add_argument('--save', type=str, default=None, help='Path to save result point cloud.')
args = parser.parse_args()

pcd = o3d.io.read_point_cloud(args.cloud)
X = np.asarray(pcd.points)

classes = GDquickshift(X, args.k, args.d, leafsize=args.leaf_size, minimum_size=args.min_size)

n = classes.max()
print("GD Quick-Shift: {}".format(n))
colors = np.random.rand(np.int64(n)+1, 3)
colors[0, :] = 0.0
pcd.colors = o3d.utility.Vector3dVector(colors[classes, :])
if args.save is not None:
    o3d.io.write_point_cloud(args.save, pcd)
o3d.visualization.draw_geometries([pcd])