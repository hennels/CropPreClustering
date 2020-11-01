import argparse
import open3d as o3d
import numpy as np
from scalablecroppreclustering.paperalgs import GDquickshiftpp

parser = argparse.ArgumentParser(description='Perform Ground Density Quickshift++ on a point cloud.')
parser.add_argument('cloud', help='Path to point cloud.')
parser.add_argument('k', type=int, help='Kth neighbor distance (density parameter). Must be integer > 1.')
parser.add_argument('beta', type=float, help='Density multiplier for finding dense regions. Domain: [0.0, 1.0]')
parser.add_argument('--min_size', type=int, default=1, help='Only show clusters larger or equal to this size.')
parser.add_argument('--leaf_size', type=int, default=16, help='Number of points in kdtree leaf nodes.')
args = parser.parse_args()

pcd = o3d.io.read_point_cloud(args.cloud)
X = np.asarray(pcd.points)

classes = GDquickshiftpp(X, args.beta, args.k, leafsize=args.leaf_size, minimum_size=args.min_size)

colors = np.random.rand(np.int64(classes.max())+1, 3)
colors[0, :] = 0.0
pcd.colors = o3d.utility.Vector3dVector(colors[classes, :])
o3d.visualization.draw_geometries([pcd])