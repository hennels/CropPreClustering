from numba import njit, prange
import numpy as np
from scipy.spatial import cKDTree as KDTree
from croppreclustering import graph
from croppreclustering import kdtree2D
from croppreclustering import kdtree3D

@njit(parallel=True)
def nRAIN(X, limit, leafsize=16, minimum_size=2):
    parents = np.empty((X.shape[0],), dtype=np.int64)
    tree_indexs, tree_array = kdtree3D.make_tree(X, leafsize=leafsize)
    limit2 = limit*limit
    kdtree3D.set_tree_lowest(X, tree_indexs, tree_array)
    for i in prange(X.shape[0]):
        parents[i] = kdtree3D.lowest_near(X, tree_indexs, tree_array, i, limit=limit2)
    G = graph.make_graph(X.shape[0])
    for i in range(X.shape[0]):
        graph.merge(G, i, parents[i])
    return graph.components_array(G, minimum_size)

@njit(parallel=True)
def negativeZquickshift(X, limit, leafsize=16, minimum_size=2):
    density = np.empty((X.shape[0],), dtype=np.float64)
    parents = np.empty((X.shape[0],), dtype=np.int64)
    tree_indexs, tree_array = kdtree3D.make_tree(X, leafsize=leafsize)
    for i in prange(X.shape[0]):
        density[i] = -1*X[i, 2]
    kdtree3D.set_tree_density(tree_indexs, tree_array, density)
    G = graph.make_graph(X.shape[0])
    limit2 = limit*limit
    for i in prange(X.shape[0]):
        parent_index, _ = kdtree3D.nearest_greater_density(X, tree_indexs, tree_array,
                                                  density, i, limit=limit2)
        if parent_index >= 0:
            parents[i] = parent_index
        else:
            parents[i] = i
    for i in range(X.shape[0]):
        graph.merge(G, i, parents[i])
    return graph.components_array(G, minimum_size)

@njit(parallel=True)
def _GDquickshift(X, k, limit, distance, leafsize=16, minimum_size=2):
    density = np.empty((X.shape[0],), dtype=np.float64)
    parents = np.empty((X.shape[0],), dtype=np.int64)
    tree_indexs, tree_array = kdtree3D.make_tree(X[:, :2], leafsize=leafsize)
    nV = X.shape[0]*np.pi
    for i in prange(X.shape[0]):
        density[i] = k/(nV*distance[i]*distance[i])
    print("Density done")
    kdtree2D.set_tree_density(tree_indexs, tree_array, density)
    G = graph.make_graph(X.shape[0])
    limit2 = limit*limit
    for i in prange(X.shape[0]):
        parents[i], _ = kdtree2D.nearest_greater_density(X, tree_indexs, tree_array,
                                                  density, i, limit=limit2)
    print("Parents found")
    for i in range(X.shape[0]):
        if parents[i] > -1:
            graph.merge(G, i, parents[i])
    return graph.components_array(G, minimum_size)

def GDquickshift(X, k, limit, leafsize=16, minimum_size=2):
    kdt = KDTree(X[:, :2])
    distance, _ = kdt.query(X[:, :2], k=[k], n_jobs=-1)
    return _GDquickshift(X, k, limit, distance[:, 0], leafsize=leafsize, minimum_size=minimum_size)

@njit(parallel=True)
def MCores2(X, beta, k, distance, leafsize=16):
    tree_indexs, tree_array = kdtree2D.make_tree(X, leafsize=leafsize)
    density = np.empty((X.shape[0],), dtype=np.float64)
    distance2 = np.empty((X.shape[0],), dtype=np.float64)
    nV = X.shape[0]*np.pi
    for i in prange(X.shape[0]):
        distance2[i] = distance[i]*distance[i]
        density[i] = k/(nV*distance2[i])
    print("density done")
    order = np.argsort(density)
    kdtree2D.set_tree_density(tree_indexs, tree_array, density)
    out_graph = graph.make_graph(X.shape[0])
    lambda_graph = graph.make_graph(X.shape[0])
    rev_len = order.shape[0] - 1
    one_minus_beta = 1.0 - beta
    count = 0
    included = 0
    for i in order[::-1]:
        if lambda_graph[i].core:
            continue
        else:
            density_min = one_minus_beta*density[i]
            # add to graph
            while included < order.shape[0]:
                to_include = order[rev_len - included]
                if density[to_include] < density_min:
                    break
                else:
                    inds, dists = kdtree2D.query_radius2_greater_density(X, tree_indexs, tree_array, density,
                                                                         X[to_include, :], distance2[to_include], density_min)
                    for k, dist in zip(inds, dists):
                        if dist < distance2[k]:
                            graph.merge_core(lambda_graph, to_include, k)
                    included += 1
            # extract component
            component_parent = graph.find_parent(lambda_graph, i)
            if not lambda_graph[component_parent].core:
                for k in range(X.shape[0]):
                    if component_parent == graph.find_parent(lambda_graph, k):
                        lambda_graph[k].core = True
                        out_graph[k].core = True
                        graph.merge(out_graph, i, k)
                count += 1
                print(count, included/order.shape[0])
    return out_graph, density

@njit(parallel=True)
def quickshiftpp3(X, density, G, leafsize=16, minimum_size=2):
    tree_indexs, tree_array = kdtree3D.make_tree(X, leafsize=leafsize)
    kdtree3D.set_tree_density(tree_indexs, tree_array, density)
    density_parent = np.empty((G.shape[0],), dtype=np.int64)
    for i in prange(X.shape[0]):
        if not G[i].core:
            density_parent[i], _ = kdtree3D.nearest_greater_density(X, tree_indexs, tree_array, density, i, limit=np.inf)
    for i in range(X.shape[0]):
        if not G[i].core:
            graph.merge(G, i, density_parent[i])
    return graph.components_array(G, minimum_size)

def GDquickshiftpp(X, beta, k, leafsize=16, minimum_size=2):
    kdt = KDTree(X[:, :2])
    distance, _ = kdt.query(X[:, :2], k=[k], n_jobs=-1)
    print("MCores")
    G, density = MCores2(X[:, :2], beta, k, distance[:, 0], leafsize=leafsize)
    print("Quickshift++")
    return quickshiftpp3(X, density, G, leafsize=leafsize, minimum_size=minimum_size)