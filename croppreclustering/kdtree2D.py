from numba import njit, from_dtype, int64, float64, prange
from numba.typed import List
import numpy as np

tree_type_signature = [
    ('leaf', np.bool_),
    ('start', np.int64),
    ('stop', np.int64),
    ('x_min', np.float64),
    ('x_max', np.float64),
    ('y_min', np.float64),
    ('y_max', np.float64),
    ('max_density', np.float64)
]

tree_type = from_dtype(np.dtype(tree_type_signature))

@njit()
def squared_distance(x, y):
    d0 = x[0] - y[0]
    d1 = x[1] - y[1]
    return d0*d0 + d1*d1

@njit()
def split_dim(data, indexs, begin, end):
    i = indexs[begin]
    x_min, x_max = data[i, 0], data[i, 0]
    y_min, y_max = data[i, 1], data[i, 1]
    for i in indexs[begin+1:end]:
        if data[i, 0] < x_min:
            x_min = data[i, 0]
        else:
            if x_max < data[i, 0]:
                x_max = data[i, 0]
        if data[i, 1] < y_min:
            y_min = data[i, 1]
        else:
            if y_max < data[i, 1]:
                y_max = data[i, 1]
    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range > y_range:
        return 0
    else:
        return 1

@njit()
def argpartition(values, indexs, value, begin, end):
    swap_index = begin
    for i in range(begin, end):
        checking = indexs[i]
        if values[checking] < value:
            indexs[i] = indexs[swap_index]
            indexs[swap_index] = checking 
            swap_index += 1
    return swap_index

@njit()
def _initialize_kdtree(data, indexs, tree_array,
                       begin, end, leafsize, i):
    tree_node = tree_array[i]
    tree_node.start = begin
    tree_node.stop = end
    tree_node.max_density = 0.0
    if end - begin > leafsize:
        tree_node.leaf = False
        left_index = 2*i + 1
        right_index = left_index + 1
        dim = split_dim(data, indexs, begin, end)
        split = argpartition(data[:, dim], indexs, np.median(data[indexs[begin:end], dim]), begin, end)
        if split == begin or split == end:
            print("bad split: performance will suffer")
            split = begin + (end - begin)//2
        _initialize_kdtree(data, indexs, tree_array, begin, split, leafsize, left_index)
        _initialize_kdtree(data, indexs, tree_array, split, end,   leafsize, right_index)
        left_tree = tree_array[left_index]
        right_tree = tree_array[right_index]
        if left_tree.x_min < right_tree.x_min:
            tree_node.x_min = left_tree.x_min
        else:
            tree_node.x_min = right_tree.x_min
        if left_tree.x_max > right_tree.x_max:
            tree_node.x_max = left_tree.x_max
        else:
            tree_node.x_max = right_tree.x_max
        if left_tree.y_min < right_tree.y_min:
            tree_node.y_min = left_tree.y_min
        else:
            tree_node.y_min = right_tree.y_min
        if left_tree.y_max > right_tree.y_max:
            tree_node.y_max = left_tree.y_max
        else:
            tree_node.y_max = right_tree.y_max
    else:
        tree_node.leaf = True
        index = indexs[begin]
        x = data[index, 0]
        y = data[index, 1]
        tree_node.x_min = x
        tree_node.x_max = x
        tree_node.y_min = y
        tree_node.y_max = y
        for index in indexs[begin+1:end]:
            x = data[index, 0]
            y = data[index, 1]
            if x < tree_node.x_min:
                tree_node.x_min = x
            else:
                if tree_node.x_max < x:
                    tree_node.x_max = x
            if y < tree_node.y_min:
                tree_node.y_min = y
            else:
                if tree_node.y_max < y:
                    tree_node.y_max = y
    return None

@njit()
def make_tree(data, leafsize=16):
    indexs = np.arange(data.shape[0], dtype=np.int64)
    tree_array = np.empty((data.shape[0],), dtype=tree_type)
    _initialize_kdtree(data, indexs, tree_array,
                       0, data.shape[0], leafsize, 0)
    return indexs, tree_array

@njit()
def _populate_density_kdtree(indexs, tree_array, density, i):
    tree_node = tree_array[i]
    if tree_node.leaf:
        current_max = density[indexs[tree_node.start]]
        for i in range(tree_node.start+1, tree_node.stop):
            value = density[indexs[i]]
            if value > current_max:
                current_max = value
        tree_node.max_density = current_max
    else:
        left_index = 2*i + 1
        right_index = left_index + 1
        _populate_density_kdtree(indexs, tree_array, density, left_index)
        _populate_density_kdtree(indexs, tree_array, density, right_index)
        left_tree = tree_array[left_index]
        right_tree = tree_array[right_index]
        if left_tree.max_density > right_tree.max_density:
            tree_node.max_density = left_tree.max_density
        else:
            tree_node.max_density = right_tree.max_density
    return None

@njit()
def set_tree_density(indexs, tree_array, density):
    _populate_density_kdtree(indexs, tree_array, density, 0)
    return None

@njit()
def dist_sq_nearest(tree_node, point):
    if point[0] < tree_node.x_min:
        tmp = tree_node.x_min - point[0]
        dist = tmp*tmp
    else:
        if tree_node.x_max < point[0]:
            tmp = point[0] - tree_node.x_max
            dist = tmp*tmp
        else:
            dist = 0.0
    if point[1] < tree_node.y_min:
        tmp = tree_node.y_min - point[1]
        dist += tmp*tmp
    else:
        if tree_node.y_max < point[1]:
            tmp = point[1] - tree_node.y_max
            dist += tmp*tmp
    return dist

@njit()
def _query_radius_greater_density(data, indexs, tree_array, density,
                                  point, radius_sq, density_minimum, i,
                                  out_index, out_dist_sq):
    tree_node = tree_array[i]
    if tree_node.leaf:
        for index in indexs[tree_node.start:tree_node.stop]:
            if density[index] > density_minimum:
                dist = squared_distance(point, data[index, :])
                if dist < radius_sq:
                    out_index.append(index)
                    out_dist_sq.append(dist)
    else:
        left_index = 2*i + 1
        right_index = left_index + 1
        left_tree = tree_array[left_index]
        right_tree = tree_array[right_index]
        if left_tree.max_density > density_minimum and dist_sq_nearest(left_tree, point) < radius_sq:
            _query_radius_greater_density(data, indexs, tree_array, density,
                                          point, radius_sq, density_minimum, left_index,
                                          out_index, out_dist_sq)
        if right_tree.max_density > density_minimum and dist_sq_nearest(right_tree, point) < radius_sq:
            _query_radius_greater_density(data, indexs, tree_array, density,
                                          point, radius_sq, density_minimum, right_index,
                                          out_index, out_dist_sq)
    return None

heap_type_signature = [
    ('index', np.int64),
    ('distance', np.float64)
]

heap_type = from_dtype(np.dtype(heap_type_signature))

@njit()
def _k_nearest(data, indexs, tree_array,
               point, i,
               heap):
    tree_node = tree_array[i]
    if tree_node.leaf:
        for i in range(tree_node.start, tree_node.stop):
            index = indexs[i]
            dist = squared_distance(point, data[index, :])
            if dist < heap[0].distance:
                heap.index[0] = index
                heap.distance[0] = dist
                # heapify down
                current = 0
                while True:
                    left = 2*current + 1
                    if left < heap.shape[0]:
                        right = left + 1
                        if right < heap.shape[0]:
                            if heap[left].distance < heap[right].distance:
                                larger = right
                            else:
                                larger = left
                        else:
                            larger = left
                        if heap[current].distance < heap[larger].distance:
                            heap[current], heap[larger] = heap[larger], heap[current]
                            current = larger
                        else:
                            break
                    else:
                        break         
    else:
        left_index = 2*i + 1
        right_index = left_index + 1
        left_dist = dist_sq_nearest(tree_array[left_index], point)
        right_dist = dist_sq_nearest(tree_array[right_index], point)
        if left_dist < right_dist:
            if left_dist < heap[0].distance:
                _k_nearest(data, indexs, tree_array,
                           point, left_index,
                           heap)
                if right_dist < heap[0].distance:
                    _k_nearest(data, indexs, tree_array,
                               point, right_index,
                               heap)
        else:
            if right_dist < heap[0].distance:
                _k_nearest(data, indexs, tree_array,
                           point, right_index,
                           heap)
                if left_dist < heap[0].distance:
                    _k_nearest(data, indexs, tree_array,
                               point, left_index,
                               heap)
    return None

@njit()
def query_radius2_greater_density(data, indexs, tree_array, density,
                                  point, radius2, density_minimum):
    out_index = List.empty_list(int64)
    out_dist_sq = List.empty_list(float64)
    _query_radius_greater_density(data, indexs, tree_array, density,
                                  point, radius2, density_minimum, 0,
                                  out_index, out_dist_sq)
    return out_index, out_dist_sq

@njit(parallel=True)
def query_kth_nearest_all(data, indexs, tree_array, k):
    kth_index = np.empty((data.shape[0],), dtype=np.int64)
    kth_dist2 = np.empty((data.shape[0],), dtype=np.float64)
    for i in prange(data.shape[0]):
        heap = np.empty((k,), dtype=heap_type)
        for j in range(k):
            heap_node = heap[j]
            heap_node.distance = np.inf
        _k_nearest(data, indexs, tree_array, data[i, :], 0, heap)
        kth_index[i] = heap[0].index
        kth_dist2[i] = heap[0].distance
    return kth_index, kth_dist2

@njit()
def _nearest_greater_density(data, indexs, tree_array, density,
                             point, density_minimum, i,
                             out_index, out_dist_sq):
    tree_node = tree_array[i]
    if tree_node.leaf:
        for i in range(tree_node.start, tree_node.stop):
            index = indexs[i]
            if density[index] > density_minimum:
                dist = squared_distance(point, data[index, :])
                if dist < out_dist_sq:
                    out_index = index
                    out_dist_sq = dist
    else:
        left_index = 2*i + 1
        right_index = left_index + 1
        left_tree = tree_array[left_index]
        right_tree = tree_array[right_index]
        if left_tree.max_density > density_minimum:
            if right_tree.max_density > density_minimum:
                # check both
                left_dist = dist_sq_nearest(left_tree, point)
                right_dist = dist_sq_nearest(right_tree, point)
                if left_dist < right_dist:
                    if left_dist < out_dist_sq:
                        out_index, out_dist_sq = _nearest_greater_density(data, indexs, tree_array, density,
                                                                          point, density_minimum, left_index,
                                                                          out_index, out_dist_sq)
                        if right_dist < out_dist_sq:
                            out_index, out_dist_sq = _nearest_greater_density(data, indexs, tree_array, density,
                                                                              point, density_minimum, right_index,
                                                                              out_index, out_dist_sq)
                else:
                    if right_dist < out_dist_sq:
                        out_index, out_dist_sq = _nearest_greater_density(data, indexs, tree_array, density,
                                                                          point, density_minimum, right_index,
                                                                          out_index, out_dist_sq)
                        if left_dist < out_dist_sq:
                            out_index, out_dist_sq = _nearest_greater_density(data, indexs, tree_array, density,
                                                                              point, density_minimum, left_index,
                                                                              out_index, out_dist_sq)
            else:
                # check left
                if dist_sq_nearest(left_tree, point) < out_dist_sq:
                    out_index, out_dist_sq = _nearest_greater_density(data, indexs, tree_array, density,
                                                                      point, density_minimum, left_index,
                                                                      out_index, out_dist_sq)
        else:
            if right_tree.max_density > density_minimum:
                # check right
                out_index, out_dist_sq = _nearest_greater_density(data, indexs, tree_array, density,
                                                                  point, density_minimum, right_index,
                                                                  out_index, out_dist_sq)
    return out_index, out_dist_sq

@njit()
def nearest_greater_density(data, indexs, tree_array, density, query_index, limit=np.inf):
    out_index, out_dist_sq = _nearest_greater_density(data, indexs, tree_array, density,
                                                      data[query_index, :], density[query_index], 0,
                                                      -1, limit*limit)
    return out_index, out_dist_sq