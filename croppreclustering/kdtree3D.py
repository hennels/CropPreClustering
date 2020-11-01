from numba import from_dtype, njit
import numpy as np

tree_type_signature = [
    ('leaf', np.bool_),
    ('start', np.int64),
    ('stop', np.int64),
    ('x_min', np.float64),
    ('x_max', np.float64),
    ('y_min', np.float64),
    ('y_max', np.float64),
    ('z_min', np.float64),
    ('z_max', np.float64),
    ('max_density', np.float64)
]

tree_type = from_dtype(np.dtype(tree_type_signature))

@njit()
def squared_distance(x, y):
    d0 = x[0] - y[0]
    d1 = x[1] - y[1]
    d2 = x[2] - y[2]
    return d0*d0 + d1*d1 + d2*d2

@njit()
def split_dim(data, indexs, begin, end):
    i = indexs[begin]
    x_min, x_max = data[i, 0], data[i, 0]
    y_min, y_max = data[i, 1], data[i, 1]
    z_min, z_max = data[i, 2], data[i, 2]
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
        if data[i, 2] < z_min:
            z_min = data[i, 2]
        else:
            if z_max < data[i, 2]:
                z_max = data[i, 2]
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    if x_range > y_range:
        if x_range > z_range:
            return 0
        else:
            return 2
    else:
        if y_range > z_range:
            return 1
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
        if left_tree.z_min < right_tree.z_min:
            tree_node.z_min = left_tree.z_min
        else:
            tree_node.z_min = right_tree.z_min
        if left_tree.z_max > right_tree.z_max:
            tree_node.z_max = left_tree.z_max
        else:
            tree_node.z_max = right_tree.z_max
    else:
        tree_node.leaf = True
        index = indexs[begin]
        x = data[index, 0]
        y = data[index, 1]
        z = data[index, 2]
        tree_node.x_min = x
        tree_node.x_max = x
        tree_node.y_min = y
        tree_node.y_max = y
        tree_node.z_min = z
        tree_node.z_max = z
        for index in indexs[begin+1:end]:
            x = data[index, 0]
            y = data[index, 1]
            z = data[index, 2]
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
            if z < tree_node.z_min:
                tree_node.z_min = z
            else:
                if tree_node.z_max < z:
                    tree_node.z_max = z
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
def _populate_lowest_kdtree(lowness, indexs, tree_array, i):
    tree_node = tree_array[i]
    if tree_node.leaf:
        current_lowest = lowness[indexs[tree_node.start]]
        for i in range(tree_node.start+1, tree_node.stop):
            value = lowness[indexs[i]]
            if value < current_lowest:
                current_lowest = value
        tree_node.max_density = current_lowest
    else:
        left_index = 2*i + 1
        right_index = left_index + 1
        _populate_lowest_kdtree(lowness, indexs, tree_array, left_index)
        _populate_lowest_kdtree(lowness, indexs, tree_array, right_index)
        left_tree = tree_array[left_index]
        right_tree = tree_array[right_index]
        if left_tree.max_density < right_tree.max_density:
            tree_node.max_density = left_tree.max_density
        else:
            tree_node.max_density = right_tree.max_density
    return None

@njit()
def set_tree_lowest(data, indexs, tree_array):
    _populate_lowest_kdtree(data[:, 2], indexs, tree_array, 0)
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
    if point[2] < tree_node.z_min:
        tmp = tree_node.z_min - point[2]
        dist += tmp*tmp
    else:
        if tree_node.z_max < point[2]:
            tmp = point[2] - tree_node.z_max
            dist += tmp*tmp
    return dist

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

@njit()
def _lowest_near(data, indexs, tree_array, limit,
                   point, i,
                   out_index):
    tree_node = tree_array[i]
    if tree_node.leaf:
        for i in range(tree_node.start, tree_node.stop):
            index = indexs[i]
            if data[index, 2] < data[out_index, 2]:
                if squared_distance(point, data[index, :]) < limit:
                    out_index = index
    else:
        left_index = 2*i + 1
        right_index = left_index + 1
        left_tree = tree_array[left_index]
        right_tree = tree_array[right_index]
        if left_tree.max_density < data[out_index,2] and dist_sq_nearest(left_tree, point) < limit:
            out_index = _lowest_near(data, indexs, tree_array, limit,
                                       point, left_index,
                                       out_index)
        if right_tree.max_density < data[out_index,2] and dist_sq_nearest(right_tree, point) < limit:
            out_index = _lowest_near(data, indexs, tree_array, limit,
                                       point, right_index,
                                       out_index)
    return out_index

@njit()
def lowest_near(data, indexs, tree_array, query_index, limit=np.inf):
    out_index = _lowest_near(data, indexs, tree_array, limit,
                             data[query_index, :], 0,
                             np.int64(query_index))
    return out_index