from numba import njit, from_dtype
import numpy as np

tree_type_signature = [
    ('parent', np.int64),
    ('size', np.int64),
    ('core', np.bool_)
]

tree_type = from_dtype(np.dtype(tree_type_signature))

@njit()
def make_graph(n):
    G = np.empty((n,), dtype=tree_type)
    for i in range(n):
        node = G[i]
        node.parent = i
        node.size = 1
        node.core = False
    return G

@njit()
def find_parent(G, i):
    node = G[i]
    if node.parent != i:
        node.parent = find_parent(G, node.parent)
    return node.parent

@njit()
def merge(G, i, j):
    i = find_parent(G, i)
    j = find_parent(G, j)
    if i != j:
        i_node = G[i]
        j_node = G[j]
        si = i_node.size
        sj = j_node.size
        if si < sj:
            # make j the parent of i
            i_node.parent = j
            i_node.size = 0
            j_node.size = si + sj
        else:
            # make i the parent of j
            j_node.parent = i
            j_node.size = 0
            i_node.size = si + sj
    return None

@njit()
def merge_core(G, i, j):
    i = find_parent(G, i)
    j = find_parent(G, j)
    if i != j:
        i_node = G[i]
        j_node = G[j]
        si = i_node.size
        sj = j_node.size
        if i_node.core:
            if j_node.core:
                if si < sj:
                    # make j the parent of i
                    i_node.parent = j
                    i_node.size = 0
                    j_node.size = si + sj
                else:
                    # make i the parent of j
                    j_node.parent = i
                    j_node.size = 0
                    i_node.size = si + sj
            else:
                # make i the parent of j
                j_node.parent = i
                j_node.size = 0
                i_node.size = si + sj
        else:
            if j_node.core:
                # make j the parent of i
                i_node.parent = j
                i_node.size = 0
                j_node.size = si + sj
            else:
                if si < sj:
                    # make j the parent of i
                    i_node.parent = j
                    i_node.size = 0
                    j_node.size = si + sj
                else:
                    # make i the parent of j
                    j_node.parent = i
                    j_node.size = 0
                    i_node.size = si + sj
    return None

@njit()
def components_array(G, minimum_size=2):
    out = dict()
    class_num = 1
    out_array = np.empty((G.shape[0],), dtype=np.int64)
    for i in range(G.shape[0]):
        p = find_parent(G, i)
        if G[p].size >= minimum_size:
            if p not in out:
                out[p] = class_num
                out_array[i] = class_num
                class_num += 1
            else:
                out_array[i] = out[p]
        else:
            out_array[i] = 0
    return out_array
