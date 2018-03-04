import numpy as np
from triangle_test import *
import time


class OctreeNode():

    def __init__(self):
        self.is_leaf = False


    def add_child(self, children):
        self.children = children


    def mark_as_leaf(self, tris):
        self.tris = tris
        self.is_leaf = True



class Octree():

    triplets = []

    def __init__(self, tris, aabb_min, aabb_max):
        self.tree_min = aabb_min
        self.tree_max = aabb_max
        self.root = OctreeNode() # just for root
        self.X = [] # list for occlussion

        Octree.triplets = np.zeros((8,3))
        Octree.triplets[0,:] = [0, 0, 0]
        Octree.triplets[1,:] = [1, 0, 0]
        Octree.triplets[2,:] = [0, 1, 0]
        Octree.triplets[3,:] = [0, 0, 1]
        Octree.triplets[4,:] = [1, 0, 1]
        Octree.triplets[5,:] = [1, 1, 0]
        Octree.triplets[6,:] = [0, 1, 1]
        Octree.triplets[7,:] = [1, 1, 1]

        starttime = time.time()
        self.recursive_tree(self.root, tris, aabb_min, aabb_max, 1)
        print("Tree constructed in: " + str(time.time() - starttime))


    def get_occlussion(self):
        return self.X


    def recursive_tree(self, node, tris, min_e, max_e, level):
        if level == 5 or tris.shape[0] < 2:
            node.mark_as_leaf(tris)
            if tris.shape[0] > 0:
                self.X.append(min_e)
            return


        # sizes of those 8 boxes
        box_size = (max_e - min_e) / 2
        middle = box_size + min_e

        box_half_size = box_size / 2
        start = min_e + box_half_size
        children = {}
        
        for i in range(0, 8):
            children[i] = {}
            children[i]["tris"] = []
            children[i]["node"] = OctreeNode()
            
            children[i]["min_e"] = min_e + box_size * Octree.triplets[i,:]
            children[i]["max_e"] = children[i]["min_e"] + box_size


        for i in range(0, tris.shape[0], 3):
            t = tris[i:i+3,:]
            for j in range(0, 8):
                center = start + box_size * Octree.triplets[j,:]
                if intersects_box(t, center, box_half_size * 1.0001):
                    children[j]["tris"].append(t[0,:])
                    children[j]["tris"].append(t[1,:])
                    children[j]["tris"].append(t[2,:])


        for i in range(0, 8):
            children[i]["tris"] = (np.array(children[i]["tris"]))
            self.recursive_tree(children[i]["node"], children[i]["tris"], children[i]["min_e"], children[i]["max_e"], level+1)

        # add nodes to parent
        node.add_child(children)