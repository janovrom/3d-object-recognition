import time
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_binvox_np(pointcloud, labels, grid_size=32, visualize=False):
    num_points = int(pointcloud.shape[0])
    # Find point cloud min and max
    min_x = np.min(pointcloud[0::3])
    min_y = np.min(pointcloud[1::3])
    min_z = np.min(pointcloud[2::3])
    max_x = np.max(pointcloud[0::3])
    max_y = np.max(pointcloud[1::3])
    max_z = np.max(pointcloud[2::3])
    # Compute sizes 
    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z
    
    max_size = np.max([size_x, size_y, size_z])
    # print("%s has size=(%f, %f, %f) meters\n" % (os.path.basename(filename), size_x, size_y, size_z))
    occupancy_grid = np.array(np.zeros((grid_size,grid_size,grid_size)), dtype=np.float32)
    label_grid = np.array(np.zeros((grid_size,grid_size,grid_size)), dtype=np.int32)
    hist_grid = np.array(np.zeros((grid_size,grid_size,grid_size)), dtype=np.object)
    max_size = np.max([size_x, size_y, size_z]) / 2
    # print("%s has size=(%f, %f, %f) meters\n" % (os.path.basename(filename), size_x, size_y, size_z))
    cx = size_x / 2 + min_x
    cy = size_y / 2 + min_y
    cz = size_z / 2 + min_z
    extent = int(grid_size / 2)

    for i in range(0,num_points,3):
        x = pointcloud[i+0]
        y = pointcloud[i+1]
        z = pointcloud[i+2]
        # idx_x = int(((x - cx) * extent / size_x * 2.0 + extent) * (grid_size - 1) / (extent * 2))
        # idx_y = int(((y - cy) * extent / size_y * 2.0 + extent) * (grid_size - 1) / (extent * 2))
        # idx_z = int(((z - cz) * extent / size_z * 2.0 + extent) * (grid_size - 1) / (extent * 2))
        idx_x = int(((x - cx) * extent / max_size + extent) * (grid_size - 1) / (extent * 2))
        idx_y = int(((y - cy) * extent / max_size + extent) * (grid_size - 1) / (extent * 2))
        idx_z = int(((z - cz) * extent / max_size + extent) * (grid_size - 1) / (extent * 2))
        occupancy_grid[idx_x, idx_y, idx_z] = 1
        if not hist_grid[idx_x, idx_y, idx_z]:
            hist_grid[idx_x, idx_y, idx_z] = []

        hist_grid[idx_x, idx_y, idx_z].append(labels[int(i/3)])

    if visualize:
        px = []
        py = []
        pz = []



    for x in range(0, grid_size):
        for y in range(0, grid_size):
            for z in range(0, grid_size):
                if hist_grid[x, y, z]:
                    label_grid[x, y, z] = np.argmax(np.bincount(np.array(hist_grid[x, y, z])))
                    if visualize:
                        px.append(idx_x)
                        py.append(idx_y)
                        pz.append(idx_z)

    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        m = plt.get_cmap("Set1")
        ax.scatter(px, py, pz, cmap=m, marker='p')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    # vol_occ = np.copy(occupancy_grid)
    # for x in range(0, grid_size):
    #     for z in range(0, grid_size):
    #         y = 0
    #         val = 0.25
    #         while y < grid_size:
    #             if occupancy_grid[x,y,z] == 1:
    #                 val *= -1
    #             else:
    #                 vol_occ[x,y,z] += val

    #             y += 1

    # for x in range(0, grid_size):
    #     for y in range(0, grid_size):
    #         z = 0
    #         val = 0.25
    #         while z < grid_size:
    #             if occupancy_grid[x,y,z] == 1:
    #                 val *= -1
    #             else:
    #                 vol_occ[x,y,z] += val
                    
    #             z += 1

    # for y in range(0, grid_size):
    #     for z in range(0, grid_size):
    #         x = 0
    #         val = 0.25
    #         while x < grid_size:
    #             if occupancy_grid[x,y,z] == 1:
    #                 val *= -1
    #             else:
    #                 vol_occ[x,y,z] += val
                    
    #             x += 1

    vol_occ = np.copy(occupancy_grid)
    # lspace = np.arange(-15.5,16,1)
    # xx,yy,zz = np.meshgrid(lspace,lspace,lspace)
    # vol_occ = np.sqrt(np.square(xx) + np.square(yy) + np.square(zz)) / np.sqrt(3)
    # vol_occ[occupancy_grid == 1] = 1

    return occupancy_grid, label_grid, 0, pointcloud, labels, vol_occ


def load_binvox(pts, sgs, grid_size=32, ogrid_size=32, label_start=0, visualize=False):
    points = []
    labels = []
    with open(pts, "r") as pts_f, open(sgs, "rb") as sgs_f:
        lines = pts_f.readlines()

        for line in lines:
            splitted = line.split(" ")
            points.append(float(splitted[0]))
            points.append(float(splitted[1]))
            points.append(float(splitted[2]))
            labels.append(int(sgs_f.read(1))-1) # indexing starts from 1 in the file
            sgs_f.read(1) # throw out newline

    pointcloud = np.array(points)
    labels = np.array(labels)
    label_count = np.max(labels) + 1
    labels = labels + label_start
    num_points = int(pointcloud.shape[0])

    # Find point cloud min and max
    min_x = np.min(pointcloud[0::3])
    min_y = np.min(pointcloud[1::3])
    min_z = np.min(pointcloud[2::3])
    max_x = np.max(pointcloud[0::3])
    max_y = np.max(pointcloud[1::3])
    max_z = np.max(pointcloud[2::3])
    # Compute sizes 
    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z
    
    max_size = np.max([size_x, size_y, size_z])
    min_size = min(min_x, min(min_y, min_z))
    # print("%s has size=(%f, %f, %f) meters\n" % (os.path.basename(filename), size_x, size_y, size_z))
    occupancy_grid = np.array(np.zeros((grid_size,grid_size,grid_size)), dtype=np.float32)
    label_grid = np.array(np.zeros((ogrid_size,ogrid_size,ogrid_size)), dtype=np.int32)
    hist_grid = np.array(np.zeros((ogrid_size,ogrid_size,ogrid_size)), dtype=np.object)
    max_size = np.max([size_x, size_y, size_z]) / 2
    # print("%s has size=(%f, %f, %f) meters\n" % (os.path.basename(filename), size_x, size_y, size_z))
    cx = size_x / 2 + min_x
    cy = size_y / 2 + min_y
    cz = size_z / 2 + min_z
    extent = int(grid_size / 2)

    if visualize:
        px = []
        py = []
        pz = []
        pv = []
        px.append(0)
        py.append(0)
        pz.append(0)
        px.append(0)
        py.append(0)
        pz.append(0)
        pv.append(0)
        pv.append(8)

    for i in range(0,num_points,3):
        x = pointcloud[i+0]
        y = pointcloud[i+1]
        z = pointcloud[i+2]
        idx_x = int(((x - cx) * extent / max_size + extent) * (grid_size - 1) / (extent * 2))
        idx_y = int(((y - cy) * extent / max_size + extent) * (grid_size - 1) / (extent * 2))
        idx_z = int(((z - cz) * extent / max_size + extent) * (grid_size - 1) / (extent * 2))
        occupancy_grid[idx_x, idx_y, idx_z] = 1

        idx_x = int(((x - cx) * extent / max_size + extent) * (ogrid_size - 1) / (extent * 2))
        idx_y = int(((y - cy) * extent / max_size + extent) * (ogrid_size - 1) / (extent * 2))
        idx_z = int(((z - cz) * extent / max_size + extent) * (ogrid_size - 1) / (extent * 2))
        if not hist_grid[idx_x, idx_y, idx_z]:
            hist_grid[idx_x, idx_y, idx_z] = []

        hist_grid[idx_x, idx_y, idx_z].append(labels[int(i/3)])

        if visualize:
            px.append(x)
            py.append(y)
            pz.append(z)
            pv.append(labels[int(i/3)] - label_start)

            

    if visualize:
        xs = []
        ys = []
        zs = []
        vs = []
        xs.append(0)
        ys.append(0)
        zs.append(0)
        xs.append(0)
        ys.append(0)
        zs.append(0)
        vs.append(0)
        vs.append(8)


    for x in range(0, ogrid_size):
        for y in range(0, ogrid_size):
            for z in range(0, ogrid_size):
                if hist_grid[x, y, z]:
                    label_grid[x, y, z] = np.argmax(np.bincount(np.array(hist_grid[x, y, z])))
                    if visualize:
                        xs.append(x)
                        ys.append(y)
                        zs.append(z)
                        vs.append(label_grid[x, y, z] - label_start)


    # viz
    if visualize:
        fig = plt.figure()
        # plt.axis("scaled")
        ax = fig.add_subplot(111, projection='3d')
        m = plt.get_cmap("Set1")
        # plt.gca().set_aspect('scaled', adjustable='box')
        # pv = np.array(pv)
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.axis('off')
        # plt.gca().set_aspect('equal', adjustable='box')
        # ax.set_xlim(min_size,max_size)
        # ax.set_ylim(min_size,max_size)
        # ax.set_zlim(min_size,max_size)
        # ax.scatter(px, py, pz, c=pv, cmap=m, marker='p')

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.scatter(xs, ys, zs, c=vs, cmap=m, marker='p')
        ax2.set_xlim(0,ogrid_size-1)
        ax2.set_ylim(0,ogrid_size-1)
        ax2.set_zlim(0,ogrid_size-1)
        ax2.set_xlabel('X Label')
        ax2.set_ylabel('Y Label')
        ax2.set_zlabel('Z Label')
        plt.show()

    # vol_occ = np.copy(occupancy_grid)
    # for x in range(0, grid_size):
    #     for z in range(0, grid_size):
    #         y = 0
    #         val = 0.25
    #         while y < grid_size:
    #             if occupancy_grid[x,y,z] == 1:
    #                 val *= -1
    #             else:
    #                 vol_occ[x,y,z] += val

    #             y += 1

    # for x in range(0, grid_size):
    #     for y in range(0, grid_size):
    #         z = 0
    #         val = 0.25
    #         while z < grid_size:
    #             if occupancy_grid[x,y,z] == 1:
    #                 val *= -1
    #             else:
    #                 vol_occ[x,y,z] += val
                    
    #             z += 1

    # for y in range(0, grid_size):
    #     for z in range(0, grid_size):
    #         x = 0
    #         val = 0.25
    #         while x < grid_size:
    #             if occupancy_grid[x,y,z] == 1:
    #                 val *= -1
    #             else:
    #                 vol_occ[x,y,z] += val
                    
    #             x += 1

    vol_occ = np.copy(occupancy_grid)
    # lspace = np.arange(-15.5,16,1)
    # xx,yy,zz = np.meshgrid(lspace,lspace,lspace)
    # vol_occ = np.sqrt(np.square(xx) + np.square(yy) + np.square(zz)) / np.sqrt(3)
    # vol_occ[occupancy_grid == 1] = 1

    return occupancy_grid, label_grid, label_count, pointcloud, labels, vol_occ
