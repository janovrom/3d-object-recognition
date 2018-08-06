import time
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_off_file(filename):
    f = open(filename, "r")
    lines = f.readlines()
    assert(lines[0] == "OFF\n")
    splitted = lines[1].split(" ")
    numVertices = int(splitted[0])
    numFaces = int(splitted[1])
    numEdges = int(splitted[2])

    npoints = np.zeros((3, numFaces * 3))
    points = []

    header_size = 2

    for i in range(header_size, numVertices+header_size):
        splitted = lines[i].split(" ")
        points.append(float(splitted[0]))
        points.append(float(splitted[1]))
        points.append(float(splitted[2]))

    faceId = 0
    for i in range(numVertices+header_size, numVertices+header_size+numFaces):
        splitted = lines[i].split(" ") 
        if len(splitted) >= 3: 
            # splitted[0] should always be 3
            assert(int(splitted[0]) == 3)
            idx1 = int(splitted[1])
            idx2 = int(splitted[2])
            idx3 = int(splitted[3])

            npoints[:, (faceId)*3+0] = points[idx1*3:idx1*3+3]
            npoints[:, (faceId)*3+1] = points[idx2*3:idx2*3+3]
            npoints[:, (faceId)*3+2] = points[idx3*3:idx3*3+3]

            faceId += 1

    return npoints


def load_off_file_to_mesh(filename):
    f = open(filename, "r")
    lines = f.readlines()
    print(lines[0])
    assert(lines[0] == "OFF\n")
    splitted = lines[1].split(" ")
    numVertices = int(splitted[0])
    numFaces = int(splitted[1])
    numEdges = int(splitted[2])

    mesh = {}
    points = []
    vertices = []
    indices = []

    header_size = 2

    for i in range(header_size, numVertices+header_size):
        splitted = lines[i].split(" ")
        points.append(float(splitted[0]))
        points.append(float(splitted[1]))
        points.append(float(splitted[2]))

    for i in range(numVertices+header_size, numVertices+header_size+numFaces):
        splitted = lines[i].split(" ") 
        if len(splitted) >= 3: 
            # splitted[0] should always be 3
            assert(int(splitted[0]) == 3)
            idx1 = int(splitted[1])
            idx2 = int(splitted[2])
            idx3 = int(splitted[3])

            indices.append([idx1, idx2, idx3])

            vertices.append(points[idx1*3:idx1*3+3])
            vertices.append(points[idx2*3:idx2*3+3])
            vertices.append(points[idx3*3:idx3*3+3])

    print(np.array(vertices).shape)
    print(np.array(indices).shape)
    mesh["vertices"] = np.array(vertices)
    mesh["indices"] = np.array(indices)

    return mesh


def load_xyz_as_occlussion(filename, voxel_size=0.025, grid_size=32):
    points = []
    name = os.path.basename(filename).split("-")[0]
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            splitted = line.split(" ")
            points.append(float(splitted[0]))
            points.append(float(splitted[1]))
            points.append(float(splitted[2]))

    pointcloud = np.array(points)
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
    print("%s has size=(%f, %f, %f) meters\n" % (os.path.basename(filename), size_x, size_y, size_z))
    occupancy_grid = np.array(np.zeros((grid_size,grid_size,grid_size)), dtype=np.float32)
    ox = min_x - (max_size - size_x) / 2
    oy = min_y - (max_size - size_y) / 2
    oz = min_z - (max_size - size_z) / 2
    for i in range(0,num_points,3):
        x = pointcloud[i+0]
        y = pointcloud[i+1]
        z = pointcloud[i+2]
        idx_x = int((x - ox) * (grid_size - 1) / max_size)
        idx_y = int((y - oy) * (grid_size - 1) / max_size)
        idx_z = int((z - oz) * (grid_size - 1) / max_size)
        occupancy_grid[idx_x, idx_y, idx_z] = 1

    return occupancy_grid

def load_xyz_as_density(filename, grid_size=32, normalize=False):
    points = []
    name = os.path.basename(filename).split("-")[0]
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            splitted = line.split(" ")
            points.append(float(splitted[0]))
            points.append(float(splitted[1]))
            points.append(float(splitted[2]))

    pointcloud = np.array(points)
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
    density_grid = np.array(np.zeros((grid_size,grid_size,grid_size)), dtype=np.float32)
    ox = min_x - (max_size - size_x) / 2
    oy = min_y - (max_size - size_y) / 2
    oz = min_z - (max_size - size_z) / 2
    for i in range(0,num_points,3):
        x = pointcloud[i+0]
        y = pointcloud[i+1]
        z = pointcloud[i+2]
        idx_x = int((x - ox) * (grid_size - 1) / max_size)
        idx_y = int((y - oy) * (grid_size - 1) / max_size)
        idx_z = int((z - oz) * (grid_size - 1) / max_size)
        density_grid[idx_x, idx_y, idx_z] = density_grid[idx_x, idx_y, idx_z] + 1

    # sanity check
    g_max = np.max(density_grid)
    if normalize:
        density_grid = density_grid / g_max

    print("sanity check. density maximum " + str(g_max))

    return density_grid    


def load_xyz_as_mean(filename, grid_size=32, normalize=False):
    points = []
    name = os.path.basename(filename).split("-")[0]
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            splitted = line.split(" ")
            points.append(float(splitted[0]))
            points.append(float(splitted[1]))
            points.append(float(splitted[2]))

    pointcloud = np.array(points)
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
    density_grid = np.array(np.zeros((grid_size,grid_size,grid_size,1)), dtype=np.float32)
    mean_grid = np.array(np.zeros((grid_size,grid_size,grid_size,3)), dtype=np.float32)
    expected_center = np.array(np.zeros((grid_size,grid_size,grid_size,3)), dtype=np.float32)
    for x in range(0, grid_size):
        for y in range(0, grid_size):
            for z in range(0, grid_size):
                expected_center[x,y,z] = np.array([x,y,z]) * max_size / grid_size

    ox = min_x - (max_size - size_x) / 2
    oy = min_y - (max_size - size_y) / 2
    oz = min_z - (max_size - size_z) / 2
    for i in range(0,num_points,3):
        x = pointcloud[i+0]
        y = pointcloud[i+1]
        z = pointcloud[i+2]
        idx_x = int((x - ox) * (grid_size - 1) / max_size)
        idx_y = int((y - oy) * (grid_size - 1) / max_size)
        idx_z = int((z - oz) * (grid_size - 1) / max_size)
        density_grid[idx_x, idx_y, idx_z] = density_grid[idx_x, idx_y, idx_z] + 1
        mean_grid[idx_x, idx_y, idx_z] = mean_grid[idx_x, idx_y, idx_z] + np.array([x,y,z])

    # sanity check
    g_max = np.max(density_grid)
    mean_grid = np.multiply(mean_grid, density_grid)
    if normalize:
        mean_grid = mean_grid - expected_center

    print("sanity check. density maximum " + str(g_max))

    return mean_grid  


def load_occ(filename, grid_size=32):
    with open(filename, "rb") as f:
        occ = np.load(f)
        xs = []
        ys = []
        zs = []
        vs = []
        for i in range(0, grid_size):
            for j in range(0, grid_size):
                for k in range(0, grid_size):
                    if occ[i,j,k] > 0:
                        xs.append(i)
                        ys.append(j)
                        zs.append(k)
                        vs.append(occ[i,j,k])

    return xs, ys, zs, vs


def load_xyz(filename):
    xs = []
    ys = []
    zs = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            splitted = line.split(" ")
            xs.append(float(splitted[0]))
            ys.append(float(splitted[1]))
            zs.append(float(splitted[2]))

    return xs, ys, zs


def load_xyz_as_variance(filename, grid_size=32, normalize=False):
    points = []
    name = os.path.basename(filename).split("-")[0]
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            splitted = line.split(" ")
            points.append(float(splitted[0]))
            points.append(float(splitted[1]))
            points.append(float(splitted[2]))

    pointcloud = np.array(points)
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
    var_grid = np.array(np.zeros((grid_size,grid_size,grid_size,1)), dtype=np.float32)
    point_grid = np.array(np.zeros((grid_size,grid_size,grid_size)), dtype=object)
    for x in range(0, grid_size):
        for y in range(0, grid_size):
            for z in range(0, grid_size):
                point_grid[x,y,z] = []

    ox = min_x - (max_size - size_x) / 2
    oy = min_y - (max_size - size_y) / 2
    oz = min_z - (max_size - size_z) / 2
    for i in range(0,num_points,3):
        x = pointcloud[i+0]
        y = pointcloud[i+1]
        z = pointcloud[i+2]
        idx_x = int((x - ox) * (grid_size - 1) / max_size)
        idx_y = int((y - oy) * (grid_size - 1) / max_size)
        idx_z = int((z - oz) * (grid_size - 1) / max_size)
        point_grid[idx_x, idx_y, idx_z].append(([x,y,z]))

    for x in range(0, grid_size):
        for y in range(0, grid_size):
            for z in range(0, grid_size):
                if len(point_grid[x,y,z]) > 1:
                    var_grid[x,y,z] = np.var(np.array(point_grid[x,y,z]))

    var_grid = var_grid / np.max(var_grid)
    # sanity check
    print("sanity check. var maximum " + str(np.max(var_grid)))

    return var_grid  


def load_xyzl(filename):
    xs = []
    ys = []
    zs = []
    name = os.path.basename(filename).split("-")[0]
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            splitted = line.split(" ")
            xs.append(-float(splitted[0]))
            ys.append(float(splitted[1]))
            zs.append(float(splitted[2]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # cm = LinearSegmentedColormap.from_list("alpha", [(0.0,0.0,0.0,0.0), (1.0,0.0,0.0,1.0)])
    ax.scatter(xs, ys, zs, c=[1.0, 0.0, 0.0, 0.8], marker='p')
    # ax.scatter(xs, ys, zs, c=vs, cmap=plt.get_cmap("Set1"), marker='p')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

# conversion label array
label_conversion = [0, 1, 2, 3, 4, 0, 0, 0, 0]
'''label_dict = {
    "none"          : 0,
    "CHAIR"         : 1,
    "DESK"          : 2,
    "COUCH"         : 3,
    "TABLE"         : 4,
    "WALL"          : 5,
    "FLOOR"         : 6,
    "WOOD"          : 7,
    "background"    : 8     
}'''

def load_xyzl_oct(filename, n_y):
    # generate two grids - 16x16x16 coarse grid for indexing
    # and 256x256x256 finer grid where each voxel represents cube 2x2x2 cm
    points = []
    labels = []
    name = os.path.basename(filename).split("-")[0]
    stime = time.time()
    cx = 0
    cy = 0
    cz = 0
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            splitted = line.split(" ")
            x = -float(splitted[0])
            y = float(splitted[1])
            z = float(splitted[2])
            cx = cx + x
            cy = cy + y
            cz = cz + z
            points.append(x)
            points.append(y)
            points.append(z)
            label = int(np.round(math.log2(float(splitted[3]))))
            labels.append(label_conversion[label])
    # print("Data readed in %f sec" % (time.time() - stime))
    
    # size is 512x512x512 cm for this grid and maximal grid is 128x128x128 voxels
    stime = time.time()
    pointcloud = np.array(points)
    num_points = int(pointcloud.shape[0])
    point_grid = np.array(np.zeros((16,16,16)), dtype=np.object)
    label_grid = np.array(np.zeros((16,16,16)), dtype=np.object)
    indices = []

    cx = cx / (num_points/3)
    cy = cy / (num_points/3)
    cz = cz / (num_points/3)

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
    cx = size_x / 2.0 + min_x
    cy = size_y / 2.0 + min_y
    cz = size_z / 2.0 + min_z
    max_size = max(size_x, max(size_y, size_z)) / 2.0

    # get the deconv labels
    deconv_hists = np.array(np.zeros((32,32,32,n_y)), dtype=np.int)
    deconv_labels = np.array(np.zeros((32,32,32)), dtype=np.int)
    for i in range(0,num_points,3):
        x = (pointcloud[i+0] - cx) / max_size
        y = (pointcloud[i+1] - cy) / max_size
        z = (pointcloud[i+2] - cz) / max_size
        # compute indices for grid 16x16x16, which is 64x64x64 cm
        idx_x = int((int(x * 16) + 16) / 32 * 31)
        idx_y = int((int(y * 16) + 16) / 32 * 31)
        idx_z = int((int(z * 16) + 16) / 32 * 31)
        if idx_x >= 0 and idx_x < 32 and idx_y >= 0 and idx_y < 32 and idx_z >= 0 and idx_z < 32:
            deconv_hists[idx_x, idx_y, idx_z, labels[int(i/3)]] = deconv_hists[idx_x, idx_y, idx_z, labels[int(i/3)]] + 1


    # make it a distribution
    deconv_labels = np.argmax(deconv_hists, axis=-1)
    # plt.imshow(deconv_labels)
    # plt.show()
    # sanity check
    # print(deconv_labels.shape)

    for i in range(0,num_points,3):
        x = (pointcloud[i+0] - cx) / max_size
        y = (pointcloud[i+1] - cy) / max_size
        z = (pointcloud[i+2] - cz) / max_size
        # compute indices for grid 16x16x16, which is 64x64x64 cm
        idx_x = int((int(x * 8) + 8) / 16 * 15)
        idx_y = int((int(y * 8) + 8) / 16 * 15)
        idx_z = int((int(z * 8) + 8) / 16 * 15)
        if idx_x >= 0 and idx_x < 16 and idx_y >= 0 and idx_y < 16 and idx_z >= 0 and idx_z < 16:
            if not point_grid[idx_x, idx_y, idx_z]:
                point_grid[idx_x, idx_y, idx_z] = []
                label_grid[idx_x, idx_y, idx_z] = []
                indices.append((idx_x, idx_y, idx_z))

            label_grid[idx_x, idx_y, idx_z].append(labels[int(i/3)])
            point_grid[idx_x, idx_y, idx_z].append(x)
            point_grid[idx_x, idx_y, idx_z].append(y)
            point_grid[idx_x, idx_y, idx_z].append(z)
    
    # print("Index grid constructed in %f sec" %(time.time() - stime))

    # generate lists of subgrids and its histogram for labels
    stime = time.time()
    sub_grids = []#np.array(np.zeros((len(indices),32,32,32,1)), dtype=np.float)
    sub_label = []#np.array(np.zeros((len(indices),32,32,32,n_y)), dtype=np.float)
    sub_indices = []
    label_lst = []#np.zeros((len(indices), n_y))
    if len(indices) == 0:
        raise Exception(filename)

    BBB = 1
    for i in range(0, len(indices)):
        index = indices[i]
        points = np.array(point_grid[index[0],index[1],index[2]])
        labels = label_grid[index[0],index[1],index[2]]

        if len(labels) < 10:
            continue

        hist = np.array(np.zeros((n_y,)), dtype=np.float)
        xs = []
        ys = []
        zs = []
        px = []
        py = []
        pz = []
        min_x = np.min(points[0::3])
        min_y = np.min(points[1::3])
        min_z = np.min(points[2::3])
        max_x = np.max(points[0::3])
        max_y = np.max(points[1::3])
        max_z = np.max(points[2::3])
        # Compute sizes 
        size_x = max_x - min_x
        size_y = max_y - min_y
        size_z = max_z - min_z
        cx = size_x / 2.0 + min_x
        cy = size_y / 2.0 + min_y
        cz = size_z / 2.0 + min_z
        max_size = max(size_x, max(size_y, size_z)) / 2.0
        sub_grid = np.zeros((32,32,32,1))
        sub_lab = np.zeros((32,32,32))
        for j in range(0,len(labels)):
            # move the points, so that they are positioned with zero corner of the grid
            # range of xyz should be [0,64)
            # convert to cms, add expected zero corner, divide by half extent, multiply by number of voxels
            x = (points[3*j+0] - cx) / max_size
            y = (points[3*j+1] - cy) / max_size
            z = (points[3*j+2] - cz) / max_size
            l = round(labels[j])
            hist[l] = hist[l] + 1
            # compute indices for grid 16x16x16, which is 64x64x64 cm
            idx_x = int((int(x * 16) + 16) / 32 * 31)
            idx_y = int((int(y * 16) + 16) / 32 * 31)
            idx_z = int((int(z * 16) + 16) / 32 * 31)
            
            sub_grid[idx_x,idx_y,idx_z,0] = 1
            sub_lab[idx_x,idx_y,idx_z] = l
            
            px.append(points[3*j+0])
            py.append(points[3*j+1])
            pz.append(points[3*j+2])
            xs.append(idx_x)
            ys.append(idx_y)
            zs.append(idx_z)

        sub_label.append(sub_lab)
        sub_grids.append(sub_grid)
        sub_indices.append(index)
        hist = hist / np.linalg.norm(hist)
        label_lst.append(hist)
        # viz
        if BBB == 0:
            fig = plt.figure()
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111, projection='3d')
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xs, ys, zs, c=[1.0, 0.0, 0.0, 0.8], marker='p')
            ax2.scatter(px, py, pz, c=[1.0, 0.0, 0.0, 0.8], marker='p')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.set_xlim(0,32)
            ax.set_ylim(0,32)
            ax.set_zlim(0,32)
            ax2.set_xlabel('X Label')
            ax2.set_ylabel('Y Label')
            ax2.set_zlabel('Z Label')
            plt.show()
            # BBB = 1


    # print("Sub-grids constructed in %f sec" % (time.time() - stime))    
    return np.array(sub_grids), np.array(label_lst), np.array(sub_label), np.array(sub_indices), deconv_labels


def load_xyzl_vis(filename, deconvolved_image, n_y):
    xs = []
    ys = []
    zs = []
    pointsx = []
    pointsy = []
    pointsz = []
    labels = []
    stime = time.time()
    cx = 0
    cy = 0
    cz = 0
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            splitted = line.split(" ")
            x = -float(splitted[0])
            y = float(splitted[1])
            z = float(splitted[2])
            cx = cx + x
            cy = cy + y
            cz = cz + z
            pointsx.append(x)
            pointsy.append(y)
            pointsz.append(z)
            label = int(np.round(math.log2(float(splitted[3]))))
            labels.append(label_conversion[label])
    # print("Data readed in %f sec" % (time.time() - stime))
    
    # size is 512x512x512 cm for this grid and maximal grid is 256x256x256 voxels
    stime = time.time()
    num_points = len(pointsx)
    vs = []
    vs_hat = []

    cx = cx / num_points
    cy = cy / num_points
    cz = cz / num_points
    min_x = np.min(pointsx)
    min_y = np.min(pointsy)
    min_z = np.min(pointsz)
    max_x = np.max(pointsx)
    max_y = np.max(pointsy)
    max_z = np.max(pointsz)
    # Compute sizes 
    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z
    cx = size_x / 2.0 + min_x
    cy = size_y / 2.0 + min_y
    cz = size_z / 2.0 + min_z
    max_size = max(size_x, max(size_y, size_z)) / 2.0

    # get the deconv labels
    for i in range(0,num_points):
        x = (pointsx[i] - cx) / max_size
        y = (pointsy[i] - cy) / max_size
        z = (pointsz[i] - cz) / max_size
        # compute indices for grid 16x16x16, which is 64x64x64 cm
        idx_x = int((int(x * 16) + 16) / 32 * 31)
        idx_y = int((int(y * 16) + 16) / 32 * 31)
        idx_z = int((int(z * 16) + 16) / 32 * 31)
        if idx_x >= 0 and idx_x < 32 and idx_y >= 0 and idx_y < 32 and idx_z >= 0 and idx_z < 32:
            xs.append(x)
            ys.append(y)
            zs.append(z)
            vs_hat.append(labels[i])
            vs.append(float(deconvolved_image[idx_x,idx_y,idx_z]))


    xs.append(0)
    ys.append(0)
    zs.append(0)
    vs.append(0)
    vs_hat.append(0)

    xs.append(0)
    ys.append(0)
    zs.append(0)
    vs.append(8)
    vs_hat.append(8)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    m = plt.get_cmap("Set1")
    plt.title("Ground truth")
    ax.scatter(xs, ys, zs, c=vs_hat, cmap=m, marker='p')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # plt.show()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    plt.title("Result")
    ax2.scatter(xs, ys, zs, c=vs, cmap=m, marker='p')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    plt.show()     


def load_xyzl_as_occlussion(filename, grid_size=32):
    points = []
    labels = []
    name = os.path.basename(filename).split("-")[0]
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            splitted = line.split(" ")
            points.append(-float(splitted[0]))
            points.append(float(splitted[1]))
            points.append(float(splitted[2]))
            label = int(np.round(math.log2(float(splitted[3]))))
            labels.append(label_conversion[label])

    pointcloud = np.array(points)
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
    max_size = np.max([size_x, size_y, size_z]) / 2
    # print("%s has size=(%f, %f, %f) meters\n" % (os.path.basename(filename), size_x, size_y, size_z))
    cx = size_x / 2 + min_x
    cy = size_y / 2 + min_y
    cz = size_z / 2 + min_z
    extent = int(grid_size / 2)

    xs = []
    ys = []
    zs = []
    px = []
    py = []
    pz = []
    BBB = 1
    for i in range(0,num_points,3):
        x = pointcloud[i+0]
        y = pointcloud[i+1]
        z = pointcloud[i+2]
        idx_x = int(((x - cx) * extent / max_size + extent) * (grid_size - 1) / (extent * 2))
        idx_y = int(((y - cy) * extent / max_size + extent) * (grid_size - 1) / (extent * 2))
        idx_z = int(((z - cz) * extent / max_size + extent) * (grid_size - 1) / (extent * 2))
        occupancy_grid[idx_x, idx_y, idx_z] = 1
        label_grid[idx_x, idx_y, idx_z] = labels[int(i/3)]
        px.append(x)
        py.append(y)
        pz.append(z)
        xs.append(idx_x)
        ys.append(idx_y)
        zs.append(idx_z)

    # viz
    if BBB == 0:
        fig = plt.figure()
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, ys, zs, c=[1.0, 0.0, 0.0, 0.8], marker='p')
        ax2.scatter(px, py, pz, c=[1.0, 0.0, 0.0, 0.8], marker='p')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim(0,grid_size-1)
        ax.set_ylim(0,grid_size-1)
        ax.set_zlim(0,grid_size-1)
        ax2.set_xlabel('X Label')
        ax2.set_ylabel('Y Label')
        ax2.set_zlabel('Z Label')
        plt.show()

    return occupancy_grid, label_grid


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

    return occupancy_grid, label_grid, 0, pointcloud, labels


def load_binvox(pts, sgs, grid_size=32, label_start=0, visualize=False):
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


    for x in range(0, grid_size):
        for y in range(0, grid_size):
            for z in range(0, grid_size):
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
        fig2 = plt.figure()
        # plt.axis("scaled")
        ax2 = fig2.add_subplot(111, projection='3d')
        ax = fig.add_subplot(111, projection='3d')
        m = plt.get_cmap("Set1")
        # plt.gca().set_aspect('scaled', adjustable='box')
        pv = np.array(pv)
        ax.scatter(px, py, pz, c=pv, cmap=m, marker='p')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax2.scatter(xs, ys, zs, c=vs, cmap=m, marker='p')
        ax2.set_xlim(0,grid_size-1)
        ax2.set_ylim(0,grid_size-1)
        ax2.set_zlim(0,grid_size-1)
        ax2.set_xlabel('X Label')
        ax2.set_ylabel('Y Label')
        ax2.set_zlabel('Z Label')
        plt.show()

    return occupancy_grid, label_grid, label_count, pointcloud, labels


# if __name__ == "__main__":
#     path_pt = ".\\3d-object-recognition\\ShapePartsData\\train\\train_data\\cap"
#     path_sg = ".\\3d-object-recognition\\ShapePartsData\\train\\train_label\\cap"
#     for pts,seg in zip(os.listdir(path_pt), os.listdir(path_sg)):
#         load_binvox(os.path.join(path_pt,pts), os.path.join(path_sg,seg), grid_size=32, label_start=0, visualize=True)