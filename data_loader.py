import time
import os
import math
import numpy as np
import matplotlib.pyplot as plt


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
label_conversion = [0, 1, 2, 3, 4, 0, 0, 0, 8]
'''label_dict = {
    "BACKGROUND"    : 0,
    "CHAIR"         : 1,
    "DESK"          : 2,
    "COUCH"         : 3,
    "TABLE"         : 4,
    "WALL"          : 5,
    "FLOOR"         : 6,
    "WOOD"          : 7,
    "NONE"          : 8     
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
    point_grid = np.array(np.zeros((8,8,8)), dtype=np.object)
    label_grid = np.array(np.zeros((8,8,8)), dtype=np.object)
    indices = []

    cx = cx / (num_points/3)
    cy = cy / (num_points/3)
    cz = cz / (num_points/3)

    # get the deconv labels
    deconv_hists = np.array(np.zeros((128,128,n_y)), dtype=np.int)
    deconv_labels = np.array(np.zeros((128,128)), dtype=np.int)
    for i in range(0,num_points,3):
        x = pointcloud[i+0] - cx
        y = pointcloud[i+1] - cy
        z = pointcloud[i+2] - cz
        # every 4cm is a voxel, 0,0,0 is in the middle
        # convert to cms, divide by half extent of the box to normalize the value, multiply by number of voxels, add center
        idx_x = int(100.0 * x / 256.0 * 64) + 64   # zero centered
        idx_y = int(100.0 * y / 256.0 * 64) + 64   # zero centered
        idx_z = int(100.0 * z / 256.0 * 64) + 64   # zero centered
        if idx_x >= 0 and idx_x < 128 and idx_y >= 0 and idx_y < 128 and idx_z >= 0 and idx_z < 128:
            deconv_hists[idx_x, idx_y, labels[int(i/3)]] = deconv_hists[idx_x, idx_y, labels[int(i/3)]] + 1

    # make it a distribution
    deconv_labels = np.argmax(deconv_hists, axis=-1)
    # sanity check
    # print(deconv_labels.shape)

    for i in range(0,num_points,3):
        x = pointcloud[i+0] - cx
        y = pointcloud[i+1] - cy
        z = pointcloud[i+2] - cz
        # every 4cm is a voxel, 0,0,0 is in the middle
        # convert to cms, divide by half extent of the box to normalize the value, multiply by number of voxels, add center
        idx_x = int(100.0 * x / 256.0 * 4) + 4   # zero centered
        idx_y = int(100.0 * y / 256.0 * 4) + 4   # zero centered
        idx_z = int(100.0 * z / 256.0 * 4) + 4   # counted from 0
        if idx_x >= 0 and idx_x < 8 and idx_y >= 0 and idx_y < 8 and idx_z >= 0 and idx_z < 8:
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
    sub_grids = np.array(np.zeros((len(indices),16,16,16,1)), dtype=np.float)
    sub_label = np.array(np.zeros((len(indices),16,16,16,n_y)), dtype=np.float)
    if len(indices) == 0:
        raise Exception(filename)

    label_lst = np.zeros((len(indices), n_y))
    for i in range(0, len(indices)):
        index = indices[i]
        points = np.array(point_grid[index[0],index[1],index[2]])
        labels = label_grid[index[0],index[1],index[2]]
        hist = np.array(np.zeros((n_y,)), dtype=np.float)
        for j in range(0,len(labels)):
            # move the points, so that they are positioned with zero corner of the grid
            # range of xyz should be [0,64)
            # convert to cms, add expected zero corner, divide by half extent, multiply by number of voxels
            x = 100.0 * points[3*j+0] - (index[0] - 4) * 64
            y = 100.0 * points[3*j+1] - (index[1] - 4) * 64
            z = 100.0 * points[3*j+2] - (index[2] - 4) * 64
            l = round(labels[j])
            hist[l] = hist[l] + 1
            # compute indices for grid 16x16x16, which is 64x64x64 cm
            idx_x = int(x / 4.0)
            idx_y = int(y / 4.0)
            idx_z = int(z / 4.0)
            sub_grids[i,idx_x,idx_y,idx_z,0] = 1
            sub_label[i,idx_x,idx_y,idx_z] = l
        
        hist = hist / np.linalg.norm(hist)
        label_lst[i,:] = hist

    # print("Sub-grids constructed in %f sec" % (time.time() - stime))    
    return sub_grids, label_lst, sub_label, indices, deconv_labels


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

    for i in range(0,num_points):
        x = pointsx[i] - cx
        y = pointsy[i] - cy
        z = pointsz[i] - cz
        # every 4cm is a voxel, 0,0,0 is in the middle
        # convert to cms, divide by half extent of the box to normalize the value, multiply by number of voxels, add center
        idx_x = int(100.0 * x / 16.0) + 16   # zero centered
        idx_y = int(100.0 * y / 16.0) + 16   # zero centered
        idx_z = int(100.0 * z / 16.0) + 16   # zero centered
        if idx_x >= 0 and idx_x < 32 and idx_y >= 0 and idx_y < 32 and idx_z >= 0 and idx_z < 32:
            xs.append(x)
            ys.append(y)
            zs.append(z)
            vs_hat.append(labels[i])
            vs.append(float(deconvolved_image[idx_x,idx_y]))

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
    ax.scatter(xs, ys, zs, c=vs_hat, cmap=m, marker='p')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # plt.show()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(xs, ys, zs, c=vs, cmap=m, marker='p')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    plt.show()     