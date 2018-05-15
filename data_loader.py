import os
import math
import numpy as np


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
            labels.append(math.log2(float(splitted[3])))

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
    
    print("%s has size=(%f, %f, %f) meters" % (os.path.basename(filename), size_x, size_y, size_z))
    occupancy_grid = np.array(np.zeros((320,128,192)), dtype=np.float32)
    labels_grid = np.array(-np.ones((320,128,192)), dtype=np.float32)

    for i in range(0,num_points,3):
        x = pointcloud[i+0]
        y = pointcloud[i+1]
        z = pointcloud[i+2]
        # every 2cm is a voxel, 0,0,0 is at (159,63,0) 
        idx_x = int(100.0 * x / 2.0) + 159
        idx_y = int(100.0 * y / 2.0) + 63
        idx_z = int(100.0 * z / 2.0)
        # add padding of 7
        if idx_x >= 7 and idx_x < 313 and idx_y >= 7 and idx_y < 121 and idx_z >= 7 and idx_z < 185:
            occupancy_grid[idx_x, idx_y, idx_z] = 1
            labels_grid[idx_x, idx_y, idx_z] = labels[int(i/3)]

    return occupancy_grid, labels_grid, np.nonzero(occupancy_grid)


def load_xyzl_oct(filename, n_y):
    # generate two grids - 16x16x16 coarse grid for indexing
    # and 256x256x256 finer grid where each voxel represents cube 2x2x2 cm
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
            labels.append(math.log2(float(splitted[3])))

    pointcloud = np.array(points)
    num_points = int(pointcloud.shape[0])
    # size is 512x512x512 cm for this grid and maximal grid is 256x256x256 voxels
    index_grid = np.array(np.zeros((16,16,16)), dtype=np.object)
    label_grid = np.array(np.zeros((16,16,16)), dtype=np.object)
    indices = []

    for i in range(0,num_points,3):
        x = pointcloud[i+0]
        y = pointcloud[i+1]
        z = pointcloud[i+2]
        # every 2cm is a voxel, 0,0,0 is at (159,63,0) 
        idx_x = int(100.0 * x / 16.0) + 8
        idx_y = int(100.0 * y / 16.0) + 8
        idx_z = int(100.0 * z / 16.0)
        if idx_x >= 0 and idx_x < 16 and idx_y >= 0 and idx_y < 16 and idx_z >= 0 and idx_z < 16:
            if not index_grid[idx_x, idx_y, idx_z]:
                index_grid[idx_x, idx_y, idx_z] = []
                label_grid[idx_x, idx_y, idx_z] = []
                indices.append((idx_x, idx_y, idx_z))

            label_grid[idx_x, idx_y, idx_z].append(labels[int(i/3)])
            index_grid[idx_x, idx_y, idx_z].append(x)
            index_grid[idx_x, idx_y, idx_z].append(y)
            index_grid[idx_x, idx_y, idx_z].append(z)
    
    # generate lists of subgrids and its histogram for labels
    sub_grids = []
    sub_label = np.zeros((len(indices), n_y))
    for i in range(0, len(indices)):
        index = indices[i]
        points = np.array(index_grid[index[0],index[1],index[2]])
        labels = label_grid[index[0],index[1],index[2]]
        # Find point cloud min and max
        min_x = np.min(points[0::3])
        min_y = np.min(points[1::3])
        min_z = np.min(points[2::3])
        max_x = np.max(points[0::3])
        max_y = np.max(points[1::3])
        max_z = np.max(points[2::3])
        hist = np.array(np.zeros((n_y,)), dtype=np.float)
        sub_grid = np.array(np.zeros((16,16,16)), dtype=np.float)
        for j in range(0,len(labels)):
            x = points[3*j+0] - min_x
            y = points[3*j+1] - min_y
            z = points[3*j+2] - min_z
            l = round(labels[j])
            hist[l] = hist[l] + 1
            # compute indices
            idx_x = int(x / 16.0)
            idx_y = int(y / 16.0)
            idx_z = int(z / 16.0)
            sub_grid[idx_x, idx_y, idx_z] = 1
        
        hist = hist / np.linalg.norm(hist)
        sub_label[i,:] = hist
        sub_grids.append(sub_grid)

    return np.array(sub_grids), sub_label