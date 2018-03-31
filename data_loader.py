import os
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