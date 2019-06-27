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

    return occupancy_grid, label_grid, 0, pointcloud, labels, vol_occ, np.array([size_x, size_y, size_z])


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

    return occupancy_grid, label_grid, label_count, pointcloud, labels, vol_occ, np.array([size_x, size_y, size_z])


def load_binvox_8_grid(pts, sgs, grid_size=32, ogrid_size=32, label_start=0, visualize=False):
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
    # make it half as we want the division below it
    grid_size = int(grid_size / 2)
    ogrid_size = int(ogrid_size / 2)

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

    os = []
    ls = []

    grid_size = int(grid_size * 2)
    ogrid_size = int(ogrid_size * 2)
    
    for a in range(0,2):
        for b in range(0,2):
            for c in range(0,2):
                os.append(occupancy_grid[a*grid_size:(a+1)*grid_size,b*grid_size:(b+1)*grid_size,c*grid_size:(c+1)*grid_size])
                ls.append(label_grid[a*ogrid_size:(a+1)*ogrid_size,b*ogrid_size:(b+1)*ogrid_size,c*ogrid_size:(c+1)*ogrid_size])


    return os, ls, label_count, pointcloud, labels


def load_npy_from_file(pts_file, grid_size=32, visualize=False):
    with open(pts_file, "rb") as f:
        pointcloud = np.load(f)
        return load_npy(pointcloud, grid_size, visualize)


def load_npy(pointcloud, grid_size=32, visualize=False):
    num_points = int(pointcloud.shape[0])

    # Find point cloud min and max
    min_x = np.min(pointcloud[:,0])
    min_y = np.min(pointcloud[:,1])
    min_z = np.min(pointcloud[:,2])
    max_x = np.max(pointcloud[:,0])
    max_y = np.max(pointcloud[:,1])
    max_z = np.max(pointcloud[:,2])
    # Compute sizes 
    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z
    
    max_size = np.max([size_x, size_y, size_z])
    min_size = min(min_x, min(min_y, min_z))
    # print("%s has size=(%f, %f, %f) meters\n" % (os.path.basename(filename), size_x, size_y, size_z))
    occupancy_grid = np.array(np.zeros((grid_size,grid_size,grid_size)), dtype=np.float32)
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


    for i in range(0,num_points,3):
        x = pointcloud[i,0]
        y = pointcloud[i,1]
        z = pointcloud[i,2]
        idx_x = int(((x - cx) * extent / max_size + extent) * (grid_size - 1) / (extent * 2))
        idx_y = int(((y - cy) * extent / max_size + extent) * (grid_size - 1) / (extent * 2))
        idx_z = int(((z - cz) * extent / max_size + extent) * (grid_size - 1) / (extent * 2))
        occupancy_grid[idx_x, idx_y, idx_z] = 1

        if visualize:
            px.append(x)
            py.append(y)
            pz.append(z)




    # viz
    if visualize:
        fig = plt.figure()
        # plt.axis("scaled")
        ax = fig.add_subplot(111, projection='3d')
        m = plt.get_cmap("Set1")
        # plt.gca().set_aspect('scaled', adjustable='box')
        # pv = np.array(pv)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        # plt.axis('off')
        plt.gca().set_aspect('equal', adjustable='box')
        ax.set_xlim(min_size,max_size)
        ax.set_ylim(min_size,max_size)
        ax.set_zlim(min_size,max_size)
        ax.scatter(px, py, pz, c="r", marker='p')

        plt.show()

    return occupancy_grid, pointcloud[:,0:3]


def viz_points_segmentation(pts_path, gt_path, res_path):
    with open(gt_path, "rb") as f:
        gt = np.load(f)
    with open(res_path, "rb") as f:
        res = np.load(f)

    points = []

    with open(pts_path, "r") as pts_f:
        lines = pts_f.readlines()

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

    maximum = max(max_x, max(max_y, max_z))
    minimum = min(min_x, min(min_y, min_z))

    xs = []
    ys = []
    zs = []
    vs = []

    for i in range(0,num_points,3):
        x = pointcloud[i+0]
        y = pointcloud[i+1]
        z = pointcloud[i+2]


        idx = int(i/3)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        if gt[idx] == res[idx]:
            vs.append((0,1,0))
        else:
            vs.append((1,0,0))
        

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c=vs, marker='p')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_xlim(minimum,maximum)
    ax.set_ylim(minimum,maximum)
    ax.set_zlim(minimum,maximum)
    plt.show()


def load_binvox_kdtree(pts, sgs, label_start=0, visualize=False, k=8, m=50):
    points = []
    labels = []
    with open(pts, "r") as pts_f, open(sgs, "rb") as sgs_f:
        lines = pts_f.readlines()

        for line in lines:
            splitted = line.split(" ")
            points.append([float(splitted[0]),float(splitted[1]),float(splitted[2])])
            labels.append(int(sgs_f.read(1))-1) # indexing starts from 1 in the file
            sgs_f.read(1) # throw out newline

    pointcloud = np.array(points)
    labels = np.array(labels)
    label_count = np.max(labels) + 1
    labels = labels + label_start
    num_points = int(pointcloud.shape[0])

    # Find point cloud min and max
    min_extent = np.min(pointcloud, axis=0)
    max_extent = np.max(pointcloud, axis=0)
    # Compute sizes 
    size_x = max_extent[0] - min_extent[0]
    size_y = max_extent[1] - min_extent[1]
    size_z = max_extent[2] - min_extent[2]
    max_size = np.max(max_extent)
    min_size = np.min(min_extent)
    # print("%s has size=(%f, %f, %f) meters\n" % (os.path.basename(filename), size_x, size_y, size_z))

    centroids = []
    

    def recursive_split(points, min_extent, max_extent, axis):
        middle = 0.5 * (max_extent[axis] - min_extent[axis]) + min_extent[axis]
        cond = points[:,axis] < middle
        L = points[cond]
        R = points[~cond]

        if L.shape[0] < m:
            if L.shape[0] > 0:
                centroids.append(np.mean(L,axis=0))
        else:
            m1 = [min_extent[0], min_extent[1], min_extent[2]]
            m2 = [max_extent[0], max_extent[1], max_extent[2]]
            m2[axis]=middle
            recursive_split(L, m1, m2, (axis+1)%3)

        if R.shape[0] < m:
            if R.shape[0] > 0:
                centroids.append(np.mean(R,axis=0))
        else:
            m1 = [min_extent[0], min_extent[1], min_extent[2]]
            m2 = [max_extent[0], max_extent[1], max_extent[2]]
            m1[axis]=middle
            recursive_split(R, m1, m2, (axis+1)%3)



    recursive_split(pointcloud, min_extent, max_extent, 0)
    centroids = np.array(centroids)

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

    data = []

    # find minimum distances
    for i in range(0,num_points):
        p = pointcloud[i]

        dists = np.sqrt(np.sum(np.square(centroids - p),axis=-1))
        idxs = np.argpartition(dists,k)
        data.append(centroids[idxs[0:k]] - p)

        if visualize:
            px.append(p[0])
            py.append(p[1])
            pz.append(p[1])
            pv.append(labels[i] - label_start)


    # viz
    if visualize:
        fig = plt.figure()
        # plt.axis("scaled")
        ax = fig.add_subplot(111, projection='3d')
        m = plt.get_cmap("Set1")
        plt.gca().set_aspect('scaled', adjustable='box')
        pv = np.array(pv)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        # plt.axis('off')
        ax.set_xlim(min_size,max_size)
        ax.set_ylim(min_size,max_size)
        ax.set_zlim(min_size,max_size)
        ax.scatter(px, py, pz, c=pv, cmap=m, marker='p')

    return np.array(data), labels, label_count, pointcloud


def xyzl_to_binvox(filename, out_dir, data_name, category_name):
    points = []
    labels = []
    name = os.path.basename(filename).split(".")[0]
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            splitted = line.split(" ")
            points.append(float(splitted[0]))
            points.append(float(splitted[1]))
            points.append(float(splitted[2]))
            label = int(np.round(math.log2(float(splitted[3]))))
            labels.append(label_conversion[label]+1)

    data_dir = os.path.join(out_dir, data_name, data_name + "_data", category_name)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        os.chmod(data_dir, 0o777)

    label_dir = os.path.join(out_dir, data_name, data_name + "_label", category_name)
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)
        os.chmod(label_dir, 0o777)
    
    pointcloud = np.array(points)
    num_points = int(pointcloud.shape[0])
    # Find point cloud min and max
    min_x = np.min(pointcloud[0::3])
    min_y = np.min(pointcloud[1::3])
    min_z = np.min(pointcloud[2::3])
    max_x = np.max(pointcloud[0::3])
    max_y = np.max(pointcloud[1::3])
    max_z = np.max(pointcloud[2::3])

    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z
    cx = size_x / 2 + min_x
    cy = size_y / 2 + min_y
    cz = size_z / 2 + min_z

    with open(os.path.join(data_dir, name + ".pts"), "w") as f, open(os.path.join(label_dir, name + ".seg"), "wb") as s:
        string = ""
        seg_string = ""
        for i in range(0, len(points), 3):
            x = points[i+0] - cx
            y = points[i+1] - cy
            z = points[i+2] - cz
            if i + 3 < len(points):
                string = string + "%f %f %f\n" % (x, y, z)
            else:
                string = string + "%f %f %f" % (x, y, z)

            seg_string += "%d\n" % labels[int(i/3)]

        f.write(string)
        s.write(seg_string.encode('utf-8'))
                

def viz_points(pts_path, seg_path, out_dir=None):
    points = []
    
    with open(seg_path, "rb") as f:
        labels = np.load(f)

    with open(pts_path, "r") as pts_f:
        lines = pts_f.readlines()

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

    maximum = max(max_x, max(max_y, max_z))
    minimum = min(min_x, min(min_y, min_z))

    xs = []
    ys = []
    zs = []
    vs = []
    # red, green, blue, orange, yellow, purple
    # bg , chair, table,couch , couch-table
    color = [(1,0,0), (0,1,0), (0,0,1), (1,0.5,0), (1,1,0), (1,0,1)]
    string = ""
    for i in range(0,num_points,24):
        x = pointcloud[i+0]
        y = pointcloud[i+1]
        z = pointcloud[i+2]
        xs.append(x)
        ys.append(y)
        zs.append(z)
        idx = labels[int(i/3)]
        c = color[idx]
        vs.append(c)

        string += "%f %f %f %f %f %f\n" % (x, y, z, c[0], c[1], c[2])
    if not out_dir == None:    
        with open(os.path.join(out_dir, pts_path.split(os.sep)[-1]), "w") as f:
            f.write(string)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c=vs, marker='p')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_xlim(minimum,maximum)
    ax.set_ylim(minimum,maximum)
    ax.set_zlim(minimum,maximum)
    plt.show()


def filter_binvox(pts_path, seg_path, min_dist=0.1, max_dist=3.0):
    point_str = ""
    seg_str = ""

    with open(pts_path, "r") as pts_f, open(seg_path, "rb") as sgs_f:
        lines = pts_f.readlines()
        iteration = 1
        for line in lines:
            splitted = line.split(" ")
            l = int(sgs_f.read(1))
            sgs_f.read(1) # throw out newline
            x = float(splitted[0])
            y = float(splitted[1])
            z = float(splitted[2])
            d = np.sqrt(x*x + y*y + z*z)
            if d >= min_dist and d <= max_dist:
                if iteration == len(lines):
                    point_str += "%f %f %f" % (x,y,z)
                else:
                    point_str += "%f %f %f\n" % (x,y,z)

                seg_str += "%d\n" % l
            
            iteration += 1

    with open(pts_path, "w") as pts_f, open(seg_path, "wb") as sgs_f:
        pts_f.truncate()
        pts_f.write(point_str)
        sgs_f.truncate()
        sgs_f.write(seg_str.encode('utf-8'))


def load_binvox_as_distance_field(pts, sgs, grid_size=32, ogrid_size=32, label_start=0, visualize=False):
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

    return occupancy_grid, label_grid, label_count, pointcloud, labels




if __name__ == "__main__":
#     # pts_path = "./tmp/012938.pts"
#     # gt_path =  "./tmp/012938.gt"
#     # res_path = "./tmp/012938.res"
#     # viz_points_segmentation(pts_path, gt_path, res_path)
    # path_pt =  ".\\3d-object-recognition\\ShapePartsData\\train\\train_data\\motorbike\\"
    # path_sg = ".\\3d-object-recognition\\ShapePartsData\\train\\train_label\\motorbike\\"
    # for pt_name,sg_name in zip(os.listdir(path_pt),os.listdir(path_sg)):
    #     pf = os.path.join(path_pt, pt_name)
    #     sf = os.path.join(path_sg, sg_name)
    #     load_binvox(pf, sf, grid_size=32, label_start=0, visualize=True)

    path_not_centered = "C:/Users/janovrom/Desktop/points_not_centered.xyz"
    path_centered = "C:/Users/janovrom/Desktop/points_centered.xyz"
    path_just_points1 = "C:/Users/janovrom/Desktop/points1.xyz"
    path_just_points2 = "C:/Users/janovrom/Desktop/points2.xyz"
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # m = plt.get_cmap("plasma")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # ax.set_xlim(min_size,max_size)
    # ax.set_ylim(min_size,max_size)
    # ax.set_zlim(min_size,max_size)
    plt.axis("equal")
    for path,color in [[path_just_points1, "r"],[path_just_points2,"b"]]:
        xs,ys,zs = load_xyz(path)
        min_x = np.min(np.array(xs))
        min_y = np.min(np.array(ys))
        min_z = np.min(np.array(zs))
        max_x = np.max(np.array(xs))
        max_y = np.max(np.array(ys))
        max_z = np.max(np.array(zs))
        # Compute sizes 
        size_x = max_x - min_x
        size_y = max_y - min_y
        size_z = max_z - min_z
        max_size = np.max([max_x, max_y, max_z])
        min_size = np.min([min_x, min_y, min_z])
        print([min_x, min_y, min_z])
        print([max_x, max_y, max_z])
        # cx = (max_x - min_x) * 0.5 + min_x
        # cy = (max_y - min_y) * 0.5 + min_y
        # cz = (max_z - min_z) * 0.5 + min_z
        cx = np.sum(np.array(xs)) / len(xs)
        cy = np.sum(np.array(ys)) / len(ys)
        cz = np.sum(np.array(zs)) / len(zs)
        print("CENTER: " + str(cx) + "," + str(cy) + "," + str(cz))

        ax.scatter(xs, ys, zs, c=color, marker='s')
        # ax.scatter([128.5,125.4], [108,99.6], [234,233], c="b", marker="s")

    plt.show()
    # for pts,seg in zip(os.listdir("./tmp/pts/"), os.listdir("./tmp/res/")):
    #     viz_points("./tmp/pts\\" + pts, "./tmp/res\\" + seg)#, out_dir="./tmp")
    
    # path_pts = "./3d-object-recognition/UnityData/test/test_data/room"
    # path_seg = "./3d-object-recognition/UnityData/segmentation/res/test/room"
    # for pts,seg in zip(os.listdir(path_pts),os.listdir(path_seg)):
    #     viz_points(os.path.join(path_pts, pts), os.path.join(path_seg, seg))

    # occ, points = load_npy("./3d-object-recognition/ModelNet40/train/tv_stand/tv_stand_0025.npy")
    # # Find point cloud min and max
    # min_x = np.min(points[:,0])
    # min_y = np.min(points[:,1])
    # min_z = np.min(points[:,2])
    # max_x = np.max(points[:,0])
    # max_y = np.max(points[:,1])
    # max_z = np.max(points[:,2])
    # # Compute sizes 
    # size_x = max_x - min_x
    # size_y = max_y - min_y
    # size_z = max_z - min_z
    
    # max_size = np.max([size_x, size_y, size_z])
    # min_size = min(min_x, min(min_y, min_z))
    # import mdp

    # gng = mdp.nodes.GrowingNeuralGasNode(max_nodes=512)
    # px = points[:,0]
    # py = points[:,1]
    # pz = points[:,2]
    # pv = np.zeros(pz.shape)

    # stime = time.time()
    # gng.train(points)
    # gng.stop_training()
    # print("GNG trained in %f sec" % (time.time() - stime))
    # positions = np.array(gng.get_nodes_position())
    # print(positions.shape)

    # px = np.concatenate([px,positions[:,0]], axis=0)
    # py = np.concatenate([py,positions[:,1]], axis=0)
    # pz = np.concatenate([pz,positions[:,2]], axis=0)
    # pv = np.concatenate([pv,np.ones(positions[:,2].shape)], axis=0)


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # m = plt.get_cmap("plasma")
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # ax.set_xlim(min_size,max_size)
    # ax.set_ylim(min_size,max_size)
    # ax.set_zlim(min_size,max_size)
    # plt.gca().set_aspect('equal', adjustable='box')
    # ax.scatter(px, py, pz, c=pv, cmap=m, marker='s', s=(pv * 10.0 + 1.0))
    # plt.show()

    # for xyzl_fname in os.listdir("./3d-object-recognition/UnityData/src-train"):
    #     xyzl_to_binvox(os.path.join("./3d-object-recognition/UnityData/src-train", xyzl_fname), "./3d-object-recognition/UnityData/", "train", "room")
    #     path_pt =  "./3d-object-recognition/UnityData/train/train_data/room/" + xyzl_fname.split(".")[0] + ".pts"
    #     path_sg =  "./3d-object-recognition/UnityData/train/train_label/room/" + xyzl_fname.split(".")[0] + ".seg"
    #     # load_binvox(path_pt, path_sg, grid_size=32, label_start=0, visualize=True)