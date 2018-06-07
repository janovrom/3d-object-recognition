import octree as octree
import random
import sys
from multiprocessing import Pool
from functools import partial
import os
import math
import shutil
import numpy as np
from timeit import default_timer as timer
from numba import vectorize
from numba import guvectorize 
from numba import float32, int32
from numba import jit, cuda
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import data_loader as dl
from triangle_test import * 
from matplotlib.colors import LinearSegmentedColormap



# change environ properties for numba
os.environ['NUMBAPRO_NVVM']=r'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\nvvm\\bin\\nvvm64_31_0.dll'
os.environ['NUMBAPRO_LIBDEVICE']=r'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\nvvm\\libdevice'

def eulerToMatrix(theta) :
    '''
    Computes rotation matrix from given Euler angles in degrees.

    Arguments:
        theta -- triplet of floats, angles in degrees

    Returns:
        R -- rotation matrix, shape=(3,3)
    '''
    theta = np.radians(theta)

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
         
         
                     
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R


def rotatePoints(points, rotMatrix):
    '''
    Rotates vertices around origin.
    Arguments:
        points -- numpy array, shape=(n*3, 3) where n is number of faces in the mesh
        rotMatrix -- numpy array, shape=(3,3)
    Returns:
        point rotated around origin
    '''
    return np.dot(rotMatrix, points)


def getAABB(points):
    '''
    Returns minimum and maximum for axis-aligned bounding box.

    Arguments:
        points -- numpy array, shape=(3, m) where m is number of vertices

    Returns:
        mins, maxs -- tuple of two triplets
    '''
    mins = np.min(points, axis=1)
    maxs = np.max(points, axis=1)
    assert(mins.shape[0] == 3)
    assert(maxs.shape[0] == 3)

    return mins, maxs


def dot2(v):
    return np.dot(v,v)


def udTriangle(p, a, b, c):
    ba = b - a
    pa = p - a
    cb = c - b
    pb = p - b
    ac = a - c
    pc = p - c
    nor = np.cross(ba, ac)

    if np.sqrt(np.sign(np.dot(np.cross(ba,nor),pa)) + np.sign(np.dot(np.cross(cb,nor),pb)) + np.sign(np.dot(np.cross(ac,nor),pc)))<2.0:
        return min( min(
    dot2(ba*np.clip(np.dot(ba,pa)/np.dot(ba, ba),0.0,1.0)-pa),
    dot2(cb*np.clip(np.dot(cb,pb)/np.dot(cb, cb),0.0,1.0)-pb) ),
    dot2(ac*np.clip(np.dot(ac,pc)/np.dot(ac, ac),0.0,1.0)-pc) )
    else:
        return np.dot(nor,pa)*np.dot(nor,pa)/dot2(nor)


def orient2d(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def points2Occ(tris, mins, maxs, pad, occ_out):
    voxel_size = (maxs - mins)/occ_out.shape
    grid_size = maxs - mins
    size = np.array(list(occ_out.shape))
    box_half_size = voxel_size / 2.0 + 0.01

    for i in range(0,tris.shape[1], 3):
        # compute aabb of triangle
        t = tris[:, i:i+3]
        t = np.transpose(t)

        # normal = np.cross(t[:,1] - t[:,0], t[:,2] - t[:,0])
        # if normal[2] <= 0:
        #     continue

        t_min = np.min(t, axis=1) - mins
        t_max = np.max(t, axis=1) - mins
        # compute absolute position in the grid
        t_min = (t_min * (size - 1) / grid_size).astype(int)
        t_max = (t_max * (size - 1) / grid_size).astype(int) + 1
        for i in range(t_min[0], t_max[0]):
            for j in range(t_min[1], t_max[1]):
                for k in range(t_min[2], t_max[2]):
                    center = (np.array([i, j, k]) + 0.5) * voxel_size + mins
                    if intersects_box(t, center, box_half_size):
                        occ_out[i,j,k] = 1
                # # add min to transform to object space
                # p = (np.array([i, j, t_max[2]])) * voxel_size + mins
                # # Determine point orientations
                # w0 = orient2d(t[:,1], t[:,2], p)
                # w1 = orient2d(t[:,2], t[:,0], p)
                # w2 = orient2d(t[:,0], t[:,1], p)

                # # If p is on or inside all edges, render pixel.
                # if (w0 >= 0 and w1 >= 0 and w2 >= 0):
                #     # compute z in object space in aabb
                #     p[2] = maxs[2]
                #     z = udTriangle(p, t[:,0], t[:,1], t[:,2])
                #     k = int((maxs[2] - mins[2] - z) * size[2] / grid_size[2])
                #     occ_out[i,j,k] = 1
         
def main():
    nps = dl.load_off_file("./3d-object-recognition/objects/off/cube.off")
    # nps = rotatePoints(nps, eulerToMatrix([45,45,45]))
    mins, maxs = getAABB(nps)
    occ_out = np.zeros((30,30,30))
    pad = np.zeros((30))
    points2Occ(nps, mins, maxs, pad, occ_out)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    print(np.sum(occ_out))
    axis = np.nonzero(occ_out)
    print(len(axis))
    xs = axis[0]
    ys = axis[1]
    zs = axis[2]
    ax.scatter(xs, ys, zs, c='r', marker='s')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim3d(0, 30)
    ax.set_ylim3d(0, 30)
    ax.set_zlim3d(0, 30)

    plt.show()


def exists_and_create(datapath):
    if not os.path.exists(datapath):
        os.mkdir(datapath)
        os.chmod(datapath, 0o777)


def create_segmentation_set(in_dir, out_dir, grid_size=32):
    for fname in os.listdir(in_dir):
        in_file = os.path.join(in_dir, fname)
        out_file = os.path.join(out_dir, os.path.basename(fname) + ".npy")

        occ = dl.load_xyz_as_occlussion(in_file, grid_size=grid_size)
        with open(out_file, "wb") as f:
            np.save(f, occ)


def create_set(filename, outpath, inpath, insize):
    print("Creating set for: " + filename)

    rots = []

    for xrot in range(-80, 81, 20):
        for yrot in range(0, 360, 10):
            rots.append([xrot, yrot])
    
    rots = np.array(rots)
    indices = np.random.permutation(len(rots))
    # print(indices[0:len(rots)-200])
    ten_percent = int(len(rots) / 10.0)
    rots_train = rots[indices[0:len(rots)-2*ten_percent]]
    rots_test = rots[indices[len(rots)-2*ten_percent:len(rots)-ten_percent]]
    rots_dev = rots[indices[len(rots)-ten_percent:len(rots)]]

    def create(set_name, rotations):
        idx = 0
        triangles = dl.load_off_file(os.path.join(inpath, filename))
        data_filename = filename.split(".")[0]
        for rot in rotations:
            xrot, yrot = rot
            tris = np.copy(triangles)
            tris = rotatePoints(tris, eulerToMatrix([0,yrot,0]))
            tris = rotatePoints(tris, eulerToMatrix([xrot,0,0]))
            mins, maxs = getAABB(tris)
            tris = np.transpose(tris)
            tree = octree.Octree(tris, mins, maxs, 4)
            X = np.array(tree.get_occlussion())
            # hash it to the regular grid
            mins, maxs = getAABB(np.transpose(X))
            X = (insize-1) * (X - mins) / (maxs - mins)
            X = X.astype(int)
            occ_grid = np.zeros((insize,insize,insize), dtype=np.int)
            for i in range(0, X.shape[0]):
                x, y, z = X[i,:]
                occ_grid[x,y,z] = 1

            name = os.path.join(outpath, set_name, data_filename + "_%d" % idx)
            with open(name, "wb") as f:
                np.save(f, occ_grid) 
            idx = idx + 1

    create("train", rots_train)
    create("test", rots_test)
    create("dev", rots_dev)
        

def create_dataset():
    train_path = "./3d-object-recognition/data-4/train"
    test_path = "./3d-object-recognition/data-4/test"
    dev_path = "./3d-object-recognition/data-4/dev"
    outpath = "./3d-object-recognition/data-4/"
    inpath = "./3d-object-recognition/objects/off"
    exists_and_create(outpath)
    exists_and_create(train_path)
    exists_and_create(test_path)
    exists_and_create(dev_path)

    p = Pool(5)

    p.map(partial(create_set, outpath=outpath,inpath=inpath, insize=4), [
        "cube.off",
        "cone.off",
        "torus.off",
        "sphere.off",
        "cylinder.off"])

    p.close()
    p.join()


def convert_scale_dataset(from_path, to_path, insize, outsize, subdir):
    exists_and_create(to_path)
    from_path = os.path.join(from_path, subdir)
    exists_and_create(from_path)
    fig_path = "./3d-object-recognition/figures/scaled-" + str(outsize) + "-from-" +str(insize)
    exists_and_create(fig_path)
    to_path = os.path.join(to_path, subdir)
    exists_and_create(from_path)

    for fname in os.listdir(from_path):
        filename = os.path.join(from_path, fname)
        grid = np.zeros((outsize, outsize, outsize))
        start = int(outsize / 2 - insize / 2 - 1)
        with open(filename, "rb") as f:
            small_grid = np.reshape(np.load(f), (insize, insize, insize))
            for i in range(0, insize):
                for j in range(0,insize):
                    for k in range(0,insize):
                        grid[i+start,j+start,k+start] = small_grid[i,j,k]

        filename = os.path.join(to_path, fname)
        with open(filename, "wb") as f:
            np.save(f, grid)

        # if "_0" in fname:
        #     # sanity check
        #     s = outsize
        #     xs = []
        #     ys = []
        #     zs = []
        #     for i in range(0, s):
        #         for j in range(0, s):
        #             for k in range(0, s):
        #                 if grid[i,j,k] == 1:
        #                     xs.append(i)
        #                     ys.append(j)
        #                     zs.append(k)

        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.scatter(xs=xs, ys=ys, zs=zs, c=[0.1,0.1,0.1,0.2], marker='o')
        #     ax.set_xlabel('X Label')
        #     ax.set_ylabel('Y Label')
        #     ax.set_zlabel('Z Label')
        #     ax.set_xlim3d(0, s)
        #     ax.set_ylim3d(0, s)
        #     ax.set_zlim3d(0, s)

        #     # plt.show() 
        #     plt.savefig(os.path.join(fig_path, fname.split(".")[0] + ".png"))   
        #     plt.close() 


def convert_scaled_dataset_to_translation(from_path, to_path, insize, outsize, subdir):
    exists_and_create(to_path)
    from_path = os.path.join(from_path, subdir)
    exists_and_create(from_path)
    fig_path = "./3d-object-recognition/figures/translated-" + str(outsize) + "-from-" +str(insize)
    exists_and_create(fig_path)
    to_path = os.path.join(to_path, subdir)
    exists_and_create(to_path)

    for fname in os.listdir(from_path):
        filename = os.path.join(from_path, fname)
        grid = np.zeros((outsize, outsize, outsize))
        end = int(outsize - insize - 1)
        x = random.randint(0, end)
        y = random.randint(0, end)
        z = random.randint(0, end)

        with open(filename, "rb") as f:
            small_grid = np.reshape(np.load(f), (insize, insize, insize))
            for i in range(0, insize):
                for j in range(0,insize):
                    for k in range(0,insize):
                        grid[i+x,j+y,k+z] = small_grid[i,j,k]

        filename = os.path.join(to_path, fname)
        with open(filename, "wb") as f:
            np.save(f, grid)

        # if "_" in fname:
        #     # sanity check
        #     s = outsize
        #     xs = []
        #     ys = []
        #     zs = []
        #     for i in range(0, s):
        #         for j in range(0, s):
        #             for k in range(0, s):
        #                 if grid[i,j,k] == 1:
        #                     xs.append(i)
        #                     ys.append(j)
        #                     zs.append(k)

        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.scatter(xs=xs, ys=ys, zs=zs, c=[0.1,0.1,0.1,0.2], marker='o')
        #     ax.set_xlabel('X Label')
        #     ax.set_ylabel('Y Label')
        #     ax.set_zlabel('Z Label')
        #     ax.set_xlim3d(0, s)
        #     ax.set_ylim3d(0, s)
        #     ax.set_zlim3d(0, s)

        #     # plt.show() 
        #     plt.savefig(os.path.join(fig_path, fname.split(".")[0] + ".png"))   
        #     plt.close()


def rename_set_files(in_dir, start_idx):
    for fname in os.listdir(in_dir):
        filename = os.path.join(in_dir, fname)
        split = fname.split("_")
        out_filename = os.path.join(in_dir, split[0] + "_" + str(int((split[1])) + int(start_idx)))
        os.rename(filename, out_filename)


def get_visible_set(in_dir, out_dir, insize, subdir):
    fig_path = "./3d-object-recognition/figures/seen-" +str(insize)
    exists_and_create(fig_path)
    exists_and_create(out_dir)
    out_dir = os.path.join(out_dir, subdir)
    in_dir = os.path.join(in_dir, subdir)
    exists_and_create(out_dir)

    for fname in os.listdir(in_dir):
        filename = os.path.join(in_dir, fname)
        grid = np.zeros((insize, insize, insize))

        with open(filename, "rb") as f:
            small_grid = np.reshape(np.load(f), (insize, insize, insize))
            for i in range(0, insize):
                for j in range(0,insize):
                    for k in range(insize-1,-1,-1):
                        if small_grid[i,j,k] == 1:
                            grid[i,j,k] = small_grid[i,j,k]
                            break # stop at first z

        filename = os.path.join(out_dir, fname)
        with open(filename, "wb") as f:
            np.save(f, grid)

        if "_0" in fname:
            # sanity check
            s = insize
            xs = []
            ys = []
            zs = []
            for i in range(0, s):
                for j in range(0, s):
                    for k in range(0, s):
                        if grid[i,j,k] == 1:
                            xs.append(i)
                            ys.append(j)
                            zs.append(k)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xs=xs, ys=ys, zs=zs, c=[0.1,0.1,0.1,0.2], marker='o')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.set_xlim3d(0, s)
            ax.set_ylim3d(0, s)
            ax.set_zlim3d(0, s)

            # plt.show() 
            plt.savefig(os.path.join(fig_path, fname.split(".")[0] + ".png"))   
            plt.close()


def get_visible_set_sparse(in_dir, out_dir, insize, subdir, step):
    fig_path = "./3d-object-recognition/figures/sparse-seen-" +str(step)
    exists_and_create(fig_path)
    exists_and_create(out_dir)
    out_dir = os.path.join(out_dir, subdir)
    in_dir = os.path.join(in_dir, subdir)
    exists_and_create(out_dir)

    for fname in os.listdir(in_dir):
        filename = os.path.join(in_dir, fname)
        grid = np.zeros((insize, insize, insize))

        with open(filename, "rb") as f:
            small_grid = np.reshape(np.load(f), (insize, insize, insize))
            for i in range(0, insize,step):
                for j in range(0,insize,step):
                    for k in range(insize-1,-1,-1):
                        if small_grid[i,j,k] == 1:
                            grid[i,j,k] = small_grid[i,j,k]
                            break # stop at first z

        filename = os.path.join(out_dir, fname)
        with open(filename, "wb") as f:
            np.save(f, grid)

        if "_0" in fname:
            # sanity check
            s = insize
            xs = []
            ys = []
            zs = []
            for i in range(0, s):
                for j in range(0, s):
                    for k in range(0, s):
                        if grid[i,j,k] == 1:
                            xs.append(i)
                            ys.append(j)
                            zs.append(k)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xs=xs, ys=ys, zs=zs, c=[0.1,0.1,0.1,0.2], marker='o')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.set_xlim3d(0, s)
            ax.set_ylim3d(0, s)
            ax.set_zlim3d(0, s)

            # plt.show() 
            plt.savefig(os.path.join(fig_path, fname.split(".")[0] + ".png"))   
            plt.close()


def convert_model_net(modelnet_path, set_path, outpath):
    outpath = os.path.join(outpath, set_path)
    for fdir in os.listdir(modelnet_path):
        for fname in os.listdir(os.path.join(modelnet_path, fdir, set_path)):
            new_name = os.path.join(outpath, os.path.basename(fname))
            new_name = new_name.replace("_", "-")
            shutil.copy(os.path.join(modelnet_path, fdir, set_path, fname), new_name)


def create_and_save_density(files, in_dir, out_dir, grid_size, normalize):
    for fname in files:
        in_file = os.path.join(in_dir, fname)
        out_file = os.path.join(out_dir, os.path.basename(fname) + ".npy")

        occ = dl.load_xyz_as_density(in_file, grid_size=grid_size, normalize=normalize)
        with open(out_file, "wb") as f:
            np.save(f, occ)


def create_density_set(in_dir, out_dir, grid_size=32, normalize=False):
    fnames = os.listdir(in_dir)
    count_per_thread = int(np.floor(len(fnames) / 8))
    f1 = fnames[0:count_per_thread]
    f2 = fnames[1*count_per_thread:2*count_per_thread]
    f3 = fnames[2*count_per_thread:3*count_per_thread]
    f4 = fnames[3*count_per_thread:4*count_per_thread]
    f5 = fnames[4*count_per_thread:5*count_per_thread]
    f6 = fnames[5*count_per_thread:6*count_per_thread]
    f7 = fnames[6*count_per_thread:7*count_per_thread]
    f8 = fnames[7*count_per_thread:len(fnames)]    

    p = Pool(8)
    p.map(partial(create_and_save_density, in_dir=in_dir, out_dir=out_dir, normalize=normalize, grid_size=grid_size), [f1,f2,f3,f4,f5,f6,f7,f8])
    p.close()
    p.join()


def create_and_save_mean(files, in_dir, out_dir, grid_size, normalize):
    for fname in files:
        in_file = os.path.join(in_dir, fname)
        out_file = os.path.join(out_dir, os.path.basename(fname) + ".npy")

        occ = dl.load_xyz_as_mean(in_file, grid_size=grid_size, normalize=normalize)
        with open(out_file, "wb") as f:
            np.save(f, occ)


def create_mean_set(in_dir, out_dir, grid_size=32, normalize=False):
    fnames = os.listdir(in_dir)
    count_per_thread = int(np.floor(len(fnames) / 8))
    f1 = fnames[0:count_per_thread]
    f2 = fnames[1*count_per_thread:2*count_per_thread]
    f3 = fnames[2*count_per_thread:3*count_per_thread]
    f4 = fnames[3*count_per_thread:4*count_per_thread]
    f5 = fnames[4*count_per_thread:5*count_per_thread]
    f6 = fnames[5*count_per_thread:6*count_per_thread]
    f7 = fnames[6*count_per_thread:7*count_per_thread]
    f8 = fnames[7*count_per_thread:len(fnames)]    

    p = Pool(8)
    p.map(partial(create_and_save_mean, in_dir=in_dir, out_dir=out_dir, normalize=normalize, grid_size=grid_size), [f1,f2,f3,f4,f5,f6,f7,f8])
    p.close()
    p.join()        


def create_and_save_var(files, in_dir, out_dir, grid_size, normalize):
    for fname in files:
        in_file = os.path.join(in_dir, fname)
        out_file = os.path.join(out_dir, os.path.basename(fname) + ".npy")

        occ = dl.load_xyz_as_variance(in_file, grid_size=grid_size, normalize=normalize)
        with open(out_file, "wb") as f:
            np.save(f, occ)


def create_var_set(in_dir, out_dir, grid_size=32, normalize=False):
    fnames = os.listdir(in_dir)
    count_per_thread = int(np.floor(len(fnames) / 8))
    f1 = fnames[0:count_per_thread]
    f2 = fnames[1*count_per_thread:2*count_per_thread]
    f3 = fnames[2*count_per_thread:3*count_per_thread]
    f4 = fnames[3*count_per_thread:4*count_per_thread]
    f5 = fnames[4*count_per_thread:5*count_per_thread]
    f6 = fnames[5*count_per_thread:6*count_per_thread]
    f7 = fnames[6*count_per_thread:7*count_per_thread]
    f8 = fnames[7*count_per_thread:len(fnames)]    

    p = Pool(8)
    p.map(partial(create_and_save_var, in_dir=in_dir, out_dir=out_dir, normalize=normalize, grid_size=grid_size), [f1,f2,f3,f4,f5,f6,f7,f8])
    p.close()
    p.join() 


if __name__ == '__main__':
    # convert_model_net("./3d-object-recognition/ModelNet-data/ModelNet10", "train", "./3d-object-recognition/ModelNet-data/data-out")
    # convert_model_net("./3d-object-recognition/ModelNet-data/ModelNet10", "test", "./3d-object-recognition/ModelNet-data/data-out")

    # create_segmentation_set("E:/janovrom/Engine/test-data-out", "./3d-object-recognition/Engine-data-32/test", grid_size=32)
    # create_segmentation_set("E:/janovrom/Engine/train-data-out", "./3d-object-recognition/Engine-data-32/train", grid_size=32)
    # create_segmentation_set("E:/janovrom/Engine/dev-data-out", "./3d-object-recognition/Engine-data-32/dev", grid_size=32)

    # create_density_set("D:/janovrom/Data/test-data-out", "./3d-object-recognition/ModelNet-data-density/test", grid_size=32)
    # create_density_set("D:/janovrom/Data/train-data-out", "./3d-object-recognition/ModelNet-data-density/train", grid_size=32)

    # create_var_set("D:/janovrom/ModelNet/test-data-out", "./3d-object-recognition/ModelNet-data-var/test", grid_size=32, normalize=True)
    # create_var_set("D:/janovrom/ModelNet/train-data-out", "./3d-object-recognition/ModelNet-data-var/train", grid_size=32, normalize=True)

    # create_dataset()
    # convert_scale_dataset("./3d-object-recognition/data-16/", "./3d-object-recognition/data-32-scaled-16/", 16, 32, "train")
    # rename_set_files("./3d-object-recognition/data-32-plus-scaled/test", 0)
    # convert_scaled_dataset_to_translation("./3d-object-recognition/data-16/", "./3d-object-recognition/data-32-scaled-16-translated/", 16, 32, "train")
    # get_visible_set("./3d-object-recognition/data-small/", "./3d-object-recognition/data-32-seen/", 32, "test")

    # get_visible_set_sparse("./3d-object-recognition/data-small/", "./3d-object-recognition/data-32-sparse-seen/", 32, "test", 5)
    # sanity check on saved data
    dl.load_xyzl("E:\\janovrom\\Python\\3d-object-recognition\\SegData\\gauc-0.xyzl")   