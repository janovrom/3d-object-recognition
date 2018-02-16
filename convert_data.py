import random
import threading
import os
import math
import numpy as np
from timeit import default_timer as timer
from numba import vectorize
from numba import guvectorize 
from numba import float32, int32
from numba import jit, cuda
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import data_loader as dl



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
    size = list(occ_out.shape)
    size[0] -= 1
    size[1] -= 1
    size[2] -= 1
    for i in range(0,tris.shape[1], 3):
        # compute aabb of triangle
        t = tris[:, i:i+3]

        # normal = np.cross(t[:,1] - t[:,0], t[:,2] - t[:,0])
        # if normal[2] <= 0:
        #     continue

        t_min = np.min(t, axis=1) - mins
        t_max = np.max(t, axis=1) - mins
        # compute absolute position in the grid
        t_min = ((t_min * size) / grid_size).astype(int)
        t_max = ((t_max) * size / grid_size).astype(int) + 1
        for i in range(t_min[0], t_max[0]):
            for j in range(t_min[1], t_max[1]):
                for k in range(t_min[2], t_max[2]):
                    p = (np.array([i, j, k])) * voxel_size + mins
                    if udTriangle(p, t[:,0], t[:,1], t[:,2]) < voxel_size[0]:
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
    nps = dl.load_off_file("./3d-object-recognition/objects/off/torus.off")
    nps = rotatePoints(nps, eulerToMatrix([45,45,45]))
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

