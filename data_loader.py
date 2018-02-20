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