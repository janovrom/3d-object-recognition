import sys
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.colors import LinearSegmentedColormap


def display_stimuli(stimuli, size):
    occ = stimuli
    xs = []
    ys = []
    zs = []
    for i in range(0, size):
        for j in range(0, size):
            for k in range(0, size):
                if occ[i,j,k] == 1:
                    xs.append(i)
                    ys.append(j)
                    zs.append(k)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=xs, ys=ys, zs=zs, c=[0.1,0.1,0.1,0.2], marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim3d(0, size)
    ax.set_ylim3d(0, size)
    ax.set_zlim3d(0, size)

    plt.show()     


def add_to_plot(ax, filter):
    xs = []
    ys = []
    zs = []
    vs = []
    xs.append(0)
    ys.append(0)
    zs.append(0)
    vs.append(1)
    xs.append(0)
    ys.append(0)
    zs.append(0)
    vs.append(0)
    for x in range(0,filter.shape[0]):
        for y in range(0,filter.shape[1]):
            for z in range(0,filter.shape[2]):
                    xs.append(x + 0.5)
                    ys.append(y + 0.5)
                    zs.append(z + 0.5)
                    vs.append(filter[x,y,z])
    

    cm = LinearSegmentedColormap.from_list("alpha", [(0.0,0.0,0.0,0.0), (1.0,0.0,0.0,1.0)])
    # ax.scatter(xs, ys, zs, color=vs, marker='o')
    ax.scatter(xs, ys, zs, vs, c=vs, cmap=cm, marker='8')
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    ax.set_xlim3d(0, filter.shape[0])
    ax.set_ylim3d(0, filter.shape[1])
    ax.set_zlim3d(0, filter.shape[2])


def conv3d_plot(filters, ignore_input=True):
    print(filters.shape)
    num_filters = filters.shape[-1]
    # filter_size = filters.shape[0]
    rows = np.floor(np.sqrt(num_filters))
    g_x = np.ceil(num_filters / rows)
    fig = plt.figure()
    zero_coutn = 0
    min_in_units = np.min(filters[0,:,:,:,:])
    filters = filters - min_in_units
    max_in_units = np.max(filters[0,:,:,:,:])
    filters = filters / max_in_units
    if max_in_units < 0.00001: 
        print("There is zero as max in filter.")
    for i in range(0, num_filters):
        ax = fig.add_subplot(rows, g_x, i+1, projection='3d')
        filter = filters[0,:,:,:,i]
        # filter = filter - np.min(filter)
        # if np.max(filter) < 0.0001: # test if all are zeros
        #     zero_coutn += 1
        # filter = filter / np.max(filter)
        # continue
        print(filter)
        add_to_plot(ax, filter)

    print("%d from %d filters has all zeros." % (zero_coutn, num_filters))
    plt.show()     
    # get specific filter
    while not ignore_input:
        print("Get specific unit:")
        input_line = sys.stdin.readline().split("\n")[0]

        if input_line == "":
            break
        else:
            idx = int(input_line)
            filter = filters[0,:,:,:,idx]
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            add_to_plot(ax, filter)
            plt.show()


def convert_to_one_hot(labels, max_label):
    one_hot = np.eye(max_label)[labels]
    return one_hot

def save_tensorboard_sprite(data, path, labels):  
    to_visualise = data
    # to_visualise = vector_to_matrix_mnist(to_visualise)
    # to_visualise = invert_grayscale(to_visualise)

    sprite_image = create_sprite_image(to_visualise)

    with open(os.path.join(path, "metadata.tsv"),'w') as f:
        f.write("Index\tLabel\n")
        for index,label in enumerate(labels):
            f.write("%d\t%d\n" % (index,label))

    plt.imsave(os.path.join(path, "sprites.png"), sprite_image, cmap='gray')
    plt.imshow(sprite_image, cmap='gray')
    plt.show()


def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    
    spriteimage = np.ones((img_h * n_plots , img_w * n_plots ))
    
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = rgb2gray(images[this_filter])
                spriteimage[i * img_h:(i + 1) * img_h, j * img_w:(j + 1) * img_w] = this_img
    
    return spriteimage


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def vector_to_matrix_mnist(mnist_digits):
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(mnist_digits,(-1,28,28))


def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 1-mnist_digits


if "__main__" == __name__:
    conv3d_plot()