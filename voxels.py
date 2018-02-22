import numpy as np 
from nn_template import *


class Voxels(dataset_template):

    label_dict = {
        "cube"      : 0,
        "cone"      : 1,
        "sphere"    : 2,
        "torus"     : 3,
        "cylinder"  : 4
    }

    def __load_dataset(self, path, data_dict):
        if os.path.exists(path):
            data_dict[dataset_template.PATH] = path
            data_dict[dataset_template.DATASET] = np.array(os.listdir(path))
            data_dict[dataset_template.NUMBER_EXAMPLES] = data_dict[dataset_template.DATASET].shape[0]
            data_dict[dataset_template.DATASET] = data_dict[dataset_template.DATASET][np.random.permutation(data_dict[dataset_template.NUMBER_EXAMPLES])]
            data_dict[dataset_template.CURRENT_BATCH] = 0
            data_dict[dataset_template.NUMBER_BATCHES] = int(data_dict[dataset_template.NUMBER_EXAMPLES] / self.batch_size) + (0 if data_dict[dataset_template.NUMBER_EXAMPLES] % self.batch_size == 0 else 1)
            print("Dataset %s has %d examples." %(path, data_dict[dataset_template.NUMBER_EXAMPLES]))


    def __init__(self, datapath, batch_size=8, ishape=[16,16,16,1],n_classes=256):
        super().__init__(datapath, batch_size, ishape, n_classes)
        # declare datasets
        self.train = {}
        self.test = {}
        self.dev = {}
        # initialize datasets
        self.__load_dataset(os.path.join(datapath, "train"), self.train)
        self.__load_dataset(os.path.join(datapath, "test"), self.test)
        self.__load_dataset(os.path.join(datapath, "dev"), self.dev)


    def load_test_scene(self, idx):
        data = []
        labels = []
        occupancy = []
        points = []

        for fname in self.test[dataset_template.DATASET]:
            if fname.startswith("scene_" + str(idx)):
                if fname.endswith("occupancy.npy"): # load occupancy
                    filename = os.path.join(self.test[dataset_template.PATH], fname)
                    with open(filename, "rb") as f:
                        occupancy.append(np.reshape(np.load(f), self.shape))
                elif fname.endswith("points.npy"): # load points
                    filename = os.path.join(self.test[dataset_template.PATH], fname)
                    with open(filename, "rb") as f:
                        points.append(np.reshape(np.load(f), (-1,3)))
                else: # load tdf
                    filename = os.path.join(self.test[dataset_template.PATH], fname)
                    with open(filename, "rb") as f:
                        data.append(np.reshape(np.load(f), self.shape))
                        labels.append(os.path.basename(fname))

        return np.array(data), np.array(labels), np.array(occupancy), np.array(points)


    def next_mini_batch(self, dataset):
        data = []
        labels = []
        start = dataset[dataset_template.CURRENT_BATCH] * self.batch_size
        end = min(dataset[dataset_template.NUMBER_EXAMPLES], start + self.batch_size)

        for fname in dataset[dataset_template.DATASET][start:end]:
            filename = os.path.join(dataset[dataset_template.PATH], fname)
            with open(filename, "rb") as f:
                data.append(np.reshape(np.load(f), self.shape))
                labels.append(int(self.label_dict[os.path.basename(fname).split("_")[0]]))

        dataset[dataset_template.CURRENT_BATCH] += 1
        return np.array(data), np.array(labels)

