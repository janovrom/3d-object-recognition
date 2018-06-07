
import numpy as np 
from nn_template import *
import data_loader as dl


class Segmentations(dataset_template):
    
    label_dict = {
        "BACKGROUND"    : 0,
        "CHAIR"         : 1,
        "DESK"          : 2,
        "COUCH"         : 3,
        "TABLE"         : 4,
        "WALL"          : 5,
        "FLOOR"         : 6,
        "WOOD"          : 7,
        "NONE"          : 8     
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


    def __init__(self, datapath, batch_size=1, ishape=[16,16,16,1],n_classes=256):
        super().__init__(datapath, batch_size, ishape, n_classes)
        # declare datasets
        self.train = {}
        self.test = {}
        self.dev = {}
        self.train_d = {}
        self.test_d = {}
        self.dev_d = {}
        # initialize datasets
        self.__load_dataset(os.path.join(datapath, "train"), self.train)
        self.__load_dataset(os.path.join(datapath, "test"), self.test)
        self.__load_dataset(os.path.join(datapath, "dev"), self.dev)
        self.__load_dataset(os.path.join(datapath, "d/train"), self.train_d)
        self.__load_dataset(os.path.join(datapath, "d/test"), self.test_d)
        self.__load_dataset(os.path.join(datapath, "d/dev"), self.dev_d)


    def next_mini_batch(self, dataset):
        start = dataset[dataset_template.CURRENT_BATCH] * self.batch_size

        fname = dataset[dataset_template.DATASET][start]
        filename = os.path.join(dataset[dataset_template.PATH], fname)
        occ, lab, mask, idxs, deconv_labels = dl.load_xyzl_oct(filename, self.num_classes)

        dataset[dataset_template.CURRENT_BATCH] += 1
        return occ, lab, mask, idxs, filename, np.reshape(deconv_labels, [1,128,128])


    @staticmethod
    def label_to_name(label):
        for key,val in Segmentations.label_dict.items():
            if val == label:
                return key

        return ""
