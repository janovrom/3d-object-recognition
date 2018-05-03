
import numpy as np 
from nn_template import *


class Segmentations(dataset_template):

    label_dict = {
        "empty"             : 0,
        "crank"             : 1,
        "crank-bushing"     : 2,
        "crank-shaft"       : 3,
        "cylinder"          : 4,
        "cylinder-pivot"    : 5,
        "flywheel"          : 6,
        "flywheel-cap"      : 7,
        "frame-bushings"    : 8,
        "hose-barb"         : 9,
        "lower-frame"       : 10,
        "piston"            : 11,
        "spring-retainer"   : 12,
        "upper-frame"       : 13        
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
        # initialize datasets
        self.__load_dataset(os.path.join(datapath, "train"), self.train)
        self.__load_dataset(os.path.join(datapath, "test"), self.test)
        self.__load_dataset(os.path.join(datapath, "dev"), self.dev)


    def get_data(self, dataset, name):
        data = []
        labels = []
        for fname in dataset[dataset_template.DATASET]:
            if name in fname:
                print("Searched for %s, found %s." % (name, fname))
                filename = os.path.join(dataset[dataset_template.PATH], fname)
                with open(filename, "rb") as f:
                    data.append(np.reshape(np.load(f), self.shape))
                    labels.append(int(self.label_dict[os.path.basename(fname).split("_")[0]]))
                break

        if len(data) == 0:
            print("No such name found: "+name)
        return np.array(data), np.array(labels)


    def next_mini_batch(self, dataset):
        data = []
        labels = []
        start = dataset[dataset_template.CURRENT_BATCH] * self.batch_size
        end = min(dataset[dataset_template.NUMBER_EXAMPLES], start + self.batch_size)

        for fname in dataset[dataset_template.DATASET][start:end]:
            filename = os.path.join(dataset[dataset_template.PATH], fname)
            with open(filename, "rb") as f:
                occ = np.load(f)
                data.append(np.reshape(occ, [occ.shape[0], occ.shape[1], occ.shape[2], 1]))
                labels.append(int(self.label_dict[os.path.basename(fname).split("_")[0]]))

        dataset[dataset_template.CURRENT_BATCH] += 1
        return np.array(data), np.array(labels)


    @staticmethod
    def label_to_name(label):
        for key,val in Segmentations.label_dict.items():
            if val == label:
                return key

        return ""
