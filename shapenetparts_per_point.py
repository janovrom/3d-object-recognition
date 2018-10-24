import convert_data as convert
import numpy as np 
from nn_template import *
import data_loader as dl


class Parts(dataset_template):
    
    label_dict = {
        "airplane"      : 0,
        "bag"           : 1,
        "cap"           : 2,
        "car"           : 3,
        "chair"         : 4,
        "earphone"      : 5,
        "guitar"        : 6,
        "knife"         : 7,
        "lamp"          : 8 ,
        "laptop"        : 9,
        "motorbike"     : 10,
        "mug"           : 11,
        "pistol"        : 12,
        "rocket"        : 13,
        "skateboard"    : 14,
        "table"         : 15
    }
    
    ORDER = "order"
    CATEGORY_NAME = "name"
    CATEGORY = "id"
    PART_COUNT = "part_count"
    PART_START = "part_start"
    IOU_GROUND_TRUTH = "iou_ground_truths"
    IOU_PREDICTION = "iou_predictions"
    Labels = {}

    def __load_dataset(self, path, data_dict, load):
        if os.path.exists(path):
            data_dict[dataset_template.PATH] = path
            cat_idx = 0
            num_parts = 0
            data_dict[dataset_template.DATASET] = []
            dict_name = data_dict["name"]
            # list all category directories in test/dev/train dataset
            for cat_dir in os.listdir(os.path.join(path, dict_name + "_data")):
                stime = time.time()
                if cat_dir not in Parts.Labels:
                    Parts.Labels[cat_dir] = {}
                    Parts.Labels[cat_dir][Parts.CATEGORY] = cat_idx
                    Parts.Labels[cat_dir][Parts.CATEGORY_NAME] = cat_dir
                    Parts.Labels[cat_dir][Parts.PART_COUNT] = 0
                    Parts.Labels[cat_dir][Parts.PART_START] = num_parts
                    Parts.Labels[str(cat_idx)] = cat_dir
                    cat_idx += 1

                if load:
                    # get all files in label and data directories
                    cat_files = os.path.join(path, dict_name + "_data", cat_dir)
                    cat_labels = os.path.join(path, dict_name + "_label", cat_dir)
                    iteration = 0
                    lst_cat_files = os.listdir(cat_files)
                    lst_cat_labels = os.listdir(cat_labels)
                    for pts,lab in zip(lst_cat_files, lst_cat_labels):
                        print("\rLoading category %s: %d %%" % (cat_dir, int(iteration/len(lst_cat_files)*100)), end='', flush=True)
                        points,labels,part_count,cloud = dl.load_binvox_kdtree(os.path.join(cat_files,pts),os.path.join(cat_labels,lab),label_start=num_parts,k=self.k,m=self.m)
                        Parts.Labels[cat_dir][Parts.PART_COUNT] = max(Parts.Labels[cat_dir][Parts.PART_COUNT], part_count)
                        data_dict[dataset_template.DATASET].append((points,labels,Parts.Labels[cat_dir][Parts.CATEGORY],cat_dir+"-"+pts,cloud))
                        iteration += 1
                
                num_parts += Parts.Labels[cat_dir][Parts.PART_COUNT]
                print("\rCategory %s loaded in %f sec" % (cat_dir, time.time() - stime))

            self.num_classes_parts = max(num_parts, self.num_classes_parts)
            self.num_classes = max(cat_idx, self.num_classes)
            data_dict[dataset_template.DATASET] = np.array(data_dict[dataset_template.DATASET])
            data_dict[dataset_template.NUMBER_EXAMPLES] = data_dict[dataset_template.DATASET].shape[0]
            data_dict[Parts.ORDER] = np.random.permutation(data_dict[dataset_template.NUMBER_EXAMPLES])
            data_dict[dataset_template.CURRENT_BATCH] = 0
            data_dict[dataset_template.NUMBER_BATCHES] = int(data_dict[dataset_template.NUMBER_EXAMPLES] / self.batch_size) + (0 if data_dict[dataset_template.NUMBER_EXAMPLES] % self.batch_size == 0 else 1)
            print("Dataset %s has %d examples." %(path, data_dict[dataset_template.NUMBER_EXAMPLES]))


    def __init__(self, datapath, batch_size=1, k=8,n_classes=256, load=True, m=50):
        super().__init__(datapath, batch_size, [k,3], n_classes)
        # declare datasets
        self.k = k
        self.m = m
        self.num_classes_parts = 0
        self.num_classes = 0
        self.train = { "name": "train" }
        self.test = { "name": "test" }
        self.dev = { "name": "dev" }
        # initialize datasets
        stime = time.time()
        self.dataset_path = datapath
        self.__load_dataset(os.path.join(datapath, "train"), self.train, load)
        self.__load_dataset(os.path.join(datapath, "test"), self.test, load)
        self.__load_dataset(os.path.join(datapath, "dev"), self.dev, load)
        print("Datasets train/dev/test loaded in %f seconds." % (time.time() - stime))
        

    def restart_mini_batches(self, dataset):
        dataset[Parts.ORDER] = np.random.permutation(dataset[dataset_template.NUMBER_EXAMPLES])
        dataset[dataset_template.CURRENT_BATCH] = 0


    def next_mini_batch(self, dataset):
        dat = []
        seg = []
        cat = []
        nam = []
        pts = []
        start = dataset[dataset_template.CURRENT_BATCH] * self.batch_size
        end = min(dataset[dataset_template.NUMBER_EXAMPLES], start + self.batch_size)

        for data_idx in dataset[Parts.ORDER][start:end]:
            data = dataset[dataset_template.DATASET][data_idx]

            dat.append(data[0])
            seg.append(data[1])
            cat.append(data[2])
            nam.append(data[3])
            pts.append(data[4])

        dataset[dataset_template.CURRENT_BATCH] += 1
        return np.array(dat),np.array(seg),np.array(cat),np.array(nam),np.array(pts)


    def clear_segmentation(self, data_dict, in_memory=True):
        if in_memory:
            for i in range(0,self.num_classes):
                parts = Parts.Labels[Parts.Labels[str(i)]]
                parts[Parts.IOU_GROUND_TRUTH] = []
                parts[Parts.IOU_PREDICTION] = []
        else:
            paths = [os.path.join(self.dataset_path, "segmentation/gt", data_dict["name"]), os.path.join(self.dataset_path, "segmentation/res", data_dict["name"])]
            for path in paths:
                for directory in os.listdir(path):
                    directory = os.path.join(path,directory)
                    for seg_file in os.listdir(directory):
                        os.remove(os.path.join(directory,seg_file))


    def save_segmentation_disc(self, segmentation_gt_pts, segmentation_res_pts, name, data_dict):
        name_split = name.split("-")
        cat_dir = name_split[0]
        fname = name_split[1].split(".")[0]
        path_gt = os.path.join(self.dataset_path, "segmentation/gt", data_dict["name"], cat_dir)
        path_res = os.path.join(self.dataset_path, "segmentation/res", data_dict["name"], cat_dir)
        if not os.path.exists(path_gt):
            os.mkdir(path_gt)
            os.chmod(path_gt, 0o777)

        if not os.path.exists(path_res):
            os.mkdir(path_res)
            os.chmod(path_res, 0o777)            

        with open(os.path.join(path_gt,fname + ".gt"), "wb") as f:
            np.save(f, segmentation_gt_pts)

        with open(os.path.join(path_res, fname + ".res"), "wb") as f:
            np.save(f, segmentation_res_pts)


    def save_segmentation_mem(self, segmentation_gt_pts, segmentation_res_pts, name, data_dict):
        name_split = name.split("-")
        cat_dir = name_split[0]
        parts = Parts.Labels[cat_dir]
        parts[Parts.IOU_GROUND_TRUTH].append(segmentation_gt_pts)
        parts[Parts.IOU_PREDICTION].append(segmentation_res_pts)
        

    def save_segmentation(self, segmentation_gt_pts, segmentation_res, name, data_dict, in_memory=True):
        if in_memory:
            return self.save_segmentation_mem(segmentation_gt_pts, segmentation_res, name, data_dict)
        else:
            return self.save_segmentation_disc(segmentation_gt_pts, segmentation_res, name, data_dict)


    def evaluate_iou_results(self, data_dict, in_memory=True):
        if in_memory:
            return self.evaluate_iou_results_mem(data_dict)
        else:
            return self.evaluate_iou_results_disc(data_dict)


    def evaluate_iou_results_mem(self, data_dict):
        print("Evaluating " + data_dict["name"])
        ncategory = self.num_classes
        nmodels = np.zeros(ncategory)
        iou_all = np.zeros(ncategory)
        eps = 0.0000001
        for i in range(0, ncategory):
            parts = Parts.Labels[Parts.Labels[str(i)]]
            l = len(parts[Parts.IOU_GROUND_TRUTH])
            nmodels[i] = l
            nparts = parts[self.PART_COUNT]
            iou_per_part = np.zeros((l,nparts))
            for k in range(0,l):
                for j in range(parts[self.PART_START],parts[self.PART_START]+nparts):
                    union = np.sum(np.logical_or(parts[Parts.IOU_PREDICTION][k]==j,parts[Parts.IOU_GROUND_TRUTH][k]==j))
                    if union < eps:
                        iou_per_part[k,j-parts[self.PART_START]] = 1.0
                    else:
                        iou_per_part[k,j-parts[self.PART_START]] = np.sum(np.logical_and(parts[Parts.IOU_PREDICTION][k]==j,parts[Parts.IOU_GROUND_TRUTH][k]==j))/union

            iou_all[i] = np.mean(iou_per_part)
            print("\rCategory %s has %d parts and IoU %f" % (Parts.Labels[str(i)],nparts,iou_all[i]))


        iou_weighted_ave = np.sum(np.multiply(iou_all,nmodels))/np.sum(nmodels)

        print("Weighted average IOU is %f" % iou_weighted_ave)

        return iou_weighted_ave,iou_all

    
    def evaluate_iou_results_disc(self, data_dict):
        print("Evaluating " + data_dict["name"])
        path = os.path.join(self.dataset_path, "segmentation")

        path_gt = os.path.join(path, "gt", data_dict["name"])
        path_res = os.path.join(path, "res", data_dict["name"])

        files_gt = os.listdir(path_gt)
        files_res = os.listdir(path_res)

        ncategory = len(files_gt)
        nmodels = np.zeros(ncategory)
        iou_all = np.zeros(ncategory)
        eps = 0.0000001
        for i in range(0, ncategory):
            print("Evaluating category %s" % Parts.Labels[str(i)],end="", flush=True)
            category = files_gt[i]
            path_gt_cat = os.path.join(path_gt, category)
            path_res_cat = os.path.join(path_res, category)

            files_gt_cat = os.listdir(path_gt_cat)
            files_res_cat = os.listdir(path_res_cat)

            # load all
            predictions = []
            ground_truths = []
            nparts = 0
            nparts_min = 50
            nparts_max = 0

            for res,gt in zip(files_res_cat,files_gt_cat):
                with open(os.path.join(path_res_cat,res), "rb") as f:
                    predictions.append(np.load(f))

                with open(os.path.join(path_gt_cat,gt), "rb") as f:
                    g = np.load(f)
                    nparts_min = min(np.min(g),nparts_min)
                    nparts_max = max(np.max(g),nparts_max)
                    ground_truths.append(g)

            nparts = nparts_max - nparts_min + 1
            # nparts = nparts + 1
            # predictions = np.array(predictions)
            # ground_truths = np.array(ground_truths)

            nmodels[i] = len(files_res_cat)

            iou_per_part = np.zeros((len(files_gt_cat),nparts))
            for k in range(0,len(files_gt_cat)):
                for j in range(0,nparts):
                    union = np.sum(np.logical_or(predictions[k]==(j+nparts_min),ground_truths[k]==(j+nparts_min)))
                    if union < eps:
                        iou_per_part[k,j] = 1.0
                    else:
                        iou_per_part[k,j] = np.sum(np.logical_and(predictions[k]==(j+nparts_min),ground_truths[k]==(j+nparts_min)))/union

            iou_all[i] = np.mean(iou_per_part)
            print("\rCategory %s has %d parts and IoU %f" % (Parts.Labels[str(i)],nparts,iou_all[i]))


        iou_weighted_ave = np.sum(np.multiply(iou_all,nmodels))/np.sum(nmodels)

        print("Weighted average IOU is %f" % iou_weighted_ave)

        return iou_weighted_ave,iou_all


# if __name__ == "__main__":
#     dataset = Parts("./3d-object-recognition/UnityData", load=False)
#     dataset.evaluate_iou_results(dataset.train)