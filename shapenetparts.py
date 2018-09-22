import convert_data as convert
import numpy as np 
from nn_template import *
import data_loader as dl


class Parts(dataset_template):
    
    # label_dict = {
    #     "room" : 0
    # }
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
                        occ,seg,part_count,cloud,labels = dl.load_binvox(os.path.join(cat_files,pts),os.path.join(cat_labels,lab),label_start=num_parts,grid_size=self.shape[0],ogrid_size=self.oshape[0])
                        Parts.Labels[cat_dir][Parts.PART_COUNT] = max(Parts.Labels[cat_dir][Parts.PART_COUNT], part_count)
                        data_dict[dataset_template.DATASET].append((np.reshape(occ, self.shape),seg,Parts.Labels[cat_dir][Parts.CATEGORY],cat_dir+"-"+pts,cloud,labels,0))
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
            # counts = np.zeros(50)
            # for i in range(0,data_dict[dataset_template.NUMBER_EXAMPLES]):
            #     data = data_dict[dataset_template.DATASET][i]
            #     for x in range(0, 32):
            #         for y in range(0, 32):
            #             for z in range(0, 32):
            #                 if data[0][x, y, z] == 1:
            #                     counts[data[1][x,y,z]] += 1

            # print(counts)


    def __init__(self, datapath, batch_size=1, ishape=[16,16,16,1],n_classes=256, load=True, oshape=[16,16,16]):
        super().__init__(datapath, batch_size, ishape, n_classes)
        # declare datasets
        self.oshape = oshape
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


    def next_mini_batch(self, dataset, update=True):
        occ = []
        seg = []
        cat = []
        nam = []
        pts = []
        lbs = []
        acc = []
        start = dataset[dataset_template.CURRENT_BATCH] * self.batch_size
        end = min(dataset[dataset_template.NUMBER_EXAMPLES], start + self.batch_size)

        for data_idx in dataset[Parts.ORDER][start:end]:
            data = dataset[dataset_template.DATASET][data_idx]

            occ.append(data[0])
            seg.append(data[1])
            cat.append(data[2])
            nam.append(data[3])
            pts.append(data[4])
            lbs.append(data[5])
            acc.append(data[6])

        if update:
            dataset[dataset_template.CURRENT_BATCH] += 1

        return np.array(occ),np.array(seg),np.array(cat),np.array(nam),np.array(pts),np.array(lbs),np.array(acc)


    def update_mini_batch(self, dataset, new_accs, alpha=0.35):
        start = dataset[dataset_template.CURRENT_BATCH] * self.batch_size
        end = min(dataset[dataset_template.NUMBER_EXAMPLES], start + self.batch_size)

        idx = 0
        for data_idx in dataset[Parts.ORDER][start:end]:
            dataset[dataset_template.DATASET][data_idx][6] = alpha * new_accs[idx] + (1 - alpha) * dataset[dataset_template.DATASET][data_idx][6]
            idx += 1

        dataset[dataset_template.CURRENT_BATCH] += 1


    def next_mini_batch_augmented(self, dataset):
        occ = []
        seg = []
        cat = []
        nam = []
        pts = []
        lbs = []
        start = dataset[dataset_template.CURRENT_BATCH] * self.batch_size
        end = min(dataset[dataset_template.NUMBER_EXAMPLES], start + self.batch_size)

        for data_idx in dataset[Parts.ORDER][start:end]:
            data = dataset[dataset_template.DATASET][data_idx]

            # append original
            occ.append(data[0])
            seg.append(data[1])
            cat.append(data[2])
            nam.append(data[3])
            pts.append(data[4])
            lbs.append(data[5])
            # add augmentation
            for rot_iter in range(0,2):
                orig_shape = data[4].shape
                nps = np.reshape(np.copy(data[4]), [3,-1])
                nps = convert.rotatePoints(nps, convert.eulerToMatrix(np.random.rand(3) * 10.0))
                nps = np.reshape(nps, orig_shape)
                occupancy_grid, label_grid, _, _, _ = dl.load_binvox_np(nps, data[5])
                occ.append(np.reshape(occupancy_grid, self.shape))
                seg.append(label_grid)
                cat.append(data[2])
                nam.append(data[3])
                pts.append(data[4])
                lbs.append(data[5])

        dataset[dataset_template.CURRENT_BATCH] += 1
        return np.array(occ),np.array(seg),np.array(cat),np.array(nam),np.array(pts),np.array(lbs)


    # @staticmethod
    # def label_to_name(label):
    #     for key,val in Segmentations.label_dict.items():
    #         if val == label:
    #             return key

    #     return ""


    def vizualise_batch(self, segmentation_gt, segmentation_res, category_gt, category_res, occupancy, name):
        xs = []
        ys = []
        zs = []
        vs = []
        colors = [(1,0,0), (0,1,0), (1,0.5,0), (1,1,0)]
        if category_res == category_gt:
            offset = 0
        else:
            offset = 2

        for x in range(0, self.shape[0]):
            for y in range(0, self.shape[1]):
                for z in range(0, self.shape[2]):
                    if occupancy[x,y,z] == 1:
                        # red for miss
                        # orange for miss with category miss
                        # green for correct
                        # yellow for correct with category miss
                        xs.append(x)
                        ys.append(y)
                        zs.append(z)

                        if segmentation_gt[x,y,z] == segmentation_res[x,y,z]:
                            vs.append(colors[1+offset])
                        else:
                            vs.append(colors[0+offset])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        m = plt.get_cmap("Set1")
        plt.title(name)
        ax.scatter(xs, ys, zs, c=vs, marker='p')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(0,self.shape[0]-1)
        ax.set_ylim(0,self.shape[1]-1)
        ax.set_zlim(0,self.shape[2]-1)
        plt.show()


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


    def save_segmentation_disc(self, segmentation_gt_pts, segmentation_res, name, points, data_dict):
        name_split = name.split("-")
        cat_dir = name_split[0]
        fname = name_split[1].split(".")[0]
        segmentation_gt_pts = segmentation_gt_pts
        segmentation_res = segmentation_res
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
            pointcloud = np.array(points)
            num_points = int(pointcloud.shape[0])
            segmentation_res_pts = []
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
            max_size = np.max([size_x, size_y, size_z]) / 2
            cx = size_x / 2 + min_x
            cy = size_y / 2 + min_y
            cz = size_z / 2 + min_z
            extent = int(self.shape[0] / 2)

            for i in range(0,num_points,3):
                x = pointcloud[i+0]
                y = pointcloud[i+1]
                z = pointcloud[i+2]
                idx_x = int(((x - cx) * extent / max_size + extent) * (self.oshape[0] - 1) / (extent * 2))
                idx_y = int(((y - cy) * extent / max_size + extent) * (self.oshape[0] - 1) / (extent * 2))
                idx_z = int(((z - cz) * extent / max_size + extent) * (self.oshape[0] - 1) / (extent * 2))
                # idx_x = int(((x - cx) * extent / size_x * 2.0 + extent) * (self.shape[0] - 1) / (extent * 2))
                # idx_y = int(((y - cy) * extent / size_y * 2.0 + extent) * (self.shape[0] - 1) / (extent * 2))
                # idx_z = int(((z - cz) * extent / size_z * 2.0 + extent) * (self.shape[0] - 1) / (extent * 2))
                segmentation_res_pts.append(segmentation_res[idx_x,idx_y,idx_z])

            segmentation_res_pts = np.array(segmentation_res_pts)
            np.save(f, segmentation_res_pts)


    def save_segmentation_mem(self, segmentation_gt_pts, segmentation_res, name, points, data_dict, interpolate=False):
        name_split = name.split("-")
        cat_dir = name_split[0]
        parts = Parts.Labels[cat_dir]
        parts[Parts.IOU_GROUND_TRUTH].append(segmentation_gt_pts)
        # convert point cloud and grid results to point cloud results
        pointcloud = np.array(points)
        num_points = int(pointcloud.shape[0])
        segmentation_res_pts = []
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
        max_size = np.max([size_x, size_y, size_z]) / 2
        cx = size_x / 2 + min_x
        cy = size_y / 2 + min_y
        cz = size_z / 2 + min_z
        extent = int(self.shape[0] / 2)

        if interpolate:
            point_grid = np.array(np.zeros((self.oshape[0],self.oshape[0],self.oshape[0])), dtype=np.object)
            mean_grid = np.array(np.zeros((self.oshape[0],self.oshape[0],self.oshape[0], 3)), dtype=np.float)

        for i in range(0,num_points,3):
            x = pointcloud[i+0]
            y = pointcloud[i+1]
            z = pointcloud[i+2]
            idx_x = int(((x - cx) * extent / max_size + extent) * (self.oshape[0] - 1) / (extent * 2))
            idx_y = int(((y - cy) * extent / max_size + extent) * (self.oshape[0] - 1) / (extent * 2))
            idx_z = int(((z - cz) * extent / max_size + extent) * (self.oshape[0] - 1) / (extent * 2))
            if interpolate:
                if not point_grid[idx_x, idx_y, idx_z]:
                    point_grid[idx_x, idx_y, idx_z] = []

                point_grid[idx_x, idx_y, idx_z].append((x,y,z))
            else:
                segmentation_res_pts.append(segmentation_res[idx_x,idx_y,idx_z])

        if interpolate:
            for x in range(0, self.oshape[0]):
                for y in range(0, self.oshape[0]):
                    for z in range(0, self.oshape[0]):
                        if point_grid[x,y,z]:
                            for i in range(0, len(point_grid[x,y,z])):
                                p = point_grid[x,y,z][i]
                                mean_grid[x,y,z,0] += p[0] 
                                mean_grid[x,y,z,1] += p[1] 
                                mean_grid[x,y,z,2] += p[2] 

                            mean_grid[x,y,z,0] /= len(point_grid[x,y,z])
                            mean_grid[x,y,z,1] /= len(point_grid[x,y,z])
                            mean_grid[x,y,z,2] /= len(point_grid[x,y,z])

            for i in range(0,num_points,3):
                px = pointcloud[i+0]
                py = pointcloud[i+1]
                pz = pointcloud[i+2]
                idx_x = int(((px - cx) * extent / max_size + extent) * (self.oshape[0] - 1) / (extent * 2))
                idx_y = int(((py - cy) * extent / max_size + extent) * (self.oshape[0] - 1) / (extent * 2))
                idx_z = int(((pz - cz) * extent / max_size + extent) * (self.oshape[0] - 1) / (extent * 2))
                closest = [0,0,0]
                closest_dist = 10000000
                for x in range(-1,1,1):
                    for y in range(-1,1,1):
                        for z in range(-1,1,1):
                            if not (idx_x + x < 0 or idx_x + x >= self.oshape[0] or idx_y + y < 0 or idx_y + y >= self.oshape[0] or idx_z + z < 0 or idx_z + z >= self.oshape[0]):
                                if mean_grid[idx_x+x,idx_y+y,idx_z+z,0] != 0 and mean_grid[idx_x+x,idx_y+y,idx_z+z,1] != 0 and mean_grid[idx_x+x,idx_y+y,idx_z+z,2] != 0:
                                    dx = mean_grid[idx_x+x,idx_y+y,idx_z+z,0] - px
                                    dy = mean_grid[idx_x+x,idx_y+y,idx_z+z,1] - py
                                    dz = mean_grid[idx_x+x,idx_y+y,idx_z+z,2] - pz

                                    d = np.sqrt(dx**2 + dy**2 + dz**2)
                                    if d < closest_dist:
                                        closest_dist = d
                                        closest = [idx_x+x,idx_y+y,idx_z+z]

                segmentation_res_pts.append(segmentation_res[closest[0], closest[1], closest[2]])
                            


        segmentation_res_pts = np.array(segmentation_res_pts)
        parts[Parts.IOU_PREDICTION].append(segmentation_res_pts)
        

    def save_segmentation(self, segmentation_gt_pts, segmentation_res, name, points, data_dict, in_memory=True):
        if in_memory:
            return self.save_segmentation_mem(segmentation_gt_pts, segmentation_res, name, points, data_dict)
        else:
            return self.save_segmentation_disc(segmentation_gt_pts, segmentation_res, name, points, data_dict)


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