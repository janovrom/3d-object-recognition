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
        "lamp"          : 8,
        "laptop"        : 9,
        "motorbike"     : 10,
        "mug"           : 11,
        "pistol"        : 12,
        "rocket"        : 13,
        "skateboard"    : 14,
        "table"         : 15
    }

    label_weights = {
        "airplane"      : 1,
        "bag"           : 1,
        "cap"           : 1,
        "car"           : 1,
        "chair"         : 1,
        "earphone"      : 1,
        "guitar"        : 1,
        "knife"         : 1,
        "lamp"          : 1,
        "laptop"        : 1,
        "motorbike"     : 1,
        "mug"           : 1,
        "pistol"        : 1,
        "rocket"        : 1,
        "skateboard"    : 1,
        "table"         : 1
    }
    
    ORDER = "order"
    CATEGORY_NAME = "name"
    CATEGORY = "id"
    PART_COUNT = "part_count"
    PART_START = "part_start"
    IOU_GROUND_TRUTH = "iou_ground_truths"
    IOU_PREDICTION = "iou_predictions"
    NUM_SMALLEST_CLASS = "NUM_SMALLEST_CLASS"
    NUM_LARGEST_CLASS = "NUM_LARGEST_CLASS"
    DATASET_DIRS = "DATASET_DIRS"
    DATASET_WEIGHTS = "DATASET_WEIGHTS"
    Labels = {}

    def __load_dataset(self, path, data_dict, load):
        if os.path.exists(path):
            data_dict[dataset_template.PATH] = path
            cat_idx = 0
            num_parts = 0
            data_dict[dataset_template.DATASET] = []
            data_dict[Parts.DATASET_DIRS] = []
            data_dict[Parts.DATASET_WEIGHTS] = []
            dict_name = data_dict["name"]
            smallest_class = 100000000
            largest_class = 0
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
                        data_dict[dataset_template.DATASET].append((np.reshape(occ, self.shape),seg,Parts.Labels[cat_dir][Parts.CATEGORY],cat_dir+"-"+pts,cloud,labels))
                        data_dict[Parts.DATASET_DIRS].append(cat_dir)
                        data_dict[Parts.DATASET_WEIGHTS].append(Parts.label_weights[cat_dir])
                        iteration += 1

                    if iteration > 0:
                        smallest_class = min(smallest_class, iteration)
                        largest_class = max(largest_class, iteration)
                
                num_parts += Parts.Labels[cat_dir][Parts.PART_COUNT]
                print("\rCategory %s loaded in %f sec" % (cat_dir, time.time() - stime))

            self.num_classes_parts = max(num_parts, self.num_classes_parts)
            self.num_classes = max(cat_idx, self.num_classes)
            data_dict[Parts.NUM_SMALLEST_CLASS] = smallest_class
            data_dict[Parts.NUM_LARGEST_CLASS] = largest_class
            data_dict[dataset_template.DATASET] = np.array(data_dict[dataset_template.DATASET])
            data_dict[Parts.DATASET_DIRS] = np.array(data_dict[Parts.DATASET_DIRS])
            data_dict[Parts.DATASET_WEIGHTS] = np.array(data_dict[Parts.DATASET_WEIGHTS])
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
        

    def restart_mini_batches(self, dataset, train=False):
        if train:
            order = []
            iter_start = 0
            idxs = np.random.permutation(dataset[Parts.NUM_LARGEST_CLASS])
            for key in Parts.label_dict.keys():
                idx = dataset[Parts.DATASET_DIRS] == key # get indices for this class
                # get first element: starting index
                start = np.nonzero(idx)[0][0]
                count = np.sum(idx)
                indices = np.mod(idxs, count) + start
                order.append(indices)

            order = np.ravel(np.column_stack(order))
            # for key in Parts.label_dict.keys():
            #     idx = dataset[Parts.DATASET_DIRS] == key # get indices for this class
            #     wgs = dataset[Parts.DATASET_WEIGHTS][idx] # get weights for class
            #     data = dataset[dataset_template.DATASET][idx] # get data for class

            #     sorted_idx = wgs.argsort() # get indices for sorted weights

            #     dataset[Parts.DATASET_WEIGHTS][idx] = wgs[sorted_idx] # update
            #     dataset[dataset_template.DATASET][idx] = data[sorted_idx] 
            #     # create order for first num_smallest_class models with lowest weights
            #     order.extend(np.arange(iter_start, iter_start+dataset[Parts.NUM_SMALLEST_CLASS]))
            #     iter_start += dataset[Parts.NUM_SMALLEST_CLASS]

            dataset[Parts.ORDER] = np.array(order)
            # dataset[Parts.ORDER] = np.random.permutation(dataset[Parts.ORDER].shape[0])
            dataset[dataset_template.NUMBER_EXAMPLES] = dataset[Parts.ORDER].shape[0]
        else:
            dataset[dataset_template.NUMBER_EXAMPLES] = dataset[dataset_template.DATASET].shape[0]
            dataset[Parts.ORDER] = np.random.permutation(dataset[dataset_template.NUMBER_EXAMPLES])

        dataset[dataset_template.NUMBER_BATCHES] = int(dataset[dataset_template.NUMBER_EXAMPLES] / self.batch_size) + (0 if dataset[dataset_template.NUMBER_EXAMPLES] % self.batch_size == 0 else 1)
        dataset[dataset_template.CURRENT_BATCH] = 0


    def next_mini_batch(self, dataset, update=True, augment=False):
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
            if augment:
                scale_range = 1.0
                per_point_noise_range = 0.0
                p = np.copy(data[4])
                orig_shape = data[4].shape
                p = np.reshape(p, [-1,3]).transpose()
                p = convert.rotatePoints(p, convert.eulerToMatrix((0,np.random.randint(0,360),0))) # random rotation
                p = p.transpose()
                p = p * (1.0 + scale_range * (np.array([np.random.randint(0,500),np.random.randint(0,500),np.random.randint(0,500)]) / 500.0 - 0.5) / 5.0) # random scale in range 1 +- scale_range*0.1
                # p = p * ((np.random.rand() * 0.2 - 0.1) * per_point_noise_range + 1.0)
                # p = p.transpose()
                p = np.reshape(p, orig_shape)
                occupancy_grid,label_grid,_,_,_ = dl.load_binvox_np(p, data[5])

                occ.append(np.reshape(occupancy_grid, self.shape))
                seg.append(label_grid)
                pts.append(p)  

                # xs = []
                # ys = []
                # zs = []
                # vs = []

                # for x in range(0, self.shape[0]):
                #     for y in range(0, self.shape[1]):
                #         for z in range(0, self.shape[2]):
                #             if occ[-1][x,y,z] == 1:
                #                 xs.append(x)
                #                 ys.append(y)
                #                 zs.append(z)
                #                 vs.append(label_grid[x,y,z])


                # fig = plt.figure()
                # ax = fig.add_subplot(121, projection='3d')
                # m = plt.get_cmap("Set1")
                # ax.scatter(xs, ys, zs, c=vs, cmap=m, marker='p')
                # ax.set_xlabel('X')
                # ax.set_ylabel('Y')
                # ax.set_zlabel('Z')
                # ax.set_xlim(0,self.shape[0]-1)
                # ax.set_ylim(0,self.shape[1]-1)
                # ax.set_zlim(0,self.shape[2]-1)
                # plt.show()              
            else:
                occ.append(data[0])
                seg.append(data[1])
                pts.append(data[4])

            cat.append(data[2])
            nam.append(data[3])
            lbs.append(data[5])
            acc.append(dataset[Parts.DATASET_WEIGHTS][data_idx])

        if update:
            dataset[dataset_template.CURRENT_BATCH] += 1

        return np.array(occ),np.array(seg),np.array(cat),np.array(nam),np.array(pts),np.array(lbs),np.array(acc)


    def update_mini_batch(self, dataset, new_accs, alpha=0.05):
        start = dataset[dataset_template.CURRENT_BATCH] * self.batch_size
        end = min(dataset[dataset_template.NUMBER_EXAMPLES], start + self.batch_size)

        idx = 0
        for data_idx in dataset[Parts.ORDER][start:end]:
            dataset[Parts.DATASET_WEIGHTS][data_idx] = alpha * new_accs[idx] + (1 - alpha) * dataset[Parts.DATASET_WEIGHTS][data_idx]
            idx += 1

        dataset[dataset_template.CURRENT_BATCH] += 1


    def next_mini_batch_augmented(self, dataset):
        occ = []
        seg = []
        cat = []
        nam = []
        pts = []
        lbs = []
        wgs = []
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
            wgs.append(data[6])
            # add augmentation
            scale_range = 3.5
            per_point_noise_range = 2.0
            for rot_iter in range(0,3):
                p = np.copy(data[4])
                orig_shape = data[4].shape
                p = np.reshape(p, [3,-1])
                p = convert.rotatePoints(p, convert.eulerToMatrix((0,np.random.randint(0,360),0))) # random rotation
                p = p.transpose()
                p = p * (1.0 + scale_range * (np.array([np.random.randint(0,500),np.random.randint(0,500),np.random.randint(0,500)]) / 500.0 - 0.5) / 5.0) # random scale in range 1 +- scale_range*0.1
                p = p * ((np.random.rand() * 0.2 - 0.1) * per_point_noise_range + 1.0)
                p = p.transpose()
                p = np.reshape(p, orig_shape)
                occupancy_grid, label_grid, _, _, _ = dl.load_binvox_np(p, data[5])

                occ.append(np.reshape(occupancy_grid, self.shape))
                seg.append(label_grid)
                cat.append(data[2])
                nam.append(data[3])
                pts.append(data[4])
                lbs.append(data[5])
                wgs.append(data[6])

        dataset[dataset_template.CURRENT_BATCH] += 1
        return np.array(occ),np.array(seg),np.array(cat),np.array(nam),np.array(pts),np.array(lbs),np.array(wgs)


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
        gs = []
        colors = [(1,0,0), (0,1,0), (1,0.5,0), (1,1,0)]
        if category_res == category_gt:
            offset = 0
        else:
            offset = 2

        for x in range(0, self.shape[0]):
            for y in range(0, self.shape[1]):
                for z in range(0, self.shape[2]):
                    if occupancy[x,y,z] == 1:
                        # green for correct
                        xs.append(x)
                        ys.append(y)
                        zs.append(z)

                        if segmentation_gt[x,y,z] == segmentation_res[x,y,z]:
                            vs.append(colors[1+offset])
                        else:
                            vs.append(colors[0+offset])

                        gs.append(segmentation_gt[x,y,z])


        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        m = plt.get_cmap("Set1")
        plt.title(name)
        ax.scatter(xs, ys, zs, c=vs, marker='p')
        ax2.scatter(xs, ys, zs, c=gs, cmap=m, marker='p')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_xlim(0,self.shape[0]-1)
        ax2.set_ylim(0,self.shape[1]-1)
        ax2.set_zlim(0,self.shape[2]-1)
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

        num_parts = 50

        vsize_x = size_x / self.oshape[0]
        start_x = vsize_x * 0.5
        vsize_y = size_y / self.oshape[0]
        start_y = vsize_y * 0.5
        vsize_z = size_z / self.oshape[0]
        start_z = vsize_z * 0.5

        for i in range(0,num_points,3):
            x = pointcloud[i+0]
            y = pointcloud[i+1]
            z = pointcloud[i+2]
            idx_x = int(((x - cx) * extent / max_size + extent) * (self.oshape[0] - 1) / (extent * 2))
            idx_y = int(((y - cy) * extent / max_size + extent) * (self.oshape[0] - 1) / (extent * 2))
            idx_z = int(((z - cz) * extent / max_size + extent) * (self.oshape[0] - 1) / (extent * 2))
            counter = np.zeros(num_parts)
            if interpolate:
                for ix in range(-1,2):
                    for iy in range(-1,2):
                        for iz in range(-1,2):
                            idx = idx_x + ix
                            idy = idx_y + iy
                            idz = idx_z + iz
                            if (idx >= 0 and idy >= 0 and idz >= 0 and idx < self.oshape[0] and idy < self.oshape[0] and idz < self.oshape[0]):
                                px = start_x + vsize_x * idx
                                py = start_y + vsize_y * idy
                                pz = start_z + vsize_z * idz
                                dx = px - x
                                dy = py - y
                                dz = pz - z
                                counter += (dx**2 + dy**2 + dz**2) * segmentation_res[idx,idy,idz]


                segmentation_res_pts.append(np.argmax(counter))
            else:
                segmentation_res_pts.append(segmentation_res[idx_x,idx_y,idx_z])


        segmentation_res_pts = np.array(segmentation_res_pts)
        parts[Parts.IOU_PREDICTION].append(segmentation_res_pts)
        

    def save_segmentation(self, segmentation_gt_pts, segmentation_res, name, points, data_dict, in_memory=True, interpolate=False):
        if in_memory:
            return self.save_segmentation_mem(segmentation_gt_pts, segmentation_res, name, points, data_dict, interpolate=interpolate)
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
        misses = np.zeros((self.num_classes_parts,self.num_classes_parts))
        part_category = [""] * self.num_classes_parts
        for i in range(0, ncategory):
            parts = Parts.Labels[Parts.Labels[str(i)]]
            l = len(parts[Parts.IOU_GROUND_TRUTH])
            nmodels[i] = l
            nparts = parts[self.PART_COUNT]
            iou_per_part = np.zeros((l,nparts))
            for k in range(0,l):
                for j in range(parts[self.PART_START],parts[self.PART_START]+nparts):
                    union = np.sum(np.logical_or(parts[Parts.IOU_PREDICTION][k]==j,parts[Parts.IOU_GROUND_TRUTH][k]==j))
                    # do some prediction statistics
                    jth_part = parts[Parts.IOU_PREDICTION][k][parts[Parts.IOU_GROUND_TRUTH][k]==j]
                    bins = np.bincount(jth_part, minlength=self.num_classes_parts)
                    misses[j] += bins

                    part_category[j] = Parts.Labels[str(i)] # assign category to each part

                    if union < eps:
                        iou_per_part[k,j-parts[self.PART_START]] = 1.0
                    else:
                        iou_per_part[k,j-parts[self.PART_START]] = np.sum(np.logical_and(parts[Parts.IOU_PREDICTION][k]==j,parts[Parts.IOU_GROUND_TRUTH][k]==j))/union

            iou_all[i] = np.mean(iou_per_part)
            print("\rCategory %s has %d parts and IoU %f" % (Parts.Labels[str(i)],nparts,iou_all[i]))


        iou_weighted_ave = np.sum(np.multiply(iou_all,nmodels))/np.sum(nmodels)

        print("Weighted average IOU is %f" % iou_weighted_ave)

        # save another statistics
        np.savetxt("./3d-object-recognition/ShapeNet/misses.csv", misses, delimiter=",", comments='', header=",".join(part_category))

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