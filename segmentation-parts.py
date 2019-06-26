import tensorflow as tf 
import numpy as np 
import json
import os
import shutil
from nn_template import *
from shapenetparts import Parts
import sys
import time
import matplotlib.pyplot as plt


class PartsNet():

    def create_placeholders(self, n_y, n_seg):
        X = tf.placeholder(dtype=tf.float32, shape=(None,self.dataset.shape[0],self.dataset.shape[1],self.dataset.shape[2],1), name="input_grid")
        X_vol = tf.placeholder(dtype=tf.float32, shape=(None,self.dataset.shape[0],self.dataset.shape[1],self.dataset.shape[2],1), name="input_grid_vol")
        weight = tf.placeholder(dtype=tf.float32, shape=(None), name="loss_weights")
        Y_seg = tf.placeholder(dtype=tf.int32, shape=(None), name="segmentation_labels")
        Y_cat = tf.placeholder(dtype=tf.int32, shape=(None), name="category_labels")
        keep_prob = tf.placeholder(dtype=tf.float32, name="keep_probability")
        bn_training = tf.placeholder(dtype=tf.bool, name="batch_norm_training")
        pos_mask =  tf.placeholder(dtype=tf.float32, shape=(None, n_seg), name="loss_weights")

        return X, Y_seg, Y_cat, keep_prob, bn_training, weight, pos_mask, X_vol

    
    def convolution(self, X, shape, strides=[1,1,1,1,1], padding="SAME", act=tf.nn.relu):
        # tf.contrib.layers.xavier_initializer(uniform=False)
        W = tf.get_variable("weights" + str(self.layer_idx), shape, initializer=tf.variance_scaling_initializer(), dtype=tf.float32, regularizer=tf.contrib.layers.l2_regularizer(self.lr_dec))
        # T = tf.get_variable("biastransform" + str(self.layer_idx), [X.shape[1], X.shape[2], X.shape[3]], initializer=tf.variance_scaling_initializer(), dtype=tf.float32, regularizer=tf.contrib.layers.l2_regularizer(self.lr_dec))
        b = tf.get_variable("biases" + str(self.layer_idx), shape[-1], initializer=tf.zeros_initializer(), dtype=tf.float32, regularizer=tf.contrib.layers.l2_regularizer(self.lr_dec))
        Z = tf.nn.conv3d(X, W, strides=strides, padding=padding) + b #T * b

        tf.summary.histogram("weights" + str(self.layer_idx), W)
        tf.summary.histogram("biases" + str(self.layer_idx), b)

        if act != None:
            Z = act(Z)

        self.layer_idx = self.layer_idx + 1
        return Z


    def tensor_shape(self, tensor):
        return [-1, tensor.shape[1], tensor.shape[2], tensor.shape[3], tensor.shape[4]]


    def block(self, X, in_size, out_size, keep_prob, bn_training, s3=3, s5=5):
        A0_2 = self.convolution(X, [s3,s3,s3,in_size,out_size], padding="SAME", act=tf.nn.leaky_relu) 
        D0_2 = tf.nn.dropout(A0_2, keep_prob)
        D0_2 = tf.layers.batch_normalization(D0_2, training=bn_training)

        A0_2 = self.convolution(D0_2, [s3,s3,s3,out_size,out_size], padding="SAME", act=tf.nn.leaky_relu) 
        D0_2 = tf.nn.dropout(A0_2, keep_prob)
        D0_2 = tf.layers.batch_normalization(D0_2, training=bn_training)

        A0 = self.convolution(D0_2, [s3,s3,s3,out_size,out_size], padding="SAME", act=tf.nn.leaky_relu) 
        A0 = tf.nn.dropout(A0, keep_prob)
        A0 = tf.layers.batch_normalization(A0, training=bn_training)
        A0 = tf.contrib.layers.maxout(A0, out_size, axis=-1)

        return A0


    def forward_propagation(self, X, n_cat, n_seg, keep_prob, bn_training):
        self.layer_idx = 0
        # imagine that the net operates over 32x32x32 blocks
        # feature vector learning
        # IN 32
        # first block
        A0 = self.block(X, 1, 32, keep_prob, bn_training)     
        M0 = self.convolution(A0, [2,2,2,32,64], strides=[1,2,2,2,1], padding="VALID", act=tf.nn.leaky_relu)     
        # M0 = tf.nn.max_pool3d(A0, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID") # to 16

        # second block
        A1 = self.block(M0, 64, 64, keep_prob, bn_training)  
        M1 = self.convolution(A1, [2,2,2,64,128], strides=[1,2,2,2,1], padding="VALID", act=tf.nn.leaky_relu)   
        # M1 = tf.nn.max_pool3d(A1, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID") # to 8

        # third block
        A2 = self.block(M1, 128, 128, keep_prob, bn_training) 
        M2 = self.convolution(A2, [2,2,2,128,256], strides=[1,2,2,2,1], padding="VALID", act=tf.nn.leaky_relu)   
        # M2 = tf.nn.max_pool3d(A2, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID") # to 4

        D3 = tf.contrib.layers.maxout(tf.reshape(M2, [-1,1,1,1,4*4*4*256]), 512, axis=-1)

        # up segmentation
        U_t = tf.tile(D3, [1,8,8,8,1])
        U_concat = tf.concat([A2, U_t], axis=-1)
        U1 = self.convolution(U_concat, [3,3,3,512+128,256], padding="SAME", act=tf.nn.leaky_relu)
        U1 = tf.layers.batch_normalization(U1, training=bn_training)        
        U1 = self.convolution(tf.concat([U1, M1], axis=-1), [3,3,3,256+128,256], padding="SAME", act=tf.nn.leaky_relu)
        U1 = tf.layers.batch_normalization(U1, training=bn_training)    
        
        U2 = tf.keras.layers.UpSampling3D([2,2,2])(U1) # to 16
        U_concat1 = tf.concat([A1, U2], axis=-1)
        U2 = self.convolution(U_concat1, [3,3,3,256+64,128], padding="SAME", act=tf.nn.leaky_relu)
        U2 = tf.layers.batch_normalization(U2, training=bn_training)        
        U2 = self.convolution(tf.concat([U2, M0], axis=-1), [3,3,3,128+64,128], padding="SAME", act=tf.nn.leaky_relu)
        U2 = tf.layers.batch_normalization(U2, training=bn_training)  

        U3 = tf.keras.layers.UpSampling3D([2,2,2])(U2) # to 32
        U_concat2 = tf.concat([A0, U3], axis=-1)
        U_mask = self.convolution(U_concat2, [3,3,3,128+32,64], padding="SAME", act=tf.nn.leaky_relu)
        U_mask = tf.layers.batch_normalization(U_mask, training=bn_training)  
        U_mask = self.convolution(tf.concat([U_mask, self.convolution(X, [3,3,3,1,16], padding="SAME")], axis=-1), [3,3,3,64+16,64], padding="SAME", act=tf.nn.leaky_relu)
        U_mask = tf.layers.batch_normalization(U_mask, training=bn_training) 

        # segmentation prediction      
        U_mask = self.convolution(U_mask, [1,1,1,64,n_seg], padding="SAME", act=None)
        mask = self.convolution(D3, [1,1,1,512,256], padding="VALID", act=tf.nn.leaky_relu)
        mask = tf.nn.dropout(mask, keep_prob)
        mask = tf.layers.batch_normalization(mask, training=bn_training)
        mask = self.convolution(mask, [1,1,1,256,128], padding="VALID", act=tf.nn.leaky_relu)
        mask = tf.nn.dropout(mask, keep_prob)
        mask = tf.layers.batch_normalization(mask, training=bn_training)
        mask = self.convolution(mask, [1,1,1,128,n_seg], padding="VALID", act=None)
        
        # category prediction
        A4 = self.convolution(D3, [1,1,1,512,256], padding="VALID", act=tf.nn.leaky_relu)
        A4 = tf.nn.dropout(A4, keep_prob)
        A4 = tf.layers.batch_normalization(A4, training=bn_training)
        seg_sum = tf.reduce_sum(U_mask, axis=3)
        seg_sum = tf.reduce_sum(seg_sum, axis=2)
        seg_sum = tf.reduce_sum(seg_sum, axis=1)
        print(seg_sum.shape)
        seg_sum = tf.reshape(seg_sum, [-1,1,1,1,n_seg])
        A_cat = self.convolution(tf.concat([A4,seg_sum], axis=-1), [1,1,1,256+n_seg,128], padding="VALID", act=tf.nn.leaky_relu)
        A_cat = tf.nn.dropout(A_cat, keep_prob)
        A_cat = tf.layers.batch_normalization(A_cat, training=bn_training)
        # A_cat = self.convolution(A4, [1,1,1,256,256], padding="VALID")
        A_cat = self.convolution(A_cat, [1,1,1,128,n_cat], padding="VALID", act=None)
        A_fv = tf.reshape(A_cat, [-1, n_cat])
        A_class = tf.argmax(tf.nn.softmax(A_fv), axis=-1)
        print(A_class.shape)

        # U_mask = U_mask * tf.nn.sigmoid(mask)
        U_class = tf.argmax(tf.nn.softmax(U_mask), axis=-1)

        return A_fv, A_class, U_mask, U_class, tf.reshape(mask, [-1,n_seg])


    def compute_cost(self, U, Y_seg, X, A, Y_cat, n_seg, weights, pos_mask, res_mask):
        U = U * X
        
        soft_U = tf.nn.softmax(U)
        Xrep = tf.reshape(X, [-1, X.shape[1], X.shape[2], X.shape[3]])
        entropy_cat = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_cat, logits=A) # used
        weighted_entropy_seg = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(Y_seg, n_seg) * X, logits=U) * Xrep
        # weight it by prediction 
        # weighted_entropy_seg = weighted_entropy_seg * (2 - tf.reduce_max(tf.one_hot(Y_seg, n_seg) * soft_U, axis=-1))
        # weighted_entropy_seg = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(Y_seg, n_seg) * X, logits=U, weights=(1 - 0.75*tf.pow(weights,8)))
        print(weighted_entropy_seg)
        weighted_entropy_seg = tf.reduce_sum(weighted_entropy_seg, axis=-1)
        weighted_entropy_seg = tf.reduce_sum(weighted_entropy_seg, axis=-1)
        weighted_entropy_seg = tf.reduce_sum(weighted_entropy_seg, axis=-1)
        weighted_entropy_seg = weighted_entropy_seg
        
        mask_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=pos_mask, logits=res_mask)

        acc = tf.cast(tf.equal(tf.argmax(soft_U, axis=-1, output_type=tf.int32), Y_seg), tf.float32) * Xrep
        print(acc)
        acc = tf.reduce_sum(acc, axis=-1)
        acc = tf.reduce_sum(acc, axis=-1)
        acc = tf.reduce_sum(acc, axis=-1)
        xsum = tf.reduce_sum(Xrep, axis=-1)
        xsum = tf.reduce_sum(xsum, axis=-1)
        xsum = tf.reduce_sum(xsum, axis=-1)

        acc = acc / xsum

        reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
        seg_loss = tf.reduce_sum(weighted_entropy_seg)
        cat_loss = tf.reduce_sum(entropy_cat)
        mask_loss = tf.reduce_sum(mask_entropy)
        c = reg_loss + seg_loss + cat_loss + mask_loss
        return c, acc



    def run_model(self, load=False, train=True,visualize=True, in_memory=True, interpolate=True, augment=True, save_train=False, save_dev=False, apply_mask=False):
        log_dir = os.path.join("", self.name)
        tf.reset_default_graph()
        n_cat = self.dataset.num_classes
        n_seg = self.dataset.num_classes_parts
        print(n_cat)
        print(n_seg)
        # get variables and placeholders
        step = tf.Variable(0, trainable=False, name="global_step")
        self.lr_dec = tf.maximum(1e-5, tf.train.exponential_decay(self.lr, step, self.decay_step, self.decay_rate, staircase=True))
        X, Y_seg, Y_cat, keep_prob, bn_training, weight, pos_mask, X_vol = self.create_placeholders(n_cat, n_seg)
        
        # get model
        A_fv, A_class, U_mask, U_class, res_mask = self.forward_propagation(X_vol, n_cat, n_seg, keep_prob, bn_training)
        U_vec = tf.nn.softmax(U_mask)
        cost,acc_op = self.compute_cost(U_mask, Y_seg, X, A_fv, Y_cat, n_seg, weight, pos_mask, res_mask)

        # declare optimizations
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_dec)
        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_ops):
            gvs = optimizer.compute_gradients(cost)
            # print(gvs)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs, global_step=step)
            # train_op = optimizer.minimize(cost)

        init = tf.global_variables_initializer()
        tf.summary.scalar("learning_rate", self.lr_dec)
        tf.summary.scalar("global_step", step)
        tf.summary.scalar("cost", cost)
        writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())
        summary_op = tf.summary.merge_all()
        writer.flush()
            

        def accuracy_test(dataset, data_dict, in_memory=True):
            acc = 0
            acc_cat = 0
            avg_time = 0
            dataset.restart_mini_batches(data_dict)
            dataset.clear_segmentation(data_dict, in_memory=in_memory)
            for i in range(dataset.num_mini_batches(data_dict)):
                stime = time.time()
                occ,seg,cat,names,points,lbs,wgs,msk,pos_msk,vol = self.dataset.next_mini_batch(data_dict)
                # deconvolved_images,d_cost = sess.run([U_class,cost], feed_dict={X: occ, Y_seg: seg, Y_cat: cat, keep_prob: 1.0, bn_training: False, weight: 1.0})
                deconvolved_images,d_cost,pred_class,seg_vec,feat_vec = sess.run([U_class,cost,A_class,U_vec,U_mask], feed_dict={X: occ, Y_seg: seg, Y_cat: cat, keep_prob: 1.0, bn_training: False, weight: wgs, pos_mask: msk, X_vol: vol})

                # mask based on class prediction
                if apply_mask:
                    for j in range(0, seg_vec.shape[0]):
                        mask = np.zeros(n_seg)
                        cat_dir = Parts.Labels[str(pred_class[j])]
                        s = Parts.Labels[cat_dir][Parts.PART_START]
                        e = s + Parts.Labels[cat_dir][Parts.PART_COUNT]
                        mask[s:e] = 1.0
                        deconvolved_images[j] = np.argmax(seg_vec[j] * mask, axis=-1)

                xresh = np.reshape(occ, [-1, occ.shape[1], occ.shape[2], occ.shape[3]])
                a = 1.0 - (np.sum((xresh * deconvolved_images) != seg)) / np.sum(xresh)
                predicted_category = pred_class
                avg_time += (time.time() - stime) / occ.shape[0]
                # print("Average interference time per mini batch example %f sec" % ((time.time() - stime) / occ.shape[0]))
                acc = acc + a
                acc_cat = acc_cat + np.sum(cat == predicted_category) / predicted_category.shape[0]
                
                for j in range(0, deconvolved_images.shape[0]):
                    if interpolate:
                        dataset.save_segmentation(lbs[j], seg_vec[j], names[j], points[j], data_dict, in_memory=in_memory, interpolate=interpolate)
                    else:
                        # dataset.save_segmentation(lbs[j], seg[j], names[j], points[j], data_dict, in_memory=in_memory, interpolate=interpolate)
                        dataset.save_segmentation(lbs[j], deconvolved_images[j], names[j], points[j], data_dict, in_memory=in_memory, interpolate=interpolate)
 
                if visualize:
                    for j in range(0, deconvolved_images.shape[0]):
                        print(names[j])
                        dataset.vizualise_batch(seg[j],deconvolved_images[j],cat[j],predicted_category[j],xresh[j],names[j])

                print("\rEvaluating %s: %d %%..." % (data_dict["name"], i*100 / dataset.num_mini_batches(data_dict)), end="")

            print("\r%s deconvolution average accuracy %f" % (data_dict["name"], acc / dataset.num_mini_batches(data_dict)))
            print("%s category accuracy %f" % (data_dict["name"], acc_cat / dataset.num_mini_batches(data_dict)))
            return float(acc / dataset.num_mini_batches(data_dict)), avg_time / dataset.num_mini_batches(data_dict)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:

            if load:
                tf.train.Saver().restore(sess, os.path.join(log_dir, self.name + ".ckpt"))
                print("Model loaded from file: %s" % os.path.join(log_dir, self.name + ".ckpt"))
            else:
                sess.run(init)
            
            trainable_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
            print("Model %s has %d trainable parameters." % (self.name, trainable_params))

            if train:
                costs = []
                train_accuracies = []
                dev_accuracies = []
                test_accuracies = []
                wious = []
                wious_dev = []
                wious_test = []
                train_times = []
                wupdate_times = []
                train_infer_time = []
                dev_infer_time = []
                test_infer_time = []

                out_header = "Training times/epoch (ms), Weight update time/epoch (ms)"
                if save_train and dataset_template.NUMBER_BATCHES in self.dataset.train:
                    out_header += ",Train accuracy,Train weighted average IOU,Train inference times (ms)"
                if save_dev and dataset_template.NUMBER_BATCHES in self.dataset.dev:
                    out_header += ",Dev accuracy,Dev weighted average IOU,Dev inference times (ms)"
                if dataset_template.NUMBER_BATCHES in self.dataset.test:
                    out_header += ",Test accuracy,Test weighted average IOU,Test inference times (ms)"

                restarts = [True] * 100
                for epoch in range(0, self.num_epochs):
                    self.dataset.restart_mini_batches(self.dataset.train, train=restarts[epoch])
                    batches = self.dataset.num_mini_batches(self.dataset.train)
                    # evaluate the scene batch
                    cc = 0
                    min_wgs = 1.0
                    max_wgs = 0.0
                    stime = time.time()
                    for i in range(batches):
                        # always train on unscaled data
                        occ,seg,cat,names,_,_,wgs,msk,pos_msk,vol = self.dataset.next_mini_batch(self.dataset.train,augment=augment)
                        wgs = (wgs - min_wgs) / (max_wgs - min_wgs) # normalize in range (0,1)
                        wgs = 1.0 - wgs # best results should have least priority
                        summary,_,d_cost = sess.run([summary_op,train_op,cost], feed_dict={X: occ, Y_cat: cat, Y_seg: seg, keep_prob: self.keep_prob, bn_training: True, weight: wgs, pos_mask: msk, X_vol: vol})
                        cc = cc + d_cost
                        print("\rBatch learning %05d/%d" % (i+1,batches),end="")


                    print("")
                    train_times.append(time.time() - stime)
                    self.dataset.restart_mini_batches(self.dataset.train)
                    stime = time.time()                    
                    batches = self.dataset.num_mini_batches(self.dataset.train)
                    # for i in range(batches):
                    #     # occ,seg,cat,names,_,_ = self.dataset.next_mini_batch_augmented(self.dataset.train)
                    #     occ,seg,cat,_,_,_,wgs = self.dataset.next_mini_batch(self.dataset.train, update=False)
                    #     out = sess.run([acc_op], feed_dict={X: occ, Y_cat: cat, Y_seg: seg, keep_prob: 1, bn_training: False, weight: wgs})
                    #     min_wgs = min(min_wgs, np.min(wgs))
                    #     max_wgs = max(max_wgs, np.max(wgs))
                    #     self.dataset.update_mini_batch(self.dataset.train, out[0])
                    #     min_wgs = min(min_wgs, np.min(wgs))
                    #     print("\rUpdate weights %05d/%d" % (i+1,batches),end="")
                    
                    writer.add_summary(summary, epoch)
                    cc = cc / (self.dataset.num_examples(self.dataset.train))
                    costs.append(cc)
                    t = time.time() - stime
                    wupdate_times.append(t)
                    print("\nEpoch %d trained in %f, cost %f" % (epoch+1, t+train_times[-1], cc))                    
                    
                    # save model
                    save_path = tf.train.Saver().save(sess, os.path.join(log_dir, self.name + str(epoch+1) + ".ckpt"))
                    print("Model saved in file: %s" % save_path)
                        
                    # plt.figure(1)
                    plt.clf()
                    plt.plot(np.squeeze(np.array(costs)))
                    plt.savefig("./ShapeNet/cost.png", format="png")

                    # evaluation of existing sets
                    def evaluate(data_dict, suffix, infer_time, accuracies, wious):
                        acc, avg_time = accuracy_test(self.dataset, data_dict)
                        infer_time.append(avg_time)
                        accuracies.append(acc)
                        plt.clf()
                        # plt.figure(2)
                        plt.plot(np.squeeze(np.array(accuracies)))
                        plt.savefig("./ShapeNet/accuracies" + suffix + ".png", format="png")

                        weighted_average_iou, per_category_iou = self.evaluate_iou_results(data_dict) # has to be after the accuracy_test, so it has saved and current values
                        wious.append(weighted_average_iou)
                        # plt.figure(3)
                        plt.clf()
                        plt.plot(np.squeeze(np.array(wious)))
                        plt.savefig("./ShapeNet/weighted_average_iou" + suffix + ".png", format="png")

                        # plt.figure(4)
                        plt.clf()
                        plt.barh(np.arange(n_cat),per_category_iou, tick_label=list(Parts.label_dict.keys()))
                        # plt.barh(np.arange(n_cat),per_category_iou, tick_label=["airplane", "bag", "cap", "car", "chair", "earphone", "guitar", "knife", "lamp", "laptop", "motorbike", "mug", "pistol", "rocket", "skateboard", "table"])
                        plt.savefig("./ShapeNet/per_category_iou" + str(epoch+1) + suffix + ".png", format="png")


                    if save_train and dataset_template.NUMBER_BATCHES in self.dataset.train:
                        evaluate(self.dataset.train, "_train", train_infer_time, train_accuracies, wious)

                    if save_dev and dataset_template.NUMBER_BATCHES in self.dataset.dev:
                        evaluate(self.dataset.dev, "_dev", dev_infer_time, dev_accuracies, wious_dev)
                    
                    if dataset_template.NUMBER_BATCHES in self.dataset.test:
                        evaluate(self.dataset.test, "_test", test_infer_time, test_accuracies, wious_test)

                    if save_train:
                        print("Max wIOU train %f" % np.max(wious))
                    if save_dev:
                        print("Max wIOU dev %f" % np.max(wious_dev))
                    
                    print("Max wIOU test %f" % np.max(wious_test))

                    print("")
                    # do check for file barrier, if so, break training cycle
                    if os.path.exists(os.path.join(log_dir, "barrier.txt")):
                        break


                array2csv = []
                array2csv.append(train_times)
                array2csv.append(wupdate_times)
                if save_train and dataset_template.NUMBER_BATCHES in self.dataset.train:
                    array2csv.append(train_accuracies)
                    array2csv.append(wious)
                    array2csv.append(train_infer_time)

                if save_dev and dataset_template.NUMBER_BATCHES in self.dataset.dev:
                    array2csv.append(dev_accuracies)
                    array2csv.append(wious_dev)
                    array2csv.append(dev_infer_time)

                if dataset_template.NUMBER_BATCHES in self.dataset.test:
                    array2csv.append(test_accuracies)
                    array2csv.append(wious_test)
                    array2csv.append(test_infer_time)

                array2csv = np.array(array2csv) # reshape so the values are in the rows
                array2csv = array2csv.transpose()
                np.savetxt("./ShapeNet/Results/" + self.name + ".csv", array2csv, delimiter=",", comments='', header=out_header)
                shutil.copy("./ShapeNet/network.json", "./ShapeNet/Results/network.json")
                shutil.copy("./segmentation-parts.py", "./" + self.name + "/Results/segmentation-parts.py")
                shutil.copy("./shapenetparts.py", "./" + self.name + "/Results/shapenetparts.py")
                
            else:
                if dataset_template.NUMBER_BATCHES in self.dataset.test:
                    accuracy_test(self.dataset, self.dataset.test, in_memory=in_memory)
                    self.evaluate_iou_results(self.dataset.test, in_memory=in_memory)
                else:
                    accuracy_test(self.dataset, self.dataset.dev, in_memory=in_memory)
                    self.evaluate_iou_results(self.dataset.dev, in_memory=in_memory)
            #     print("Evaluate on train dataset")
            #     accuracy_test(self.dataset, self.dataset.train)
            #     self.evaluate_iou_results(self.dataset.train)
            

            # acc_train = accuracy_test(self.dataset, self.dataset.train)
            # acc_test = accuracy_test(self.dataset, self.dataset.test)
            # acc_dev = accuracy_test(self.dataset, self.dataset.dev)
            
            # print("Train/Dev/Test accuracies %f/%f/%f" %(acc_train, acc_dev, acc_test))
            # plt.show()


    def evaluate_iou_results(self, data_dict={ "name" : "train"}, in_memory=True):
        return self.dataset.evaluate_iou_results(data_dict, in_memory=in_memory)



    def __init__(self, model_name, datapath):
        assert os.path.exists(os.path.join(".", model_name, "network.json"))

        with open(os.path.join(".", model_name, "network.json"), "r") as f:
            jparams = json.load(f)["params"]
            self.name =         model_name
            self.lr =           jparams["learning_rate"]
            self.lrd =          jparams["learning_rate_deconv"]
            self.num_epochs =   jparams["num_epochs"]
            self.l2_reg_w =     jparams["l2_reg_weights"]
            self.l2_reg_b =     jparams["l2_reg_biases"] 
            self.decay_step =   jparams["decay_step"] 
            self.decay_rate =   jparams["decay_rate"] 
            self.min_prob =     jparams["min_prob"] 
            self.keep_prob =    jparams["keep_prob"] 
            self.dataset =      Parts(datapath, batch_size=jparams["batch_size"], ishape=jparams["input_shape"], oshape=jparams["output_shape"])


if __name__ == "__main__":
    s = PartsNet("ShapeNet", "./ShapePartsData/dataset")
    s.run_model(load=False, train=True,visualize=False, in_memory=True,interpolate=False,augment=True, apply_mask=False)

