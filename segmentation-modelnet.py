import tensorflow as tf
import numpy as np
import json
import os
import shutil
from nn_utils import *
from modelnet_set import Parts
import sys
import time
import matplotlib.pyplot as plt


class Net():

    def create_placeholders(self, n_y, n_seg):
        X = tf.placeholder(dtype=tf.float32, shape=(None,self.dataset.shape[0],self.dataset.shape[1],self.dataset.shape[2],1), name="input_grid")
        weight = tf.placeholder(dtype=tf.float32, shape=(None), name="loss_weights")
        Y_cat = tf.placeholder(dtype=tf.int32, shape=(None), name="category_labels")
        keep_prob = tf.placeholder(dtype=tf.float32, name="keep_probability")
        bn_training = tf.placeholder(dtype=tf.bool, name="batch_norm_training")

        return X, Y_cat, keep_prob, bn_training, weight


    def convolution(self, X, shape, strides=[1,1,1,1,1], padding="SAME", act=tf.nn.relu):
        # tf.contrib.layers.xavier_initializer(uniform=False)
        W = tf.get_variable("weights" + str(self.layer_idx), shape, initializer=tf.variance_scaling_initializer(), dtype=tf.float32, regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_w))
        b = tf.get_variable("biases" + str(self.layer_idx), shape[-1], initializer=tf.zeros_initializer(), dtype=tf.float32, regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_b))
        Z = tf.nn.conv3d(X, W, strides=strides, padding=padding) + b

        tf.summary.histogram("weights" + str(self.layer_idx), W)
        tf.summary.histogram("biases" + str(self.layer_idx), b)

        if act != None:
            Z = act(Z)

        self.layer_idx = self.layer_idx + 1
        return Z


    def tensor_shape(self, tensor):
        return [-1, tensor.shape[1], tensor.shape[2], tensor.shape[3], tensor.shape[4]]


    def block(self, X, in_size, out_size, keep_prob, bn_training, s3=3, s5=5):
        A0_5 = self.convolution(X, [s5,s5,s5,in_size,out_size], padding="SAME")
        D0_5 = tf.nn.dropout(A0_5, keep_prob)
        D0_5 = tf.layers.batch_normalization(D0_5, training=bn_training)

        A0_3 = self.convolution(X, [s3,s3,s3,in_size,out_size], padding="SAME")
        D0_3 = tf.nn.dropout(A0_3, keep_prob)
        D0_3 = tf.layers.batch_normalization(D0_3, training=bn_training)
        A0 = tf.concat([D0_3,D0_5], axis=-1)
        M0 = tf.nn.max_pool3d(A0, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID")

        return A0, M0


    def forward_propagation(self, X, n_cat, n_seg, keep_prob, bn_training):
        self.layer_idx = 0
        # imagine that the net operates over 32x32x32 blocks
        # feature vector learning
        # IN 32
        # first block
        A0_5 = self.convolution(X, [5,5,5,1,32], padding="SAME")
        D0_5 = tf.nn.dropout(A0_5, keep_prob)
        D0_5 = tf.layers.batch_normalization(D0_5, training=bn_training)

        A0_3 = self.convolution(X, [3,3,3,1,32], padding="SAME")
        D0_3 = tf.nn.dropout(A0_3, keep_prob)
        D0_3 = tf.layers.batch_normalization(D0_3, training=bn_training)
        A0 = tf.concat([D0_3,D0_5], axis=-1)
        M0 = tf.nn.max_pool3d(A0, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID") # to 16

        # second block
        # A1_5 = self.convolution(M0, [5,5,5,64,64], padding="SAME")
        A1_5 = self.convolution(M0, [5,5,5,64,32], padding="SAME")
        D1_5 = tf.nn.dropout(A1_5, keep_prob)
        D1_5 = tf.layers.batch_normalization(D1_5, training=bn_training)

        A1_3 = self.convolution(M0, [3,3,3,64,32], padding="SAME")
        # A1_3 = self.convolution(M0, [3,3,3,64,64], padding="SAME")
        D1_3 = tf.nn.dropout(A1_3, keep_prob)
        D1_3 = tf.layers.batch_normalization(D1_3, training=bn_training)
        A1 = tf.concat([D1_3,D1_5], axis=-1)
        M1 = tf.nn.max_pool3d(A1, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID") # to 8

        # third block
        A2_5 = self.convolution(M1, [5,5,5,64,32], padding="SAME")
        # A2_5 = self.convolution(M1, [5,5,5,128,128], padding="SAME")
        D2_5 = tf.nn.dropout(A2_5, keep_prob)
        D2_5 = tf.layers.batch_normalization(D2_5, training=bn_training)

        A2_3 = self.convolution(M1, [3,3,3,64,32], padding="SAME")
        # A2_3 = self.convolution(M1, [3,3,3,128,128], padding="SAME")
        D2_3 = tf.nn.dropout(A2_3, keep_prob)
        D2_3 = tf.layers.batch_normalization(D2_3, training=bn_training)
        A2 = tf.concat([D2_3,D2_5], axis=-1)
        M2 = tf.nn.max_pool3d(A2, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID") # to 4

        # TODO try conv 3 and final max pool
        A3 = self.convolution(M2, [4,4,4,64,512], padding="VALID") # to 1 
        # A3 = self.convolution(M2, [4,4,4,256,512], padding="VALID") # to 1
        D3 = tf.nn.dropout(A3, keep_prob)
        D3 = tf.layers.batch_normalization(D3, training=bn_training)

        A4 = self.convolution(D3, [1,1,1,512,256], padding="VALID")
        A_cat = self.convolution(A4, [1,1,1,256,n_cat], padding="VALID", act=None)
        A_fv = tf.reshape(A_cat, [-1, n_cat])
        A_class = tf.argmax(tf.nn.softmax(A_fv), axis=-1)
        print(A_class.shape)

        return A_fv, A_class



    def compute_cost(self, A, Y_cat, n_seg, weights):
        entropy_cat = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_cat, logits=A) # used
        print(entropy_cat)
        # entropy_cat = (1 - 0.75 * tf.pow(weights,3)) * entropy_cat

        acc = tf.cast(tf.equal(tf.argmax(tf.nn.softmax(A), axis=-1, output_type=tf.int32), Y_cat), tf.float32)

        reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
        cat_loss = tf.reduce_sum(entropy_cat)
        c = reg_loss + cat_loss

        return c, acc



    def run_model(self, load=False, train=True,visualize=True, in_memory=True):
        log_dir = os.path.join("./3d-object-recognition", self.name)
        tf.reset_default_graph()
        n_cat = self.dataset.num_classes
        n_seg = self.dataset.num_classes_parts
        print(n_cat)
        print(n_seg)
        # get variables and placeholders
        step = tf.Variable(0, trainable=False, name="global_step")
        self.lr_dec = tf.maximum(1e-5, tf.train.exponential_decay(self.lr, step, self.decay_step, self.decay_rate, staircase=True))
        X, Y_cat, keep_prob, bn_training, weight = self.create_placeholders(n_cat, n_seg)

        # get model
        A_fv, A_class = self.forward_propagation(X, n_cat, n_seg, keep_prob, bn_training)
        cost, acc_op = self.compute_cost(A_fv, Y_cat, n_seg, weight)

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
            acc_cat = 0
            avg_time = 0
            dataset.restart_mini_batches(data_dict)
            dataset.clear_segmentation(data_dict, in_memory=in_memory)
            for i in range(dataset.num_mini_batches(data_dict)):
                stime = time.time()
                occ,cat,names,points,wgs = self.dataset.next_mini_batch(data_dict)
                # deconvolved_images,d_cost = sess.run([U_class,cost], feed_dict={X: occ, Y_seg: seg, Y_cat: cat, keep_prob: 1.0, bn_training: False, weight: 1.0})
                d_cost,pred_class = sess.run([cost,A_class], feed_dict={X: occ, Y_cat: cat, keep_prob: 1.0, bn_training: False, weight: wgs})

                predicted_category = pred_class
                avg_time += (time.time() - stime) / occ.shape[0]
                # print("Average interference time per mini batch example %f sec" % ((time.time() - stime) / occ.shape[0]))
                acc_cat = acc_cat + np.sum(cat == predicted_category) / predicted_category.shape[0]

                if visualize:
                    for j in range(0, occ.shape[0]):
                        print(names[j])
                        dataset.vizualise_batch(occ[j],occ[j],cat[j],predicted_category[j],occ[j],names[j])

                print("\rEvaluating %s: %d %%..." % (data_dict["name"], i*100 / dataset.num_mini_batches(data_dict)), end="")

            print("\r%s category accuracy %f" % (data_dict["name"], acc_cat / dataset.num_mini_batches(data_dict)))
            return float(acc_cat / dataset.num_mini_batches(data_dict)), avg_time / dataset.num_mini_batches(data_dict)

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
                train_times = []
                wupdate_times = []
                train_infer_time = []
                dev_infer_time = []

                for epoch in range(0, self.num_epochs):
                    self.dataset.restart_mini_batches(self.dataset.train)
                    batches = self.dataset.num_mini_batches(self.dataset.train)
                    # evaluate the scene batch
                    cc = 0
                    min_wgs = 0.0
                    stime = time.time()
                    for i in range(batches):
                        occ,cat,_,_,wgs = self.dataset.next_mini_batch_augmented(self.dataset.train)
                        summary,_,d_cost = sess.run([summary_op,train_op,cost], feed_dict={X: occ, Y_cat: cat, keep_prob: self.keep_prob, bn_training: True, weight: wgs - min_wgs})
                        cc = cc + d_cost
                        print("\rBatch learning %05d/%d" % (i+1,batches),end="")

                    print("")
                    train_times.append(time.time() - stime)
                    self.dataset.restart_mini_batches(self.dataset.train)
                    stime = time.time()
                    for i in range(batches):
                        # occ,seg,cat,names,_,_ = self.dataset.next_mini_batch_augmented(self.dataset.train)
                        occ,cat,_,_,wgs = self.dataset.next_mini_batch(self.dataset.train, update=False)
                        out = sess.run([acc_op], feed_dict={X: occ, Y_cat: cat, keep_prob: 1, bn_training: False, weight: wgs})
                        min_wgs = min(min_wgs, np.min(wgs))
                        self.dataset.update_mini_batch(self.dataset.train, out[0])
                        print("\rUpdate weights %05d/%d" % (i+1,batches),end="")

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
                    plt.savefig(os.path.join("./3d-object-recognition/", self.name, "cost.png"), format="png")

                    acc, avg_time = accuracy_test(self.dataset, self.dataset.train)
                    train_infer_time.append(avg_time)
                    train_accuracies.append(acc)
                    plt.clf()
                    # plt.figure(2)
                    plt.plot(np.squeeze(np.array(train_accuracies)))
                    plt.savefig(os.path.join("./3d-object-recognition/", self.name, "train_accuracies.png"), format="png")

                    acc, avg_time = accuracy_test(self.dataset, self.dataset.dev, in_memory=in_memory)
                    dev_infer_time.append(avg_time)
                    dev_accuracies.append(acc)
                    plt.clf()
                    plt.plot(np.squeeze(np.array(dev_accuracies)))
                    plt.savefig(os.path.join("./3d-object-recognition/", self.name, "dev_accuracies.png"), format="png")

                    print("")
                    # do check for file barrier, if so, break training cycle
                    if os.path.exists(os.path.join(log_dir, "barrier.txt")):
                        break


                array2csv = []
                array2csv.append(train_accuracies)
                array2csv.append(dev_accuracies)
                array2csv.append(train_times)
                array2csv.append(wupdate_times)
                array2csv.append(train_infer_time)
                array2csv.append(dev_infer_time)
                array2csv = np.array(array2csv) # reshape so the values are in the rows
                array2csv = array2csv.transpose()
                np.savetxt(os.path.join("./3d-object-recognition/", self.name, "Results", self.name + ".csv"), array2csv, delimiter=",", comments='', header="Train accuracy,Dev accuracy,"\
                "Training times/epoch (ms), Weight update time/epoch (ms),"\
                "Train inference times (ms),Dev inference times (ms)")
                shutil.copy(os.path.join("./3d-object-recognition/", self.name, "network.json"), os.path.join("./3d-object-recognition/", self.name, "Results", "network.json"))
                print("Max acc train %f" % np.max(train_accuracies))
                print("Max acc dev %f" % np.max(dev_accuracies))
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
            plt.show()


    def evaluate_iou_results(self, data_dict={ "name" : "train"}, in_memory=True):
        return self.dataset.evaluate_iou_results(data_dict, in_memory=in_memory)



    def __init__(self, model_name, datapath):
        assert os.path.exists(os.path.join("./3d-object-recognition", model_name, "network.json"))

        with open(os.path.join("./3d-object-recognition", model_name, "network.json"), "r") as f:
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
    s = Net("ModelNet", "./3d-object-recognition/ModelNet40")
    s.run_model(load=False, train=True,visualize=False, in_memory=True)
