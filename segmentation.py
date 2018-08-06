import tensorflow as tf 
import numpy as np 
import json
import os
from nn_utils import *
from segmentation_set import Segmentations
import sys
import time
import matplotlib.pyplot as plt


class SegmentationNet():

    def create_placeholders(self, n_y):
        X = tf.placeholder(dtype=tf.float32, shape=(None,32,32,32,1), name="input_grid")
        # X = tf.placeholder(dtype=tf.float32, shape=(1,320,128,192,1), name="input_grid")
        Y = tf.placeholder(dtype=tf.int32, shape=(None,32,32,32), name="labels")
        # Y = tf.placeholder(dtype=tf.int32, shape=(1,320,128,192,1), name="input_labels")
        keep_prob = tf.placeholder(dtype=tf.float32, name="keep_probability")
        bn_training = tf.placeholder(dtype=tf.bool, name="batch_norm_training")

        return X, Y, keep_prob, bn_training


    def upsample_filter(self):
        filter = np.ones((3,3,3))
        filter = filter / 16
        filter[1,1,1] = 0.5

        return filter


    def convolution2d(self, X, shape, strides=[1,1,1,1], padding="SAME", act=tf.nn.relu):
        W = tf.get_variable("weights2d" + str(self.layer_idx), shape, initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        b = tf.get_variable("biases2d" + str(self.layer_idx), shape[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
        Z = tf.nn.conv2d(X, W, strides=strides, padding=padding) + b
        if act != None:
            Z = act(Z)

        self.layer_idx = self.layer_idx + 1
        return Z

    
    def convolution(self, X, shape, strides=[1,1,1,1,1], padding="SAME", act=tf.nn.relu):
        W = tf.get_variable("weights" + str(self.layer_idx), shape, initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        b = tf.get_variable("biases" + str(self.layer_idx), shape[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
        Z = tf.nn.conv3d(X, W, strides=strides, padding=padding) + b
        if act != None:
            Z = act(Z)

        self.layer_idx = self.layer_idx + 1
        return Z


    def forward_propagation(self, X, n_y, keep_prob, bn_training):
        self.layer_idx = 0
        # imagine that the net operates over 32x32x32 blocks
        # feature vector learning
        # IN 32
        A0 = self.convolution(X, [5,5,5,1,32], padding="SAME") 
        D0 = tf.nn.dropout(A0, keep_prob)
        D0 = tf.layers.batch_normalization(D0, training=bn_training)
        M0 = tf.nn.max_pool3d(D0, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID") # to 16

        A1 = self.convolution(M0, [5,5,5,32,64], padding="SAME")
        D1 = tf.nn.dropout(A1, keep_prob)        
        D1 = tf.layers.batch_normalization(D1, training=bn_training)
        M1 = tf.nn.max_pool3d(D1, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID") # to 8

        A2 = self.convolution(M1, [3,3,3,64,128], padding="VALID") # to 6
        D2 = tf.nn.dropout(A2, keep_prob)
        D2 = tf.layers.batch_normalization(D2, training=bn_training)
        M2 = tf.nn.max_pool3d(D2, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID") # to 3

        A3 = self.convolution(M2, [3,3,3,128,256], padding="VALID") #1x1x1x256
        A_fv = self.convolution(A3, [1,1,1,256,128], padding="VALID") #1x1x1x256
        U0 = self.convolution(A_fv, [1,1,1,128,256], padding="VALID") #1x1x1x256

        U_t = tf.tile(U0, [1,8,8,8,1])
        print(U_t)

        U_concat = tf.concat([M1, U_t], axis=-1)
        U1 = self.convolution(U_concat, [3,3,3,320,256], padding="SAME")
        U1 = tf.layers.batch_normalization(U1, training=bn_training)
        U1 = self.convolution(U1, [3,3,3,256,256], padding="SAME")
        U1 = tf.layers.batch_normalization(U1, training=bn_training)
        
        U2 = tf.keras.layers.UpSampling3D([2,2,2])(U1) # to 16
        U_concat1 = tf.concat([M0, U2], axis=-1)
        print(U_concat1)
        U2 = self.convolution(U_concat1, [3,3,3,288,128], padding="SAME")
        U2 = tf.layers.batch_normalization(U2, training=bn_training)
        U2 = self.convolution(U2, [3,3,3,128,128], padding="SAME")
        U2 = tf.layers.batch_normalization(U2, training=bn_training)

        U3 = tf.keras.layers.UpSampling3D([2,2,2])(U2) # to 32
        U_mask = self.convolution(U3, [3,3,3,128,128], padding="SAME")
        U_mask = self.convolution(U_mask, [1,1,1,128,64], padding="SAME")
        U_mask = self.convolution(U_mask, [1,1,1,64,n_y], padding="SAME", act=None)
        U_class = tf.argmax(tf.nn.softmax(U_mask), axis=-1)

        return A_fv, U_mask, U_class


    def compute_cost(self, U, Y, X):
        # U = U * X
        # print(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=U)*tf.reshape(X, [-1, X.shape[1], X.shape[2], X.shape[3]]))
        Xrep = tf.reshape(X, [-1, X.shape[1], X.shape[2], X.shape[3]])
        weighted_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=U) * Xrep
        
        # define summation filter
        f = 1
        F = tf.get_variable("sum_conv", [3,3,3,1,1], initializer=tf.constant_initializer(f), dtype=tf.float32, trainable=False)

        # define correct classifications
        classifications = tf.cast(tf.equal(tf.argmax(tf.nn.softmax(U), output_type=tf.int32, axis=-1),Y), tf.float32)
        classifications = tf.reshape(classifications, [-1,classifications.shape[1],classifications.shape[2],classifications.shape[3],1])
        print(classifications)
        # # count local number of non-empty voxels
        # non_empty = tf.nn.conv3d(X, F, strides=[1,1,1,1,1], padding="SAME")
        # corr_sum = tf.nn.conv3d(classifications, F, strides=[1,1,1,1,1], padding="SAME")
        # # 0-1 defines how much correctly classified
        # # local_correct = tf.divide(corr_sum, non_empty) # there might be division by zero, but we don't care, since full voxels will always have at least 1 - aaaand it is a problem
        # local_correct = corr_sum / 27.0 #tf.divide(corr_sum, non_empty+0.0001)
        # # change the range from [0,1] to [1,alpha+1]
        # alpha = 1
        # local_correct = local_correct * alpha + 1
        # local_correct = tf.reshape(local_correct, [-1, local_correct.shape[1], local_correct.shape[2],local_correct.shape[3]])
        # print(local_correct)
        # print(weighted_entropy)
        # weighted_entropy = tf.multiply(tf.multiply(weighted_entropy, local_correct), Xrep)
        # c = tf.reduce_sum(weighted_entropy)

        ## Full conv attempt for loss
        weighted_entropy = tf.reshape(weighted_entropy, [-1,weighted_entropy.shape[1],weighted_entropy.shape[2],weighted_entropy.shape[3],1])
        we_sum = tf.nn.conv3d(weighted_entropy, F, strides=[1,1,1,1,1], padding="VALID") # 30
        we_sum = tf.nn.conv3d(we_sum, F, strides=[1,3,3,3,1], padding="VALID") # 10
        we_sum = tf.nn.conv3d(we_sum, F, strides=[1,1,1,1,1], padding="VALID") # 8
        we_sum = tf.nn.conv3d(we_sum, F, strides=[1,1,1,1,1], padding="VALID") # 6
        we_sum = tf.nn.conv3d(we_sum, F, strides=[1,1,1,1,1], padding="VALID") # 4
        we_sum = tf.nn.conv3d(we_sum, F, strides=[1,1,1,1,1], padding="VALID") # 2
        c = tf.reduce_sum(we_sum)


        print(c)# try multiplying the result by a weight
        return c, U



    def run_model(self, load=False, train=True,visualize=True):
        log_dir = os.path.join("./3d-object-recognition", self.name)
        tf.reset_default_graph()
        n_y = self.dataset.num_classes
        X, Y, keep_prob, bn_training = self.create_placeholders(n_y)
        A_fv, U_mask, U_class = self.forward_propagation(X, n_y, keep_prob, bn_training)
        cost, tmp_test = self.compute_cost(U_mask, Y, X)

        # fv part
        step = tf.Variable(0, trainable=False, name="global_step")
        lr_dec = tf.train.exponential_decay(self.lr, step, self.decay_step, self.decay_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_dec)
        gvs = optimizer.compute_gradients(cost)
        # print(gvs)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)
        # train_op = optimizer.minimize(cost)

        init = tf.global_variables_initializer()
        writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())
        writer.flush()
            

        def accuracy_test(dataset, data_dict):
            acc = 0
            acc_cat = 0
            dataset.restart_mini_batches(data_dict)
            for i in range(dataset.num_mini_batches(data_dict)):
                stime = time.time()
                x,y,names = self.dataset.next_class_mini_batch(data_dict)
                deconvolved_images,d_cost = sess.run([U_class,cost], feed_dict={X: x, Y: y, keep_prob: 1.0, bn_training: False})

                xresh = np.reshape(x, [-1, x.shape[1], x.shape[2], x.shape[3]])
                a = np.sum((xresh * deconvolved_images) == y) / np.sum(x)
                # print("Average interference time per mini batch example %f sec" % ((time.time() - stime) / x.shape[0]))
                acc = acc + a
                if visualize:
                    for j in range(0, deconvolved_images.shape[0]):
                        print(names[j])
                        dl.load_xyzl_vis(names[j], deconvolved_images[j], n_y)

            print("Deconvolution average accuracy %f" % (acc / dataset.num_mini_batches(data_dict)))    
            # print("Deconvolution average category accuracy %f" % (acc_cat / dataset.num_mini_batches(data_dict)))
            return float(acc / dataset.num_mini_batches(data_dict))


        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init)

            if load:
                tf.train.Saver().restore(sess, os.path.join(log_dir, self.name + ".ckpt"))
                print("Model loaded from file: %s" % os.path.join(log_dir, self.name + ".ckpt"))

            costs = []
            if train:
                for epoch in range(0, self.num_epochs):
                    self.dataset.restart_mini_batches(self.dataset.train_d)
                    self.dataset.restart_mini_batches(self.dataset.train)
                    self.dataset.restart_mini_batches(self.dataset.test_d)
                    self.dataset.restart_mini_batches(self.dataset.dev_d)
                    

                    batches_d = self.dataset.num_mini_batches(self.dataset.train_d)
                    cc = 0
                    stime = time.time()
                    # evaluate the scene batch
                    for i in range(batches_d):
                        x,y,_ = self.dataset.next_class_mini_batch(self.dataset.train_d)
                        _,d_cost,tmp = sess.run([train_op,cost,tmp_test], feed_dict={X: x, Y: y, keep_prob: self.keep_prob, bn_training: True})
                        cc = cc + d_cost
                        print("\rBatch %03d/%d" % (i+1,batches_d),end="")


                    cc = cc / (self.dataset.num_examples(self.dataset.train_d))
                    costs.append(cc)
                    print("\nEpoch %d trained in %f, cost %f" % (epoch+1, time.time() - stime, cc))
                    

                    # acc_train = accuracy_test(self.dataset, self.dataset.train)
                    # acc_test = accuracy_test(self.dataset, self.dataset.test)
                    # acc_dev = accuracy_test(self.dataset, self.dataset.dev)
                    
                    # print("Train/Dev/Test accuracies %f/%f/%f" %(acc_train, acc_dev, acc_test))
                    # do check for file barrier, if so, break training cycle
                    if os.path.exists(os.path.join(log_dir, "barrier.txt")):
                        break
                    
                    # save model
                    save_path = tf.train.Saver().save(sess, os.path.join(log_dir, self.name + str(epoch+1) + ".ckpt"))
                    print("Model saved in file: %s\n" % save_path)

            print(accuracy_test(self.dataset, self.dataset.train_d))
            plt.plot(np.squeeze(np.array(costs)))
            plt.show()


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
            self.dataset =      Segmentations(datapath, batch_size=jparams["batch_size"], ishape=jparams["input_shape"], n_classes=jparams["num_classes"])


if __name__ == "__main__":
    s = SegmentationNet("SegNet", "./3d-object-recognition/SegData")
    s.run_model(load=True, train=False,visualize=True)
