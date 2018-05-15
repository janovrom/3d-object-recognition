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
        X = tf.placeholder(dtype=tf.float32, shape=(None,15,15,15,1), name="input_grid")
        # X = tf.placeholder(dtype=tf.float32, shape=(1,320,128,192,1), name="input_grid")
        Y = tf.placeholder(dtype=tf.float32, shape=(None,n_y), name="input_distributions")
        # Y = tf.placeholder(dtype=tf.int32, shape=(1,320,128,192,1), name="input_labels")

        return X, Y


    def upsample_filter(self):
        filter = np.ones((3,3,3))
        filter = filter / 16
        filter[1,1,1] = 0.5

        return filter

    
    def convolution(self, X, shape, strides=[1,1,1,1,1], padding="SAME", act=tf.nn.relu):
        W = tf.get_variable("weights" + str(self.layer_idx), shape, initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        b = tf.get_variable("biases" + str(self.layer_idx), shape[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
        Z = tf.nn.conv3d(X, W, strides=strides, padding=padding) + b
        if act != None:
            Z = act(Z)

        self.layer_idx = self.layer_idx + 1
        return Z


    def deconvolution(self, X, shape, out_shape, act=tf.nn.relu):
        Z = tf.layers.conv3d_transpose(X, shape[-2], shape[0:3], name="deconv"+ str(self.layer_idx), strides=[1,1,1], padding="VALID", activation=act, trainable=False, kernel_initializer=tf.constant_initializer(self.upsample_filter()))

        self.layer_idx = self.layer_idx + 1
        print(Z)
        return Z


    def forward_propagation_fv(self, X, n_y):
        self.layer_idx = 0
        # imagine that the net operates over 16x16x16 blocks
        # IN 15, OUT 11
        A0 = self.convolution(X, [5,5,5,1,32], padding="VALID")
        # IN 11, OUT 7
        A1 = self.convolution(A0, [5,5,5,32,64], padding="VALID")
        # IN 7, OUT 5
        A2 = self.convolution(A1, [3,3,3,64,128], padding="VALID")
        # IN 5, OUT 3
        A3 = self.convolution(A2, [3,3,3,128,128], padding="VALID")
        # IN 3, OUT 1
        A4 = self.convolution(A3, [3,3,3,128,256], padding="VALID")
        A_fv = self.convolution(A4, [1,1,1,256,512], padding="VALID")
        A_class = self.convolution(A_fv, [1,1,1,512,n_y], act=None)

        return A_fv, A_class


    def compute_cost_fv(self, Z, Y, n_y):
        return tf.norm(Y-Z) # compute norm


    def compute_predictions(self, Zl, Y):
        softmax = tf.nn.softmax(logits=Zl)
        return tf.argmax(softmax, axis=-1)


    def run_model(self):
        log_dir = os.path.join("./3d-object-recognition", self.name)
        tf.reset_default_graph()
        n_y = self.dataset.num_classes
        X, Y = self.create_placeholders(n_y)
        A_fv, Z_class = self.forward_propagation_fv(X, n_y)
        cost = self.compute_cost_fv(Z_class, Y, n_y)
        # pred_op = self.compute_predictions(Z_classif, Y)

        step = tf.Variable(0, trainable=False, name="global_step")
        lr_dec = tf.train.exponential_decay(self.lr, step, 10000, 0.96)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_dec)
        gvs = optimizer.compute_gradients(cost)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)
        init = tf.global_variables_initializer()
        writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())
        sub_batches = 64

        def accuracy_test(dataset, data_dict):
            acc = 0
            dataset.restart_mini_batches(data_dict)
            for j in range(dataset.num_mini_batches(data_dict)):
                x,y,idxs = dataset.next_mini_batch(data_dict)
                stime = time.time()
                accum_time_extraction = 0
                for j in range(0, idxs[0].shape[0],sub_batches):
                    batch = []
                    blabs = []
                    m = min(idxs[0].shape[0] - j, sub_batches)
                    for k in range(j, j+m):
                        at = time.time()
                        xi = idxs[0][k]
                        yi = idxs[1][k]
                        zi = idxs[2][k]
                        batch.append(x[0,xi-7:xi+8,yi-7:yi+8,zi-7:zi+8])
                        blab = y[0,xi-7:xi+8,yi-7:yi+8,zi-7:zi+8]
                        y_hat, _ = np.histogram(blab.flatten(), [0,1,2,3,4,5,6,7,8,9])
                        if x[0,xi,yi,zi] != 1:
                            raise Exception("Some weird shit happened - index for zero occupancy")

                        if sum(y_hat) == 0:
                            raise Exception("another weird shit happened - there are no labels")
                        y_hat = y_hat / np.linalg.norm(y_hat)
                        blabs.append(y_hat)
                        accum_time_extraction = accum_time_extraction + time.time() - at

                    c = sess.run([cost], feed_dict={X: np.array(batch), Y: np.array(blabs)})
                print("Cost computation time over one sample was %f sec in average" % ((time.time() - stime)))
                print("From that %f sec was extraction time" % (accum_time_extraction))
                
            return float(acc / data_dict[dataset.NUMBER_EXAMPLES])

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init)

            costs = []
            for epoch in range(0, self.num_epochs):
                self.dataset.restart_mini_batches(self.dataset.train)
                self.dataset.restart_mini_batches(self.dataset.dev)
                self.dataset.restart_mini_batches(self.dataset.test)
                cc = 0
                stime = time.time()
                batches = self.dataset.num_mini_batches(self.dataset.train)
                for j in range(batches):
                    batch_stime = time.time()
                    x,y,idxs = self.dataset.next_mini_batch(self.dataset.train)
                    print("\rBatch loaded in %f" % (time.time() - batch_stime))
                    for j in range(0, idxs[0].shape[0],sub_batches):
                        batch = []
                        blabs = []
                        m = min(idxs[0].shape[0] - j, sub_batches)
                        for k in range(j, j+m):
                            xi = idxs[0][k]
                            yi = idxs[1][k]
                            zi = idxs[2][k]
                            batch.append(x[0,xi-7:xi+8,yi-7:yi+8,zi-7:zi+8])
                            blab = y[0,xi-7:xi+8,yi-7:yi+8,zi-7:zi+8]
                            y_hat, _ = np.histogram(blab.flatten(), [0,1,2,3,4,5,6,7,8,9])
                            if x[0,xi,yi,zi] != 1:
                                raise Exception("Some weird shit happened - index for zero occupancy")

                            if sum(y_hat) == 0:
                                raise Exception("another weird shit happened - there are no labels")
                            y_hat = y_hat / np.linalg.norm(y_hat)
                            blabs.append(y_hat)

                        _,c = sess.run([train_op,cost], feed_dict={X: np.array(batch), Y: np.array(blabs)})
                        cc = cc + c
                    print("\rBatch trained in %f" % (time.time() - batch_stime))

                costs.append(cc / (batches*idxs[0].shape[0]))
                print("\nEpoch trained in %f" % (time.time() - stime))
                

                # acc_train = accuracy_test(self.dataset, self.dataset.train)
                # acc_test = accuracy_test(self.dataset, self.dataset.test)
                # acc_dev = accuracy_test(self.dataset, self.dataset.dev)
                
                # print("Train/Dev/Test accuracies %f/%f/%f" %(acc_train, acc_dev, acc_test))
                print("epoch %d" %(epoch+1))

            accuracy_test(self.dataset, self.dataset.train)
            plt.plot(np.squeeze(np.array(costs)))
            plt.show()
            # save model
            save_path = tf.train.Saver().save(sess, os.path.join(log_dir, self.name + ".ckpt"))
            print("Model saved in file: %s" % save_path)


    def __init__(self, model_name, datapath):
        assert os.path.exists(os.path.join("./3d-object-recognition", model_name, "network.json"))

        with open(os.path.join("./3d-object-recognition", model_name, "network.json"), "r") as f:
            jparams = json.load(f)["params"]
            self.name =         model_name
            self.lr =           jparams["learning_rate"]
            self.num_epochs =   jparams["num_epochs"]
            self.l2_reg_w =     jparams["l2_reg_weights"]
            self.l2_reg_b =     jparams["l2_reg_biases"] 
            self.decay_step =   jparams["decay_step"] 
            self.decay_rate =   jparams["decay_rate"] 
            self.min_prob =     jparams["min_prob"] 
            self.keep_prob =    jparams["keep_prob"] 
            self.dataset =      Segmentations(datapath, batch_size=1, ishape=jparams["input_shape"], n_classes=jparams["num_classes"])


if __name__ == "__main__":
    s = SegmentationNet("SegNet", "./3d-object-recognition/SegData")
    s.run_model()