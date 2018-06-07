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
        X = tf.placeholder(dtype=tf.float32, shape=(None,16,16,16,1), name="input_grid")
        # X = tf.placeholder(dtype=tf.float32, shape=(1,320,128,192,1), name="input_grid")
        Y = tf.placeholder(dtype=tf.float32, shape=(None,n_y), name="input_distributions")
        U_deconv = tf.placeholder(dtype=tf.float32, shape=(1,8,8,256), name="input_deconv")
        Y_deconv = tf.placeholder(dtype=tf.int32, shape=(1,128,128), name="deconv_mask")
        # Y = tf.placeholder(dtype=tf.int32, shape=(1,320,128,192,1), name="input_labels")

        return X, Y, Y_deconv, U_deconv


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


    def deconvolution(self, X, shape, out_shape, act=tf.nn.relu):
        Z = tf.layers.conv3d_transpose(X, shape[-2], shape[0:3], name="deconv"+ str(self.layer_idx), strides=[1,1,1], padding="VALID", activation=act, trainable=False, kernel_initializer=tf.constant_initializer(self.upsample_filter()))

        self.layer_idx = self.layer_idx + 1
        print(Z)
        return Z


    def forward_propagation_deconv(self, U, n_y):
        # get global correspondce by merging 7-neighbourhood
        U0 = self.convolution2d(U, [3,3,256,256], padding="SAME")
        U0 = self.convolution2d(U0, [3,3,256,256], padding="SAME")
        U0 = self.convolution2d(U0, [3,3,256,256], padding="SAME")
        #
        U1 = tf.image.resize_bilinear(U0, [16,16])
        U1 = self.convolution2d(U1, [3,3,256,256], padding="SAME")
        U1 = self.convolution2d(U1, [3,3,256,128], padding="SAME")
        #
        U2 = tf.image.resize_bilinear(U1, [32,32])
        U2 = self.convolution2d(U2, [3,3,128,128], padding="SAME")
        U2 = self.convolution2d(U2, [3,3,128,64], padding="SAME")
        #
        U3 = tf.image.resize_bilinear(U2, [64,64])
        U3 = self.convolution2d(U3, [3,3,64,64], padding="SAME")
        U3 = self.convolution2d(U3, [3,3,64,32], padding="SAME")
        #
        U4 = tf.image.resize_bilinear(U3, [128,128])
        U4 = self.convolution2d(U4, [3,3,32,32], padding="SAME")
        U_mask = self.convolution2d(U4, [3,3,32,n_y], padding="SAME", act=None)

        return U_mask


    def forward_propagation_fv(self, X, n_y):
        self.layer_idx = 0
        # imagine that the net operates over 16x16x16 blocks
        # IN 16
        A0 = self.convolution(X, [5,5,5,1,32], padding="SAME") 
        A0 = tf.nn.avg_pool3d(A0, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID") # to 8x8x8
        A1 = self.convolution(A0, [5,5,5,32,64], padding="SAME")
        A1 = tf.nn.avg_pool3d(A1, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID") # to 4x4x4
        A2 = self.convolution(A1, [3,3,3,64,128], padding="SAME") 
        A2 = tf.nn.avg_pool3d(A2, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID") # to 2x2x2
        A_fv = self.convolution(A2, [2,2,2,128,256], padding="VALID") #1x1x1x256
        A_class = self.convolution(A_fv, [1,1,1,256,n_y], padding="VALID", act=None) #1x1x1x256
        print(A_fv)
        return A_fv, A_class


    def compute_cost_deconv(self, U, Y):
        print(U)
        print(Y)
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=U)


    def compute_cost_fv(self, Z, Y):
        print(Z)
        print(Y)
        return tf.nn.l2_loss(Z - Y)  # compute cost between distributions


    def compute_predictions(self, U_labels):
        return tf.argmax(U_labels, axis=-1)


    def run_model(self, load=False, train=True,visualize=True):
        log_dir = os.path.join("./3d-object-recognition", self.name)
        tf.reset_default_graph()
        n_y = self.dataset.num_classes
        X, Y, Y_deconv, U_deconv = self.create_placeholders(n_y)
        A_fv, A_class = self.forward_propagation_fv(X, n_y)
        A_deconv = self.forward_propagation_deconv(U_deconv, n_y)
        cost_fv = self.compute_cost_fv(A_class, Y)
        cost_d = self.compute_cost_deconv(A_deconv, Y_deconv)
        pred_op = self.compute_predictions(A_deconv)
        print(pred_op)

        # fv part
        step_fv = tf.Variable(0, trainable=False, name="global_step_fv")
        lr_dec_fv = tf.train.exponential_decay(self.lr, step_fv, self.decay_step, self.decay_rate)
        optimizer_fv = tf.train.AdamOptimizer(learning_rate=lr_dec_fv)
        # gvs_fv = optimizer_fv.compute_gradients(cost_fv)
        # print(gvs_fv)
        # capped_gvs_fv = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs_fv]
        # train_op_fv = optimizer_fv.apply_gradients(capped_gvs_fv)
        train_op_fv = optimizer_fv.minimize(cost_fv)
        # deconv part
        step_d = tf.Variable(0, trainable=False, name="global_step_d")
        lr_dec_d = tf.train.exponential_decay(self.lr, step_d, self.decay_step, self.decay_rate)
        optimizer_d = tf.train.AdamOptimizer(learning_rate=lr_dec_d)
        # gvs_d = optimizer_d.compute_gradients(cost_d)
        # capped_gvs_d = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs_d]
        # train_op_d = optimizer_d.apply_gradients(capped_gvs_d)
        train_op_d = optimizer_d.minimize(cost_d)

        init = tf.global_variables_initializer()
        writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())
        sub_batches = 32
        writer.flush()
            


        def accuracy_test(dataset, data_dict):
            acc = 0
            dataset.restart_mini_batches(data_dict)
            for i in range(dataset.num_mini_batches(data_dict)):
                ltime = time.time()
                x,y,_,idxs,filename,d_labels = self.dataset.next_mini_batch(data_dict)
                ltime_end = time.time()
                stime = time.time()
                D = np.zeros((1,8,8,256), dtype=np.float)
                # print("\rBatch loaded in %f\n*********************\n" % (time.time() - batch_stime))
                for j in range(0, y.shape[0],sub_batches):
                    start = j
                    end = min(y.shape[0],sub_batches+j)
                    A = sess.run([A_fv], feed_dict={X: x[start:end], Y: y[start:end]})
                    A = np.array(A)[0]
                    for k in range(0,min(sub_batches, A.shape[0])):
                        idx = idxs[k]
                        D[0,idx[0],idx[1],:] = A[k,0,0,0,:]

                deconvolved_image = sess.run([tf.argmax(A_deconv, axis=-1)], feed_dict={U_deconv: D, Y_deconv: d_labels})
                deconvolved_image = np.array(deconvolved_image)[0,0]
                if visualize:
                    dl.load_xyzl_vis(filename, deconvolved_image, n_y)

                print("\rData loaded in \t %f sec, Convolved in \t %f sec" % (ltime_end - ltime, time.time() - stime), end="")

            print("")    
            return float(acc / data_dict[dataset.NUMBER_EXAMPLES])

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
                    self.dataset.restart_mini_batches(self.dataset.train)
                    self.dataset.restart_mini_batches(self.dataset.test)
                    self.dataset.restart_mini_batches(self.dataset.dev)
                    self.dataset.restart_mini_batches(self.dataset.train_d)
                    self.dataset.restart_mini_batches(self.dataset.test_d)
                    self.dataset.restart_mini_batches(self.dataset.dev_d)
                    cc = 0
                    stime = time.time()
                    batches = self.dataset.num_mini_batches(self.dataset.train)
                    batches_d = self.dataset.num_mini_batches(self.dataset.train_d)
                    for i in range(batches):
                        batch_stime = time.time()
                        x,y,mask,idxs,_,d_labels = self.dataset.next_mini_batch(self.dataset.train)
                        # print("\rBatch loaded in %f\n*********************\n" % (time.time() - batch_stime))
                        # train on the one example
                        for j in range(0, y.shape[0],sub_batches):
                            start = j
                            end = min(y.shape[0],sub_batches+j)
                            _,c = sess.run([train_op_fv,cost_fv], feed_dict={X: x[start:end], Y: y[start:end]})

                            cc = cc + c / y.shape[0]

                        # train the deconvolution
                        #_,d_cost = sess.run([train_op_d,cost_d], feed_dict={U_deconv: D, Y_deconv: d_labels})
                        # print("\rBatch trained in %f" % (time.time() - batch_stime), end="")

                    for i in range(batches_d):
                        # evaluate the scene batch
                        # evaluate the last layer after the training so we have reasonable results
                        # flatten the grid 8x8x8 to image 8x8
                        D = np.zeros((1,8,8,256), dtype=np.float)
                        x,y,mask,idxs,_,d_labels = self.dataset.next_mini_batch(self.dataset.train_d)
                        for j in range(0, y.shape[0],sub_batches):
                            start = j
                            end = min(y.shape[0],sub_batches+j)
                            A = sess.run([A_fv], feed_dict={X: x[start:end], Y: y[start:end]})
                            A = np.array(A)
                            for k in range(0,min(sub_batches, A.shape[0])):
                                idx = idxs[k]
                                D[0,idx[0],idx[1],:] = A[k,0,0,0,:]

                    costs.append(cc / (batches))
                    print("\nEpoch %d trained in %f, cost %f" % (epoch+1, time.time() - stime, costs[-1]))
                    

                    # acc_train = accuracy_test(self.dataset, self.dataset.train)
                    # acc_test = accuracy_test(self.dataset, self.dataset.test)
                    # acc_dev = accuracy_test(self.dataset, self.dataset.dev)
                    
                    # print("Train/Dev/Test accuracies %f/%f/%f" %(acc_train, acc_dev, acc_test))
                    # do check for file barrier, if so, break training cycle
                    if os.path.exists(os.path.join(log_dir, "barrier.txt")):
                        break

            accuracy_test(self.dataset, self.dataset.train_d)
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
    s.run_model(load=False, train=True,visualize=False)
