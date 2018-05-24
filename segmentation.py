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
        Y_mask = tf.placeholder(dtype=tf.int32, shape=(None,16,16,16), name="deconv_mask")
        # Y = tf.placeholder(dtype=tf.int32, shape=(1,320,128,192,1), name="input_labels")

        return X, Y, Y_mask


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
        # IN 16, OUT 16
        A0 = self.convolution(X, [5,5,5,1,32], padding="SAME")
        # IN 16, OUT 8
        A1 = self.convolution(A0, [3,3,3,32,64], padding="SAME")
        A1 = tf.nn.max_pool3d(A1, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID")
        # IN 8, OUT 6, MP to 3
        A2 = self.convolution(A1, [3,3,3,64,128], padding="VALID")
        A2 = tf.nn.max_pool3d(A2, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID")
        # IN 3, OUT 1
        A3 = self.convolution(A2, [3,3,3,128,256], padding="VALID")
        # fully-conv layer
        A_fv = self.convolution(A3, [1,1,1,256,512], padding="VALID")
        A_class = self.convolution(A_fv, [1,1,1,512,n_y], act=None)

        # IN 1, OUT 4
        U0 = tf.keras.layers.UpSampling3D(size=(4,4,4))(A_fv)
        U0 = self.convolution(U0, [3,3,3,512,256], padding="SAME")
        # IN 4, OUT 8
        U1 = tf.keras.layers.UpSampling3D(size=(2,2,2))(U0)
        U1 = self.convolution(U1, [3,3,3,256,64], padding="SAME")
        # IN 8, OUT 16
        U2 = tf.keras.layers.UpSampling3D(size=(2,2,2))(U1)
        U2 = self.convolution(U2, [3,3,3,64,32], padding="SAME")
        # Convert to one hot and smooth
        U_mask = self.convolution(U2, [3,3,3,32,n_y], padding="SAME")

        return A_fv, A_class, U_mask


    def compute_cost_fv(self, Z, Y, n_y, Y_mask, U_mask):
        # penalize for non-normalized vector Z
        norm_penalty = tf.abs(tf.reduce_sum(Z**2) - 1.0)
        # compute norm for feature vector
        n = tf.nn.l2_loss(Z - Y)
        print(n)
        # compute pixel-wise deconv error
        softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_mask, logits=U_mask)
        print(softmax)
        cost = norm_penalty + n + 2.0 * tf.reduce_sum(softmax)
        print(cost)
        return cost  # compute cost


    def compute_predictions(self, U_labels):
        return tf.argmax(U_labels, axis=-1)


    def run_model(self, load=False, train=True,visualize=True):
        log_dir = os.path.join("./3d-object-recognition", self.name)
        tf.reset_default_graph()
        n_y = self.dataset.num_classes
        X, Y, Y_mask = self.create_placeholders(n_y)
        A_fv, Z_class, U_mask = self.forward_propagation_fv(X, n_y)
        cost = self.compute_cost_fv(Z_class, Y, n_y, Y_mask, U_mask)
        pred_op = self.compute_predictions(U_mask)
        print(pred_op)

        step = tf.Variable(0, trainable=False, name="global_step")
        lr_dec = tf.train.exponential_decay(self.lr, step, 10000, 0.96)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_dec)
        gvs = optimizer.compute_gradients(cost)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)
        init = tf.global_variables_initializer()
        writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())
        sub_batches = 128

        def accuracy_test(dataset, data_dict):
            acc = 0
            dataset.restart_mini_batches(data_dict)
            for i in range(dataset.num_mini_batches(data_dict)):
                ltime = time.time()
                x,y,mask,idxs,filename = dataset.next_mini_batch(data_dict)
                ltime_end = time.time()
                stime = time.time()
                labels=[]
                for j in range(0, y.shape[0],sub_batches):
                    start = j
                    end = min(y.shape[0],sub_batches+j)
                    c, ls = sess.run([cost, pred_op], feed_dict={X: x[start:end], Y: y[start:end], Y_mask: mask[start:end]})
                    labels.extend(ls)
                
                if visualize:
                    labels_dict = {}
                    for k in range(0, len(labels)):
                        labels_dict[idxs[k]] = labels[k]

                    dl.load_xyzl_vis(filename, labels_dict, n_y)

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
                    self.dataset.restart_mini_batches(self.dataset.dev)
                    self.dataset.restart_mini_batches(self.dataset.test)
                    cc = 0
                    stime = time.time()
                    batches = self.dataset.num_mini_batches(self.dataset.train)
                    for i in range(batches):
                        batch_stime = time.time()
                        x,y,mask,_,_ = self.dataset.next_mini_batch(self.dataset.train)
                        # print("\rBatch loaded in %f\n*********************\n" % (time.time() - batch_stime))
                        for j in range(0, y.shape[0],sub_batches):
                            start = j
                            end = min(y.shape[0],sub_batches+j)
                            _,c = sess.run([train_op,cost], feed_dict={X: x[start:end], Y: y[start:end], Y_mask: mask[start:end]})
                            cc = cc + c

                        cc = cc / y.shape[0]
                        # print("\rBatch trained in %f" % (time.time() - batch_stime), end="")

                    costs.append(cc / (batches))
                    print("\nEpoch trained in %f" % (time.time() - stime))
                    

                    # acc_train = accuracy_test(self.dataset, self.dataset.train)
                    # acc_test = accuracy_test(self.dataset, self.dataset.test)
                    # acc_dev = accuracy_test(self.dataset, self.dataset.dev)
                    
                    # print("Train/Dev/Test accuracies %f/%f/%f" %(acc_train, acc_dev, acc_test))
                    print("epoch %d" %(epoch+1))

                    # do check for file barrier, if so, break training cycle
                    if os.path.exists(os.path.join(log_dir, "barrier.txt")):
                        break

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
    s.run_model(load=False, train=True,visualize=False)