import tensorflow as tf 
import numpy as np 
import json
import os
from nn_utils import *
from segmentation_set import Segmentations
import sys


class SegmentationNet():

    def create_placeholders(self, n_y):
        X = tf.placeholder(dtype=tf.float32, shape=(1, 15,15,15, 1), name="input_grid")
        Y = tf.placeholder(dtype=tf.int32, shape=(1, n_y), name="input_labels")

        return X, Y


    def upsample_filter(self):
        filter = np.ones((3,3,3))
        filter = filter / 16
        filter[1,1,1] = 0.5

        return filter

    
    def convolution(self, X, shape, strides=[1,1,1,1,1], padding="VALID", act=tf.nn.relu):
        W = tf.get_variable("weights" + str(self.layer_idx), shape, initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        b = tf.get_variable("biases" + str(self.layer_idx), shape[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
        Z = tf.nn.conv3d(X, W, strides=strides, padding=padding) + b
        if act != None:
            Z = act(Z)

        self.layer_idx = self.layer_idx + 1
        return Z


    def deconvolution(self, X, shape, out_shape, act=tf.nn.relu):
        Z = tf.layers.conv3d_transpose(X, shape[-2], shape[0:3], strides=[1,1,1], padding="VALID", activation=act, kernel_initializer=tf.constant_initializer(self.upsample_filter()))

        self.layer_idx = self.layer_idx + 1
        print(Z)
        return Z


    def forward_propagation(self, X, n_y):
        self.layer_idx = 0
        # conv1 from 32x32x32 to 16x16x16
        Z = self.convolution(X, [5,5,5,1,16], padding="SAME")
        Z = tf.nn.max_pool3d(Z, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID")
        # conv2 from 13x13x13 to 11x11x11
        Z = self.convolution(Z, [3,3,3,64,64])
        # conv3 from 11x11x11 to 9x9x9
        Z = self.convolution(Z, [3,3,3,64,128])

        # # conv4 from 9x9x9 to 7x7x7
        # Z = self.convolution(Z, [3,3,3,128,128])
        # # conv5 from 7x7x7 to 5x5x5
        # Z = self.convolution(Z, [3,3,3,128,256])
        # # conv6 from 5x5x5 to 3x3x3
        # Z = self.convolution(Z, [3,3,3,256,256])
        # # conv7 from 3x3x3 to 1x1x1
        # Z = self.convolution(Z, [3,3,3,256,512])

        # FC part
        Z = self.convolution(Z, [1,1,1,128,512])
        Z = self.convolution(Z, [1,1,1,512,512])

        # do classification part
        Z_classif = self.convolution(Z, [1,1,1,512,256])
        Z_classif = tf.contrib.layers.flatten(self.convolution(Z_classif, [1,1,1,256,n_y], act=None))

        # self.layer_idx = 0
        # d = Z.shape[1]
        # w = Z.shape[2]
        # h = Z.shape[3]
        # Z = self.deconvolution(Z, [3,3,3,256,512], [1,d+2,w+2,h+2,256])
        # Z = self.deconvolution(Z, [3,3,3,256,256], [1,d+4,w+4,h+4,256])
        # Z = self.deconvolution(Z, [3,3,3,128,256], [1,d+6,w+6,h+6,128])
        # Z = self.deconvolution(Z, [3,3,3,128,128], [1,d+8,w+8,h+8,128])
        # Z = self.deconvolution(Z, [3,3,3,64,128],  [1,d+10,w+10,h+10,64])
        # Z = self.deconvolution(Z, [3,3,3,64,64],   [1,d+12,w+12,h+12,64])
        # Z_seg = self.deconvolution(Z, [3,3,3,1,64],[1,d+14,w+14,h+14,1], act=None)

        return Z_classif#, Z_seg


    def compute_cost(self, Z, Y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z, labels=Y))


    def compute_predictions(self, Zl, Y):
        softmax = tf.nn.softmax(logits=Zl)
        return tf.argmax(softmax, axis=1)


    def run_model(self):
        log_dir = os.path.join("./3d-object-recognition", self.name)
        tf.reset_default_graph()
        n_y = self.dataset.num_classes
        X, Y = self.create_placeholders(n_y)
        Z_classif = self.forward_propagation(X, n_y)
        cost = self.compute_cost(Z_classif, Y)
        pred_op = self.compute_predictions(Z_classif, Y)

        step = tf.Variable(0, trainable=False, name="global_step")
        lr_dec = tf.train.exponential_decay(0.001, step, 10000, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_dec)
        gvs = optimizer.compute_gradients(cost)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)
        init = tf.global_variables_initializer()
        writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

        def accuracy_test(dataset, data_dict):
            acc = 0
            dataset.restart_mini_batches(data_dict)
            for j in range(dataset.num_mini_batches(data_dict)):
                x,y = dataset.next_mini_batch(data_dict)
                y_hot = convert_to_one_hot(y, dataset.num_classes)
                pred,zcl = sess.run([pred_op,Z_classif], feed_dict={X: x, Y: y_hot})
                # print(zcl)
                # print(pred)
                # print(y)
                # sys.stdin.read(1)
                if pred == y:
                    acc = acc + 1

            return float(acc / data_dict[dataset.NUMBER_EXAMPLES])


        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(0, self.num_epochs):
                self.dataset.restart_mini_batches(self.dataset.train)
                self.dataset.restart_mini_batches(self.dataset.dev)
                self.dataset.restart_mini_batches(self.dataset.test)
                cc = 0
                for j in range(self.dataset.num_mini_batches(self.dataset.train)):
                    x,y = self.dataset.next_mini_batch(self.dataset.train)
                    y_hot = convert_to_one_hot(y, self.dataset.num_classes)
                    cc = sess.run([train_op], feed_dict={X: x, Y: y_hot})

                acc_train = accuracy_test(self.dataset, self.dataset.train)
                acc_test = accuracy_test(self.dataset, self.dataset.test)
                acc_dev = accuracy_test(self.dataset, self.dataset.dev)
                
                print("Train/Dev/Test accuracies %f/%f/%f" %(acc_train, acc_dev, acc_test))
                print("epoch %d" %(epoch+1))


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
    s = SegmentationNet("SegNet", "./3d-object-recognition/Engine-data-15")
    s.run_model()