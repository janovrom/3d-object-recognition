import tensorflow as tf 
import numpy as np 
import json
import os
import shutil
from nn_utils import *
from shapenetparts import Parts
import sys
import time
import matplotlib.pyplot as plt


class PartsNet():

    def create_placeholders(self, n_y, n_seg):
        X = tf.placeholder(dtype=tf.float32, shape=(None,self.dataset.shape[0],self.dataset.shape[1],self.dataset.shape[2],1), name="input_grid")
        weight = tf.placeholder(dtype=tf.float32, shape=(None), name="loss_weights")
        Y_seg = tf.placeholder(dtype=tf.int32, shape=(None), name="segmentation_labels")
        Y_cat = tf.placeholder(dtype=tf.int32, shape=(None), name="category_labels")
        keep_prob = tf.placeholder(dtype=tf.float32, name="keep_probability")
        bn_training = tf.placeholder(dtype=tf.bool, name="batch_norm_training")

        return X, Y_seg, Y_cat, keep_prob, bn_training, weight

    
    def convolution(self, X, shape, strides=[1,1,1,1,1], padding="SAME", act=tf.nn.relu):
        # tf.contrib.layers.xavier_initializer(uniform=False)
        W = tf.get_variable("weights" + str(self.layer_idx), shape, initializer=tf.variance_scaling_initializer(), dtype=tf.float32, regularizer=tf.contrib.layers.l2_regularizer(self.lr_dec))
        b = tf.get_variable("biases" + str(self.layer_idx), shape[-1], initializer=tf.zeros_initializer(), dtype=tf.float32, regularizer=tf.contrib.layers.l2_regularizer(self.lr_dec))
        Z = tf.nn.conv3d(X, W, strides=strides, padding=padding) + b

        tf.summary.histogram("weights" + str(self.layer_idx), W)
        tf.summary.histogram("biases" + str(self.layer_idx), b)

        if act != None:
            Z = act(Z)

        self.layer_idx = self.layer_idx + 1
        return Z


    def forward_propagation_great_small_but_efficient(self, X, n_cat, n_seg, keep_prob, bn_training):
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
        A1_5 = self.convolution(M0, [5,5,5,64,32], padding="SAME")
        D1_5 = tf.nn.dropout(A1_5, keep_prob)        
        D1_5 = tf.layers.batch_normalization(D1_5, training=bn_training) 

        A1_3 = self.convolution(M0, [3,3,3,64,32], padding="SAME")
        D1_3 = tf.nn.dropout(A1_3, keep_prob)        
        D1_3 = tf.layers.batch_normalization(D1_3, training=bn_training)      
        A1 = tf.concat([D1_3,D1_5], axis=-1)     
        M1 = tf.nn.max_pool3d(A1, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID") # to 8

        # third block
        A2_5 = self.convolution(M1, [5,5,5,64,32], padding="SAME")
        D2_5 = tf.nn.dropout(A2_5, keep_prob)        
        D2_5 = tf.layers.batch_normalization(D2_5, training=bn_training) 

        A2_3 = self.convolution(M1, [3,3,3,64,32], padding="SAME")
        D2_3 = tf.nn.dropout(A2_3, keep_prob)        
        D2_3 = tf.layers.batch_normalization(D2_3, training=bn_training)      
        A2 = tf.concat([D2_3,D2_5], axis=-1)     
        M2 = tf.nn.max_pool3d(A2, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID") # to 4

        A3 = self.convolution(M2, [4,4,4,64,128], padding="VALID") # to 1
        D3 = tf.nn.dropout(A3, keep_prob)
        D3 = tf.layers.batch_normalization(D3, training=bn_training)        
        
        # TODO try and remove the 3,3,3 conv and use reshape instead to large vector
        A4 = self.convolution(D3, [1,1,1,128,256], padding="VALID")
        A5 = self.convolution(A4, [1,1,1,256,256], padding="VALID")
        A_cat = self.convolution(A5, [1,1,1,256,n_cat], padding="VALID", act=None)
        A_fv = tf.reshape(A_cat, [-1, n_cat])
        A_class = tf.argmax(tf.nn.softmax(A_fv), axis=-1)
        print(A_class.shape)

        # U0 = self.convolution(A4, [1,1,1,512,256], padding="VALID") #1x1x1x256
        # TODO use A5 as input or A_cat
        U_t = tf.tile(D3, [1,8,8,8,1])
        U_concat = tf.concat([A2, U_t], axis=-1)
        U1 = self.convolution(U_concat, [3,3,3,128+64,128], padding="SAME")
        U1 = tf.layers.batch_normalization(U1, training=bn_training)        
        U1 = self.convolution(U1, [3,3,3,128,128], padding="SAME")
        U1 = tf.layers.batch_normalization(U1, training=bn_training)        
        
        U2 = tf.keras.layers.UpSampling3D([2,2,2])(U1) # to 16
        U_concat1 = tf.concat([A1, U2], axis=-1)
        U2 = self.convolution(U_concat1, [3,3,3,128+64,64], padding="SAME")
        U2 = tf.layers.batch_normalization(U2, training=bn_training)        
        U2 = self.convolution(U2, [3,3,3,64,64], padding="SAME",act=tf.nn.elu)
        U2 = tf.layers.batch_normalization(U2, training=bn_training)        

        U3 = tf.keras.layers.UpSampling3D([2,2,2])(U2) # to 32
        U_concat2 = tf.concat([A0, U3], axis=-1)
        U_mask = self.convolution(U_concat2, [3,3,3,64+64,64], padding="SAME")
        U_mask = tf.layers.batch_normalization(U_mask, training=bn_training)        
        U_mask = self.convolution(U_mask, [1,1,1,64,n_seg], padding="SAME", act=None)
        U_class = tf.argmax(tf.nn.softmax(U_mask), axis=-1)

        return A_fv, A_class, U_mask, U_class


    def forward_propagation3(self, X, n_cat, n_seg, keep_prob, bn_training):
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
        A2_1 = self.convolution(M1, [3,3,3,64,128], padding="SAME") # to 8
        A2 = self.convolution(A2_1, [3,3,3,128,128], padding="VALID") # to 6
        D2 = tf.nn.dropout(A2, keep_prob)
        D2 = tf.layers.batch_normalization(D2, training=bn_training)        
        
        M2 = tf.nn.max_pool3d(D2, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID") # to 3
        A3 = self.convolution(M2, [3,3,3,128,256], padding="VALID") #1x1x1x256
        A4 = self.convolution(A3, [1,1,1,256,512], padding="VALID") #1x1x1x256
        A5 = self.convolution(A4, [1,1,1,512,256], padding="VALID")
        A_cat = self.convolution(A5, [1,1,1,256,n_cat], padding="VALID", act=None)
        A_fv = tf.reshape(A_cat, [-1, n_cat])
        A_class = tf.argmax(tf.nn.softmax(A_fv), axis=-1)
        print(A_class.shape)

        # U0 = self.convolution(A4, [1,1,1,512,256], padding="VALID") #1x1x1x256

        U_t = tf.tile(A4, [1,8,8,8,1])
        U_concat = tf.concat([A2_1, U_t], axis=-1)
        U1 = self.convolution(U_concat, [3,3,3,640,512], padding="SAME")
        U1 = tf.layers.batch_normalization(U1, training=bn_training)        
        U1 = self.convolution(U1, [3,3,3,512,256], padding="SAME")
        U1 = tf.layers.batch_normalization(U1, training=bn_training)        
        
        U2 = tf.keras.layers.UpSampling3D([2,2,2])(U1) # to 16
        U_concat1 = tf.concat([D1, U2], axis=-1)
        U2 = self.convolution(U_concat1, [3,3,3,320,256], padding="SAME")
        U2 = tf.layers.batch_normalization(U2, training=bn_training)        
        U2 = self.convolution(U2, [3,3,3,256,128], padding="SAME")
        U2 = tf.layers.batch_normalization(U2, training=bn_training)        

        U3 = tf.keras.layers.UpSampling3D([2,2,2])(U2) # to 32
        U_concat2 = tf.concat([D0, U3], axis=-1)
        U_mask = self.convolution(U_concat2, [3,3,3,160,256], padding="SAME")
        U_mask = tf.layers.batch_normalization(U_mask, training=bn_training)        
        U_mask = self.convolution(U_mask, [1,1,1,256,128], padding="SAME")
        U_mask = self.convolution(U_mask, [1,1,1,128,n_seg], padding="SAME", act=None)
        U_class = tf.argmax(tf.nn.softmax(U_mask), axis=-1)

        return A_fv, A_class, U_mask, U_class


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
        A1_5 = self.convolution(M0, [5,5,5,64,64], padding="SAME")
        D1_5 = tf.nn.dropout(A1_5, keep_prob)        
        D1_5 = tf.layers.batch_normalization(D1_5, training=bn_training) 

        A1_3 = self.convolution(M0, [3,3,3,64,64], padding="SAME")
        D1_3 = tf.nn.dropout(A1_3, keep_prob)        
        D1_3 = tf.layers.batch_normalization(D1_3, training=bn_training)      
        A1 = tf.concat([D1_3,D1_5], axis=-1)     
        M1 = tf.nn.max_pool3d(A1, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID") # to 8

        # third block
        A2_5 = self.convolution(M1, [5,5,5,128,128], padding="SAME")
        D2_5 = tf.nn.dropout(A2_5, keep_prob)        
        D2_5 = tf.layers.batch_normalization(D2_5, training=bn_training) 

        A2_3 = self.convolution(M1, [3,3,3,128,128], padding="SAME")
        D2_3 = tf.nn.dropout(A2_3, keep_prob)        
        D2_3 = tf.layers.batch_normalization(D2_3, training=bn_training)      
        A2 = tf.concat([D2_3,D2_5], axis=-1)     
        M2 = tf.nn.max_pool3d(A2, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID") # to 4

        A3 = self.convolution(M2, [4,4,4,256,512], padding="VALID") # to 1
        D3 = tf.nn.dropout(A3, keep_prob)
        D3 = tf.layers.batch_normalization(D3, training=bn_training)        
        
        A4 = self.convolution(D3, [1,1,1,512,256], padding="VALID")
        A_cat = self.convolution(A4, [1,1,1,256,n_cat], padding="VALID", act=None)
        A_fv = tf.reshape(A_cat, [-1, n_cat])
        A_class = tf.argmax(tf.nn.softmax(A_fv), axis=-1)
        print(A_class.shape)

        U_t = tf.tile(D3, [1,8,8,8,1])
        U_concat = tf.concat([A2, U_t], axis=-1)
        U1 = self.convolution(U_concat, [1,1,1,512+256,256], padding="SAME")
        U1 = tf.layers.batch_normalization(U1, training=bn_training)        
        U1 = self.convolution(U1, [3,3,3,256,256], padding="SAME")
        U1 = tf.layers.batch_normalization(U1, training=bn_training)        
        
        U2 = tf.keras.layers.UpSampling3D([2,2,2])(U1) # to 16
        U_concat1 = tf.concat([A1, U2], axis=-1)
        U2 = self.convolution(U_concat1, [3,3,3,256+128,128], padding="SAME")
        U2 = tf.layers.batch_normalization(U2, training=bn_training)        
        U2 = self.convolution(U2, [3,3,3,128,128], padding="SAME")
        U2 = tf.layers.batch_normalization(U2, training=bn_training)        

        U3 = tf.keras.layers.UpSampling3D([2,2,2])(U2) # to 32
        U_concat2 = tf.concat([A0, U3], axis=-1)
        U_mask = self.convolution(U_concat2, [3,3,3,128+64,64], padding="SAME")
        U_mask = tf.layers.batch_normalization(U_mask, training=bn_training)        
        U_mask = self.convolution(U_mask, [1,1,1,64,n_seg], padding="SAME", act=None)
        U_class = tf.argmax(tf.nn.softmax(U_mask), axis=-1)

        return A_fv, A_class, U_mask, U_class


    def forward_propagation_best(self, X, n_cat, n_seg, keep_prob, bn_training):
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
        A1_5 = self.convolution(M0, [5,5,5,64,64], padding="SAME")
        D1_5 = tf.nn.dropout(A1_5, keep_prob)        
        D1_5 = tf.layers.batch_normalization(D1_5, training=bn_training) 

        A1_3 = self.convolution(M0, [3,3,3,64,64], padding="SAME")
        D1_3 = tf.nn.dropout(A1_3, keep_prob)        
        D1_3 = tf.layers.batch_normalization(D1_3, training=bn_training)      
        A1 = tf.concat([D1_3,D1_5], axis=-1)     
        M1 = tf.nn.max_pool3d(A1, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID") # to 8

        # third block
        A2_5 = self.convolution(M1, [5,5,5,128,128], padding="SAME")
        D2_5 = tf.nn.dropout(A2_5, keep_prob)        
        D2_5 = tf.layers.batch_normalization(D2_5, training=bn_training) 

        A2_3 = self.convolution(M1, [3,3,3,128,128], padding="SAME")
        D2_3 = tf.nn.dropout(A2_3, keep_prob)        
        D2_3 = tf.layers.batch_normalization(D2_3, training=bn_training)      
        A2 = tf.concat([D2_3,D2_5], axis=-1)     
        M2 = tf.nn.max_pool3d(A2, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID") # to 4

        A3 = self.convolution(M2, [4,4,4,256,512], padding="VALID") # to 1
        D3 = tf.nn.dropout(A3, keep_prob)
        D3 = tf.layers.batch_normalization(D3, training=bn_training)        
        
        # TODO try and remove the 3,3,3 conv and use reshape instead to large vector
        A4 = self.convolution(D3, [1,1,1,512,256], padding="VALID")
        A_cat = self.convolution(A4, [1,1,1,256,n_cat], padding="VALID", act=None)
        A_fv = tf.reshape(A_cat, [-1, n_cat])
        A_class = tf.argmax(tf.nn.softmax(A_fv), axis=-1)
        print(A_class.shape)

        # U0 = self.convolution(A4, [1,1,1,512,256], padding="VALID") #1x1x1x256
        # TODO use A5 as input or A_cat
        U_t = tf.tile(D3, [1,8,8,8,1])
        U_concat = tf.concat([A2, U_t], axis=-1)
        U1 = self.convolution(U_concat, [1,1,1,512+256,256], padding="SAME")
        U1 = tf.layers.batch_normalization(U1, training=bn_training)        
        U1 = self.convolution(U1, [3,3,3,256,256], padding="SAME")
        U1 = tf.layers.batch_normalization(U1, training=bn_training)        
        
        U2 = tf.keras.layers.UpSampling3D([2,2,2])(U1) # to 16
        U_concat1 = tf.concat([A1, U2], axis=-1)
        U2 = self.convolution(U_concat1, [3,3,3,256+128,128], padding="SAME")
        U2 = tf.layers.batch_normalization(U2, training=bn_training)        
        U2 = self.convolution(U2, [3,3,3,128,128], padding="SAME")
        U2 = tf.layers.batch_normalization(U2, training=bn_training)        

        U3 = tf.keras.layers.UpSampling3D([2,2,2])(U2) # to 32
        U_concat2 = tf.concat([A0, U3], axis=-1)
        U_mask = self.convolution(U_concat2, [3,3,3,128+64,64], padding="SAME")
        U_mask = tf.layers.batch_normalization(U_mask, training=bn_training)        
        U_mask = self.convolution(U_mask, [1,1,1,64,n_seg], padding="SAME", act=None)
        U_class = tf.argmax(tf.nn.softmax(U_mask), axis=-1)

        return A_fv, A_class, U_mask, U_class


    def compute_cost(self, U, Y_seg, X, A, Y_cat, n_seg, weights):
        U = U * X 
        Xrep = tf.reshape(X, [-1, X.shape[1], X.shape[2], X.shape[3]])
        entropy_cat = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_cat, logits=A) # used
        weighted_entropy_seg = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(Y_seg, n_seg) * X, logits=U) * Xrep
        # weighted_entropy_seg = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(Y_seg, n_seg) * X, logits=U, weights=(1 - 0.75*tf.pow(weights,8)))
        print(weighted_entropy_seg)
        weighted_entropy_seg = tf.reduce_sum(weighted_entropy_seg, axis=-1)
        weighted_entropy_seg = tf.reduce_sum(weighted_entropy_seg, axis=-1)
        weighted_entropy_seg = tf.reduce_sum(weighted_entropy_seg, axis=-1)
        weighted_entropy_seg = weighted_entropy_seg
        # weighted_entropy_seg = (1.0 - 0.95*tf.pow(weights,3)) * weighted_entropy_seg
        # weighted_entropy_seg = weights * weighted_entropy_seg
        # weighted_entropy_seg = (1.0 - 0.75*tf.pow(weights,3)) * weighted_entropy_seg

        acc = tf.cast(tf.equal(tf.argmax(tf.nn.softmax(U), axis=-1, output_type=tf.int32), Y_seg), tf.float32) * Xrep
        print(acc)
        acc = tf.reduce_sum(acc, axis=-1)
        acc = tf.reduce_sum(acc, axis=-1)
        acc = tf.reduce_sum(acc, axis=-1)
        xsum = tf.reduce_sum(Xrep, axis=-1)
        xsum = tf.reduce_sum(xsum, axis=-1)
        xsum = tf.reduce_sum(xsum, axis=-1)

        acc = acc / xsum

        # weighted_entropy_seg1 = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(Y_seg, n_seg) * X, logits=U_mask1 * X)
        # weighted_entropy_seg2 = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(Y_seg, n_seg) * X, logits=U_mask2 * X)
        reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
        seg_loss = tf.reduce_sum(weighted_entropy_seg)
        cat_loss = tf.reduce_sum(entropy_cat)
        c = reg_loss + seg_loss + cat_loss #+ (tf.reduce_sum(weighted_entropy_seg1) + tf.reduce_sum(weighted_entropy_seg2)) * 0.01

        return c, acc



    def run_model(self, load=False, train=True,visualize=True, in_memory=True, interpolate=True, augment=True):
        log_dir = os.path.join("./3d-object-recognition", self.name)
        tf.reset_default_graph()
        n_cat = self.dataset.num_classes
        n_seg = self.dataset.num_classes_parts
        print(n_cat)
        print(n_seg)
        # get variables and placeholders
        step = tf.Variable(0, trainable=False, name="global_step")
        self.lr_dec = tf.maximum(1e-5, tf.train.exponential_decay(self.lr, step, self.decay_step, self.decay_rate, staircase=True))
        X, Y_seg, Y_cat, keep_prob, bn_training, weight = self.create_placeholders(n_cat, n_seg)
        
        # get model
        A_fv, A_class, U_mask, U_class = self.forward_propagation(X, n_cat, n_seg, keep_prob, bn_training)
        U_vec = tf.nn.softmax(U_mask)
        cost,acc_op = self.compute_cost(U_mask, Y_seg, X, A_fv, Y_cat, n_seg, weight)

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
                occ,seg,cat,names,points,lbs,wgs = self.dataset.next_mini_batch(data_dict)
                # deconvolved_images,d_cost = sess.run([U_class,cost], feed_dict={X: occ, Y_seg: seg, Y_cat: cat, keep_prob: 1.0, bn_training: False, weight: 1.0})
                deconvolved_images,d_cost,pred_class,seg_vec,feat_vec = sess.run([U_class,cost,A_class,U_vec,U_mask], feed_dict={X: occ, Y_seg: seg, Y_cat: cat, keep_prob: 1.0, bn_training: False, weight: wgs})

                # for j in range(0, seg_vec.shape[0]):
                #     mask = np.zeros(n_seg)
                #     cat_dir = Parts.Labels[str(pred_class[j])]
                #     s = Parts.Labels[cat_dir][Parts.PART_START]
                #     e = s + Parts.Labels[cat_dir][Parts.PART_COUNT]
                #     mask[s:e] = 1.0
                #     deconvolved_images[j] = np.argmax(seg_vec[j] * mask, axis=-1)

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
                wious = []
                wious_dev = []
                train_times = []
                wupdate_times = []
                train_infer_time = []
                dev_infer_time = []

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
                        # train for other cases (the augmented ones)
                        if augment:
                            for _ in range(0, 3):
                                occ,seg,cat,names,_,_,wgs = self.dataset.next_mini_batch(self.dataset.train,augment=augment,update=False)
                                wgs = (wgs - min_wgs) / (max_wgs - min_wgs) # normalize in range (0,1)
                                wgs = 1.0 - wgs # best results should have least priority
                                summary,_,d_cost = sess.run([summary_op,train_op,cost], feed_dict={X: occ, Y_cat: cat, Y_seg: seg, keep_prob: self.keep_prob, bn_training: True, weight: wgs})
                                cc = cc + d_cost

                        # always train on unscaled data
                        occ,seg,cat,names,_,_,wgs = self.dataset.next_mini_batch(self.dataset.train)
                        wgs = (wgs - min_wgs) / (max_wgs - min_wgs) # normalize in range (0,1)
                        wgs = 1.0 - wgs # best results should have least priority
                        summary,_,d_cost = sess.run([summary_op,train_op,cost], feed_dict={X: occ, Y_cat: cat, Y_seg: seg, keep_prob: self.keep_prob, bn_training: True, weight: wgs})
                        cc = cc + d_cost
                        print("\rBatch learning %05d/%d" % (i+1,batches),end="")


                    print("")
                    train_times.append(time.time() - stime)
                    self.dataset.restart_mini_batches(self.dataset.train)
                    stime = time.time()                    
                    batches = self.dataset.num_mini_batches(self.dataset.train)
                    for i in range(batches):
                        # occ,seg,cat,names,_,_ = self.dataset.next_mini_batch_augmented(self.dataset.train)
                        occ,seg,cat,_,_,_,wgs = self.dataset.next_mini_batch(self.dataset.train, update=False)
                        out = sess.run([acc_op], feed_dict={X: occ, Y_cat: cat, Y_seg: seg, keep_prob: 1, bn_training: False, weight: wgs})
                        min_wgs = min(min_wgs, np.min(wgs))
                        max_wgs = max(max_wgs, np.max(wgs))
                        self.dataset.update_mini_batch(self.dataset.train, out[0])
                        min_wgs = min(min_wgs, np.min(wgs))
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
                    plt.savefig("./3d-object-recognition/ShapeNet/cost.png", format="png")

                    acc, avg_time = accuracy_test(self.dataset, self.dataset.train)
                    train_infer_time.append(avg_time)
                    train_accuracies.append(acc)
                    plt.clf()
                    # plt.figure(2)
                    plt.plot(np.squeeze(np.array(train_accuracies)))
                    plt.savefig("./3d-object-recognition/ShapeNet/train_accuracies.png", format="png")

                    weighted_average_iou, per_category_iou = self.evaluate_iou_results() # has to be after the accuracy_test, so it has saved and current values
                    wious.append(weighted_average_iou)
                    # plt.figure(3)
                    plt.clf()
                    plt.plot(np.squeeze(np.array(wious)))
                    plt.savefig("./3d-object-recognition/ShapeNet/weighted_average_iou.png", format="png")

                    # plt.figure(4)
                    plt.clf()
                    plt.barh(np.arange(n_cat),per_category_iou, tick_label=list(Parts.label_dict.keys()))
                    # plt.barh(np.arange(n_cat),per_category_iou, tick_label=["airplane", "bag", "cap", "car", "chair", "earphone", "guitar", "knife", "lamp", "laptop", "motorbike", "mug", "pistol", "rocket", "skateboard", "table"])
                    plt.savefig("./3d-object-recognition/ShapeNet/per_category_iou" + str(epoch+1) + ".png", format="png")
                    
                    acc, avg_time = accuracy_test(self.dataset, self.dataset.dev, in_memory=in_memory)
                    dev_infer_time.append(avg_time)
                    dev_accuracies.append(acc)
                    plt.clf()
                    plt.plot(np.squeeze(np.array(dev_accuracies)))
                    plt.savefig("./3d-object-recognition/ShapeNet/dev_accuracies.png", format="png")

                    weighted_average_iou, per_category_iou = self.evaluate_iou_results(self.dataset.dev, in_memory=in_memory)
                    wious_dev.append(weighted_average_iou)
                    # plt.figure(3)
                    plt.clf()
                    plt.plot(np.squeeze(np.array(wious_dev)))
                    plt.savefig("./3d-object-recognition/ShapeNet/weighted_average_iou_dev.png", format="png")
                    
                    print("")
                    # do check for file barrier, if so, break training cycle
                    if os.path.exists(os.path.join(log_dir, "barrier.txt")):
                        break


                array2csv = []
                array2csv.append(train_accuracies)
                array2csv.append(dev_accuracies)
                array2csv.append(wious)
                array2csv.append(wious_dev)
                array2csv.append(train_times)
                array2csv.append(wupdate_times)
                array2csv.append(train_infer_time)
                array2csv.append(dev_infer_time)
                array2csv = np.array(array2csv) # reshape so the values are in the rows
                array2csv = array2csv.transpose()
                np.savetxt("./3d-object-recognition/ShapeNet/Results/" + self.name + ".csv", array2csv, delimiter=",", comments='', header="Train accuracy,Dev accuracy,"\
                "Train weighted average IOU,Dev weighted average IOU,Training times/epoch (ms), Weight update time/epoch (ms),"\
                "Train inference times (ms),Dev inference times (ms)")
                shutil.copy("./3d-object-recognition/ShapeNet/network.json", "./3d-object-recognition/ShapeNet/Results/network.json")
                print("Max wIOU train %f" % np.max(wious))
                print("Max wIOU dev %f" % np.max(wious_dev))
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
    # s = PartsNet("ShapeNet", "./3d-object-recognition/UnityData")
    s = PartsNet("ShapeNet", "./3d-object-recognition/ShapePartsData")
    s.run_model(load=True, train=True,visualize=False, in_memory=True,interpolate=False,augment=True)
