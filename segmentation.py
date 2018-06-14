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
        Y = tf.placeholder(dtype=tf.int32, shape=(None), name="input_distributions")
        U_deconv = tf.placeholder(dtype=tf.float32, shape=(1,16,16,16,512), name="input_deconv")
        Y_deconv = tf.placeholder(dtype=tf.int32, shape=(1,32,32,32), name="deconv_mask")
        # Y = tf.placeholder(dtype=tf.int32, shape=(1,320,128,192,1), name="input_labels")
        keep_prob = tf.placeholder(dtype=tf.float32, name="keep_probability")

        return X, Y, Y_deconv, U_deconv, keep_prob


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
        # U0 = self.convolution2d(U, [3,3,256,256], padding="SAME")
        # U0 = self.convolution2d(U0, [3,3,256,256], padding="SAME")
        # U0 = self.convolution2d(U0, [3,3,256,256], padding="SAME")
        # feature combining
        U0 = self.convolution(U, [3,3,3,512,512], padding="SAME")
        U1 = tf.keras.layers.UpSampling3D([2,2,2])(U0) # to 32
        U1 = self.convolution(U1, [3,3,3,512,256], padding="SAME")
        U1 = self.convolution(U1, [3,3,3,256,256], padding="SAME")
        # classification part
        U_mask = self.convolution(U1, [1,1,1,256,128], padding="VALID")
        U_mask = self.convolution(U_mask, [1,1,1,128,64], padding="VALID")
        U_mask = self.convolution(U_mask, [1,1,1,64,n_y], padding="VALID")

        return U_mask


    def forward_propagation_fv(self, X, n_y, keep_prob):
        self.layer_idx = 0
        # imagine that the net operates over 32x32x32 blocks
        # feature vector learning
        # IN 32, to 28
        A0 = self.convolution(X, [5,5,5,1,64], padding="VALID") 
        A0 = tf.nn.dropout(A0, keep_prob)
        A0 = tf.nn.max_pool3d(A0, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID") # to 14
        A1 = self.convolution(A0, [5,5,5,64,128], padding="VALID") # to 10
        A1 = tf.nn.dropout(A1, keep_prob)        
        A1 = tf.nn.max_pool3d(A1, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID") # to 5
        A2 = self.convolution(A1, [3,3,3,128,256], padding="VALID") # to 3 
        A2 = tf.nn.dropout(A2, keep_prob)
        A_fv = self.convolution(A2, [3,3,3,256,512], padding="VALID") #1x1x1x256

        # classification learning
        A_class1 = self.convolution(A_fv, [1,1,1,512,256], padding="VALID") #1x1x1x256
        A_class2 = self.convolution(A_class1, [1,1,1,256,128], padding="VALID") #1x1x1x256
        A_class = self.convolution(A_class2, [1,1,1,128,n_y], padding="VALID", act=None) #1x1x1x256
        print(A_fv)
        return A_fv, A_class


    def compute_cost_deconv(self, U, Y):
        print(U)
        print(Y)
        c = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=U))
        print(c)
        return c


    def compute_cost_fv(self, Z, Y):
        print(Z)
        print(Y)
        return tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=Z))  # compute cost between distributions


    def compute_predictions(self, U_labels):
        return tf.argmax(U_labels, axis=-1)


    def run_model(self, load=False, train_fv=True, train_d=True,visualize=True):
        log_dir = os.path.join("./3d-object-recognition", self.name)
        tf.reset_default_graph()
        n_y = self.dataset.num_classes
        X, Y, Y_deconv, U_deconv, keep_prob = self.create_placeholders(n_y)
        A_fv, A_class = self.forward_propagation_fv(X, n_y, keep_prob)
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
        lr_dec_d = tf.train.exponential_decay(self.lrd, step_d, self.decay_step, self.decay_rate)
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
                D = np.zeros((1,16,16,16,512), dtype=np.float)
                # print("\rBatch loaded in %f\n*********************\n" % (time.time() - batch_stime))
                for j in range(0, y.shape[0],sub_batches):
                    start = j
                    end = min(y.shape[0],sub_batches+j)
                    A = sess.run([A_fv], feed_dict={X: x[start:end], Y: y[start:end], keep_prob: 1.0})
                    A = np.array(A)[0]
                    for k in range(0,min(sub_batches, A.shape[0])):
                        idx = idxs[k]
                        D[0,idx[0],idx[1],idx[2],:] = A[k,0,0,0,:] + D[0,idx[0],idx[1],idx[2],:]

                deconvolved_image, d_cost = sess.run([tf.argmax(tf.nn.softmax(A_deconv, axis=-1), axis=-1), cost_d], feed_dict={U_deconv: D, Y_deconv: d_labels, keep_prob: 1.0})
                deconvolved_image = np.array(deconvolved_image)[0]
                acc = acc + d_cost
                if visualize:
                    dl.load_xyzl_vis(filename, deconvolved_image, n_y)

                print("\rData loaded in \t %f sec, Convolved in \t %f sec" % (ltime_end - ltime, time.time() - stime), end="")

            print("Deconvolution average cost %f" % (acc / data_dict[dataset.NUMBER_EXAMPLES]))    
            return float(acc / data_dict[dataset.NUMBER_EXAMPLES])

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init)

            if load:
                tf.train.Saver().restore(sess, os.path.join(log_dir, self.name + ".ckpt"))
                print("Model loaded from file: %s" % os.path.join(log_dir, self.name + ".ckpt"))

            if train_d or train_fv:
                costs = []
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
                    if train_fv:
                        accuracy = 0
                        for i in range(batches):
                            acc = 0
                            batch_stime = time.time()
                            x,y,names = self.dataset.next_class_mini_batch(self.dataset.train)
                            # print("\rBatch loaded in %f\n*********************\n" % (time.time() - batch_stime))
                            _,c,labs = sess.run([train_op_fv,cost_fv, tf.argmax(tf.nn.softmax(A_class, axis=-1), axis=-1)], feed_dict={X: x, Y: y, keep_prob: self.keep_prob})
                            cc = cc + c / y.shape[0]
                            for j in range(0, y.shape[0]):
                                if y[j] == labs[j,0,0,0]:
                                    acc = acc + 1
                            accuracy = accuracy + acc  / y.shape[0]
                            print("\rBatch \t %d/%d \t trained in \t %f" % ((i+1), batches, time.time() - batch_stime), end="")

                        accuracy = accuracy / batches
                        print("\nAccuracy of class classification %f and cost %f" % (accuracy, cc/batches))

                    costs.append(cc / (batches))

                    batches = self.dataset.num_examples(self.dataset.train)
                    batches_d = self.dataset.num_examples(self.dataset.train_d)
                    if train_d:
                        cc = 0
                        # evaluate the object batch
                        self.dataset.restart_mini_batches(self.dataset.train)
                        for i in range(batches):
                            # evaluate the last layer after the training so we have reasonable results
                            # flatten the grid 8x8x8 to image 8x8
                            D = np.zeros((1,16,16,16,512), dtype=np.float)
                            x,y,mask,idxs,_,d_labels = self.dataset.next_mini_batch(self.dataset.train)
                            for j in range(0, y.shape[0],sub_batches):
                                start = j
                                end = min(y.shape[0],sub_batches+j)
                                A = sess.run([A_fv], feed_dict={X: x[start:end], Y: y[start:end], keep_prob: 1.0})
                                A = np.array(A)
                                for k in range(0,min(sub_batches, A.shape[0])):
                                    idx = idxs[k]
                                    D[0,idx[0],idx[1],idx[2],:] = A[k,0,0,0,:] + D[0,idx[0],idx[1],idx[2],:]
                            
                            # for j in range(0, y.shape[0]):
                            #     # vizualize x
                            #     xs = []
                            #     ys = []
                            #     zs = []
                            #     for a in range(0,32):
                            #         for b in range(0,32):
                            #             for c in range(0,32):
                            #                 if x[j,a,b,c,0] > 0:
                            #                     xs.append(a)
                            #                     ys.append(b)
                            #                     zs.append(c)
                                
                            #     fig = plt.figure()
                            #     ax = fig.add_subplot(111, projection='3d')
                            #     ax.scatter(xs, ys, zs, c=[1.0, 0.0, 0.0, 0.8], marker='p')
                            #     ax.set_xlabel('X Label')
                            #     ax.set_ylabel('Y Label')
                            #     ax.set_zlabel('Z Label')
                            #     ax.set_xlim(0,32)
                            #     ax.set_ylim(0,32)
                            #     ax.set_zlim(0,32)
                            #     plt.show()

                            _,d_cost = sess.run([train_op_d,cost_d], feed_dict={U_deconv: D, Y_deconv: d_labels, keep_prob: self.keep_prob})
                            cc = cc + d_cost
                            print("\rBatch \t %03d/%d" % ((i+1), batches), end="")
                            

                        print("\nObject deconvolution cost %f" % (cc / batches))
                        cc = 0
                        # evaluate the scene batch
                        for i in range(batches_d):
                            # evaluate the last layer after the training so we have reasonable results
                            # flatten the grid 8x8x8 to image 8x8
                            D = np.zeros((1,16,16,16,512), dtype=np.float)
                            x,y,mask,idxs,_,d_labels = self.dataset.next_mini_batch(self.dataset.train_d)
                            for j in range(0, y.shape[0],sub_batches):
                                start = j
                                end = min(y.shape[0],sub_batches+j)
                                A = sess.run([A_fv], feed_dict={X: x[start:end], Y: y[start:end], keep_prob: 1.0})
                                A = np.array(A)
                                for k in range(0,min(sub_batches, A.shape[0])):
                                    idx = idxs[k]
                                    D[0,idx[0],idx[1],idx[2],:] = np.maximum(A[k,0,0,0,:],D[0,idx[0],idx[1],idx[2],:])

                            _,d_cost = sess.run([train_op_d,cost_d], feed_dict={U_deconv: D, Y_deconv: d_labels, keep_prob: self.keep_prob})
                            cc = cc + d_cost

                        cc = cc / batches_d

                        print("Epoch %d trained in %f, cost %f, dcost %f" % (epoch+1, time.time() - stime, costs[-1], cc))
                        

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

            print(accuracy_test(self.dataset, self.dataset.train))
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
    s.run_model(load=True, train_fv=False, train_d=False,visualize=True)
