import tensorflow as tf 
import numpy as np 
import json
import os
from nn_utils import *
from shapenetparts_per_point import Parts
import sys
import time
import matplotlib.pyplot as plt


class PartsNet():

    def create_placeholders(self, n_y):
        X = tf.placeholder(dtype=tf.float32, shape=(None,self.dataset.shape[0],self.dataset.shape[1]), name="input_grid")
        for_concat = tf.placeholder(dtype=tf.float32, shape=(None,1,1), name="for_concat_global")
        Y_seg = tf.placeholder(dtype=tf.int32, shape=(None), name="segmentation_labels")
        Y_cat = tf.placeholder(dtype=tf.int32, shape=(None), name="category_labels")
        keep_prob = tf.placeholder(dtype=tf.float32, name="keep_probability")
        bn_training = tf.placeholder(dtype=tf.bool, name="batch_norm_training")

        return X, Y_seg, Y_cat, keep_prob, bn_training, for_concat

    
    def convolution(self, X, shape, strides=1, padding="SAME", act=tf.nn.relu):
        # tf.contrib.layers.xavier_initializer(uniform=False)
        W = tf.get_variable("weights" + str(self.layer_idx), shape, initializer=tf.variance_scaling_initializer(), dtype=tf.float32, regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_w))
        b = tf.get_variable("biases" + str(self.layer_idx), shape[-1], initializer=tf.zeros_initializer(), dtype=tf.float32, regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_b))
        Z = tf.nn.conv1d(X, W, stride=strides, padding=padding) + b

        tf.summary.histogram("weights" + str(self.layer_idx), W)
        tf.summary.histogram("biases" + str(self.layer_idx), b)

        if act != None:
            Z = act(Z)

        self.layer_idx = self.layer_idx + 1
        return Z


    def block(self, X, filter_size, in_size, out_size, keep_prob, bn_training, activation=tf.nn.relu):
        A = self.convolution(X, [filter_size,in_size,out_size], act=activation, padding="VALID")
        D = tf.nn.dropout(A, keep_prob)
        # D = tf.layers.batch_normalization(D, training=bn_training)
        print(D)

        return D


    def forward_propagation(self, X, n_cat, n_seg, keep_prob, bn_training, for_concat):
        self.layer_idx = 0
        # imagine that the net operates over nxkx3 points
        # feature vector learning
        print(X)
        L0 = self.block(X, self.dataset.k, 3, self.dataset.k, keep_prob, bn_training)
        L1 = self.block(L0, 1, self.dataset.k, 64, keep_prob, bn_training)
        L2 = self.block(L1, 1, 64, 128, keep_prob, bn_training)
        L3 = self.block(L2, 1, 128, 256, keep_prob, bn_training)

        # M3 = self.convolution(tf.reshape(L3, [-1,256*self.dataset.k]), [1,256*self.dataset.k,512], padding="VALID")
        M3 = self.block(L3, 1, 256, 512, keep_prob, bn_training, )
        print(M3)

        L4 = tf.layers.dense(M3, 512, kernel_initializer=tf.variance_scaling_initializer(), activation=tf.nn.relu)
        print(L4)
        L5 = tf.reduce_max(L4, axis=0, keepdims=True)
        print(L5)
        L6 = tf.layers.dense(L5, 256, kernel_initializer=tf.variance_scaling_initializer(), activation=tf.nn.relu)
        print(L6)
        A_fv = tf.layers.dense(L6, n_cat, kernel_initializer=tf.variance_scaling_initializer())
        print(A_fv)
        A_class = tf.argmax(tf.nn.softmax(A_fv), axis=-1)
        print(A_class)
        # A_class = tf.squeeze(A_class)
        # print(A_class)

        C = tf.reshape(M3, [-1,1,512])
        C = tf.concat([self.convolution(C, [1,512,512]),tf.reshape(L5,[512])*for_concat],axis=-1)
        C = self.convolution(C, [1,1024,512])
        C = self.convolution(C, [1,512,256])
        U_mask = self.convolution(C, [1,256,n_seg], act=None)  
        print(U_mask)   
        U_class = tf.argmax(tf.nn.softmax(U_mask), axis=-1)
        print(U_class)
        return A_fv, A_class, U_mask, U_class




    def compute_cost(self, U, Y_seg, X, A, Y_cat, n_seg):
        entropy_cat = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_cat, logits=A)
        weighted_entropy_seg = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(Y_seg, n_seg), logits=U)

        reg_losses = tf.losses.get_regularization_losses()
        c = tf.reduce_sum(reg_losses) + tf.reduce_sum(weighted_entropy_seg) + tf.reduce_sum(entropy_cat)

        return c, c



    def run_model(self, load=False, train=True,visualize=True, in_memory=True):
        log_dir = os.path.join("./3d-object-recognition", self.name)
        tf.reset_default_graph()
        n_cat = self.dataset.num_classes
        n_seg = self.dataset.num_classes_parts
        print(n_cat)
        print(n_seg)
        X, Y_seg, Y_cat, keep_prob, bn_training, for_concat = self.create_placeholders(n_cat)
        A_fv, A_class, U_mask, U_class = self.forward_propagation(X, n_cat, n_seg, keep_prob, bn_training, for_concat)
        cost, tmp_test = self.compute_cost(U_mask, Y_seg, X, A_fv, Y_cat, n_seg)

        # fv part
        step = tf.Variable(0, trainable=False, name="global_step")
        lr_dec = tf.maximum(1e-5, tf.train.exponential_decay(self.lr, step, self.decay_step, self.decay_rate, staircase=True))
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_dec)
        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_ops):
            gvs = optimizer.compute_gradients(cost)
            # print(gvs)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs, global_step=step)
            # train_op = optimizer.minimize(cost)

        init = tf.global_variables_initializer()
        tf.summary.scalar("learning_rate", lr_dec)
        tf.summary.scalar("global_step", step)
        tf.summary.scalar("cost", cost)
        writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())
        summary_op = tf.summary.merge_all()
        writer.flush()
            

        def accuracy_test(dataset, data_dict, in_memory=True):
            acc = 0
            acc_cat = 0
            dataset.restart_mini_batches(data_dict)
            dataset.clear_segmentation(data_dict, in_memory=in_memory)
            for i in range(dataset.num_mini_batches(data_dict)):
                stime = time.time()
                dat,seg,cat,names,pts = self.dataset.next_mini_batch(data_dict)
                # deconvolved_images,d_cost = sess.run([U_class,cost], feed_dict={X: occ, Y_seg: seg, Y_cat: cat, keep_prob: 1.0, bn_training: False, weight: 1.0})
                deconvolved_images,d_cost,pred_class = sess.run([U_class,cost,A_class], feed_dict={X: dat[0], Y_seg: seg[0], Y_cat: cat, keep_prob: 1.0, bn_training: False, for_concat: np.ones((dat[0].shape[0],1,1))})

                # print("Average interference time per mini batch example %f sec" % ((time.time() - stime) / occ.shape[0]))
                acc = acc + np.sum(seg[0] == np.reshape(deconvolved_images,[-1])) / seg[0].shape[0]
                acc_cat = acc_cat + np.sum(cat[0] == pred_class) / cat.shape[0]

                for j in range(0, dat.shape[0]):
                    dataset.save_segmentation(seg[j], np.reshape(deconvolved_images,[-1]), names[j], data_dict, in_memory=in_memory)
 
                if visualize:
                    for j in range(0, deconvolved_images.shape[0]):
                        print(names[j])

                print("\rEvaluating %s: %d %%..." % (data_dict["name"], i*100 / dataset.num_mini_batches(data_dict)), end="")

            print("\r%s deconvolution average accuracy %f" % (data_dict["name"], acc / dataset.num_mini_batches(data_dict)))
            print("%s category accuracy %f" % (data_dict["name"], acc_cat / dataset.num_mini_batches(data_dict)))
            return float(acc / dataset.num_mini_batches(data_dict))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:

            if load:
                tf.train.Saver().restore(sess, os.path.join(log_dir, self.name + ".ckpt"))
                print("Model loaded from file: %s" % os.path.join(log_dir, self.name + ".ckpt"))
            else:
                sess.run(init)

            if train:
                costs = []
                train_accuracies = []
                wious = []
                wious_dev = []

                for epoch in range(0, self.num_epochs):
                    self.dataset.restart_mini_batches(self.dataset.train)
                    batches = self.dataset.num_mini_batches(self.dataset.train)
                    # evaluate the scene batch
                    cc = 0
                    stime = time.time()
                    for i in range(batches):
                        # occ,seg,cat,names,_,_ = self.dataset.next_mini_batch_augmented(self.dataset.train)
                        dat,seg,cat,names,pts = self.dataset.next_mini_batch(self.dataset.train)
                        summary,_,d_cost,tmp = sess.run([summary_op,train_op,cost,tmp_test], feed_dict={X: dat[0], Y_cat: cat, Y_seg: seg[0], keep_prob: self.keep_prob, bn_training: True, for_concat: np.ones((dat[0].shape[0],1,1))})
                        cc = cc + d_cost
                        print("\rBatch %03d/%d" % (i+1,batches),end="")
                    
                    writer.add_summary(summary, epoch)
                    cc = cc / (self.dataset.num_examples(self.dataset.train))
                    costs.append(cc)
                    print("\nEpoch %d trained in %f, cost %f" % (epoch+1, time.time() - stime, cc))
                    
                    
                    # save model
                    save_path = tf.train.Saver().save(sess, os.path.join(log_dir, self.name + str(epoch+1) + ".ckpt"))
                    print("Model saved in file: %s" % save_path)
                    # plt.figure(1)
                    plt.clf()
                    plt.plot(np.squeeze(np.array(costs)))
                    plt.savefig("./3d-object-recognition/PerPointNet/cost.png", format="png")

                    train_accuracies.append(accuracy_test(self.dataset, self.dataset.train))
                    plt.clf()
                    # plt.figure(2)
                    plt.plot(np.squeeze(np.array(train_accuracies)))
                    plt.savefig("./3d-object-recognition/PerPointNet/train_accuracies.png", format="png")

                    weighted_average_iou, per_category_iou = self.evaluate_iou_results() # has to be after the accuracy_test, so it has saved and current values
                    wious.append(weighted_average_iou)
                    # plt.figure(3)
                    plt.clf()
                    plt.plot(np.squeeze(np.array(wious)))
                    plt.savefig("./3d-object-recognition/PerPointNet/weighted_average_iou.png", format="png")

                    # plt.figure(4)
                    plt.clf()
                    plt.barh(np.arange(n_cat),per_category_iou, tick_label=list(Parts.label_dict.keys()))
                    # plt.barh(np.arange(n_cat),per_category_iou, tick_label=["airplane", "bag", "cap", "car", "chair", "earphone", "guitar", "knife", "lamp", "laptop", "motorbike", "mug", "pistol", "rocket", "skateboard", "table"])
                    plt.savefig("./3d-object-recognition/PerPointNet/per_category_iou" + str(epoch+1) + ".png", format="png")
                    
                    accuracy_test(self.dataset, self.dataset.dev, in_memory=in_memory)
                    weighted_average_iou, per_category_iou = self.evaluate_iou_results(self.dataset.dev, in_memory=in_memory)
                    wious_dev.append(weighted_average_iou)
                    # plt.figure(3)
                    plt.clf()
                    plt.plot(np.squeeze(np.array(wious_dev)))
                    plt.savefig("./3d-object-recognition/PerPointNet/weighted_average_iou_dev.png", format="png")
                    print("")
                    # do check for file barrier, if so, break training cycle
                    if os.path.exists(os.path.join(log_dir, "barrier.txt")):
                        break
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
            self.dataset =      Parts(datapath, batch_size=jparams["batch_size"], k=jparams["k"], m=jparams["m"])


if __name__ == "__main__":
    s = PartsNet("PerPointNet", "./3d-object-recognition/ShapePartsData")
    s.run_model(load=False, train=True,visualize=False, in_memory=True)
