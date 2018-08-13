import tensorflow as tf 
import numpy as np 
import json
import os
from nn_utils import *
from shapenetparts import Parts
import sys
import time
import matplotlib.pyplot as plt


class PartsNet():

    def create_placeholders(self, n_y):
        X = tf.placeholder(dtype=tf.float32, shape=(None,32,32,32,1), name="input_grid")
        Y_seg = tf.placeholder(dtype=tf.int32, shape=(None,32,32,32), name="segmentation_labels")
        Y_cat = tf.placeholder(dtype=tf.int32, shape=(None), name="category_labels")
        keep_prob = tf.placeholder(dtype=tf.float32, name="keep_probability")
        bn_training = tf.placeholder(dtype=tf.bool, name="batch_norm_training")

        return X, Y_seg, Y_cat, keep_prob, bn_training

    
    def convolution(self, X, shape, strides=[1,1,1,1,1], padding="SAME", act=tf.nn.relu):
        W = tf.get_variable("weights" + str(self.layer_idx), shape, initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        b = tf.get_variable("biases" + str(self.layer_idx), shape[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
        Z = tf.nn.conv3d(X, W, strides=strides, padding=padding) + b
        if act != None:
            Z = act(Z)

        self.layer_idx = self.layer_idx + 1
        return Z


    def forward_propagation(self, X, n_cat, n_seg, keep_prob, bn_training):
        self.layer_idx = 0

        A0 = self.convolution(X, [5,5,5,1,32], padding="SAME") 
        D0 = tf.nn.dropout(A0, keep_prob)
        D0 = tf.layers.batch_normalization(D0, training=bn_training)
        A1 = self.convolution(D0, [5,5,5,32,32], padding="SAME")
        D1 = tf.nn.dropout(A1, keep_prob)
        D1 = tf.layers.batch_normalization(D1, training=bn_training)        

        A2 = self.convolution(D1, [5,5,5,32,32], padding="SAME")
        D2 = tf.nn.dropout(A2, keep_prob)
        D2 = tf.layers.batch_normalization(D2, training=bn_training)        

        A3 = self.convolution(D2, [5,5,5,32,32], padding="SAME")
        D3 = tf.nn.dropout(A3, keep_prob)
        D3 = tf.layers.batch_normalization(D3, training=bn_training)        

        A4 = self.convolution(D3, [3,3,3,32,32], padding="SAME")
        D4 = tf.nn.dropout(A4, keep_prob) 
        D4 = tf.layers.batch_normalization(D4, training=bn_training)        

        # category
        A5 = self.convolution(D4, [5,5,5,32,128], padding="VALID") # 28
        D5 = tf.nn.dropout(A5, keep_prob) 
        D5 = tf.layers.batch_normalization(D5, training=bn_training)        
        M5 = tf.nn.max_pool3d(D5, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID") # to 14

        A6 = self.convolution(M5, [5,5,5,128,256], padding="VALID") # 10
        D6 = tf.nn.dropout(A6, keep_prob) 
        D6 = tf.layers.batch_normalization(D6, training=bn_training)        
        M6 = tf.nn.max_pool3d(D6, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID") # to 5

        A_fv = self.convolution(M6, [5,5,5,256,512], padding="VALID") # to 1
        A_class = self.convolution(A_fv, [1,1,1,512,256], padding="VALID")
        A_class = tf.reshape(self.convolution(A_class, [1,1,1,256,n_cat], padding="VALID", act=None), [-1, n_cat])
        print(A_class.shape)

        # per pixel category
        A7 = self.convolution(D4, [1,1,1,32,64], padding="VALID")
        D7 = tf.nn.dropout(A7, keep_prob)
        D7 = tf.layers.batch_normalization(D7, training=bn_training)        

        A8 = self.convolution(D7, [1,1,1,64,64], padding="VALID")
        D8 = tf.nn.dropout(A8, keep_prob) 
        D8 = tf.layers.batch_normalization(D8, training=bn_training)        

        A9 = self.convolution(D8, [1,1,1,64,64], padding="VALID")
        D9 = tf.nn.dropout(A9, keep_prob)
        D9 = tf.layers.batch_normalization(D9, training=bn_training)        

        A10 = self.convolution(D9, [1,1,1,64,64], padding="VALID")
        D10 = tf.nn.dropout(A10, keep_prob)   
        D10 = tf.layers.batch_normalization(D10, training=bn_training)        

        U_mask = self.convolution(D10, [1,1,1,64,n_seg], padding="SAME", act=None)
        U_class = tf.argmax(tf.nn.softmax(U_mask), axis=-1)

        # return  U_mask, U_class, U_mask, U_class        
        return A_fv, A_class, U_mask, U_class        


    def forward_propagation2(self, X, n_cat, n_seg, keep_prob, bn_training):
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
        A_fv = self.convolution(A3, [1,1,1,256,512], padding="VALID") #1x1x1x256
        A_class = self.convolution(A_fv, [1,1,1,512,256], padding="VALID")
        A_class = tf.reshape(self.convolution(A_class, [1,1,1,256,n_cat], padding="VALID", act=None), [-1, n_cat])
        print(A_class.shape)

        U0 = self.convolution(A_fv, [1,1,1,512,256], padding="VALID") #1x1x1x256

        U_t = tf.tile(U0, [1,8,8,8,1])
        U_concat = tf.concat([M1, U_t], axis=-1)
        U1 = self.convolution(U_concat, [3,3,3,320,256], padding="SAME")
        U1 = tf.layers.batch_normalization(U1, training=bn_training)        
        U1 = self.convolution(U1, [3,3,3,256,256], padding="SAME")
        U1 = tf.layers.batch_normalization(U1, training=bn_training)        
        
        U2 = tf.keras.layers.UpSampling3D([2,2,2])(U1) # to 16
        U_concat1 = tf.concat([M0, U2], axis=-1)
        U2 = self.convolution(U_concat1, [3,3,3,288,128], padding="SAME")
        U2 = tf.layers.batch_normalization(U2, training=bn_training)        
        U2 = self.convolution(U2, [3,3,3,128,128], padding="SAME")
        U2 = tf.layers.batch_normalization(U2, training=bn_training)        

        U3 = tf.keras.layers.UpSampling3D([2,2,2])(U2) # to 32
        U_mask = self.convolution(U3, [3,3,3,128,128], padding="SAME")
        U_mask = self.convolution(U_mask, [1,1,1,128,128], padding="SAME")
        U_mask = self.convolution(U_mask, [1,1,1,128,n_seg], padding="SAME", act=None)
        U_class = tf.argmax(tf.nn.softmax(U_mask), axis=-1)

        return A_fv, A_class, U_mask, U_class


    def compute_cost(self, U, Y_seg, X, A, Y_cat, n_seg):
        # U = U * X
        # print(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=U)*tf.reshape(X, [-1, X.shape[1], X.shape[2], X.shape[3]]))
        # Xrep = tf.reshape(X, [-1, 1])
        Xrep = tf.reshape(X, [-1, X.shape[1], X.shape[2], X.shape[3]])
        one_hot_labels = tf.reshape(tf.one_hot(Y_seg, n_seg), [-1, n_seg])
        # segmentation_weights = tf.reduce_sum(one_hot_labels, axis=0)
        # segmentation_weights = 2 - segmentation_weights / tf.reduce_sum(segmentation_weights)
        # # make it more drastic with cubic and exp
        # segmentation_weights = segmentation_weights * segmentation_weights * segmentation_weights
        # segmentation_weights = tf.exp(segmentation_weights)
        # segmentation_weights = segmentation_weights * Xrep
        predictions = tf.reshape(U, [-1,n_seg])
        entropy_cat = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_cat, logits=A)

        # # precomputed in compute_weights.py, depends on number of points in category, some cubic and exp scaling
        # weights = [ 4.80975095e-02, 9.90898234e-02, 3.85734095e-01, 5.01347729e-01,
        #             9.88436474e-01, 8.11319237e-01, 8.92995723e-01, 9.62735516e-01,
        #             8.62868961e-01, 8.19563785e-01, 6.46282996e-01, 1.56735874e-01,
        #             4.04370035e-02, 2.24167763e-02, 1.20006511e-01, 6.74435177e-01,
        #             8.93703008e-01, 9.55680156e-01, 9.79336081e-01, 8.47372667e-01,
        #             6.88981213e-01, 2.53654299e-01, 6.36732805e-01, 6.45198989e-01,
        #             5.83778654e-01, 1.16268283e-01, 9.56457696e-01, 4.68781083e-01,
        #             4.87983729e-01, 5.35777850e-01, 9.78915904e-01, 9.82482226e-01,
        #             8.86548432e-01, 9.95355861e-01, 1.00000000e+00, 7.12441534e-01,
        #             9.69314553e-01, 5.95564071e-01, 5.81312538e-01, 7.79297715e-01,
        #             9.57090876e-01, 8.91458273e-01, 9.73534657e-01, 9.84085331e-01,
        #             9.58625410e-01, 7.17993818e-01, 9.74575434e-01, 9.15025316e-04,
        #             4.20287163e-02, 6.19380876e-01]
        # weights = np.array(weights)
        # segmentation_weights = tf.convert_to_tensor(weights, dtype=tf.float32) * Xrep

        weighted_entropy_seg = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_seg, logits=U) * Xrep
        c = tf.reduce_sum(weighted_entropy_seg) + tf.reduce_sum(entropy_cat)
        # c = tf.losses.mean_pairwise_squared_error(one_hot_labels, predictions, weights=segmentation_weights) + tf.reduce_sum(entropy_cat)


        print(c)# try multiplying the result by a weight
        return c, c



    def run_model(self, load=False, train=True,visualize=True):
        log_dir = os.path.join("./3d-object-recognition", self.name)
        tf.reset_default_graph()
        n_cat = self.dataset.num_classes
        n_seg = self.dataset.num_classes_parts
        print(n_cat)
        print(n_seg)
        X, Y_seg, Y_cat, keep_prob, bn_training = self.create_placeholders(n_cat)
        A_fv, A_class, U_mask, U_class = self.forward_propagation(X, n_cat, n_seg, keep_prob, train)
        cost, tmp_test = self.compute_cost(U_mask, Y_seg, X, A_class, Y_cat, n_seg)

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
            dataset.clear_segmentation(data_dict)
            for i in range(dataset.num_mini_batches(data_dict)):
                stime = time.time()
                occ,seg,cat,names,points,lbs = self.dataset.next_mini_batch(data_dict)
                deconvolved_images,d_cost,pred_class = sess.run([U_class,cost,A_class], feed_dict={X: occ, Y_seg: seg, Y_cat: cat, keep_prob: 1.0, bn_training: False})

                xresh = np.reshape(occ, [-1, occ.shape[1], occ.shape[2], occ.shape[3]])
                a = np.sum((xresh * deconvolved_images) == seg) / np.sum(xresh)
                predicted_category = np.argmax(pred_class, axis=-1)
                # print("Average interference time per mini batch example %f sec" % ((time.time() - stime) / occ.shape[0]))
                acc = acc + a
                acc_cat = acc_cat + np.sum(cat == predicted_category) / predicted_category.shape[0]
                for j in range(0, deconvolved_images.shape[0]):
                    dataset.save_segmentation(lbs[j], deconvolved_images[j], names[j], points[j], data_dict)

                if visualize:
                    for j in range(0, deconvolved_images.shape[0]):
                        print(names[j])
                        dataset.vizualise_batch(seg[j],deconvolved_images[j],cat[j],predicted_category[j],xresh[j],names[j])

            print("Deconvolution average accuracy %f" % (acc / dataset.num_mini_batches(data_dict)))
            print("Deconvolution average category accuracy %f" % (acc_cat / dataset.num_mini_batches(data_dict)))
            return float(acc / dataset.num_mini_batches(data_dict))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init)

            if load:
                tf.train.Saver().restore(sess, os.path.join(log_dir, self.name + ".ckpt"))
                print("Model loaded from file: %s" % os.path.join(log_dir, self.name + ".ckpt"))

            if train:
                costs = []
                train_accuracies = []
                wious = []
                for epoch in range(0, self.num_epochs):
                    self.dataset.restart_mini_batches(self.dataset.train)
                    batches = self.dataset.num_mini_batches(self.dataset.train)
                    # evaluate the scene batch
                    cc = 0
                    stime = time.time()
                    for i in range(batches):
                        occ,seg,cat,names,_,_ = self.dataset.next_mini_batch(self.dataset.train)
                        # occ,seg,cat,names,_,_ = self.dataset.next_mini_batch(self.dataset.train)
                        _,d_cost,tmp = sess.run([train_op,cost,tmp_test], feed_dict={X: occ, Y_cat: cat, Y_seg: seg, keep_prob: self.keep_prob, bn_training: True})
                        cc = cc + d_cost
                        print("\rBatch %03d/%d" % (i+1,batches),end="")

                    cc = cc / (self.dataset.num_examples(self.dataset.train))
                    costs.append(cc)
                    print("\nEpoch %d trained in %f, cost %f" % (epoch+1, time.time() - stime, cc))
                    
                    # do check for file barrier, if so, break training cycle
                    if os.path.exists(os.path.join(log_dir, "barrier.txt")):
                        break
                    
                    # save model
                    save_path = tf.train.Saver().save(sess, os.path.join(log_dir, self.name + str(epoch+1) + ".ckpt"))
                    print("Model saved in file: %s\n" % save_path)
                    # plt.figure(1)
                    plt.clf()
                    plt.plot(np.squeeze(np.array(costs)))
                    plt.savefig("./3d-object-recognition/ShapeNet/cost.png", format="png")

                    train_accuracies.append(accuracy_test(self.dataset, self.dataset.train))
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
                    plt.barh(np.arange(n_cat),per_category_iou, tick_label=["airplane", "bag", "cap", "car", "chair", "earphone", "guitar", "knife", "lamp", "laptop", "motorbike", "mug", "pistol", "rocket", "skateboard", "table"])
                    plt.savefig("./3d-object-recognition/ShapeNet/per_category_iou" + str(epoch+1) + ".png", format="png")



            # acc_train = accuracy_test(self.dataset, self.dataset.train)
            # acc_test = accuracy_test(self.dataset, self.dataset.test)
            # acc_dev = accuracy_test(self.dataset, self.dataset.dev)
            
            # print("Train/Dev/Test accuracies %f/%f/%f" %(acc_train, acc_dev, acc_test))
            plt.show()


    def evaluate_iou_results(self, data_dict={ "name" : "train"}):
        return self.dataset.evaluate_iou_results(data_dict)



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
            self.dataset =      Parts(datapath, batch_size=jparams["batch_size"], ishape=jparams["input_shape"])


if __name__ == "__main__":
    s = PartsNet("ShapeNet", "./3d-object-recognition/ShapePartsData")
    s.run_model(load=False, train=True,visualize=False)
    # s.evaluate_iou_results()
