from timeit import default_timer as timer
import tensorflow as tf 
import numpy as np 
import json
import os
from nn_utils import *
from voxels import Voxels


class Net3D():

    def create_placeholders(self, n_H0, n_W0, n_C0, n_y, n_d):
        '''
        Creates input placeholders for labels, input and descriptor matrix. 

        Arguments:
            n_H0 -- scalar, height of input image
            n_W0 -- scalar, width of input image 
            n_C0 -- scalar, number of channels, ie. 3 for RGB
            n_y -- scalar, number of labels
        Returns:
            X -- placeholder for data input, shape (None, n_H0, n_W0, n_C0, 1)
            Y -- placeholder for label input, shape (None, n_y)
        '''

        with tf.name_scope("input_placeholders"):
            X = tf.placeholder(dtype=tf.float32, shape=(None, n_H0, n_W0, n_C0, n_d), name="input_voxel")
            Y = tf.placeholder(dtype=tf.float32, shape=(None, n_y), name="input_label")
            keep_prob = tf.placeholder(dtype=tf.float32, name="keep_probability")

        return X, Y, keep_prob

    
    def forward_propagation(self, X, keep_prob, path):
        '''
        Initializes filter weights and biases. 

        IN > classify using softmax
        
        Arguments:
            X -- input dataset placeholder of shape (number of examples, input size)
            keep_prob -- placeholder, keep probability for dropout
            path -- string, path to json file with network description

        Returns:
            Z -- last layer before softmax
            activations -- dictionary, contains names as keys and tensors of activation layers as values
        '''
        activations = {}
        with open(path, "r") as f:
            layers = json.load(f)["layers"]

            Z = X
            with tf.name_scope("model"):
                for l in layers:
                    with tf.variable_scope(l["name"]) as scope:
                        shape = l["filter"]
                        W = tf.get_variable("weights", shape, initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_w), dtype=tf.float32)
                        b = tf.get_variable("bias", [shape[-1]], initializer=tf.zeros_initializer(), regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_b), dtype=tf.float32)
                        tf.summary.histogram(l["name"]+":weights", W)
                        tf.summary.histogram(l["name"]+":biases", b)
                        s = l["stride"]
                        p = "VALID"
                        if "padding" in l:
                            p = l["padding"].upper()

                        Z = tf.nn.conv3d(Z, W, strides=s, padding=p)
                        Z = Z + b
                        if "activation" in l and l["activation"].lower() == "relu":
                            Z = tf.nn.relu(Z)
                        elif "activation" in l and l["activation"].lower() == "elu":
                            Z = tf.nn.elu(Z)
                        elif "activation" in l and l["activation"].lower() == "softplus":
                            Z = tf.nn.softplus(Z)
                        elif "activation" in l and l["activation"].lower() == "softsign":
                            Z = tf.nn.softsign(Z)
                        elif "activation" in l and l["activation"].lower() == "crelu":
                            Z = tf.nn.crelu(Z)
                        elif "activation" in l and l["activation"].lower() == "tanh":
                            Z = tf.nn.tanh(Z)
                        elif "activation" not in l:
                            pass
                        else:
                            raise "Wrong activation provided"
                        # add layer after activation    
                        activations[l["name"] + "/activations"] = Z

                        if "dropout" in l:
                            Z = tf.nn.dropout(Z, keep_prob)
                        if "maxpool" in l:
                            Z = tf.nn.max_pool3d(Z, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="VALID")


                Z = tf.contrib.layers.flatten(Z)
            
        return Z, activations


    def compute_cost(self, Zl, Y):
        '''
        Computes the cost for output and labels. 

        Arguments:
            Zl -- output of the forward propagation of shape (n_y, number of examples)
            Y -- true labels vector placeholder, same shape as Zl
        
        Returns:
            cost -- tensor of the cost function
        '''

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Zl, labels=Y))

        return cost


    def compute_predictions(self, Zl, Y):
        """
        Creates prediction. Returns label or -1 if softmax is not certain enough.
        
        Arguments:
            Zl -- tensor, last flatten layer before softmax
            Y -- label placeholder, contains one-hot vector

        Returns:
            pred_op -- tensor, label prediction operator
            accuracy_op -- tensor, number of correctly classified
        """

        softmax = tf.nn.softmax(logits=Zl)
        max_prob = tf.reduce_max(softmax, axis=1)
        cond = tf.greater(max_prob, self.min_prob)
        arg_max_prob = tf.argmax(softmax, axis=1)
        pred_op = tf.where(cond, arg_max_prob, (-1) * tf.ones(tf.shape(arg_max_prob), dtype=tf.int64))
        accuracy_op = tf.equal(arg_max_prob, pred_op)
        accuracy_op = tf.reduce_sum(tf.cast(accuracy_op, tf.int32))
        return arg_max_prob, accuracy_op


    def run_model (self, print_cost=True, load=False, train=True, show_activations=False):
        # initialize some commonly used variables
        log_dir = os.path.join("./3d-object-recognition", self.name)

        tf.reset_default_graph()
        m, n_H0, n_W0, n_C0, n_d = self.dataset.input_shape(self.dataset.train)
        n_y = self.dataset.num_classes
        X, Y, keep_prob = self.create_placeholders(n_H0, n_W0, n_C0, n_y, n_d)
        Zl, Ws = self.forward_propagation(X, keep_prob, os.path.join(log_dir, "network.json"))
        cost = self.compute_cost(Zl, Y)
        pred_op, accuracy_op = self.compute_predictions(Zl, Y)
        step = tf.Variable(0, trainable=False, name="global_step")
        lr_dec = tf.train.exponential_decay(self.lr, step, self.decay_step, self.decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_dec)
        gvs = optimizer.compute_gradients(cost)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)
        init = tf.global_variables_initializer()

        # create summaries
        tf.summary.scalar("learning_rate", lr_dec)
        tf.summary.scalar("global_step", step)
        tf.summary.scalar("cost", cost)
        dev_writer = tf.summary.FileWriter(os.path.join(log_dir, "dev"), graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter(os.path.join(log_dir, "test"), graph=tf.get_default_graph())
        train_writer = tf.summary.FileWriter(os.path.join(log_dir, "train"), graph=tf.get_default_graph())
        summary_op = tf.summary.merge_all()

        with tf.Session() as sess:
            # initialize all variables before loading, so they won't be overwritten
            sess.run(init)
            
            if load:
                tf.train.Saver().restore(sess, os.path.join(log_dir, self.name + ".ckpt"))
                print("Model loaded from file: %s" % os.path.join(log_dir, self.name + ".ckpt"))


            def accuracy_test(dataset, data_dict, writer, idx, msg):
                dataset.restart_mini_batches(data_dict)
                sum_acc = 0
                sum_acc_argmax = 0
                for j in range(dataset.num_mini_batches(data_dict)):
                    x,y = dataset.next_mini_batch(data_dict)
                    y_hot = convert_to_one_hot(y, dataset.num_classes)
                    pred, acc, c = sess.run([pred_op, accuracy_op, cost], feed_dict={X: x, Y: y_hot, keep_prob: 1})
                    sum_acc += acc
                    sum_acc_argmax += np.sum(pred == y)
                # write only last summary after mini batch
                if summary:
                    writer.add_summary(summary, idx)
                # print('Accuracy of %s at step %s: %s' % (msg, i, sum_acc / data_dict[dataset.NUMBER_EXAMPLES]))
                print('Accuracy argmax of %s at step %s: %s' % (msg, idx, sum_acc_argmax / data_dict[dataset.NUMBER_EXAMPLES]))
                return sum_acc_argmax / data_dict[dataset.NUMBER_EXAMPLES], c


            # run the training cycle
            if train:
                # full trace for training summaries
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                accuracies = []
                train_accuracies = []
                costs = []

                for i in range(self.num_epochs):
                    starttime = timer()
                    # restart dataset for each of the sets
                    self.dataset.restart_mini_batches(self.dataset.train)
                    self.dataset.restart_mini_batches(self.dataset.test)
                    self.dataset.restart_mini_batches(self.dataset.dev)

                    for j in range(self.dataset.num_mini_batches(self.dataset.train)):
                        x,y = self.dataset.next_mini_batch(self.dataset.train)
                        y = convert_to_one_hot(y, self.dataset.num_classes)
                        # train and get summary
                        summary, _ = sess.run([summary_op, train_op], feed_dict={X: x, Y: y, keep_prob: self.keep_prob},
                              options=run_options,
                              run_metadata=run_metadata)
                        # write only last summary after mini batch
                        train_writer.add_summary(summary, i * j + j)
                        print("\rTrain batch %d/%d" % ((j+1), self.dataset.num_mini_batches(self.dataset.train)), end="")
                    duration = timer() - starttime
                    print("")
                    print("Epoch trained in " + str(duration))

                    if i % 2 == 0:  # Record summaries and train-set accuracy
                        acc_t, cc = accuracy_test(self.dataset, self.dataset.train, train_writer, i, "train")
                        acc, _ = accuracy_test(self.dataset, self.dataset.test, test_writer, i, "test")
                        accuracies.append(acc)
                        train_accuracies.append(acc_t)
                        costs.append(cc)
                        # accuracy_test(self.dataset, self.dataset.dev, dev_writer, i, "dev")
                        # plot and save training costs so you can decide whether to use barrier
                        plt.figure(0)
                        plt.plot(np.squeeze(costs))
                        plt.ylabel("train costs")
                        plt.xlabel("iterations")
                        plt.title("Model "  + self.name)
                        plt.savefig(os.path.join(log_dir, self.name + "-costs" + ".png"), bbox_inches='tight')
                        # plot and save test accuracies so you can decide whether to use barrier
                        plt.figure(1)
                        plt.plot(np.squeeze(accuracies), "r")
                        plt.ylabel("accuracies")
                        plt.xlabel("iterations")
                        plt.title("Model "  + self.name)
                        plt.plot(np.squeeze(train_accuracies), "b")
                        plt.savefig(os.path.join(log_dir, self.name + "-accs" + ".png"), bbox_inches='tight')

                        # do check for file barrier, if so, break training cycle
                        if os.path.exists(os.path.join(log_dir, "barrier.txt")):
                            break
                        print("##################################################")
            
            # save model
            save_path = tf.train.Saver().save(sess, os.path.join(log_dir, self.name + ".ckpt"))
            print("Model saved in file: %s" % save_path)

            if train == False:
                # print at least something
                summary = None
                accuracy_test(self.dataset, self.dataset.test, test_writer, -1, "test")

            # skip the rest, if no activations should be displayed
            if not show_activations:
                return

            def getActivations(layer,stimuli, label, ignore_input=True):
                units = sess.run(layer,feed_dict={X:np.reshape(stimuli,[1,n_H0,n_W0,n_C0,1],order='F'), Y: np.reshape(label, [1,n_y]),keep_prob:1.0})
                conv3d_plot(units, ignore_input=ignore_input)

            # display activations
            self.dataset.restart_mini_batches(self.dataset.test)
            for name in ["sphere", "torus", "cube", "cone", "cylinder"]:
                x,y = self.dataset.get_data(self.dataset.test, name)
                y_hot = convert_to_one_hot(y, self.dataset.num_classes)
                pred, acc = sess.run([pred_op, accuracy_op], feed_dict={X: x, Y: y_hot, keep_prob: 1})
                for i in range(0, x.shape[0]):
                    # display stimuli
                    display_stimuli(x[i], x[i].shape[0])
                    with open(os.path.join(log_dir, "network.json"), "r") as f:
                        layers = json.load(f)["layers"]
                        for l in layers:
                            getActivations(Ws[l["name"]+"/activations"], x[i], y_hot[i], ignore_input=True)
                    
                    predicted_label = self.dataset.label_to_name(pred[i])
                    print("Predicted %s and expected %s" % (predicted_label, self.dataset.label_to_name(y[i])))


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
            self.dataset =      Voxels(datapath, batch_size=jparams["batch_size"], ishape=jparams["input_shape"], n_classes=jparams["num_classes"])


if __name__ == "__main__":
    # model = Net3D("Net3D-32-scaled", "./3d-object-recognition/data-32-plus-scaled")
    model = Net3D("Net3D", "./3d-object-recognition/ModelNet-data-mean-norm")
    model.run_model(print_cost=True, load=True, train=True, show_activations=False)

