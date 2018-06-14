import time
import os
import abc
import tensorflow as tf 
from nn_utils import *
import matplotlib.pyplot as plt 
import numpy as np


class dataset_template:

    __metaclass__ = abc.ABCMeta
    
    PATH = "path"
    DATASET = "dataset"
    NUMBER_EXAMPLES = "examples"
    CURRENT_BATCH = "current_batch"
    NUMBER_BATCHES = "number_batches"


    def __init__(self, datapath, batch_size = 64, ishape = [332,332,3], n_classes = 6):
        self.shape = ishape
        self.batch_size = batch_size
        self.num_classes = n_classes
        


    @abc.abstractmethod
    def __load_dataset(self, path, data_dict):
        """
        Loads datasets in path to python dictionary data_dict. 

        Arguments:
            path -- string, path to directory with data
            data_dict -- python dictionary, contains path, dataset, number of examples, batches and current batch
        """
        raise NotImplementedError("method __load_dataset not yet implemented")        


    def next_mini_batch(self, dataset):
        """
        Get next mini batch from given dataset.

        Arguments:
            dataset -- python dictionary containing train/dev/test dataset

        Returns:
            data -- numpy array, contains mini batch data
            labels -- numpy array, contains mini batch labels
        """
        raise NotImplementedError("method next_mini_batch not yet implemented")


    def restart_mini_batches(self, dataset):
        dataset[dataset_template.DATASET] = dataset[dataset_template.DATASET][np.random.permutation(dataset[dataset_template.NUMBER_EXAMPLES])]
        dataset[dataset_template.CURRENT_BATCH] = 0


    def input_shape(self, dataset):
        return dataset[dataset_template.NUMBER_EXAMPLES], self.shape[0], self.shape[1], self.shape[2], self.shape[3]


    def num_mini_batches(self, dataset):
        return dataset[dataset_template.NUMBER_BATCHES]

    def num_examples(self, dataset):
        return dataset[dataset_template.NUMBER_EXAMPLES]


class nn_template:

    def create_placeholders(self, n_H0, n_W0, n_C0, n_y):
        '''
        Creates placeholder for the tensorflow session.

        Arguments:
            n_H0 -- scalar, height of input image
            n_W0 -- scalar, width of input image 
            n_C0 -- scalar, number of channels, ie. 3 for RGB

        Returns:
            X -- placeholder for the data input of shape [None, n_H0, n_W0, n_C0] and dtype=tf.float32
            Y -- placeholder for the label input of shape [None, n_y] and dtype=tf.float32
        '''
        raise NotImplementedError("method create_placeholders not implemented")


    def initialize_parameters(self):
        '''
        Initializes weight parameters to build a neural network LeNet-5 with tensorflow.abs

        Shapes (example):
            W1 -- [5, 5, 3, 6]
            W2 -- [5, 5, 6, 16]
            

        Returns:
            parameters -- python dictionary with weight parameters tensors
        '''
        raise NotImplementedError("method initialize_parameters not implemented")


    def forward_propagation(self, X, parameters):
        '''
        (example)
        Implements the forward propagation for LeNet-5 model. 
        IN (32x32x3) > 
                conv2d > relu > max pool > 
                conv2d > relu > max pool > 
                fc > relu > 
                fc > relu >
                fc (for softmax)

        Arguments:
            X -- input dataset placeholder of shape (input size, number of examples)
            parameters -- python dictionary with weight parameters tensors

        Returns:
            Zl -- output of the last linear unit
        '''
        raise NotImplementedError("method forward_propagation not implemented")


    def compute_cost(self, Zl, Y):
        ''' 
        Computes the cost for output and labels. 

        Arguments:
            Zl -- output of the forward propagation of shape (6, number of examples)
            Y -- true labels vector placeholder, same shape as Zl
        
        Returns:
            cost -- tensor of the cost function
        '''
        raise NotImplementedError("method compute_cost not implemented")


    def model (self, dataset, model_name, learning_rate=0.001, num_epochs=100, minibatch_size=16):
        '''
        Defines neural network model in Tensorflow - optimizer, variable initialization,
        cost function and other.  

        Arguments:
            signs -- class containing dataset of six hand signs
            learning_rate -- learning rate of the optimization
            num_epochs -- number of epochs of the optimization
            minibatch_size -- size of a minibatch

        Return:
            model -- python dictionary, contains X, Y, Zl tensors, cost,
                    number of epochs, optimizer, initializer, dataset and model name
        '''
        raise NotImplementedError("method model not yet implemented")


    def train_model(self, model, log_dir, keep_prob=0.5, print_cost=True, train=True, load=False):
        # create log dir
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_dir)
        if os.path.exists(log_dir):
            print("exists")
        else:
            os.makedirs(log_dir, mode=0o777, exist_ok=True)
            os.chmod(log_dir, 0o777)

        init_op = model["init"]
        cost_op = model["cost"]
        optimizer_op = model["optimizer"]
        num_epochs = model["epochs"]
        X, Y, Zl = model["X"], model["Y"], model["Zl"]
        name = model["name"]
        dataset = model["dataset"]

        with tf.Session() as sess:
            # save costs for plot and accuracy test
            costs = []
            train_time = 0
            if load == True:
                print("Loading " + os.path.join(log_dir, name + ".ckpt"))
                tf.train.Saver().restore(sess, os.path.join(log_dir, name + ".ckpt"))
            else:
                sess.run(init_op)

            if train == True:
                train_time = time.time()
                for epoch in range(num_epochs):
                    minibatch_cost = 0
                    dataset.restart_mini_batches(dataset.train)

                    for i in range(dataset.num_mini_batches(dataset.train)):
                        x, y = dataset.next_mini_batch(dataset.train)
                        y = convert_to_one_hot(y, dataset.num_classes)
                        if "KP" in model:
                            _, temp_cost = sess.run([optimizer_op, cost_op], {X: x, Y: y, model["KP"]: keep_prob})
                        else:
                            _, temp_cost = sess.run([optimizer_op, cost_op], {X: x, Y: y})
                        minibatch_cost += temp_cost / dataset.num_mini_batches(dataset.train)
                    
                    if print_cost == True and epoch % 5 == 0:
                        print("Cost after epoch %d: %f" % (epoch, minibatch_cost))
                        save_path = tf.train.Saver().save(sess, os.path.join(log_dir, name + ".ckpt"), global_step=epoch)
                    if print_cost == True and epoch % 1 == 0:
                        costs.append(minibatch_cost)

                train_time = time.time() - train_time
                print("Training %f seconds" % (train_time))
                save_path = tf.train.Saver().save(sess, os.path.join(log_dir, name + ".ckpt"))
                print("Model saved in file: %s" % save_path)

                train_writer = tf.summary.FileWriter(log_dir, sess.graph)
                train_writer.flush()
                train_writer.close()
                plt.plot(np.squeeze(costs))
                plt.ylabel("cost")
                plt.xlabel("iterations (per one epoch)")
                plt.title("Model " + name)
                # plt.show()
                plt.savefig(os.path.join(log_dir, name + ".png"), bbox_inches='tight')

            # run testing
            # calculate correct predictions
            predictions = model["predictions"]
            predict_op = tf.argmax(predictions, 1)
            correct_predictions_op = tf.equal(predict_op, tf.argmax(Y, 1))
            # calculate accuracy
            accuracy_op = tf.reduce_mean(tf.cast(correct_predictions_op, "float"))
            def run_accuracy_test(dataset, data_dict, accuracy_op, print_predictions=False):
                dataset.restart_mini_batches(data_dict)
                accuracies = []
                for i in range(dataset.num_mini_batches(data_dict)):
                    x, y = dataset.next_mini_batch(data_dict)
                    y = convert_to_one_hot(y, dataset.num_classes)
                    if print_predictions == True:
                        plt.imshow(x[0])
                        plt.show()
                    if "KP" in model:
                        accuracies.append(accuracy_op.eval({X: x, Y: y, model["KP"]: 1}))
                    else:
                        accuracies.append(accuracy_op.eval({X: x, Y: y}))

                if len(accuracies) == 0:
                    return 0

                return sum(accuracies) / len(accuracies)

            # get train/dev/test error
            train_accuracy = 0
            dev_accuracy = 0
            test_accuracy = 0
            start_time = time.time()
            train_accuracy = run_accuracy_test(dataset, dataset.train, accuracy_op)
            duration = (time.time() - start_time)  / dataset.train[dataset_template.NUMBER_EXAMPLES]
            print(duration)
            dev_accuracy = run_accuracy_test(dataset, dataset.dev, accuracy_op)
            test_accuracy = run_accuracy_test(dataset, dataset.test, accuracy_op)
            print("Train accuracy: " + str(train_accuracy))
            print("Dev accuracy: " + str(dev_accuracy))
            print("Test accuracy: " + str(test_accuracy))
            f = open(os.path.join(log_dir, "stats.txt"), "w")
            f.write("Execution time for one test sample: %f sec" % (duration))
            f.write("Execution time for training: %f sec" % (train_time))
            f.write("Train accuracy: %f\nDev accuracy: %f\nTest accuracy: %f" % (train_accuracy, dev_accuracy, test_accuracy))
