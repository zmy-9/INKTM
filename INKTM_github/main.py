import math
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from time import time
import argparse
import load_data as DATA
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from sklearn.metrics import roc_auc_score
from scipy.sparse import load_npz
from sklearn.model_selection import KFold
import logging


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run Neural FM.")
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=128,
                        help='Number of hidden factors.')
    parser.add_argument('--layers', nargs='?', default='[32]',
                        help="Size of each layer.")
    parser.add_argument('--keep_prob', nargs='?', default='[0.4,0.3]',
                        help='Keep probability (i.e., 1-dropout_ratio) for each deep layer and the Bi-Interaction layer. 1: no dropout. Note that the last index is for the Bi-Interaction layer.')
    parser.add_argument('--lamda', type=float, default=0.1,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--early_stop', type=int, default=1,
                        help='Whether to perform early stop (0 or 1)')
    parser.add_argument('--fenlayers', nargs='?', default='[256]',
                        help="Size of FEN layer.")
    parser.add_argument('--num_field', type=int, default=5,
                        help='the feature field')
    parser.add_argument('--keep_deep', nargs='?', default='[0.4,0.2]',
                        help='Keep probability (i.e., 1-dropout_ratio) for each deep layer')

    return parser.parse_args()


class INKTM(BaseEstimator, TransformerMixin):
    def __init__(self, features_M, hidden_factor, layers, epoch, batch_size, learning_rate,
                 lamda_bilinear,keep_prob, optimizer_type, verbose, early_stop, fenlayes,
                 random_seed=2016):
        # bind params to class
        self.batch_size = batch_size
        self.hidden_factor = hidden_factor
        self.layers = layers
        self.features_M = features_M
        self.lamda_bilinear = lamda_bilinear
        self.epoch = epoch
        self.random_seed = random_seed
        self.keep_prob = np.array(keep_prob)
        self.keep_deep = np.array(eval(args.keep_deep))
        self.no_dropout = np.array([1 for i in range(len(keep_prob))])
        self.no_dropout_deep = np.array([1 for i in range(len(eval(args.keep_deep)))])
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.batch_norm = True
        self.verbose = verbose
        self.early_stop = early_stop
        self.fenlayers = fenlayes
        self.num_field = args.num_field
        # performance of each epoch
        self.train_acc, self.test_acc, self.train_auc, self.test_auc = [], [], [], []
        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)

            # train_features: student, question, and skills -> 01000110000
            self.train_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            self.train_nums = tf.placeholder(tf.float32, shape=[None, None])  # None * features_M

            # wins_features: the number of students correctly answering on each skill
            # wins_features: 010000..., wins_nums: 030000..., where 3 denotes the exercise number
            self.wins_features = tf.placeholder(tf.int32, shape=[None, None])
            self.wins_nums = tf.placeholder(tf.float32, shape=[None, None])

            # fails_features: the number of students incorrectly answering on each skill
            self.fails_features = tf.placeholder(tf.int32, shape=[None, None])
            self.fails_nums = tf.placeholder(tf.float32, shape=[None, None])

            self.train_labels = tf.placeholder(tf.float32, shape=[None, 1])  # None * 1
            self.dropout_keep = tf.placeholder(tf.float32, shape=[None])
            self.dropout_deep = tf.placeholder(tf.float32, shape=[None])
            self.train_phase = tf.placeholder(tf.bool)

            # Variables.
            self.weights, self.weights2, self.weights3 = self._initialize_weights()
            mn = tf.zeros([1, self.hidden_factor])
            self.weights['feature_embeddings'] = tf.concat([self.weights['feature_embeddings'], mn], 0)
            mn1 = tf.zeros([1, self.hidden_factor])
            self.weights['feature_bias'] = tf.concat([self.weights['feature_bias'], mn1], 0)
            self.num = tf.concat([self.train_nums, self.wins_nums], 1)
            self.num = tf.concat([self.num, self.fails_nums], 1)
            self.num = tf.reduce_sum(self.num, 1, keepdims=True)

            # _________ feature 2-order part _____________
            nonzero_embeddings_one = tf.multiply(
                tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features),
                tf.expand_dims(self.train_nums, 2))
            nonzero_embeddings_wins = tf.multiply(
                tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.wins_features),
                tf.expand_dims(self.wins_nums, 2))
            nonzero_embeddings_fails = tf.multiply(
                tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.fails_features),
                tf.expand_dims(self.fails_nums, 2))

            nonzero_embeddings = tf.concat([nonzero_embeddings_one, nonzero_embeddings_wins], 1)
            nonzero_embeddings = tf.concat([nonzero_embeddings, nonzero_embeddings_fails], 1)
            self.summed_features_emb = tf.reduce_sum(nonzero_embeddings, 1)  # None * K
            # get the element-multiplication
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K
            # square_sum part
            self.squared_features_emb = tf.square(nonzero_embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K
            # FM - 2-order
            self.FM = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # None * K
            if self.batch_norm:
                self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase,
                                                scope_bn='bn_fm')  # None * layer[i] * 1
            # self.FM = tf.nn.relu(self.FM)
            self.FM = tf.nn.dropout(self.FM, self.dropout_keep[-1])  # dropout at each Deep layer
            Bilinear = tf.reduce_sum(self.FM, 1, keep_dims=True)  # None * 1

            # _________ feature 1-order part combined with the attention mechanism _____________
            self.Feature_bias = tf.multiply(tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features),
                                            tf.expand_dims(self.train_nums, 2))  # None * 1
            self.skill_bias = tf.reduce_sum(self.Feature_bias[:, 2:, :], 1, keepdims=True)
            self.Feature_bias_wins = tf.reduce_sum(
                tf.multiply(tf.nn.embedding_lookup(self.weights['feature_bias'], self.wins_features),
                            tf.expand_dims(self.wins_nums, 2)), 1, keepdims=True)  # None * 1
            self.Feature_bias_fails = tf.reduce_sum(
                tf.multiply(tf.nn.embedding_lookup(self.weights['feature_bias'], self.fails_features),
                            tf.expand_dims(self.fails_nums, 2)), 1, keepdims=True)  # None * 1
            self.user_item = self.Feature_bias[:, 0:2, :]
            self.bias = tf.concat([self.user_item, self.skill_bias], 1)
            self.bias = tf.concat([self.bias, self.Feature_bias_wins], 1)
            self.bias = tf.concat([self.bias, self.Feature_bias_fails], 1)

            # ________ Attention Layers __________
            dnn_nonzero_embeddings = tf.reshape(self.bias,
                                                shape=[-1, self.num_field * self.hidden_factor])
            self.dnn = tf.add(tf.matmul(dnn_nonzero_embeddings, self.weights2['fenlayer_0']),
                              self.weights2['fenbias_0'])  # None * layer[i] * 1
            if self.batch_norm:
                self.dnn = self.batch_norm_layer(self.dnn, train_phase=self.train_phase,
                                                 scope_bn='bn_0')  # None * layer[i] * 1
            self.dnn = tf.nn.relu(self.dnn)
            self.dnn = tf.nn.dropout(self.dnn, self.dropout_keep[0])  # dropout at each Deep layer
            for i in range(1, len(self.fenlayers)):
                self.dnn = tf.add(tf.matmul(self.dnn, self.weights2['fenlayer_%d' % i]),
                                  self.weights2['fenbias_%d' % i])  # None * layer[i] * 1
                if self.batch_norm:
                    self.dnn = self.batch_norm_layer(self.dnn, train_phase=self.train_phase,
                                                     scope_bn='bn_%d' % i)  # None * layer[i] * 1
                self.dnn = tf.nn.relu(self.dnn)
                self.dnn = tf.nn.dropout(self.dnn, self.dropout_keep[i])  # dropout at each Deep layer
            self.dnn_out = tf.matmul(self.dnn, self.weights2['prediction_dnn'])  # None * 10
            self.outm = tf.constant(float(5)) * tf.nn.softmax(self.dnn_out)

            self.nonzero_embeddings_m = tf.multiply(self.bias, tf.expand_dims(self.outm, 2))
            self.bias = tf.reduce_sum(self.nonzero_embeddings_m, 1)
            if self.batch_norm:
                self.FM = self.batch_norm_layer(self.bias, train_phase=self.train_phase, scope_bn='bn_bias')

            self.FM = tf.nn.dropout(self.FM, self.dropout_deep[-1])  # dropout at the bilinear interactin layer
            
            # ________ Deep Layers __________
            for i in range(0, len(self.layers)):
                self.FM = tf.add(tf.matmul(self.FM, self.weights3['layer_%d' % i]),
                                 self.weights3['bias_%d' % i])  # None * layer[i] * 1

                if self.batch_norm:
                    self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase,
                                                    scope_bn='bn_deep%d' % i)  # None * layer[i] * 1

                self.FM = tf.nn.relu(self.FM)
                self.FM = tf.nn.dropout(self.FM, self.dropout_deep[i])  # dropout at each Deep layer
            self.FM = tf.matmul(self.FM, self.weights3['prediction'])  # None * 1
            self.Bilinear_bias = tf.reduce_sum(self.FM, 1, keep_dims=True)  # None * 1
            
            # self.Bilinear_bias = tf.reduce_sum(self.bias, 1, keep_dims=True)  # None * 1
            Bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # None * 1
            self.out = tf.add_n([Bilinear, self.Bilinear_bias, Bias])  # None * 1

            # Compute the loss.
            if self.lamda_bilinear > 0:
                self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.train_labels) \
                            + tf.add_n(
                    [tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(self.weights2[v]) for v in
                     self.weights2]) + tf.add_n(
                    [tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(self.weights3[v]) for v in
                     self.weights3])
                # + tf.contrib.layers.l2_regularizer(
                # self.lamda_bilinear)(self.weights['feature_embeddings']) # regulizer
                '''
                for i in range(len(self.fenlayers)):
                    self.loss += tf.contrib.layers.l2_regularizer(
                        self.lamda_bilinear)(self.weights2['fenlayer_%d' % i])
                self.loss += tf.contrib.layers.l2_regularizer(
                        self.lamda_bilinear)(self.weights2['prediction_dnn'])
                for i in range(len(self.layers)):
                    self.loss += tf.contrib.layers.l2_regularizer(
                        self.lamda_bilinear)(self.weights2['fenlayer_%d' % i])
                self.loss += tf.contrib.layers.l2_regularizer(
                        self.lamda_bilinear)(self.weights2['prediction'])
                '''
            else:
                self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.train_labels)
            self.out = tf.nn.sigmoid(self.out)
            self.loss = tf.reduce_mean(self.loss)

            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()  # shape is an array of tf.Dimension
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)

    def _initialize_weights(self):
        all_weights = dict()
        weights = dict()
        weights1 = dict()
        all_weights['feature_embeddings'] = tf.Variable(
                tf.random_normal([self.features_M - 1, self.hidden_factor], 0.0, 0.01),
                name='feature_embeddings')  # features_M * K
        all_weights['feature_bias'] = tf.Variable(
            tf.random_normal([self.features_M - 1, self.hidden_factor], 0.0, 0.01),
            name='feature_bias')  # features_M * 1
        all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')  # 1 * 1

        num_fenlayer = len(self.fenlayers)
        if num_fenlayer > 0:
            glorot = np.sqrt(2.0 / (self.hidden_factor + self.fenlayers[0]))

            weights['fenlayer_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot,
                                 size=(self.hidden_factor*self.num_field, self.fenlayers[0])),
                dtype=np.float32)
            weights['fenbias_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.fenlayers[0])),
                dtype=np.float32)  # 1 * layers[0]

            for i in range(1, num_fenlayer):
                glorot = np.sqrt(2.0 / (self.fenlayers[i - 1] + self.fenlayers[i]))
                weights['fenlayer_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.fenlayers[i - 1], self.fenlayers[i])),
                    dtype=np.float32)  # layers[i-1]*layers[i]
                weights['fenbias_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.fenlayers[i])),
                    dtype=np.float32)  # 1 * layer[i]
            # prediction layer
            glorot = np.sqrt(2.0 / (self.fenlayers[-1] + 1))

            weights['prediction_dnn'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.fenlayers[-1], self.num_field)),
                dtype=np.float32)  # layers[-1] * 1

        num_layer = len(self.layers)
        if num_layer > 0:
            glorot = np.sqrt(2.0 / (self.hidden_factor + self.layers[0]))
            weights1['layer_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.hidden_factor, self.layers[0])), dtype=np.float32)
            weights1['bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.layers[0])),
                                             dtype=np.float32)  # 1 * layers[0]
            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.layers[i - 1] + self.layers[i]))
                weights1['layer_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.layers[i - 1], self.layers[i])),
                    dtype=np.float32)  # layers[i-1]*layers[i]
                weights1['bias_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.layers[i])), dtype=np.float32)  # 1 * layer[i]
            # prediction layer
            glorot = np.sqrt(2.0 / (self.layers[-1] + 1))
            weights1['prediction'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.layers[-1], 1)),
                                                 dtype=np.float32)  # layers[-1] * 1
        else:
            weights1['prediction'] = tf.Variable(
                np.ones((self.hidden_factor, 1), dtype=np.float32))  # hidden_factor * 1

        return all_weights, weights, weights1

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.train_features: data['X_one'], self.train_labels: data['Y'],
                     self.wins_features: data['X_wins'], self.wins_nums: data['X_wins_nums'],
                     self.fails_features: data['X_fails'], self.fails_nums: data['X_fails_nums'],
                     self.dropout_keep: self.keep_prob, self.train_nums: data['X_one_nums'],
                     self.dropout_deep: self.keep_deep, self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, data, batch_size):  # generate a random block of training data
        start_index = np.random.randint(0, len(data['Y']) - batch_size)
        X_one, Y, X_one_nums, X_wins, X_wins_nums, X_fails, X_fails_nums = [], [], [], [], [], [], []
        # forward get sample
        i = start_index
        while len(X_one) < batch_size and i < len(data['X_one']):
            if len(data['X_one'][i]) == len(data['X_one'][start_index]):
                Y.append([data['Y'][i]])
                X_one.append(data['X_one'][i])
                X_wins.append(data['X_wins'][i])
                X_fails.append(data['X_fails'][i])
                X_one_nums.append(data['X_one_nums'][i])
                X_wins_nums.append(data['X_wins_nums'][i])
                X_fails_nums.append(data['X_fails_nums'][i])
                i = i + 1
            else:
                break
        # backward get sample
        i = start_index
        while len(X_one) < batch_size and i >= 0:
            if len(data['X_one'][i]) == len(data['X_one'][start_index]):
                Y.append([data['Y'][i]])
                X_one.append(data['X_one'][i])
                X_wins.append(data['X_wins'][i])
                X_fails.append(data['X_fails'][i])
                X_one_nums.append(data['X_one_nums'][i])
                X_wins_nums.append(data['X_wins_nums'][i])
                X_fails_nums.append(data['X_fails_nums'][i])
                i = i - 1
            else:
                break
        return {'X_one': X_one, 'Y': Y, 'X_one_nums': X_one_nums, 'X_wins': X_wins, 'X_wins_nums': X_wins_nums,
                'X_fails': X_fails, 'X_fails_nums': X_fails_nums}

    def shuffle_in_unison_scary(self, a, b, c, d, e, f, g):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)
        np.random.set_state(rng_state)
        np.random.shuffle(e)
        np.random.set_state(rng_state)
        np.random.shuffle(f)
        np.random.set_state(rng_state)
        np.random.shuffle(g)

    def train(self, Train_data, Test_data):  # fit a dataset
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Train_data['X_one'], Train_data['Y'], Train_data['X_one_nums'],
                                         Train_data['X_wins'],
                                         Train_data['X_wins_nums'], Train_data['X_fails'], Train_data['X_fails_nums'])
            total_batch = int(len(Train_data['Y']) / self.batch_size)
            for i in range(total_batch):
                # generate a batch
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size)
                # Fit training
                self.partial_fit(batch_xs)
            t2 = time()

            # output validation
            train_acc, train_auc = self.evaluate(Train_data)
            test_acc, test_auc = self.evaluate(Test_data)

            self.train_acc.append(train_acc)
            self.test_acc.append(test_acc)
            self.train_auc.append(train_auc)
            self.test_auc.append(test_auc)
            # self.test_rmse.append(test_rmse)
            # self.test_mae.append(test_mae)
            if self.verbose > 0 and epoch % self.verbose == 0:
                print("Epoch %d [%.1f s]\ttrain_acc=%.4f, test_acc=%.4f [%.1f s]"
                      % (epoch + 1, t2 - t1, train_acc, test_acc, time() - t2))
                print("Epoch %d [%.1f s]\ttrain_auc=%.4f, test_auc=%.4f [%.1f s]"
                      % (epoch + 1, t2 - t1, train_auc, test_auc, time() - t2))
                print("Epoch %d [%.1f s]\ttest_rmse=%.4f, test_mae=%.4f [%.1f s]"
                      % (epoch + 1, t2 - t1, test_rmse, test_mae, time() - t2))
                logger.info("Epoch %d [%.1f s]\ttrain_acc=%.4f, test_acc=%.4f [%.1f s]"
                            % (epoch + 1, t2 - t1, train_acc, test_acc, time() - t2))
                logger.info("Epoch %d [%.1f s]\ttrain_auc=%.4f, test_auc=%.4f [%.1f s]"
                            % (epoch + 1, t2 - t1, train_auc, test_auc, time() - t2))
                '''
                logger.info("Epoch %d [%.1f s]\ttest_rmse=%.4f, test_mae=%.4f, test_nll=%.4f [%.1f s]"
                      % (epoch + 1, t2 - t1, test_rmse, test_mae, test_nll, time() - t2))
                '''
            if self.early_stop > 0 and self.eva_termination(self.test_auc):
                # print "Early stop at %d based on validation result." %(epoch+1)
                break

    def eva_termination(self, valid):
        if len(valid) > 5:
            if valid[-1] < valid[-2] and valid[-2] < valid[-3] and valid[-3] < valid[-4] and valid[-4] < valid[-5]:
                return True
        return False

    def evaluate(self, data):  # evaluate the results for an input set
        num_example = len(data['Y'])
        feed_dict = {self.train_features: data['X_one'], self.train_labels: [[y] for y in data['Y']],
                     self.train_nums: data['X_one_nums'],
                     self.wins_nums: data['X_wins_nums'], self.wins_features: data['X_wins'],
                     self.fails_nums: data['X_fails_nums'], self.fails_features: data['X_fails'],
                     self.dropout_keep: self.no_dropout, self.dropout_deep: self.no_dropout_deep,
                     self.train_phase: False}
        predictions, loss = self.sess.run((self.out, self.loss), feed_dict=feed_dict)
        y_pred = np.reshape(predictions, (num_example,))
        y_true = np.reshape(data['Y'], (num_example,))
        # predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
        # predictions_bounded = np.minimum(predictions_bounded,
        #                                  np.ones(num_example) * max(y_true))  # bound the higher values
        # RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
        # MAE= mean_absolute_error(y_true, predictions_bounded)
        auc = roc_auc_score(y_true, y_pred)
        acc = np.mean(y_true == np.round(y_pred))
        return acc, auc


if __name__ == '__main__':
    # Data loading
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('INKTM-assist2009.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    X = load_npz('assist09/One-uiswf.npz')
    X_wins = load_npz('assist09/Wins-uiswf.npz')
    X_fails = load_npz('assist09/Fails-uiswf.npz')
    y = np.load('assist09/Label-uiswf.npy')
    kf = KFold(n_splits=5, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train_one, X_test_one = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train_wins, X_test_wins = X_wins[train_index], X_wins[test_index]
        X_train_fails, X_test_fails = X_fails[train_index], X_fails[test_index]
        args = parse_args()
        data = DATA.LoadData(X_train_one, y_train, X_test_one, y_test, X_train_wins, X_test_wins, X_train_fails,
                             X_test_fails)
        if args.verbose > 0:
            print(
                "INKTM: hidden_factor=%d, dropout_keep=%s, layers=%s, epoch=%d, batch=%d, lr=%.4f, lambda=%.4f, optimizer=%s, early_stop=%d"
                % (args.hidden_factor, args.keep_prob, args.layers, args.epoch,
                   args.batch_size, args.lr, args.lamda, args.optimizer, args.early_stop))
            logger.info(
                "INKTM: hidden_factor=%d, dropout_keep=%s, layers=%s, epoch=%d, batch=%d, lr=%.4f, lambda=%.4f, optimizer=%s, early_stop=%d"
                % (args.hidden_factor, args.keep_prob, args.layers, args.epoch,
                   args.batch_size, args.lr, args.lamda, args.optimizer, args.early_stop))

        # Training
        t1 = time()
        model = INKTM(data.features_M, args.hidden_factor, eval(args.layers), args.epoch, args.batch_size, args.lr,
                      args.lamda,
                      eval(args.keep_prob), args.optimizer, args.verbose, args.early_stop, eval(args.fenlayers))
        model.train(data.Train_data, data.Test_data)

        best_auc_score = max(model.test_auc)
        best_epoch_auc = model.test_auc.index(best_auc_score)

        print("Best Iter(test_acc)= %d\t test(acc) = %.4f, test(auc) = %.4f [%.1f s]"
              % (best_epoch_auc + 1, model.test_acc[best_epoch_auc], model.test_auc[best_epoch_auc], time() - t1))
        logger.info("Best Iter(test_acc)= %d\t test(acc) = %.4f, test(auc) = %.4f [%.1f s]"
                    % (best_epoch_auc + 1, model.test_acc[best_epoch_auc], model.test_auc[best_epoch_auc], time() - t1))