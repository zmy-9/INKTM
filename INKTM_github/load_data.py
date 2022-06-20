import numpy as np
import os
from sklearn.datasets import dump_svmlight_file
import tempfile


class LoadData(object):
    '''given the path of data, return the data format for DeepFM
    :param path
    return:
    Train_data: a dictionary, 'Y' refers to a list of y values; 'X' refers to a list of features_M dimension vectors with 0 or 1 entries
    Test_data: same as Train_data
    Validation_data: same as Train_data
    '''

    # Three files are needed in the path
    def __init__(self, X_train_one, y_train, X_test_one, y_test, X_train_wins, X_test_wins, X_train_fails, X_test_fails):
        TMP_SUFFIX = '.libfm'
        train_fd_one = tempfile.NamedTemporaryFile(suffix=TMP_SUFFIX, dir=None)
        test_fd_one = tempfile.NamedTemporaryFile(suffix=TMP_SUFFIX, dir=None)
        train_fd_wins = tempfile.NamedTemporaryFile(suffix=TMP_SUFFIX, dir=None)
        test_fd_wins = tempfile.NamedTemporaryFile(suffix=TMP_SUFFIX, dir=None)
        train_fd_fails = tempfile.NamedTemporaryFile(suffix=TMP_SUFFIX, dir=None)
        test_fd_fails = tempfile.NamedTemporaryFile(suffix=TMP_SUFFIX, dir=None)

        # converts train and test data to libSVM format
        dump_svmlight_file(X_train_one, y_train, train_fd_one)
        train_fd_one.seek(0)
        dump_svmlight_file(X_test_one, y_test, test_fd_one)
        test_fd_one.seek(0)
        dump_svmlight_file(X_train_wins, y_train, train_fd_wins)
        train_fd_wins.seek(0)
        dump_svmlight_file(X_test_wins, y_test, test_fd_wins)
        test_fd_wins.seek(0)
        dump_svmlight_file(X_train_fails, y_train, train_fd_fails)
        train_fd_fails.seek(0)
        dump_svmlight_file(X_test_fails, y_test, test_fd_fails)
        test_fd_fails.seek(0)

        self.trainfile_one = train_fd_one.name
        self.testfile_one = test_fd_one.name
        self.trainfile_wins = train_fd_wins.name
        self.testfile_wins = test_fd_wins.name
        self.trainfile_fails = train_fd_fails.name
        self.testfile_fails = test_fd_fails.name

        self.features_M = self.map_features()
        self.features_M += 1
        self.Train_data, self.Test_data = self.construct_data()
        self.num_variable, self.num_variable_wins, self.num_variable_fails = self.truncate_features()

    def map_features(self):  # map the feature entries in all files, kept in self.features dictionary
        self.features = {}
        self.nums = {}
        self.read_features(self.trainfile_one, self.trainfile_wins, self.trainfile_fails)
        self.read_features(self.testfile_one, self.testfile_wins, self.testfile_fails)
        # print("features_M:", len(self.features))
        return len(self.features)

    def read_features(self, file1, file2, file3):  # read a feature file
        f1 = open(file1)
        f2 = open(file2)
        f3 = open(file3)
        line1 = f1.readline()
        line2 = f2.readline()
        line3 = f3.readline()
        i = len(self.features)
        while line1:
            items = line1.strip().split(' ')
            for item in items[1:]:
                loc, num = item.rstrip("'").split(":")
                if item not in self.nums:
                    self.nums[item] = float(num)
                if loc not in self.features:
                    self.features[loc] = i
                    i = i + 1
            items1 = line2.strip().split(' ')
            for item in items1[1:]:
                loc, num = item.rstrip("'").split(":")
                if item not in self.nums:
                    self.nums[item] = float(num)
                if loc not in self.features:
                    self.features[loc] = i
                    i = i + 1
            items2 = line3.strip().split(' ')
            for item in items2[1:]:
                loc, num = item.rstrip("'").split(":")
                if item not in self.nums:
                    self.nums[item] = float(num)
                if loc not in self.features:
                    self.features[loc] = i
                    i = i + 1
            line1 = f1.readline()
            line2 = f2.readline()
            line3 = f3.readline()
        f1.close()
        f2.close()
        f3.close()

    def construct_data(self):
        X_one, Y_for_logloss, X_one_nums, X_wins, X_wins_nums, X_fails, X_fails_nums = self.read_data(self.trainfile_one,
                                                                                                      self.trainfile_wins, self.trainfile_fails)
        Train_data = self.construct_dataset(X_one, Y_for_logloss, X_one_nums, X_wins, X_wins_nums, X_fails, X_fails_nums)
        print("# of training:", len(Y_for_logloss))

        X_one, Y_for_logloss, X_one_nums, X_wins, X_wins_nums, X_fails, X_fails_nums = self.read_data(self.testfile_one,
                                                                                                      self.testfile_wins, self.testfile_fails)
        Test_data = self.construct_dataset(X_one, Y_for_logloss, X_one_nums, X_wins, X_wins_nums, X_fails, X_fails_nums)
        print("# of test:", len(Y_for_logloss))

        return Train_data, Test_data

    def read_data(self, file1, file2, file3):
        # read a data file. For a row, the first column goes into Y_;
        # the other columns become a row in X_one and entries are maped to indexs in self.features
        f1 = open(file1)
        f2 = open(file2)
        f3 = open(file3)
        X_one = []
        X_one_nums = []
        X_wins, X_wins_nums = [], []
        X_fails, X_fails_nums = [], []
        Y_for_logloss = []
        line1 = f1.readline()
        line2 = f2.readline()
        line3 = f3.readline()
        while line1:
            items = line1.strip().split(' ')
            items2 = line2.strip().split(' ')
            items3 = line3.strip().split(' ')
            if float(items[0]) > 0:  # > 0 as 1; others as 0
                v = 1.0
            else:
                v = 0.0
            Y_for_logloss.append(v)
            X_one.append([self.features[item.rstrip("'").split(":")[0]] for item in items[1:]])
            X_one_nums.append([self.nums[item] for item in items[1:]])
            X_wins.append([self.features[item.rstrip("'").split(":")[0]] for item in items2[1:]])
            X_wins_nums.append([self.nums[item] for item in items2[1:]])
            X_fails.append([self.features[item.rstrip("'").split(":")[0]] for item in items3[1:]])
            X_fails_nums.append([self.nums[item] for item in items3[1:]])
            line1 = f1.readline()
            line2 = f2.readline()
            line3 = f3.readline()
        f1.close()
        f2.close()
        f3.close()
        return X_one, Y_for_logloss, X_one_nums, X_wins, X_wins_nums, X_fails, X_fails_nums

    def construct_dataset(self, X_one, Y_, X_one_nums, X_wins, X_wins_nums, X_fails, X_fails_nums):
        Data_Dic = {}
        X_lens = [len(line) for line in X_one]
        indexs = np.argsort(X_lens)
        Data_Dic['Y'] = [Y_[i] for i in indexs]
        Data_Dic['X_one'] = [X_one[i] for i in indexs]
        Data_Dic['X_one_nums'] = [X_one_nums[i] for i in indexs]
        Data_Dic['X_wins'] = [X_wins[i] for i in indexs]
        Data_Dic['X_wins_nums'] = [X_wins_nums[i] for i in indexs]
        Data_Dic['X_fails'] = [X_fails[i] for i in indexs]
        Data_Dic['X_fails_nums'] = [X_fails_nums[i] for i in indexs]
        return Data_Dic

    def truncate_features(self):
        """
        Make sure each feature vector is of the same length
        """
        num_variable = len(self.Train_data['X_one'][0])
        for i in range(len(self.Train_data['X_one'])):
            num_variable = max([num_variable, len(self.Train_data['X_one'][i])])
        for i in range(len(self.Test_data['X_one'])):
            num_variable = max([num_variable, len(self.Test_data['X_one'][i])])
        # truncate train, validation and test
        for i in range(len(self.Train_data['X_one'])):
            self.Train_data['X_one'][i] = self.Train_data['X_one'][i][0:] + [self.features_M - 1] * (
                    num_variable - len(self.Train_data['X_one'][i]))
        for i in range(len(self.Test_data['X_one'])):
            self.Test_data['X_one'][i] = self.Test_data['X_one'][i][0:] + [self.features_M - 1] * (
                    num_variable - len(self.Test_data['X_one'][i]))

        for i in range(len(self.Train_data['X_one_nums'])):
            self.Train_data['X_one_nums'][i] = self.Train_data['X_one_nums'][i][0:] + [0] * (
                    num_variable - len(self.Train_data['X_one_nums'][i]))
        for i in range(len(self.Test_data['X_one_nums'])):
            self.Test_data['X_one_nums'][i] = self.Test_data['X_one_nums'][i][0:] + [0] * (
                    num_variable - len(self.Test_data['X_one_nums'][i]))

        # wins filed
        num_variable_wins = len(self.Train_data['X_wins'][0])
        for i in range(len(self.Train_data['X_wins'])):
            num_variable_wins = max([num_variable_wins, len(self.Train_data['X_wins'][i])])
        for i in range(len(self.Test_data['X_wins'])):
            num_variable_wins = max([num_variable_wins, len(self.Test_data['X_wins'][i])])
        # truncate train, validation and test
        for i in range(len(self.Train_data['X_wins'])):
            self.Train_data['X_wins'][i] = self.Train_data['X_wins'][i][0:] + [self.features_M - 1] * (
                    num_variable_wins - len(self.Train_data['X_wins'][i]))
        for i in range(len(self.Test_data['X_wins'])):
            self.Test_data['X_wins'][i] = self.Test_data['X_wins'][i][0:] + [self.features_M - 1] * (
                    num_variable_wins - len(self.Test_data['X_wins'][i]))

        for i in range(len(self.Train_data['X_wins_nums'])):
            self.Train_data['X_wins_nums'][i] = self.Train_data['X_wins_nums'][i][0:] + [0] * (
                    num_variable_wins - len(self.Train_data['X_wins_nums'][i]))
        for i in range(len(self.Test_data['X_wins_nums'])):
            self.Test_data['X_wins_nums'][i] = self.Test_data['X_wins_nums'][i][0:] + [0] * (
                    num_variable_wins - len(self.Test_data['X_wins_nums'][i]))

        # fails field
        num_variable_fails = len(self.Train_data['X_fails'][0])
        for i in range(len(self.Train_data['X_fails'])):
            num_variable_fails = max([num_variable_fails, len(self.Train_data['X_fails'][i])])
        for i in range(len(self.Test_data['X_fails'])):
            num_variable_fails = max([num_variable_fails, len(self.Test_data['X_fails'][i])])
        # truncate train, validation and test
        for i in range(len(self.Train_data['X_fails'])):
            self.Train_data['X_fails'][i] = self.Train_data['X_fails'][i][0:] + [self.features_M - 1] * (
                    num_variable_fails - len(self.Train_data['X_fails'][i]))
        for i in range(len(self.Test_data['X_fails'])):
            self.Test_data['X_fails'][i] = self.Test_data['X_fails'][i][0:] + [self.features_M - 1] * (
                    num_variable_fails - len(self.Test_data['X_fails'][i]))

        for i in range(len(self.Train_data['X_fails_nums'])):
            self.Train_data['X_fails_nums'][i] = self.Train_data['X_fails_nums'][i][0:] + [0] * (
                    num_variable_fails - len(self.Train_data['X_fails_nums'][i]))
        for i in range(len(self.Test_data['X_fails_nums'])):
            self.Test_data['X_fails_nums'][i] = self.Test_data['X_fails_nums'][i][0:] + [0] * (
                    num_variable_fails - len(self.Test_data['X_fails_nums'][i]))
        return num_variable, num_variable_wins, num_variable_fails
