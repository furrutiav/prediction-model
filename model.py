"""
    Class Prediction Model:
    Felipe Urrutia V.
    Dec-2020
"""

# import library
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, bernoulli, uniform
import scipy.integrate as integrate
import pandas as pd
import typing


class Predictor(object):

    def __init__(self, data, attributes, predict, index_predict, index_attributes):
        self._data = data
        self._attributes = attributes
        self._predict = predict
        self._labels = sorted(list(set(data[predict[0]].tolist())))
        print(self._labels)
        self._size = [len(data), len(attributes), len(self._labels)]
        self._fit_size = self._size[0]
        self._fitting: Fitting
        self._columns = []
        for attrib in attributes:
            self._columns.append(data[attrib].tolist())
        self._index_data = {}
        self._data_predict = {}
        for label in self._labels:
            self._index_data[label] = []
            self._data_predict[label] = []
        self._rows = []
        for index, row in data.iterrows():
            row_list = row.tolist()
            self._rows.append(row_list)
            label = row_list[index_predict]
            self._index_data[label].append(index)
        for label in self._labels:
            for index in self._index_data[label]:
                self._data_predict[label].append(self._rows[index][index_attributes[0]:index_attributes[1]+1])
        self._clean_data = []
        for col in self._columns:
            col_data = []
            for label in self._labels:
                label_data = []
                for index in self._index_data[label]:
                    label_data.append(col[index])
                col_data.append(label_data)
            self._clean_data.append(col_data)

    def size(self):
        return self._size

    def fit(self, size=-1):
        if self._size[0] >= size > 0:
            self._fit_size = size
            self._fitting = Fitting(self, size)
        else:
            self._fitting = Fitting(self, self._size[0])
        print('Successful Training')

    def get_data(self):
        return self._clean_data

    def get_index(self):
        return self._index_data

    def prediction(self, value, epsilon=0.001, show=True):
        p = self._fitting.precision(value, epsilon)
        N = 1 / sum(p)
        label: int = int(np.argmax(p))
        if show:
            print(f'prediction: {self._labels[label]}, precision: {np.round(p[label] * N * 100, 2)}, '
                  f'more: {[np.round(p[_] * N * 100, 2) for _ in range(self._size[-1])]}')
        return self._labels[label]

    def performance(self, epsilon=0.001):
        data_predict = self._data_predict
        count = 0
        for label in data_predict.keys():
            for value in data_predict[label]:
                count += 1 if self.prediction(value, epsilon=epsilon, show=False) == label else 0
        print(f'performance: {100 * count / self._size[0]}%, fail: {self._size[0]-count}')


class Fitting(object):

    def __init__(self, predictor, size):
        print('Training...')
        self._size = size
        self._predictor = predictor
        self._data = predictor.get_data()
        self._index = predictor.get_index()
        self._size = len(self._data), len(self._index)
        self._parameters = ['mean', 'std']
        self._mean = []
        self._std = []
        for attrib in range(self._size[0]):
            mean_attrib = []
            std_attrib = []
            for label in range(self._size[1]):
                sub_data = self._data[attrib][label]
                mean_attrib.append(np.mean(sub_data))
                std_attrib.append(np.std(sub_data))
            self._mean.append(mean_attrib)
            self._std.append(std_attrib)
        self._SI_normalized = []
        for attrib in range(self._size[0]):
            SI = [self.segregation_index(attrib, 0, 1),
                  self.segregation_index(attrib, 0, 2),
                  self.segregation_index(attrib, 1, 2)]
            N = 1 / sum(SI)
            self._SI_normalized.append([SI[_] * N for _ in range(self._size[1])])

    def posterior_predictive(self, value, attrib, label, distribution='norm'):
        if distribution != 'norm':
            if distribution == 'Bernoulli':
                return bernoulli.pmf(value, self._mean[attrib][label])
        else:
            return norm.pdf(value, self._mean[attrib][label], self._std[attrib][label])

    def segregation_index(self, attrib, label_1, label_2):
        mean_1 = self._mean[attrib][label_1]
        mean_2 = self._mean[attrib][label_2]
        std_1 = self._std[attrib][label_1]
        std_2 = self._std[attrib][label_2]
        return abs(mean_1 - mean_2) / (std_1 + std_2)

    def general_predictive(self, value, epsilon):
        predictive_arr = []
        for attrib in range(self._size[0]):
            row = []
            for label in range(self._size[1]):
                epsilon_std = epsilon / self._std[attrib][label]
                value_attrib = value[attrib]
                p = self.posterior_predictive(value_attrib, attrib, label) * 2 * epsilon_std
                row.append(p)
            predictive_arr.append(row)
        return predictive_arr

    def beta_factor(self, value, epsilon):
        p_arr = self.general_predictive(value, epsilon)
        b_arr = []
        for attrib in range(self._size[0]):
            SI_N = self._SI_normalized
            beta_0 = (1 - p_arr[attrib][1]) * SI_N[attrib][0] + (1 - p_arr[attrib][2]) * SI_N[attrib][1]
            beta_1 = (1 - p_arr[attrib][0]) * SI_N[attrib][0] + (1 - p_arr[attrib][2]) * SI_N[attrib][2]
            beta_2 = (1 - p_arr[attrib][0]) * SI_N[attrib][1] + (1 - p_arr[attrib][1]) * SI_N[attrib][2]
            b_arr.append([beta_0, beta_1, beta_2])
        return b_arr, p_arr

    def certainty_attrib(self, attrib, value, epsilon):
        b_arr, p_arr = self.beta_factor(value, epsilon)
        return [(b_arr[attrib][0] + b_arr[attrib][1] + b_arr[attrib][2]) * p_arr[attrib][i]
                for i in range(self._size[1])]

    def certainty(self, value, epsilon):
        return [self.certainty_attrib(attrib, value, epsilon) for attrib in range(self._size[0])]

    def precision(self, value, epsilon):
        c = self.certainty(value, epsilon)
        return [sum([c[j][i] for j in range(self._size[0])]) for i in range(self._size[1])]
