"""
    Class Prediction Model:
    Felipe Urrutia V.
    Dec-2020
"""

# import library
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
from scipy.stats import norm
import scipy.integrate as integrate
import pandas as pd
import typing


class Predictor(object):

    def __init__(self, data, attributes, predict):
        self._data = data
        self._attributes = attributes
        self._predict = predict
        self._labels = sorted(list(set(data[predict[0]].tolist())))
        self._size = [len(data), len(attributes), len(self._labels)]
        self._fit_size = self._size[0]
        self._fitting = None
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
            label = row_list[-1]
            self._index_data[label].append(index)
        for label in self._labels:
            for index in self._index_data[label]:
                self._data_predict[label].append(self._rows[index][:-1])
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

    def get_data(self):
        return self._clean_data

    def get_index(self):
        return self._index_data


class Fitting(object):

    def __init__(self, predictor, size):
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

    def posterior_predictive(self, value, attrib, label):
        return norm.pdf(value, self._mean[attrib][label], self._std[attrib][label])
