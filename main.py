"""
    Main Prediction Model:
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
# import class and data-iris
from model import Predictor
from data import iris

columns_name = iris.columns.tolist()

attributes = columns_name[:-1]
predict = columns_name[-1:]

p = Predictor(iris, attributes, predict)
p.fit()



