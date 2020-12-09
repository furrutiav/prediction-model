"""
    Main Prediction Model:
    Felipe Urrutia V.
    Dec-2020
"""

# import class and data-iris
from model import Predictor
from data import iris

columns_name = iris.columns.tolist()

attributes = columns_name[:-1]
predict = columns_name[-1:]

p = Predictor(iris, attributes, predict)
p.fit()

flower = [5.1, 3.5, 1.4, 0.2]
p.prediction(flower)

p.performance()

