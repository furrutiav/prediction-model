"""
    Main Prediction Model:
    Felipe Urrutia V.
    Dec-2020
"""

# import class and data-iris
from model import Predictor
from data import penguins

columns_name = penguins.columns.tolist()

attributes = columns_name[2:-1]
predict = [columns_name[0]]

print(attributes, predict)

p = Predictor(penguins, attributes, predict, 0, [2, 5])
p.fit()

flower = [39.1, 18.7, 181, 3750]
p.prediction(flower)

p.performance()
