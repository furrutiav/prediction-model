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
predict = columns_name[:1]

p = Predictor(penguins, attributes, predict, 0, [2, 5])
p.fit()

# flower = [50.2,14.3,218,5700]
# p.prediction(flower)

p.performance()
