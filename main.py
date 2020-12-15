"""
    Main Prediction Model:
    Felipe Urrutia V.
    Dec-2020
"""

# import class and data
from model import Predictor
from data import penguins, iris

# iris
columns_name = iris.columns.tolist()

attributes = columns_name[0:4]
predict = [columns_name[4]]

i = Predictor(iris, attributes, predict, 4, [0, 3])
i.fit()

flower = [4.9, 5.3, 3.1, 4.0]
i.prediction(flower)

i.performance()

# penguins
columns_name = penguins.columns.tolist()

attributes = columns_name[2:6]
predict = [columns_name[0]]

p = Predictor(penguins, attributes, predict, 0, [2, 5])
p.fit()

penguin = [39.1, 18.7, 181, 3750]
p.prediction(penguin)

p.performance()
