import pandas as pd

print('Loading data...')
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']
iris = pd.read_csv(url, names=names)

print('Iris data is available for training')
