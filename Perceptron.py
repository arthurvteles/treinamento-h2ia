import random

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

seed = 42
random.seed(seed)
dataset = load_iris()
dataset_df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
dataset_df['target'] = dataset.target
novo_dataset_df = dataset_df[[
    'sepal length (cm)', 'sepal width (cm)', 'target']]
novo_dataset_df = novo_dataset_df[~(novo_dataset_df.target == 1)]
novo_dataset_df.target = novo_dataset_df.target.map(
    {0: 'setosa', 2: 'virginica'})
novo_dataset_df = novo_dataset_df.sample(frac=1)
novo_dataset_df.columns = [c.replace(' ', '_') for c in novo_dataset_df]
novo_dataset_df.columns = [c.lstrip() for c in novo_dataset_df]
novo_dataset_df.columns = [c.rstrip() for c in novo_dataset_df]
novo_dataset_df.to_csv("dataset_iris_simplificado.csv")


class Perceptron():
    def __init__(self, influency=0.01, n_iters=300000):
        self.influency = influency
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, x, y):
        n_samples, n_features = x.shape

        # inicializando os pesos
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            y_predict = np.dot(x, self.w) + self.b

            delta_w = (1/n_samples) * np.dot(x.T, (y - y_predict))
            delta_b = (1/n_samples) * np.sum(y - y_predict)
            self.w += self.influency * delta_w
            self.b += self.influency * delta_b

    def predict(self, x):
        pred = np.dot(x, self.w) + self.b
        y_pred = np.where(pred > 0, 1, 0)
        return y_pred


def accuracy(y, y_pred):
    accuracy = np.sum(y == y_pred) / len(y)
    return accuracy


x = novo_dataset_df[['sepal_length_(cm)', 'sepal_width_(cm)']]
y = novo_dataset_df['target']
y = np.array([0 if value == 'setosa' else 1 for value in y])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
per = Perceptron()
per.fit(x_train, y_train)
prediction = per.predict(x_test)
print(f'a acuracia foi de:  {accuracy(y_test,prediction)}')
