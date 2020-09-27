# -*- coding: utf-8 -*-
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling


print(tf.__version__)

"""First download the dataset."""
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

# dataset_path = "/Users/chixu/.keras/datasets/auto-mpg.data"

"""Import it using pandas"""
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(
    dataset_path,
    names=column_names,
    na_values="?",
    comment='\t',
    sep=" ",
    skipinitialspace=True
)


dataset = raw_dataset.copy()

print(dataset.tail())

"""Clean the data"""
print(dataset.isna().sum())

"""Drop rows"""
dataset = dataset.dropna()

dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

"""
get_dummies
columns : list-like, default None
        Column names in the DataFrame to be encoded.
        If `columns` is None then all the columns with
        `object` or `category` dtype will be converted.
"""
print('Origin_dtype:', dataset['Origin'].dtype)

dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')

print(dataset.tail())

"""
Split the data into train and test

Use the test set in the final evaluation of our model.
"""

train_dataset = dataset.sample(frac=0.8, random_state=0)

test_dataset = dataset.drop(train_dataset.index)

g = sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

g.map(plt.scatter, alpha=0.)

plt.show()

"""
Also look at the overall statistics:
整体统计信息
count   最大值
mean    平均值
std     标准差
min     最小值
max     最大值
"""
train_stats = train_dataset.describe()

train_stats.pop("MPG")

train_stats = train_stats.transpose()

print(train_stats)

"""Split feature from labels 通过特征分割标签"""
train_labels = train_dataset.pop('MPG')

test_labels = test_dataset.pop('MPG')

"""Normalize the data 归一化"""
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

"""
需要将此处用于标准化输入的统计信息（均值和标准差）与我们之前所做的“one-hot”一起应用于喂入模型的任何其他数据。
在生产中使用该模型时，其中包括测试集以及实时数据。
"""

"""Build the model"""
def bulid_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    return model

model = bulid_model()

"""Inspect the model检查模型"""

model.summary()

"""
现在尝试模型。从训练数据中选取10个示例，然后调用model.predict。
"""

example_batch = normed_train_data[:10]

example_result = model.predict(example_batch)

print(example_result)

"""
训练模型
"""
EPOCHS = 1000

history = model.fit(
    normed_train_data,
    train_labels,
    epochs=EPOCHS,
    validation_split = 0.2,
    verbose=0,
    callbacks=[tfdocs.modeling.EpochDots()]
)

"""
通过历史信息查看训练进度
"""
hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

print(hist.tail())

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

"""平均绝对误差MAE（mean absolute error）是绝对误差的平均值，它其实是更一般形式的误差平均值。"""
plotter.plot({'Basic': history}, metric="mae")

plt.ylim([0, 10])

plt.ylabel('MAE [MPG]')

plt.show()

"""均方误差（MSE）是用于回归问题的常见损失函数（不同的损失函数用于分类问题）。"""
plotter.plot({'Basic': history}, metric="mse")

plt.ylim([0, 20])

plt.ylabel('MSE [MPG^2]')

plt.show()

"""
此图显示约100个纪元后，验证错误几乎没有改善，甚至降级了。让我们更新model.fit调用，以在验证得分没有提高时自动停止训练。我们将使用EarlyStopping回调来测试每个时期的训练条件。如果经过了一定数量的时期但没有显示出改善，则自动停止训练。
"""
model = bulid_model()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

early_history = model.fit(
    normed_train_data,
    train_labels,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[early_stop, tfdocs.modeling.EpochDots()]
)

plotter.plot({'Early Stopping': early_history}, metric='mae')

plt.ylim([0, 10])

plt.ylabel('MAE [MPG]')

plt.show()

"""
在训练过程中没有用到的测试数据对模型的概括程度
"""
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

"""
最后，使用测试集中的数据预测MPG值：
"""
test_predictions = model.predict(normed_test_data).flatten()

a = plt.axes(aspect='equal')

plt.scatter(test_labels, test_predictions)

plt.xlabel('True Value [MPG]')

plt.ylabel('Predictions [MPG]')

lims = [0, 50]

plt.xlim(lims),

plt.ylim(lims)

_ = plt.plot(lims, lims)

plt.show()

"""
查看错误分布
"""
error = test_predictions - test_labels

plt.hist(error, bins=25)

plt.xlabel('Prediction Error [MPG]')

_ = plt.ylabel('Count')

plt.show()

"""
结果没有显示高斯分布，但是由于数据量小，高斯分布是可以预见的，
"""