# -*- coding: utf-8 -*-
"""
tensorflow影评文本二分类
使用了 tf.keras，它是一个 Tensorflow 中用于构建和训练模型的高级API，
此外还使用了 TensorFlow Hub，一个用于迁移学习的库和平台
Error: SavedModel file does not exist at:... (mac 重启)
"""

# from __future__ import absolute_import, division, print_function, unicode_literals

import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

"""
下载 IMDB 数据集
IMDB数据集可以在 Tensorflow 数据集处获取。以下代码将 IMDB 数据集下载至您的机器（或 colab 运行时环境）中：
"""

# 将训练集按照 6:4 的比例进行切割，从而最终我们将得到 15,000
# 个训练样本, 10,000 个验证样本以及 25,000 个测试样本

# train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True
)

"""
探索数据
    让我们花一点时间来了解数据的格式。
    每一个样本都是一个表示电影评论和相应标签的句子。
    该句子不以任何方式进行预处理。
    标签是一个值为 0 或 1 的整数，其中 0 代表消极评论，1 代表积极评论。
我们来打印下前十个样本。
"""

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

print(train_examples_batch)

print(train_labels_batch)

"""
文本嵌入向量
"""
embedding = "https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim/1"

hub_layer = hub.KerasLayer(
    embedding,
    input_shape=[],
    dtype=tf.string,
    trainable=True
)

print(hub_layer(train_examples_batch[:3]))

model = tf.keras.Sequential()

model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

print(model.summary())

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

checkpoint_path = "training_tf_hub/cp.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.fit(
    train_data.shuffle(10000).batch(512),
    epochs=20,
    validation_data=validation_data.batch(512),
    verbose=1,
    # callbacks=[cp_callback]
)

results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))