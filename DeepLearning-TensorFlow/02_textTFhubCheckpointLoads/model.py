# -*- coding: utf-8 -*-
"""
加载训练保存点
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


class Product(object):
    def __init__(self, data):

        self.model = None
        self.ckpt = "/Users/chixu/Desktop/DeepLearning-TensorFlow/loadTextBinaryClassification/training_tf_hub/cp.ckpt"
        self.data = data
        self.test_data = data

    def create_model(self):
        """
        创建模型对象
        :return: model
        """

        # print(hub_layer(train_examples_batch[:3]))

        embedding = "https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim/1"

        hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)

        model = tf.keras.Sequential()

        model.add(hub_layer)

        model.add(tf.keras.layers.Dense(16, activation='relu'))

        model.add(tf.keras.layers.Dense(1))

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        model.load_weights(self.ckpt)

        self.model = model

    def run(self):
        self.create_model()

        text, text_lable = next(iter(test_data.batch(1)))

        text[0:1]

        predictions = self.model.predict(text)

        for name, value in zip(self.model.metrics_names, predictions):
            print("%s: %.3f" % (name, value))


if __name__ == '__main__':
    import tensorflow_datasets as tfds

    train_data, validation_data, test_data = tfds.load(
        name="imdb_reviews",
        split=('train[:60%]', 'train[60%:]', 'test'),
        as_supervised=True)

    prod = Product(test_data.batch(512))

    prod.run()