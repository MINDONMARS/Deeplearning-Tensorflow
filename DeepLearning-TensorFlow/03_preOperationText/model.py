# -*- coding: utf-8 -*-
"""
预处理文本分类
积极消极2分类
"""
import tensorflow as tf

from tensorflow import keras

import tensorflow_datasets as tfds

tfds.disable_progress_bar()

import numpy as np

from matplotlib import pyplot as plt


print(tf.__version__)

# dataset
(train_data, test_data), info = tfds.load(
    # Use the version pre-encoded with an ~8k vocabulary.
    'imdb_reviews/subwords8k',
    # Return the train/test datasets as a tuple.
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    # Return (example, label) pairs from the dataset (instead of a dictionary)
    as_supervised=True,
    # Also return the `info` structure
    with_info=True
)

# encoder
encoder = info.features['text'].encoder

print("Vocabulary size: {}".format(encoder.vocab_size))

"""encoder example"""
sample_string = "Hello Tensorflow."

encoded_string = encoder.encode(sample_string)  # encode

print('Encoded string is {}'.format(encoded_string))

original_string = encoder.decode(encoded_string)  # decode

print('Decode string is {}'.format(original_string))
#
"""
The encoder encodes the string by breaking it into subwords or characters if the word is not in its dictionary.
So the more a string resembles the dataset, the shorter the encoded representation will be
"""
for ts in encoded_string:
    print('{}------>{}'.format(ts, encoder.decode([ts])))


"""Here's what the first review looks like:"""
for train_example, train_label in train_data.take(1):

    print('Encoded text:', train_example[0:10].numpy())

    print('Label:', train_label.numpy())

    """The info structure contains the encoder/decoder. The encoder can be used to recover the original text:"""
    print(encoder.decode(train_example))


"""
Prepare the data for training.
You will want to create batches of training data for your model. 
The reviews are all different lengths, so use padded_batch to zero pad the sequences while batching:
"""
BUFFFER_SIZE = 1000

train_batches = (train_data.shuffle(BUFFFER_SIZE).padded_batch(32, padded_shapes=([None], [])))

test_batches = (test_data.padded_batch(32))

"""
Each batch will have a shape of (batch_size, sequence_length) because the padding is dynamic each batch will have a different length:
"""
for example_batch, label_batch in train_batches.take(2):

    print("Batch shape:", example_batch.shape)

    print("label shape:", label_batch.shape)

"""Build the model"""
model = keras.Sequential([
    keras.layers.Embedding(encoder.vocab_size, 16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(1)
])

"""
The layers are stacked sequentially to build the classifier:

    1.The first layer is an Embedding layer. This layer takes the integer-encoded vocabulary and looks up the embedding vector for each word-index. These vectors are learned as the model trains. The vectors add a dimension to the output array. The resulting dimensions are: (batch, sequence, embedding). To learn more about embeddings, see the word embedding tutorial.
    
    2.Next, a GlobalAveragePooling1D layer returns a fixed-length output vector for each example by averaging over the sequence dimension. This allows the model to handle input of variable length, in the simplest way possible.
    
    3.This fixed-length output vector is piped through a fully-connected (Dense) layer with 16 hidden units.
    
    4.The last layer is densely connected with a single output node. This uses the default linear activation function that outputs logits for numerical stability. Another option is to use the sigmoid activation function that returns a float value between 0 and 1, representing a probability, or confidence level.
"""

model.summary()

"""
The above model has two intermediate or "hidden" layers, between the input and output. The number of outputs (units, nodes, or neurons) is the dimension of the representational space for the layer. In other words, the amount of freedom the network is allowed when learning an internal representation.

If a model has more hidden units (a higher-dimensional representation space), and/or more layers, then the network can learn more complex representations. However, it makes the network more computationally expensive and may lead to learning unwanted patterns—patterns that improve performance on training data but not on the test data. This is called overfitting, and we'll explore it later.
"""

"""Now, configure the model to use an optimizer and a loss function:"""
model.compile(
    optimizer='adam',
    loss=tf.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

"""
Train the model
Train the model by passing the Dataset object to the model's fit function. Set the number of epochs.
"""
history = model.fit(
    train_batches,
    epochs=10,
    validation_data=test_batches,
    validation_steps=30
)

"""
Evaluate the model(评估模型)
And let's see how the model performs. Two values will be returned. Loss (a number which represents our error, lower values are better), and accuracy.
（损失越低越好）
"""
loss, accuracy = model.evaluate(test_batches)

print("Loss: ", loss)
print("accuracy: ", accuracy)

"""Create a graph of accuracy and loss over time"""

"""model.fit() returns a History object that contains a dictionary with everything that happened during training:"""
history_dict = history.history

print(history_dict.keys())

"""
There are four entries: one for each monitored metric during training and validation. We can use these to plot the training and validation loss for comparison, as well as the training and validation accuracy:
"""

acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
loss = history_dict["loss"]
val_loss = history_dict["val_loss"]

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# "b" is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()