import os

import tensorflow as tf

from tensorflow import keras

print(tf.version.VERSION)

"""
获取示例数据集:
为了演示如何保存和加载权重，您将使用MNIST数据集。为了加快运行速度，使用前1000个示例：
"""

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0

test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

"""
定义模型
首先，建立一个简单的顺序模型：
"""
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model

# 创建模型
model = create_model()

# 显示摘要
model.summary()

"""
在训练期间保存检查点
可以使用受过训练的模型，而不必重新训练它，也可以在您停下来的地方接受训练，以防训练过程中断。通过tf.keras.callbacks.ModelCheckpoint回调，可以在训练期间和训练结束时连续保存模型。
"""

# 检查点回调的用法
# 创建一个tf.keras.callbacks.ModelCheckpoint回调，仅在训练期间保存权重：
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 创建一个保存模型权重的回调
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# 使用新的回调训练模型
model.fit(train_images,
          train_labels,
          epochs=10,
          validation_data=(test_images,test_labels),
          callbacks=[cp_callback])  # 将回调传递给培训

# 这可能会生成与保存优化器状态有关的警告。这些警告（以及整个笔记本中的类似警告）在适当位置以防止过时的使用，可以将其忽略。
# 这将创建一个TensorFlow检查点文件的单个集合，这些集合将在每个纪元结束时进行更新：

"""
创建一个新的未经训练的模型。从仅权重还原模型时，必须具有与原始模型具有相同体系结构的模型。由于它是相同的模型架构，因此尽管它是模型的不同实例，但是您可以共享权重。
现在重建一个新的，未经训练的模型，并在测试集上对其进行评估。未经训练的模型将以概率层面（〜10％的准确性）执行：
"""
model2 = create_model()

# 评估模型
loss, acc = model2.evaluate(test_images,  test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# 然后从检查点加载权重并重新评估：
latest = tf.train.latest_checkpoint(checkpoint_dir)

model2.load_weights(latest)

loss, acc = model2.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

"""
这些文件是什么？
上面的代码将权重存储到检查点格式文件的集合中，这些文件仅包含经过训练的权重（二进制格式）。检查点包含：

一个或多个包含模型权重的碎片。
一个索引文件，指示哪些权重存储在哪个分片中。
如果仅在单台机器上训练模型，则后缀为一个分片：.data-00000-of-00001
"""

"""
手动保存权重：
您已经了解了如何将权重加载到模型中。
手动保存它们与使用Model.save_weights方法一样简单。
默认情况下，tf.keras（尤其是save_weights）使用带有.ckpt扩展名的TensorFlow检查点格式（“保存和序列化模型”指南中介绍了使用.h5扩展名保存在HDF5中）：
"""

# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Create a new model instance
model = create_model()

# Restore the weights
model.load_weights('./checkpoints/my_checkpoint')

# Evaluate the model
loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

"""
保存整个模型：
调用model.save将模型的架构，权重和训练配置保存在单个文件/文件夹中。
这使您可以导出模型，以便可以在不访问原始Python代码*的情况下使用它。
由于优化器状态已恢复，因此您可以从上次中断的地方继续进行训练。
# 整个模型可以两种不同的文件格式保存（SavedModel和HDF5）。
# 请注意，TensorFlow SavedModel格式是TF2.x中的默认文件格式。但是，模型可以HDF5格式保存。下面介绍了以两种文件格式保存整个模型的更多详细信息。
保存功能齐全的模型非常有用-您可以将它们加载到TensorFlow.js（保存的模型，HDF5）中，然后在Web浏览器中训练和运行它们，或者使用TensorFlow Lite（保存的模型，HDF5）将它们转换为在移动设备上运行）
*自定义对象（例如，子类化模型或图层）在保存和加载时需要特别注意。请参阅下面的“保存自定义对象”部分
"""

"""
SavedModel format
SavedModel格式是序列化模型的另一种方法。
可以使用tf.keras.models.load_model还原以这种格式保存的模型，并且与TensorFlow Serving兼容。 
SavedModel指南详细介绍了如何提供/检查SavedModel。以下部分说明了保存和还原模型的步骤。
"""

# Create and train a new model instance.
model = create_model()

model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a SavedModel.

model.save('saved_model/my_model')

"""
SavedModel格式是一个包含protobuf二进制文件和Tensorflow检查点的文件夹。
"""

"""
从已保存的模型中重新加载新的Keras模型：
"""

new_model = tf.keras.models.load_model('saved_model/my_model')

new_model.summary()

"""
还原的模型使用与原始模型相同的参数进行编译。
"""

# 尝试使用加载的模型运行评估和预测：
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)

print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

print(new_model.predict(test_images).shape)

"""
HDF5 format
Keras使用HDF5标准提供了一种基本的保存格式。
"""

model = create_model()

model.fit(train_images, train_labels, epochs=5)

"""
将整个模型保存到HDF5文件中。 
扩展名'.h5'表示应将模型保存到HDF5。
"""

model.save('my_model.h5')

"""
现在，从该文件重建模型：
"""

# 重新创建完全相同的模型，包括权重和优化器
new_model = tf.keras.models.load_model('my_model.h5')

# 显示模型架构
new_model.summary()

# 检查准确性
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)

print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

"""
Keras通过检查架构来保存模型。这种技术可以保存一切：

权重值
模型的架构
模型的训练配置（传递给您的编译信息）
优化器及其状态（如果有）（这使您可以从离开的地方重新开始训练）
Keras无法保存v1.x优化器（来自tf.compat.v1.train），因为它们与检查点不兼容。对于v1.x优化器，您需要在加载后重新编译模型-失去优化器的状态。

保存自定义对象
如果使用的是SavedModel格式，则可以跳过此部分。 
HDF5和SavedModel之间的主要区别在于，HDF5使用对象配置保存模型体系结构，而SavedModel保存执行图。因此，SavedModels能够保存自定义对象，例如子类化模型和自定义层，而无需原始代码。

要将自定义对象保存到HDF5，必须执行以下操作：

1.在对象中定义一个get_config方法，以及可选的from_config类方法。
    get_config(self)返回一个JSON可序列化的字典，其中包含重新创建对象所需的参数。
    from_config(cls，config)使用从get_config返回的配置来创建新对象。默认情况下，此函数将使用config作为初始化变量（返回cls（** config））。
2.加载模型时，将对象传递给custom_objects参数。参数必须是将字符串类名称映射到Python类的字典。 E.g. tf.keras.models.load_model(path, custom_objects={'CustomLayer': CustomLayer})
"""