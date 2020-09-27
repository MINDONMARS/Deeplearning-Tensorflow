"""
Keras Tuner是一个库，可帮助您为TensorFlow程序选择最佳的超参数集。
为您的机器学习（ML）应用程序选择正确的超参数集的过程称为超参数调整或超调整。

超参数是控制训练过程和ML模型的拓扑的变量。这些变量在训练过程中保持不变，并直接影响ML程序的性能。超参数有两种类型：
1.影响模型选择的模型超参数，例如隐藏层的数量和宽度
2.影响学习算法的速度和质量的算法超参数，例如随机梯度下降（SGD）的学习率和k个最近邻（KNN）分类器的最近邻居数

在本教程中，您将使用Keras Tuner对图像分类应用程序执行超调。
"""

import tensorflow as tf
from tensorflow import keras

import IPython

import kerastuner as kt


# 下载并准备数据集
# 在本教程中，您将使用Keras Tuner为机器学习模型找到最佳的超参数，该模型对Fashion MNIST数据集中的服装图像进行分类。

(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()

# 归一化介于0和1之间的像素值
img_train = img_train.astype('float32') / 255.0

img_test = img_test.astype('float32') / 255.0

"""
定义模型：
当构建用于超调的模型时，除了模型体系结构之外，您还定义了超参数搜索空间。您为超调设置的模型称为超模型。
您可以通过两种方法定义超模型：
    1.通过使用model builder函数
    2.通过子类化Keras Tuner API的HyperModel类
您还可以将两个预定义的HyperModel类，HyperXception和HyperResNet用于计算机视觉应用程序。

在本教程中，您将使用模型构建器功能来定义图像分类模型。
模型构建器函数返回已编译的模型，并使用您内联定义的超参数对模型进行超调。
"""

def model_builder(hp):

    model = keras.Sequential()

    model.add(keras.layers.Flatten(input_shape=(28, 28)))

    # 调整第一个Dense层中的单位数
    # 在32-512之间选择一个最佳值
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(10))

    # 调整优化器的学习率
    # 从0.01、0.001或0.0001中选择一个最佳值
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

"""
实例化调谐器以执行超调谐。 
Keras调谐器有四个可用的调谐器:
    RandomSearch，
    Hyperband，
    BayesianOptimization，
    Sklearn。
在本教程中，您将使用Hyperband调谐器。

要实例化Hyperband调谐器，必须指定超模型，要优化的目标以及要训练的最大时期数（max_epochs）。
"""

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt',
                     overwrite=True)

"""
超调整算法使用自适应资源分配和提前停止来快速收敛到高性能模型上。这是使用锦标赛风格的框架完成的。
该算法在几个时期内训练了大量模型，并且仅将性能最高的一半模型进行到下一轮。
超带宽通过计算1 + logfactor（max_epochs）并将其四舍五入到最接近的整数来确定要在框架中训练的模型的数量。
"""

# 在运行超参数搜索之前，定义一个回调以在每个训练步骤结束时清除训练输出。
class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)

tuner.search(img_train, label_train, epochs = 10, validation_data = (img_test, label_test), callbacks = [ClearTrainingOutput()])

# 获取最佳超参数
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

model = tuner.hypermodel.build(best_hps)

model.fit(img_train, label_train, epochs=10, validation_data=(img_test, label_test))

"""
my_dir/intro_to_kt目录包含在超参数搜索期间运行的每个试验（模型配置）的详细日志和检查点。
如果重新运行超参数搜索，Keras Tuner将使用这些日志中的现有状态来继续搜索。
要禁用此行为，请在实例化调谐器时传递一个额外的overwrite=True参数。
"""