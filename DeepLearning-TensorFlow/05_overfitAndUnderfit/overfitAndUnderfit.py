# -*- coding: utf-8 -*-

"""
过拟合与欠拟合
过拟合：
在前面的两个示例（对文本进行分类和预测燃油效率）中，我们看到了在验证数据上的模型的准确性在经过多个时期的训练后将达到峰值，然后停滞或开始下降。
欠拟合：
模型的功能不够强大，被过度规范化，或者仅仅是没有足够经过长时间的训练，这意味着网络尚未学习训练数据中的相关模式。
--------
但是，如果训练时间过长，则模型将开始过拟合并从训练数据中学习无法推广到测试数据的模式。
我们需要保持平衡。如下所述，了解如何训练适当的时期是一项有用的技能。
为了防止过度拟合，最好的解决方案是使用更完整的训练数据。数据集应涵盖模型应处理的所有输入范围。仅当涉及新的有趣案例时，其他数据才有用。
经过更完整数据训练的模型自然会更好地推广。当这不再可能时，下一个最佳解决方案是使用  正则化  之类的技术。
这些都限制了模型可以存储的信息的数量和类型。
如果一个网络只能存储少量模式，那么优化过程将迫使它专注于最突出的模式，这些模式有更好的概括机会。
"""

"""常见的 正则化 技术，并使用它们对分类模型进行改进"""

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import regularizers

print(tf.__version__)

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from IPython import display
from matplotlib import pyplot as plt

import numpy as np

import pathlib
import shutil
import tempfile

logdir = pathlib.Path(tempfile.mkdtemp()) / "tensorboard_logs"

shutil.rmtree(logdir, ignore_errors=True)


"""
The Higgs Dataset 希格斯数据集
本教程的目的不是做粒子物理学，所以不要关注数据集的细节。它包含11 000 000个示例，每个示例具有28个特征以及一个二分类标签。
"""
gz = tf.keras.utils.get_file("HIGGS.csv.gz", 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')

FEATURES = 28

"""
tf.data.experimental.CsvDataset类可用于直接从gzip文件读取csv记录，而无需中间的解压缩步骤。
"""
ds = tf.data.experimental.CsvDataset(gz, [float(), ] * (FEATURES + 1), compression_type="GZIP")

def pack_row(*row):

    label = row[0]

    features = tf.stack(row[1:], 1)

    return features, label

"""
当处理大量数据时，TensorFlow效率最高。
因此，与其单独重新包装每一行，不如创建一个新的数据集，该数据集采用10000个示例的批次，对每个批次应用pack_row函数，然后将这些批次拆分回各个记录：
"""

print(ds.batch(1))

packed_ds = ds.batch(10000).map(pack_row).unbatch()


"""
看一下这个新的packed_ds的一些记录。
这些功能尚未完全标准化，但这足以满足本教程的要求。
"""

for features,label in packed_ds.batch(1000).take(1):

    print(features[0])

    print(features.numpy().shape)

    plt.hist(features.numpy().flatten(), bins=101)

    plt.show()

"""
为了使本教程相对简短，仅使用前1000个样本进行验证，然后使用10000个样本进行培训：
"""

N_VALIDATION = int(1e3)

N_TRAIN = int(1e4)

BUFFER_SIZE = int(1e4)

BATCH_SIZE = 500

STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE

validate_ds = packed_ds.take(N_VALIDATION).cache()

train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()

print(train_ds)

"""
这些数据集返回单个示例。使用.batch方法可创建适当大小的批次进行训练。批处理之前，还记得.shuffle和.repeat训练集。
"""

validate_ds = validate_ds.batch(BATCH_SIZE)

train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

"""
演示过拟合
防止过度拟合的最简单方法是从一个小的模型开始：一个具有少量可学习参数（由层数和每层单位数确定）的模型。在深度学习中，模型中可学习参数的数量通常称为模型的“容量”。直观地讲，具有更多参数的模型将具有更多的“记忆能力”，因此将能够轻松学习训练样本与其目标之间的完美的字典式映射，这种映射没有任何泛化能力，但是在进行预测时这将是无用的根据以前看不见的数据。始终牢记这一点：深度学习模型往往擅长拟合训练数据，但真正的挑战是泛化而不是拟合。另一方面，如果网络的存储资源有限，则将无法轻松地学习映射。为了最大程度地减少损失，它必须学习具有更强预测能力的压缩表示形式。同时，如果您使模型过小，将难以拟合训练数据。 “容量过多”和“容量不足”之间存在平衡。不幸的是，没有神奇的公式来确定模型的正确大小或体系结构（根据层数或每层的正确大小）。您将不得不尝试使用一系列不同的体系结构。为了找到合适的模型大小，最好从相对较少的图层和参数开始，然后开始增加图层的大小或添加新的图层，直到看到验证损失的收益递减为止。从仅使用图层的简单模型开始，以密集为基准，然后创建较大的版本并进行比较。
"""

"""
训练过程
如果您逐渐减少训练期间的学习率，许多模型的训练效果会更好。使用optimizers.schedules随着时间的推移降低学习率：
"""

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=STEPS_PER_EPOCH * 1000,
    decay_rate=1,
    staircase=False
)

def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)

"""
上面的代码设置了一个schedules.InverseTimeDecay，以双曲线的方式将学习速率在1000个周期降低到基本速率的1/2，在2000个周期降低1/3，依此类推。
"""
step = np.linspace(0, 100000)

print(step)

lr = lr_schedule(step)

plt.figure(figsize = (8,6))

plt.plot(step / STEPS_PER_EPOCH, lr)

plt.ylim([0, max(plt.ylim())])

plt.xlabel('Epoch')

_ = plt.ylabel("Learning Rate")

plt.show()

"""
本教程中的每个模型都将使用相同的训练配置。因此，从回调列表开始，以可重用的方式设置它们。

本教程的训练持续了很短的时间。要减少日志记录噪音，请使用tfdocs.EpochDots，对于每个周期，以及每100个周期的完整指标，它只打印一个'.'。

接下来导入callbacks.EarlyStopping以避免冗长和不必要的训练时间。请注意，此回调设置为监视val_binary_crossentropy，而不是val_loss。这种差异稍后将变得很重要。

使用callbacks.TensorBoard生成用于训练的TensorBoard日志。
"""
def get_callbacks(name):
    return [
        tfdocs.modeling.EpochDots(),
        tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
        tf.keras.callbacks.TensorBoard(logdir/name),
    ]

"""
同样，每个模型将使用相同的Model.compile和Model.fit设置：
"""

def compile_and_fit(model, name, optimizer=None, max_epochs=10000):

    if optimizer is None:

        optimizer = get_optimizer()

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            tf.keras.losses.BinaryCrossentropy(
                from_logits=True,
                name='binary_crossentropy'
            ),
            'accuracy'
        ]
    )

    model.summary()

    history = model.fit(
        train_ds,
        steps_per_epoch = STEPS_PER_EPOCH,
        epochs=max_epochs,
        validation_data=validate_ds,
        callbacks=get_callbacks(name),
        verbose=0
    )
    return history

"""
小模型
"""
tiny_model = tf.keras.Sequential([
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(1)
])

size_histories = {}

size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')

"""
现在检查模型如何工作：
"""

plotter = tfdocs.plots.HistoryPlotter(metric='binary_crossentropy', smoothing_std=10)

plotter.plot(size_histories)

plt.ylim([0.5, 0.7])

plt.show()

"""
小模型2
"""

# small_model = tf.keras.Sequential([
#     # `input_shape` is only required here so that `.summary` works.
#     layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
#     layers.Dense(16, activation='elu'),
#     layers.Dense(1)
# ])

# size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small')

"""
中等模型
3个包含64个单元的隐藏层
"""

# medium_model = tf.keras.models.Sequential([
#     layers.Dense(64, activation='elu', input_shape=(FEATURES,)),
#     layers.Dense(64, activation='elu'),
#     layers.Dense(64, activation='elu'),
#     layers.Dense(1)
# ])

"""
用同样的数据训练中等模型
"""

# size_histories['Medium'] = compile_and_fit(medium_model, 'sizes/Medium')


"""
大模型
作为练习，您可以创建一个更大的模型，并查看它开始过拟合的速度。接下来，让我们将具有更大容量的网络添加到此基准中，远远超出问题所能保证的范围：
"""
# large_model = tf.keras.Sequential([
#     layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
#     layers.Dense(512, activation='elu'),
#     layers.Dense(512, activation='elu'),
#     layers.Dense(512, activation='elu'),
#     layers.Dense(1)
# ])

"""
用同样的数据训练中等模型
"""

# size_histories['large'] = compile_and_fit(large_model, "sizes/large")

"""
绘制训练和验证损失
实线表示训练损失，而虚线表示验证损失（请记住：验证损失越小表示模型越好）。

虽然构建较大的模型可以提供更多功能，但是如果不以某种方式限制此功能，则可以轻松地将其过度拟合到训练集。

在此示例中，通常，只有“ Tiny”模型设法避免完全过拟合，而每个较大的模型都更快地过拟合数据。对于“大型”模型而言，这变得如此严重，以至于您需要将绘图切换为对数比例才能真正看到正在发生的事情。

如果您将验证指标与训练指标进行比较并进行比较，这很明显。

差异很小是正常的。
如果两个指标都朝着同一方向发展，那么一切都很好。
如果在训练指标继续提高的同时，验证指标开始停滞不前，那么您可能已接近过度拟合。
如果验证指标的方向错误，则表明该模型过度拟合。
"""

plotter.plot(size_histories)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
plt.ylim([0.5, 0.7])
plt.xlabel("Epochs [Log Scale]")
plt.show()

"""
防止过拟合的策略
"""

"""
复制上面“ Tiny”模型中的培训日志，以用作比较的基准。
"""
shutil.rmtree(logdir/'regularizers/Tiny', ignore_errors=True)

shutil.copytree(logdir/'sizes/Tiny', logdir/'regularizers/Tiny')

pathlib.PosixPath('/tmp/tmpqdi_f3ae/tensorboard_logs/regularizers/Tiny')

regularizer_histories = {}

regularizer_histories['Tiny'] = size_histories['Tiny']

"""
增加权重调整
奥卡姆剃刀原理：（如无必要，勿增实体）做出最少假设
这也适用于通过神经网络学习的模型：
给定一些训练数据和网络体系结构，可以使用多组权重值（多个模型）来解释数据，并且较简单的模型比复杂的模型发生过拟合的可能性更小。
在这种情况下，“简单模型”是参数值的分布具有较小熵的模型（或如上节所述，或具有总共较少参数的模型）。
因此，减轻过度拟合的一种常用方法是通过仅使网络的权重取小的值来对网络的复杂性施加约束，这使得权重值的分布更加“规则”。这称为“权重调整”，它是通过向网络的损失函数中添加与权重较大相关的成本来完成的。

此成本有两方面：
L1正则化，其中增加的成本与权重系数的绝对值成正比（即，权重的所谓“L1范数”）。
L2正则化，其中增加的成本与权重系数的值的平方成正比（即与权重的平方的“L2范数”平方成正比）。 L2正则化在神经网络中也称为权重衰减。不要让其他名称使您感到困惑：权重衰减在数学上与L2正则化完全相同。
L1正则化将权重推向正好为零，从而鼓励了稀疏模型。
L2正则化将惩罚权重参数而不会使其稀疏，因为对于小权重，惩罚变为零。 这也是L2更常见的原因之一。

在tf.keras中，通过将 权重正则化器 实例作为关键字参数传递给图层来添加权重正则化。让我们现在添加L2权重正则化。
"""

l2_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001), input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1)
])

regularizer_histories['l2'] = compile_and_fit(l2_model, "regularizers/l2")

"""
l2（0.001）表 示该层权重矩阵中的每个系数将为网络的总损耗增加 0.001 * weight_coefficient_value ** 2。

这就是为什么我们直接监视binary_crossentropy。因为它没有混入此正则化组件。

因此，相同“大型”的模型，具有L2正则化惩罚的性能要好得多：
"""

plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
plt.show()

"""
如图所见，“L2”正则化模型现在比“ Tiny”模型更具竞争力。尽管具有相同数量的参数，但“ L2”模型也比其基于的“大”模型更耐过拟合。
"""

"""
这种正则化有两点要注意。

首先：如果您正在编写自己的训练循环，则需要确保向模型询问其正则化损失。

第二：此实现通过将权重损失添加到模型的损失中，然后在此之后应用标准优化过程来工作。

还有第二种方法，它只对原始损耗运行优化器，然后在应用计算出的步骤时，优化器还会应用一些权重衰减。这种“解耦的权重衰减”可在诸如Optimizer.FTRL和Optimizer.AdamW之类的优化器中看到。
"""

result = l2_model(features)

regularization_loss=tf.add_n(l2_model.losses)

"""
Add dropout
Dropout是Hinton和他在多伦多大学的学生开发的最有效，最常用的神经网络正则化技术之一。
Dropout的直观解释是，由于网络中的各个节点不能依赖于其他节点的输出，因此每个节点必须输出自己有用的功能。

Dropout, 应用于图层的过程包括在训练过程中随机“dropping out”（即设置为零）该图层的许多输出特征。
假设在训练过程中，给定的图层通常会为给定的输入样本返回向量[0.2、0.5、1.3、0.8、1.1]；
应用删除后，此向量将有一些零个条目随机分布，        例如[0，0.5，1.3，0，1.1]。

“丢出率”是被清零的特征的一部分。通常设置在 0.2 到 0.5 之间。在测试时，不会丢失任何单元，而是将图层的输出值按等于丢失率的比例缩小，以平衡一个活跃的单元（而不是训练时）的事实。

在tf.keras中，您可以通过Dropout层在网络中引入Dropout，该层将立即应用于该层的输出。

让我们在网络中添加两个Dropout层，看看它们在减少过度拟合方面的表现如何：
"""

dropout_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

regularizer_histories['dropout'] = compile_and_fit(dropout_model, "regularizers/dropout")

plotter.plot(regularizer_histories)

plt.ylim([0.5, 0.7])

plt.show()

"""
从该图中可以明显看出，这两种正则化方法都可以改善“大”模型的行为。但这仍然没有超过“Tiny模型”基线。
"""