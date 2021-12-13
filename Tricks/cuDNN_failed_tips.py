# coding:utf-8

#########################################################################
# 1、问题描述
# tf在win的Anaconda虚拟环境下使用GPU加速，出现
# UnknownError: Failed to get convolution algorithm. This is probably because cuDNN failed to initiali


# 解决：
# 在虚拟环境中安装tf-gpu，只带安装cuda，因此不是版本不匹配的问题
# 需在import tf时指定运行设置为GPU

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
#
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
#########################################################################
