# -*- conding:utf-8 -*-
import numpy as np
import tensorflow.compat.v1 as tf
import sys
import os

tf.disable_eager_execution()

sys.path.append('..')
all_data_size = 1000
input_x_size = 20
field_size = 2

vector_dimension = 3

total_train_steps = 10000
batch_size = 1
lr = 0.01
MODEL_SAVE_PATH = "TFModel"
MODEL_NAME = "FFM"


def gen_data():
    labels = [-1, 1]
    y = [np.random.choice(a=labels, size=1)[0] for _ in range(all_data_size)]
    x = np.random.randint(low=0, high=2, size=(all_data_size, input_x_size))
    x_field = [i // 10 for i in range(input_x_size)]
    return x, y, x_field


def create_zero_dimension_weight():
    weights = tf.truncated_normal(shape=[1])
    tf_weights = tf.Variable(weights)
    return tf_weights


def create_one_dimension_weight(input_x_size):
    """
    低维特征权重（一维）
    :param input_x_size: 训练数据维度
    :return: 一次项特征权重
    """
    weights = tf.truncated_normal(shape=[input_x_size])
    tf_weights = tf.Variable(weights)
    return tf_weights


def create_two_dimension_weight(input_x_size, field_size, vector_dimension):
    """
    特征交叉权重
    :param input_x_size:
    :param field_size:
    :param vector_dimension:
    :return: 二次项特征权重
    """
    weights = tf.truncated_normal(shape=[input_x_size, field_size, vector_dimension])
    tf_weights = tf.Variable(weights)
    return tf_weights


def inference(input_x, input_x_field, zero_weight, one_weight, two_weight):
    """
    计算回归模型输出的值
    :param input_x:
    :param input_x_field:
    :param zero_weight: 偏置项
    :param one_weight: 一次项权重
    :param two_weight: 二次项权重
    :return: target (prediction)
    """
    second_value = tf.reduce_sum(tf.multiply(one_weight, input_x, name="second_value"))
    first_second_value = tf.add(zero_weight, second_value, name="first_second_value")
    third_value = tf.Variable(0.0, dtype=tf.float32)
    input_shape = input_x_size  # 20

    for i in range(input_shape):
        feature_index_1 = i
        field_index_1 = int(input_x_field[i])
        for j in range(i + 1, input_shape):
            feature_index_2 = j
            field_index_2 = int(input_x_field[j])
            # 特征处理
            vector_left = tf.convert_to_tensor([[feature_index_1, feature_index_2, i] for i in range(vector_dimension)])
            weight_left = tf.gather_nd(two_weight, vector_left)
            weight_left_after_cut = tf.squeeze(weight_left)

            # field处理
            vector_right = tf.convert_to_tensor([[field_index_1, field_index_2, i] for i in range(vector_dimension)])
            weight_right = tf.gather_nd(two_weight, vector_right)
            weight_right_after_cut = tf.squeeze(weight_right)

            temp_value = tf.reduce_sum(tf.multiply(weight_left_after_cut, weight_right_after_cut))

            indices2 = [i]
            indices3 = [j]

            xi = tf.squeeze(tf.gather_nd(input_x, indices2))
            xj = tf.squeeze(tf.gather_nd(input_x, indices3))

            product = tf.reduce_sum(tf.multiply(xi, xj))

            second_item_val = tf.multiply(temp_value, product)
            tf.assign(third_value, tf.add(third_value, second_item_val))
    return tf.add(first_second_value, third_value)


if __name__ == "__main__":
    global_step = tf.Variable(0, trainable=False)
    train_x, train_y, train_field = gen_data()

    input_x = tf.placeholder(dtype=tf.float32, shape=[input_x_size])
    input_y = tf.placeholder(tf.float32)

    lambda_w = tf.constant(0.001, name='lambda_w')
    lambda_v = tf.constant(0.001, name='lambda_v')

    zero_weight = create_zero_dimension_weight()
    one_weight = create_one_dimension_weight(input_x_size)
    two_weight = create_two_dimension_weight(input_x_size, field_size, vector_dimension)  # n * f * k

    y_ = inference(input_x, train_field, zero_weight, one_weight, two_weight)

    l2_norm = tf.reduce_sum(tf.add(tf.multiply(lambda_w, tf.pow(one_weight, 2)),
                                   tf.reduce_sum(tf.multiply(lambda_v, tf.pow(two_weight, 2)), axis=[1, 2])
                                   )
                            )

    loss = tf.log(1 + tf.exp(-input_y * y_)) + l2_norm

    train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(total_train_steps):
            for t in range(all_data_size):
                input_x_batch = train_x[t]
                input_y_batch=  train_y[t]
                predict_loss, _, steps = sess.run([loss, train_step, global_step],
                                                  feed_dict={input_x: input_x_batch, input_y: input_y_batch})
                print("After  {step} training   step(s)   ,   loss    on    training    batch   is  {predict_loss} "
                      .format(step=steps, predict_loss=predict_loss))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=steps)
                writer = tf.summary.FileWriter(os.path.join(MODEL_SAVE_PATH, MODEL_NAME), tf.get_default_graph())
                writer.close()



