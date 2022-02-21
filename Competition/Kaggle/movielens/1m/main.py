# coding:utf-8

import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import random
import datetime
from tensorflow import keras
from tensorflow.python.ops import summary_ops_v2
import time
import os
from sklearn.model_selection import train_test_split

# load data
title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = \
    pickle.load(open('preprocess.p', mode='rb'))
sentences_size = 15

MODEL_DIR = "./models"

# Number of Epochs
num_epochs = 5
# Batch Size
batch_size = 256

dropout_keep = 0.5
# Learning Rate
learning_rate = 0.0001
# Show stats for every n number of batches
show_every_n_batches = 20

save_dir = './save'

# 嵌入矩阵的维度
embed_dim = 32
# 用户ID个数
uid_max = max(features.take(0, 1)) + 1  # 6040
# 性别个数
gender_max = max(features.take(2, 1)) + 1  # 1 + 1 = 2
# 年龄类别个数
age_max = max(features.take(3, 1)) + 1  # 6 + 1 = 7
# 职业个数
job_max = max(features.take(4, 1)) + 1  # 20 + 1 = 21

# 电影ID个数
movie_id_max = max(features.take(1, 1)) + 1  # 3952
# 电影类型个数
movie_categories_max = max(genres2int.values()) + 1  # 18 + 1 = 19
# 电影名单词个数
movie_title_max = len(title_set)  # 5216

# 对电影类型嵌入向量做加和操作的标志，考虑过使用mean做平均，但是没实现mean
combiner = "sum"

# 电影名长度
sentences_size = title_count  # = 15
# 文本卷积滑动窗口，分别滑动2, 3, 4, 5个单词
window_sizes = {2, 3, 4, 5}
# 文本卷积核数量
filter_num = 8

# 电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}


class mv_network(object):
    def __init__(self, batch_size=256):
        self.batch_size = batch_size
        self.best_loss = 9999
        self.losses = {'train': [], 'test': []}

        # 获取输入占位符
        uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles = get_inputs()
        # 获取User的4个嵌入向量
        uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer = get_user_embedding(uid, user_gender,
                                                                                                   user_age, user_job)
        # 得到用户特征
        user_combine_layer, user_combine_layer_flat = get_user_feature_layer(uid_embed_layer, gender_embed_layer,
                                                                             age_embed_layer, job_embed_layer)
        # 获取电影ID的嵌入向量
        movie_id_embed_layer = get_movie_id_embed_layer(movie_id)
        # 获取电影类型的嵌入向量
        movie_categories_embed_layer = get_movie_categories_layers(movie_categories)
        # 获取电影名的特征向量
        pool_layer_flat, dropout_layer = get_movie_cnn_layer(movie_titles)
        # 得到电影特征
        movie_combine_layer, movie_combine_layer_flat = get_movie_feature_layer(movie_id_embed_layer,
                                                                                movie_categories_embed_layer,
                                                                                dropout_layer)
        # 计算出评分
        # 将用户特征和电影特征做矩阵乘法得到一个预测评分的方案
        inference = tf.keras.layers.Lambda(lambda layer:
                                           tf.reduce_sum(layer[0] * layer[1], axis=1), name="inference")(
            (user_combine_layer_flat, movie_combine_layer_flat))
        inference = tf.keras.layers.Lambda(lambda layer: tf.expand_dims(layer, axis=1))(inference)

        # 将用户特征和电影特征作为输入，经过全连接，输出一个值的方案
        # inference_layer = tf.keras.layers.concatenate([user_combine_layer_flat, movie_combine_layer_flat],
        #  )  # (?, 400)
        # 你可以使用下面这个全连接层，试试效果
        # inference_dense = tf.keras.layers.Dense(64, kernel_regularizer=tf.nn.l2_loss, activation='relu')(
        # inference_layer)
        # inference = tf.keras.layers.Dense(1, name="inference")(inference_layer)  # inference_dense

        self.model = tf.keras.Model(
            inputs=[uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles],
            outputs=[inference])

        self.model.summary()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        # MSE损失，将计算值回归到评分
        self.ComputeLoss = tf.keras.losses.MeanSquaredError()
        self.ComputeMetrics = tf.keras.metrics.MeanAbsoluteError()

        if tf.io.gfile.exists(MODEL_DIR):
            # print('Removing existing model dir: {}'.format(MODEL_DIR))
            # tf.io.gfile.rmtree(MODEL_DIR)
            pass
        else:
            tf.io.gfile.makedirs(MODEL_DIR)

        train_dir = os.path.join(MODEL_DIR, 'summaries', 'train')
        test_dir = os.path.join(MODEL_DIR, 'summaries', 'eval')

        # self.train_summary_writer = summary_ops_v2.create_file_writer(train_dir, flush_millis=10000)
        # self.test_summary_writer = summary_ops_v2.create_file_writer(test_dir, flush_millis=10000, name='test')

        checkpoint_dir = os.path.join(MODEL_DIR, 'checkpoints')
        self.checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

        # Restore variables on creation if a checkpoint exists.
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    def compute_loss(self, labels, logits):
        return tf.reduce_mean(tf.keras.losses.mse(labels, logits))

    def compute_metrics(self, labels, logits):
        return tf.keras.metrics.mae(labels, logits)  #

    @tf.function
    def train_step(self, x, y):
        # Record the operations used to compute the loss, so that the gradient
        # of the loss with respect to the variables can be computed.
        # metrics = 0
        with tf.GradientTape() as tape:
            logits = self.model([x[0],
                                 x[1],
                                 x[2],
                                 x[3],
                                 x[4],
                                 x[5],
                                 x[6]], training=True)
            loss = self.ComputeLoss(y, logits)
            # loss = self.compute_loss(labels, logits)
            self.ComputeMetrics(y, logits)
            # metrics = self.compute_metrics(labels, logits)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, logits

    def training(self, features, targets_values, epochs=5, log_freq=50):

        for epoch_i in range(epochs):
            # 将数据集分成训练集和测试集，随机种子不固定
            train_X, test_X, train_y, test_y = train_test_split(features,
                                                                targets_values,
                                                                test_size=0.2,
                                                                random_state=0)

            train_batches = get_batches(train_X, train_y, self.batch_size)
            batch_num = (len(train_X) // self.batch_size)

            train_start = time.time()
            # with self.train_summary_writer.as_default():
            if True:
                start = time.time()
                # Metrics are stateful. They accumulate values and return a cumulative
                # result when you call .result(). Clear accumulated values with .reset_states()
                avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
                # avg_mae = tf.keras.metrics.Mean('mae', dtype=tf.float32)

                # Datasets can be iterated over like any other Python iterable.
                for batch_i in range(batch_num):
                    x, y = next(train_batches)
                    categories = np.zeros([self.batch_size, 18])
                    for i in range(self.batch_size):
                        categories[i] = x.take(6, 1)[i]

                    titles = np.zeros([self.batch_size, sentences_size])
                    for i in range(self.batch_size):
                        titles[i] = x.take(5, 1)[i]

                    loss, logits = self.train_step([np.reshape(x.take(0, 1), [self.batch_size, 1]).astype(np.float32),
                                                    np.reshape(x.take(2, 1), [self.batch_size, 1]).astype(np.float32),
                                                    np.reshape(x.take(3, 1), [self.batch_size, 1]).astype(np.float32),
                                                    np.reshape(x.take(4, 1), [self.batch_size, 1]).astype(np.float32),
                                                    np.reshape(x.take(1, 1), [self.batch_size, 1]).astype(np.float32),
                                                    categories.astype(np.float32),
                                                    titles.astype(np.float32)],
                                                   np.reshape(y, [self.batch_size, 1]).astype(np.float32))
                    avg_loss(loss)
                    # avg_mae(metrics)
                    self.losses['train'].append(loss)

                    if tf.equal(self.optimizer.iterations % log_freq, 0):
                        # summary_ops_v2.scalar('loss', avg_loss.result(), step=self.optimizer.iterations)
                        # summary_ops_v2.scalar('mae', self.ComputeMetrics.result(), step=self.optimizer.iterations)
                        # summary_ops_v2.scalar('mae', avg_mae.result(), step=self.optimizer.iterations)

                        rate = log_freq / (time.time() - start)
                        print('Step #{}\tEpoch {:>3} Batch {:>4}/{}   Loss: {:0.6f} mae: {:0.6f} ({} steps/sec)'.format(
                            self.optimizer.iterations.numpy(),
                            epoch_i,
                            batch_i,
                            batch_num,
                            loss, (self.ComputeMetrics.result_cate()), rate))
                        # print('Step #{}\tLoss: {:0.6f} mae: {:0.6f} ({} steps/sec)'.format(
                        # self.optimizer.iterations.numpy(), loss, (avg_mae.result()), rate))
                        avg_loss.reset_states()
                        self.ComputeMetrics.reset_states()
                        # avg_mae.reset_states()
                        start = time.time()

            train_end = time.time()
            print(
                '\nTrain time for epoch #{} ({} total steps): {}'.format(epoch_i + 1, self.optimizer.iterations.numpy(),
                                                                         train_end - train_start))
            # with self.test_summary_writer.as_default():
            self.testing((test_X, test_y), self.optimizer.iterations)
            # self.checkpoint.save(self.checkpoint_prefix)
        self.export_path = os.path.join(MODEL_DIR, 'export')
        tf.saved_model.save(self.model, self.export_path)

    def testing(self, test_dataset, step_num):
        test_X, test_y = test_dataset
        test_batches = get_batches(test_X, test_y, self.batch_size)

        """Perform an evaluation of `model` on the examples from `dataset`."""
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        # avg_mae = tf.keras.metrics.Mean('mae', dtype=tf.float32)

        batch_num = (len(test_X) // self.batch_size)
        for batch_i in range(batch_num):
            x, y = next(test_batches)
            categories = np.zeros([self.batch_size, 18])
            for i in range(self.batch_size):
                categories[i] = x.take(6, 1)[i]

            titles = np.zeros([self.batch_size, sentences_size])
            for i in range(self.batch_size):
                titles[i] = x.take(5, 1)[i]

            logits = self.model([np.reshape(x.take(0, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(2, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(3, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(4, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(1, 1), [self.batch_size, 1]).astype(np.float32),
                                 categories.astype(np.float32),
                                 titles.astype(np.float32)], training=False)
            test_loss = self.ComputeLoss(np.reshape(y, [self.batch_size, 1]).astype(np.float32), logits)
            avg_loss(test_loss)
            # 保存测试损失
            self.losses['test'].append(test_loss)
            self.ComputeMetrics(np.reshape(y, [self.batch_size, 1]).astype(np.float32), logits)
            # avg_loss(self.compute_loss(labels, logits))
            # avg_mae(self.compute_metrics(labels, logits))

        print('Model test set loss: {:0.6f} mae: {:0.6f}'.format(avg_loss.result_cate(), self.ComputeMetrics.result_cate()))
        # print('Model test set loss: {:0.6f} mae: {:0.6f}'.format(avg_loss.result(), avg_mae.result()))
        #         summary_ops_v2.scalar('loss', avg_loss.result(), step=step_num)
        #         summary_ops_v2.scalar('mae', self.ComputeMetrics.result(), step=step_num)
        # summary_ops_v2.scalar('mae', avg_mae.result(), step=step_num)

        if avg_loss.result_cate() < self.best_loss:
            self.best_loss = avg_loss.result_cate()
            print("best loss = {}".format(self.best_loss))
            self.checkpoint.save(self.checkpoint_prefix)

    def forward(self, xs):
        predictions = self.model(xs)
        # logits = tf.nn.softmax(predictions)

        return predictions


def save_params(params):
    pickle.dump(params, open('params.p', mode='wb'))


def load_params():
    return pickle.load(open('params.p', mode='rb'))


def get_inputs():
    uid = tf.keras.layers.Input(shape=(1,), dtype='int32', name='uid')
    user_gender = tf.keras.layers.Input(shape=(1,), dtype='int32', name='user_gender')
    user_age = tf.keras.layers.Input(shape=(1,), dtype='int32', name='user_age')
    user_job = tf.keras.layers.Input(shape=(1,), dtype='int32', name='user_job')

    movie_id = tf.keras.layers.Input(shape=(1,), dtype='int32', name='movie_id')
    movie_categories = tf.keras.layers.Input(shape=(18,), dtype='int32', name='movie_categories')
    movie_titles = tf.keras.layers.Input(shape=(15,), dtype='int32', name='movie_titles')
    return uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles


def get_user_embedding(uid, user_gender, user_age, user_job):
    uid_embed_layer = tf.keras.layers.Embedding(uid_max, embed_dim, input_length=1, name='uid_embed_layer')(uid)
    gender_embed_layer = tf.keras.layers.Embedding(gender_max, embed_dim // 2, input_length=1,
                                                   name='gender_embed_layer')(user_gender)
    age_embed_layer = tf.keras.layers.Embedding(age_max, embed_dim // 2, input_length=1, name='age_embed_layer')(
        user_age)
    job_embed_layer = tf.keras.layers.Embedding(job_max, embed_dim // 2, input_length=1, name='job_embed_layer')(
        user_job)
    return uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer


def get_user_feature_layer(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer):
    # 第一层全连接
    uid_fc_layer = tf.keras.layers.Dense(embed_dim, name="uid_fc_layer", activation='relu')(uid_embed_layer)
    gender_fc_layer = tf.keras.layers.Dense(embed_dim, name="gender_fc_layer", activation='relu')(gender_embed_layer)
    age_fc_layer = tf.keras.layers.Dense(embed_dim, name="age_fc_layer", activation='relu')(age_embed_layer)
    job_fc_layer = tf.keras.layers.Dense(embed_dim, name="job_fc_layer", activation='relu')(job_embed_layer)

    # 第二层全连接
    user_combine_layer = tf.keras.layers.concatenate([uid_fc_layer, gender_fc_layer, age_fc_layer, job_fc_layer],
                                                     2)  # (?, 1, 128)
    user_combine_layer = tf.keras.layers.Dense(200, activation='tanh')(user_combine_layer)  # (?, 1, 200)

    user_combine_layer_flat = tf.keras.layers.Reshape([200], name="user_combine_layer_flat")(user_combine_layer)
    return user_combine_layer, user_combine_layer_flat


def get_movie_id_embed_layer(movie_id):
    movie_id_embed_layer = tf.keras.layers.Embedding(movie_id_max, embed_dim, input_length=1,
                                                     name='movie_id_embed_layer')(movie_id)
    return movie_id_embed_layer


def get_movie_categories_layers(movie_categories):
    movie_categories_embed_layer = tf.keras.layers.Embedding(movie_categories_max, embed_dim, input_length=18,
                                                             name='movie_categories_embed_layer')(movie_categories)
    movie_categories_embed_layer = tf.keras.layers.Lambda(lambda layer: tf.reduce_sum(layer, axis=1, keepdims=True))(
        movie_categories_embed_layer)
    # movie_categories_embed_layer = tf.keras.layers.Reshape([1, 18 * embed_dim])(movie_categories_embed_layer)

    return movie_categories_embed_layer


def get_movie_cnn_layer(movie_titles):
    # 从嵌入矩阵中得到电影名对应的各个单词的嵌入向量
    movie_title_embed_layer = tf.keras.layers.Embedding(movie_title_max, embed_dim, input_length=15,
                                                        name='movie_title_embed_layer')(movie_titles)
    sp = movie_title_embed_layer.shape
    movie_title_embed_layer_expand = tf.keras.layers.Reshape([sp[1], sp[2], 1])(movie_title_embed_layer)
    # 对文本嵌入层使用不同尺寸的卷积核做卷积和最大池化
    pool_layer_lst = []
    for window_size in window_sizes:
        conv_layer = tf.keras.layers.Conv2D(filter_num, (window_size, embed_dim), 1, activation='relu')(
            movie_title_embed_layer_expand)
        maxpool_layer = tf.keras.layers.MaxPooling2D(pool_size=(sentences_size - window_size + 1, 1), strides=1)(
            conv_layer)
        pool_layer_lst.append(maxpool_layer)
    # Dropout层
    pool_layer = tf.keras.layers.concatenate(pool_layer_lst, 3, name="pool_layer")
    max_num = len(window_sizes) * filter_num
    pool_layer_flat = tf.keras.layers.Reshape([1, max_num], name="pool_layer_flat")(pool_layer)

    dropout_layer = tf.keras.layers.Dropout(dropout_keep, name="dropout_layer")(pool_layer_flat)
    return pool_layer_flat, dropout_layer


def get_movie_feature_layer(movie_id_embed_layer, movie_categories_embed_layer, dropout_layer):
    # 第一层全连接
    movie_id_fc_layer = tf.keras.layers.Dense(embed_dim, name="movie_id_fc_layer", activation='relu')(
        movie_id_embed_layer)
    movie_categories_fc_layer = tf.keras.layers.Dense(embed_dim, name="movie_categories_fc_layer", activation='relu')(
        movie_categories_embed_layer)

    # 第二层全连接
    movie_combine_layer = tf.keras.layers.concatenate([movie_id_fc_layer, movie_categories_fc_layer, dropout_layer], 2)
    movie_combine_layer = tf.keras.layers.Dense(200, activation='tanh')(movie_combine_layer)

    movie_combine_layer_flat = tf.keras.layers.Reshape([200], name="movie_combine_layer_flat")(movie_combine_layer)
    return movie_combine_layer, movie_combine_layer_flat


def get_batches(Xs, ys, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]


def rating_movie(mv_net, user_id_val, movie_id_val):
    categories = np.zeros([1, 18])
    categories[0] = movies.values[movieid2idx[movie_id_val]][2]

    titles = np.zeros([1, sentences_size])
    titles[0] = movies.values[movieid2idx[movie_id_val]][1]

    inference_val = mv_net.model_cate([np.reshape(users.values[user_id_val - 1][0], [1, 1]),
                                       np.reshape(users.values[user_id_val - 1][1], [1, 1]),
                                       np.reshape(users.values[user_id_val - 1][2], [1, 1]),
                                       np.reshape(users.values[user_id_val - 1][3], [1, 1]),
                                       np.reshape(movies.values[movieid2idx[movie_id_val]][0], [1, 1]),
                                       categories,
                                       titles])

    return (inference_val.numpy())


def recommend_same_type_movie(movie_id_val, top_k=20):
    norm_movie_matrics = tf.sqrt(tf.reduce_sum(tf.square(movie_matrics), 1, keepdims=True))
    normalized_movie_matrics = movie_matrics / norm_movie_matrics

    # 推荐同类型的电影
    probs_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
    probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_movie_matrics))
    sim = (probs_similarity.numpy())
    #     results = (-sim[0]).argsort()[0:top_k]
    #     print(results)

    print("您看的电影是：{}".format(movies_orig[movieid2idx[movie_id_val]]))
    print("以下是给您的推荐：")
    p = np.squeeze(sim)
    p[np.argsort(p)[:-top_k]] = 0
    p = p / np.sum(p)
    results = set()
    while len(results) != 5:
        c = np.random.choice(3883, 1, p=p)[0]
        results.add(c)
    for val in (results):
        print(val)
        print(movies_orig[val])

    return results


def recommend_other_favorite_movie(movie_id_val, top_k=20):
    probs_movie_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
    probs_user_favorite_similarity = tf.matmul(probs_movie_embeddings, tf.transpose(users_matrics))
    favorite_user_id = np.argsort(probs_user_favorite_similarity.numpy())[0][-top_k:]
    #     print(normalized_users_matrics.numpy().shape)
    #     print(probs_user_favorite_similarity.numpy()[0][favorite_user_id])
    #     print(favorite_user_id.shape)

    print("您看的电影是：{}".format(movies_orig[movieid2idx[movie_id_val]]))

    print("喜欢看这个电影的人是：{}".format(users_orig[favorite_user_id - 1]))
    probs_users_embeddings = (users_matrics[favorite_user_id - 1]).reshape([-1, 200])
    probs_similarity = tf.matmul(probs_users_embeddings, tf.transpose(movie_matrics))
    sim = (probs_similarity.numpy())
    #     results = (-sim[0]).argsort()[0:top_k]
    #     print(results)

    #     print(sim.shape)
    #     print(np.argmax(sim, 1))
    p = np.argmax(sim, 1)
    print("喜欢看这个电影的人还喜欢看：")

    if len(set(p)) < 5:
        results = set(p)
    else:
        results = set()
        while len(results) != 5:
            c = p[random.randrange(top_k)]
            results.add(c)
    for val in (results):
        print(val)
        print(movies_orig[val])

    return results


def recommend_your_favorite_movie(user_id_val, top_k=10):
    # 推荐您喜欢的电影
    probs_embeddings = (users_matrics[user_id_val - 1]).reshape([1, 200])

    probs_similarity = tf.matmul(probs_embeddings, tf.transpose(movie_matrics))
    sim = (probs_similarity.numpy())
    #     print(sim.shape)
    #     results = (-sim[0]).argsort()[0:top_k]
    #     print(results)

    #     sim_norm = probs_norm_similarity.eval()
    #     print((-sim_norm[0]).argsort()[0:top_k])

    print("以下是给您的推荐：")
    p = np.squeeze(sim)
    p[np.argsort(p)[:-top_k]] = 0
    p = p / np.sum(p)
    results = set()
    while len(results) != 5:
        c = np.random.choice(3883, 1, p=p)[0]
        results.add(c)
    for val in (results):
        print(val)
        print(movies_orig[val])

    return results


if __name__ == "__main__":
    mv_net = mv_network()
    mv_net.training(features=features, targets_values=targets_values, epochs=5)

    import matplotlib.pyplot as plt

    # 显示训练Loss
    plt.plot(mv_net.losses['train'], label='Training loss')
    plt.legend()
    _ = plt.ylim()

    # 显示测试Loss
    plt.plot(mv_net.losses['test'], label='Test loss')
    plt.legend()
    _ = plt.ylim()

    ###############生成Movie特征矩阵
    movie_layer_model = keras.models.Model(inputs=[mv_net.model.input[4], mv_net.model.input[5], mv_net.model.input[6]],
                                           outputs=mv_net.model.get_layer("movie_combine_layer_flat").output)
    movie_matrics = []

    for item in movies.values:
        categories = np.zeros([1, 18])
        categories[0] = item.take(2)

        titles = np.zeros([1, sentences_size])
        titles[0] = item.take(1)

        movie_combine_layer_flat_val = movie_layer_model([np.reshape(item.take(0), [1, 1]), categories, titles])
        movie_matrics.append(movie_combine_layer_flat_val)

    pickle.dump((np.array(movie_matrics).reshape(-1, 200)), open('movie_matrics.p', 'wb'))
    movie_matrics = pickle.load(open('movie_matrics.p', mode='rb'))
    ###############生成Movie特征矩阵

    ###############生成User特征矩阵
    user_layer_model = keras.models.Model(
        inputs=[mv_net.model.input[0], mv_net.model.input[1], mv_net.model.input[2], mv_net.model.input[3]],
        outputs=mv_net.model.get_layer("user_combine_layer_flat").output)
    users_matrics = []

    for item in users.values:
        user_combine_layer_flat_val = user_layer_model([np.reshape(item.take(0), [1, 1]),
                                                        np.reshape(item.take(1), [1, 1]),
                                                        np.reshape(item.take(2), [1, 1]),
                                                        np.reshape(item.take(3), [1, 1])])
        users_matrics.append(user_combine_layer_flat_val)

    pickle.dump((np.array(users_matrics).reshape(-1, 200)), open('users_matrics.p', 'wb'))
    users_matrics = pickle.load(open('users_matrics.p', mode='rb'))

    ###############生成User特征矩阵

    # 指定用户和电影进行评分
    rating_movie(mv_net, 234, 1401)
    # 推荐同类型的电影
    recommend_same_type_movie(1401, 20)
    # 推荐您喜欢的电影
    recommend_your_favorite_movie(234, 10)
    # 看过这个电影的人还看了（喜欢）哪些电影
    recommend_other_favorite_movie(1401, 20)

    # Train time for epoch #5 (15625 total steps): 41.225642919540405
    # Model test set loss: 0.800273 mae: 0.705819
    # best loss = 0.8002728819847107


    # 您看的电影是：[1401 'Ghosts of Mississippi (1996)' 'Drama']
    # 以下是给您的推荐：
    # 3333
    # [3402 'Turtle Diary (1985)' 'Drama']
    # 2891
    # [2960 'Beefcake (1999)' 'Drama']
    # 1771
    # [1840 'He Got Game (1998)' 'Drama']
    # 1484
    # [1519 'Broken English (1996)' 'Drama']
    # 182
    # [184 'Nadja (1994)' 'Drama']
    # 以下是给您的推荐：
    # 847
    # [858 'Godfather, The (1972)' 'Action|Crime|Drama']
    # 2128
    # [2197 'Firelight (1997)' 'Drama']
    # 1242
    # [1262 'Great Escape, The (1963)' 'Adventure|War']
    # 315
    # [318 'Shawshank Redemption, The (1994)' 'Drama']
    # 1950
    # [2019
    #  'Seven Samurai (The Magnificent Seven) (Shichinin no samurai) (1954)'
    #  'Action|Drama']
    # 您看的电影是：[1401 'Ghosts of Mississippi (1996)' 'Drama']
    # 喜欢看这个电影的人是：[[760 'M' 56 15]
    #  [214 'M' 18 20]
    #  [3323 'M' 35 14]
    #  [1985 'M' 45 12]
    #  [4200 'M' 45 7]
    #  [566 'M' 25 17]
    #  [1669 'F' 25 17]
    #  [100 'M' 35 17]
    #  [5767 'M' 25 2]
    #  [371 'M' 18 4]
    #  [4085 'F' 25 6]
    #  [2693 'M' 56 13]
    #  [4800 'M' 18 4]
    #  [5861 'F' 50 1]
    #  [1855 'M' 18 4]
    #  [3901 'M' 18 14]
    #  [446 'F' 50 0]
    #  [2338 'M' 45 17]
    #  [282 'M' 25 17]
    #  [2154 'M' 25 12]]
    # 喜欢看这个电影的人还喜欢看：
    # 2434
    # [2503 'Apple, The (Sib) (1998)' 'Drama']
    # 2693
    # [2762 'Sixth Sense, The (1999)' 'Thriller']
    # 2836
    # [2905 'Sanjuro (1962)' 'Action|Adventure']
    # 315
    # [318 'Shawshank Redemption, The (1994)' 'Drama']
    # 1023
    # [1036 'Die Hard (1988)' 'Action|Thriller']