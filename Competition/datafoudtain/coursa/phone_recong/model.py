# 导入一些必要的数据包
import os

import numpy as np
import pandas as pd
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.backend as K
import warnings
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# 本次模型主要基于EfficientNet，需要pip install efficientnet
from efficientnet.tfkeras import EfficientNetB4

from warnings import simplefilter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

os.environ.setdefault('TF_KERAS', '1')
simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# 定义题目需要的评价函数
def score(y_true, y_pred):
    # 自定义的f1 socre
    return 0.4 * f1_score(y_true, y_pred, pos_label=1) + 0.6 * f1_score(y_true, y_pred, pos_label=0)


# 构建五折交叉验证
def get_s(model):
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    oof_train = np.zeros((len(data), nb_classes))
    oof_test = np.zeros((len(test), nb_classes))
    for index, (tra_index, val_index) in enumerate(skf.split(data['id'].values, data['label'].values)):
        K.clear_session()
        print('========== {} =========='.format(index))
        train = pd.DataFrame({'id': data.iloc[tra_index]['id'].values,
                              'label': data.iloc[tra_index]['label'].values})
        print(train.head())
        train['label'] = train['label'].astype(str)

        valid = pd.DataFrame({'id': data.iloc[val_index]['id'].values,
                              'label': data.iloc[val_index]['label'].values})

        valid['label'] = valid['label'].astype(str)

        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train,
            directory=r".\data\train_total",
            x_col="id",
            y_col="label",
            batch_size=batch_size // 4,
            shuffle=True,
            class_mode="categorical",
            target_size=(img_size, img_size),
            save_format='JPEG',
            verbose=False
        )
        valid_generator = test_datagen.flow_from_dataframe(
            dataframe=valid,
            directory=r".\data\train_total",
            x_col="id",
            y_col="label",
            batch_size=1,
            shuffle=False,
            class_mode="categorical",
            verbose=False,
            target_size=(img_size, img_size),
            save_format='JPEG'
        )

        model_final = get_model()
        if index == 0: model_final.summary()
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, min_lr=0.0001, verbose=True)
        earlystopping = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=3, verbose=True, mode='max')
        checkpoint = ModelCheckpoint("load_{}_{}.h5".format(model, index), monitor='val_acc', verbose=False,
                                     save_best_only=True, save_weights_only=True, mode='max')
        model_final.fit_generator(generator=train_generator,
                                  steps_per_epoch=train_generator.n // train_generator.batch_size,
                                  validation_data=valid_generator,
                                  validation_steps=valid_generator.n // valid_generator.batch_size,
                                  epochs=nb_epochs,
                                  callbacks=[checkpoint, earlystopping, reduce_lr],
                                  verbose=True)
        test_generator = test_datagen.flow_from_dataframe(
            dataframe=test,
            directory=r".\data\test_images_a",
            x_col="id",
            target_size=(img_size, img_size),
            batch_size=1,
            # save_format='JPEG',
            shuffle=False,
            class_mode=None
        )
        print('predict', index)
        model_final.load_weights("load_{}_{}.h5".format(model, index))

        oof_test += model_final.predict_generator(test_generator) / skf.n_splits

        predict = model_final.predict_generator(valid_generator)
        oof_train[val_index] = predict

    return oof_train, oof_test, train_generator


# 定义模型
def get_model():
    models = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3),
                            backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
    # models.load_weights(path)
    x = models.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model_final = Model(inputs=models.input, outputs=predictions)
    model_final.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy',
                        metrics=['acc'])
    return model_final


def invert_dict(d):
    return {v: k for k, v in d.items()}


def new_index(x):
    p = []
    for i in x:
        p.append(int(map_index[i]))
    return p


nb_classes = 2  # 二分类
batch_size = 128  # 训练时的batch size大小
img_size = 224  # 图片的尺寸
nb_epochs = 2  # 训练的次数
# 文件的路径
train_path = r'E:\DataSet\DataFountain\学习赛\工业安全生产环境违规使用手机的识别\train'
test_path = r'E:\DataSet\DataFountain\学习赛\工业安全生产环境违规使用手机的识别\test_images_a'

# 利用了keras的内置的图片处理方法，对训练数据进行增强包括：旋转角度，水平平移等
# 测试数据只进行归一化，不进行增强，其实可以利用tts方法，增强测试数据，然后对同一组测试数据的标签进行投票
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # 归一化
    rotation_range=45,  # 旋转角度
    width_shift_range=0.1,  # 水平偏移
    height_shift_range=0.1,  # 垂直偏移
    shear_range=0.1,  # 随机错切变换的角度
    zoom_range=0.25,  # 随机缩放的范围
    horizontal_flip=True,  # 随机将一半图像水平翻转
    fill_mode='nearest'
)  # 填充像素的方

test_datagen = ImageDataGenerator(
    rescale=1. / 255
)

if __name__ == "__main__":
    # 对数据进行简单处理_
    # 这里创建一个文件夹，将所有训练数据图片放在一个文件夹，也可以代码实现。
    train_0 = pd.DataFrame()
    train_0_name = [x for x in os.listdir(train_path + os.sep + '0_phone/JPEGImages/') if x not in ['Thumbs.db']]
    train_0['id'] = train_0_name
    train_0['label'] = 0

    train_1 = pd.DataFrame()
    train_1_name = [x for x in os.listdir(train_path + os.sep + '1_no_phone/') if x not in ['Thumbs.db']]
    train_1['id'] = train_1_name
    train_1['label'] = 1

    train = pd.concat([train_0, train_1]).sample(frac=1).reset_index(drop=True)

    # del train['Thumbs.db']
    # 构建提交数据集的样例
    file_name = [x for x in os.listdir(test_path) if x not in ['Thumbs.db', '.ipynb_checkpoints']]
    test = pd.DataFrame()
    test['id'] = file_name
    test['label'] = -1
    print(len(file_name))
    print(test.sort_values('id'))
    data = train.copy()
    print(data.groupby(['label']).size())

    print(test.shape, data.shape)
    oof_train = np.zeros((len(data), nb_classes))
    oof_test = np.zeros((len(test), nb_classes))

    print(oof_train.shape)
    print(oof_test.shape)

    oof_train4, oof_test4, train_generator = get_s('EfficientNetB4')
    # 保存结果
    oof_train = oof_train4
    oof_test = oof_test4

    map_index = train_generator.class_indices
    print(map_index)
    data['label'] = data['label'].astype(str)
    data['label'] = data['label'].map(map_index)
    print(data['label'].unique())

    map_index = invert_dict(map_index)
    print(map_index)

    print(oof_train)
    base_score = score(data['label'].values, np.argmax(oof_train, axis=1))
    print(base_score)
    predicted_class_indices = new_index(np.argmax(oof_test, axis=1))
    test['label'] = list(predicted_class_indices)
    test.columns = ['image_name', 'class_id']
    test.to_csv('./submit_{}.csv'.format(str(base_score).replace('.', '_')), index=False)
