#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng 
@license: Apache Licence 
@file: pyspark_config.py 
@time: 2022/02/07
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""
import os
import sys

# spark配置信息
from pyspark import SparkConf
# 测试存储的模型
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession
# 调整默认数据类型
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, FloatType

# 特征编码配置
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

os.environ['JAVA_HOME'] = '/usr/lib/jvm/java'
os.environ['SPARK_HOME'] = '/opt/spark/spark-standalone'
os.environ['PYSPARK_PYTHON'] = '/home/hadoop/anaconda3/envs/bigdata/bin/python3.9'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/home/hadoop/anaconda3/envs/bigdata/bin/python3.9'
sys.path.append('/opt/spark/spark-standalone/python')
sys.path.append('/opt/spark/spark-standalone/python/lib/py4j-0.10.9.2-src.zip')
sys.path.append('/home/hadoop/anaconda3/envs/bigdata/bin/python3.9')

SPARK_APP_NAME = "ALSRecommend"
SPARK_URL = "spark://192.168.137.128:7077"

conf = SparkConf()
config = (
    ("spark.app.name", SPARK_APP_NAME),
    ("spark.executor.memory", "512M"),
    ("spark.master", SPARK_URL),
    ("spark.executor.cores", "1"),
    ("spark.dynamicAllocation.enabled", True),
    ("spark.dynamicAllocation.initialExecutors", "1"),
    ("spark.shuffle.service.enabled", True),
    ("spark.sql.pivotMaxValues", "99999"),
    ("spark.reducer.maxReqsInFlight", "10"),
    ("spark.reducer.maxBlocksInFlightPerAddress", "10"),
    ("spark.network.timeout", "3000")
)
conf.setAll(config)

spark = SparkSession.builder.config(conf=conf).getOrCreate()
# spark ml是基于内存的训练，如果数据量过大，空间过小，迭代次数多的话，很容易造成内存溢出，报错，设置checkPoint是将数据落盘，就算中途出问题，也可以基于上次的训练节点继续训练
# 但针对大量数据的训练，还是只能增大内存，减少数据规模和迭代次数
spark.sparkContext.setCheckpointDir("hdfs://master:9000/user/hadoop/user/icss/RecommendSystem/checkPoint/")
spark.sparkContext.addPyFile("/opt/spark/spark-standalone/python/lib/redis.zip")

# 调整数据类型
schema = StructType([
    StructField("userId", IntegerType()),
    StructField("timestamp", LongType()),
    StructField("btag", StringType()),
    StructField("cateId", IntegerType()),
    StructField("brandId", IntegerType())
])
# 从hdfs加载csv文件为DataFrame
# behavior_log_df = spark.read.csv("hdfs://master:9000/user/hadoop/user/icss/RecommendSystem/dataset/behavior_log_sample.csv")
behavior_log_df = spark.read.csv(
    "hdfs://master:9000/user/hadoop/user/icss/RecommendSystem/dataset/behavior_log_sample.csv", header=True,
    schema=schema)

# 查看数据，默认显示前20条
print(behavior_log_df.show())
print(behavior_log_df.count())
# 打印数据类型
print(behavior_log_df.printSchema())
# root
#  |-- _c0: string (nullable = true)
#  |-- _c1: string (nullable = true)
#  |-- _c2: string (nullable = true)
#  |-- _c3: string (nullable = true)
#  |-- _c4: string (nullable = true)

# print("查看userId的数据情况：", behavior_log_df.groupby("userId").count().collect())
# print("查看btag的数据情况：", behavior_log_df.groupby("btag").count().collect())
# print("查看cateId的数据情况：", behavior_log_df.groupby("cateId").count().collect())
# print("查看brandId的数据情况：", behavior_log_df.groupby("brandId").count().collect())
# print("判断数据是否有空值：", behavior_log_df.count(), behavior_log_df.dropna().count())


# 透视操作
# cate_count_df = behavior_log_df.groupby(behavior_log_df.userId, behavior_log_df.cateId).pivot("btag", ["pv", "fav", "cart", "buy"]).count()
# brand_count_df = behavior_log_df.groupby(behavior_log_df.userId, behavior_log_df.brandId).pivot("btag", ["pv", "fav", "cart", "buy"]).count()
# cate_count_df.write.csv(path="hdfs://master:9000/user/hadoop/user/icss/RecommendSystem/preprocessing_dataset/cate_count.csv", header=True)
# brand_count_df.write.csv(path="hdfs://master:9000/user/hadoop/user/icss/RecommendSystem/preprocessing_dataset/brand_count.csv", header=True)
# 从hdfs加载csv文件
schema_cate = StructType([
    StructField("userId", IntegerType()),
    StructField("cateId", IntegerType()),
    StructField("pv", IntegerType()),
    StructField("fav", IntegerType()),
    StructField("cart", IntegerType()),
    StructField("buy", IntegerType())
])
cate_count_df = spark.read.csv(
    "hdfs://master:9000/user/hadoop/user/icss/RecommendSystem/preprocessing_dataset/cate_count.csv", header=True,
    schema=schema_cate)
brand_count_df = spark.read.csv(
    "hdfs://master:9000/user/hadoop/user/icss/RecommendSystem/preprocessing_dataset/brand_count.csv", header=True,
    schema=schema_cate)
brand_count_df.show(5)
cate_count_df.show(5)


def process_row(r):
    """
    打分规则
    pv: if m <= 20: scores=0.2*m; else score=4
    fav: if m <= 20: scores=0.4*m; else score=8
    cart: if m <= 20: scores=0.4*m; else score=8
    buy: if m <= 20: scores=1*m; else score=20
    Args:
        r:

    Returns:

    """
    pv_count = r.pv if r.pv else 0.0
    fav_count = r.fav if r.fav else 0.0
    cart_count = r.cart if r.cart else 0.0
    buy_count = r.buy if r.buy else 0.0

    # 打分规则
    pv_score = 0.2 * pv_count if pv_count <= 20 else 4.0
    fav_score = 0.2 * fav_count if fav_count <= 20 else 8.0
    cart_score = 0.2 * cart_count if cart_count <= 20 else 12.0
    buy_score = 0.2 * buy_count if buy_count <= 20 else 20.0

    # 最终得分
    rating = pv_score + fav_score + cart_score + buy_score
    return r.userId, r.cateId, rating


# 用户对商品类别的打分数据
# map返回的结果是rdd类型，需要调用toDF方法转换为DataFrame
cate_rating_df = cate_count_df.rdd.map(process_row).toDF(['userId', 'cateId', 'rating'])
# 用户对品牌的打分数据
brand_rating_df = brand_count_df.rdd.map(process_row).toDF(['userId', 'brandId', 'rating'])
cate_count_df.show(5)
brand_rating_df.show(5)

# 建立模型
from pyspark.ml.recommendation import ALS

# 利用打分数据，训练ALS模型，checkPointInterval是每训练几步缓存一次
als_cate = ALS(userCol='userId', itemCol='cateId', ratingCol='rating', checkpointInterval=5)
als_brand = ALS(userCol='userId', itemCol='brandId', ratingCol='rating', checkpointInterval=5)
model_cate = als_cate.fit(cate_rating_df)
model_brand = als_brand.fit(brand_rating_df)

# model.recommendForAllUsers(N)给所有用户推荐TOP-N个物品
ret_cate = model_cate.recommendForAllUsers(3)
ret_brand = model_brand.recommendForAllUsers(3)
# 由于是给所有用户进行推荐，运算时间较长
ret_cate.show(truncate=False)  # 不截断，全部显示
ret_brand.show(truncate=False)  # 不截断，全部显示
# 推荐结果存放在recommendations列中
ret_cate.select("recommendations").show()
ret_cate.show(5, truncate=False)
ret_brand.select("recommendations").show()
ret_brand.show(5, truncate=False)

# 模型保存
# transform中提供userId和cateId可以对打分进行预测，利用打分结果排序后，同样可以实现TOP-N推荐


model_cate.transform
model_brand.transform
# 将模型存储
# model.save("hdfs://master:9000/user/hadoop/user/icss/RecommendSystem/models/userCateRatingALSModel.obj")
# model_brand.save("hdfs://master:9000/user/hadoop/user/icss/RecommendSystem/models/userBrandRatingALSModel.obj")

# 从hdfs加载之前存储的模型
als_model_cate = ALSModel.load(
    "hdfs://master:9000/user/hadoop/user/icss/RecommendSystem/models/userCateRatingALSModel.obj")
als_model_brand = ALSModel.load(
    "hdfs://master:9000/user/hadoop/user/icss/RecommendSystem/models/userBrandRatingALSModel.obj")
# model.recommendForAllUsers(N)给用户推荐TOP-N物品
result_cate = als_model_cate.recommendForAllUsers(3)
# result_cate.show()
result_brand = als_model_brand.recommendForAllUsers(3)
# result_brand.show()

# import redis
#
# host = "192.168.137.128"
# port = 6379
# # 召回到redis 存储
# def recall_cate_by_cf(partition):
#     # 建立redis连接池
#     pool = redis.ConnectionPool(host=host, port=port, db='0')
#     # 建立redis客户端
#     client = redis.Redis(connection_pool=pool)
#     # 键值对形式存储起来，键：用户id， 值：推荐的商品类别
#     for row in partition:
#         client.hset("recall_cate", row.userId, [i.cateId for i in row.recommendations])
#
#
# # 召回到redis 存储
# def recall_brand_by_cf(partition):
#     # 建立redis连接池
#     pool = redis.ConnectionPool(host=host, port=port, db='0')
#     # 建立redis客户端
#     client = redis.Redis(connection_pool=pool)
#     # 键值对形式存储起来，键：用户id， 值：推荐的商品类别
#     for row in partition:
#         client.hset("recall_brand", row.userId, [i.brandId for i in row.recommendations])
#
#
# result_cate.foreachPartition(recall_cate_by_cf)
# result_brand.foreachPartition(recall_brand_by_cf)
# # 总的条目数，查看redis中总的条目数是否一致
# result_cate.count()
# result_brand.count()


# raw_sample的数据分析和处理
df_raw = spark.read.csv("hdfs://master:9000/user/hadoop/user/icss/RecommendSystem/dataset/raw_sample_sample.csv",
                        header=True)
df_raw.show(5)
print("样本数据集总条目数：", df_raw.count())  # 1152053
print("用户user总数：", df_raw.groupby("user").count().count())  # 50000
print("广告id adgroup_id总数：", df_raw.groupby("adgroup_id").count().count())  # 250837
print("广告展示位pid情况：", df_raw.groupby(
    "pid").count().collect())  # [Row(pid='430548_1007', count=713621), Row(pid='430539_1007', count=438432)]
print("广告点击数据情况click：",
      df_raw.groupby("clk").count().collect())  # [Row(clk='0', count=1092268), Row(clk='1', count=59785)]
# 数据格式转化
raw_sample_df = df_raw.withColumn("user", df_raw.user.cast(IntegerType())).withColumnRenamed("user", "userId"). \
    withColumn("time_stamp", df_raw.time_stamp.cast(LongType())).withColumnRenamed("time_stamp", "timestamp"). \
    withColumn("adgroup_id", df_raw.adgroup_id.cast(IntegerType())).withColumnRenamed("adgroup_id", "adgroupId"). \
    withColumn("pid", df_raw.pid.cast(StringType())). \
    withColumn("nonclk", df_raw.nonclk.cast(IntegerType())). \
    withColumn("clk", df_raw.clk.cast(IntegerType()))
# 打印数据字段的格式
raw_sample_df.printSchema()

# 特征处理
# StringIndexer对指定字符进行特征处理
stringIndexer_pid = StringIndexer(inputCol="pid", outputCol="pid_feature")
# 对处理出来的特征列进行独热编码
encoder_pid = OneHotEncoder(dropLast=False, inputCol="pid_feature", outputCol="pid_value")
# 利用一个管道对每个数据进行独热编码处理
pipeline_pid = Pipeline(stages=[stringIndexer_pid, encoder_pid])
pipeline_model_pid = pipeline_pid.fit(raw_sample_df)
new_df_raw = pipeline_model_pid.transform(raw_sample_df)
new_df_raw.show(5)

# 查看最大时间
new_df_raw.sort("timestamp", ascending=False).show()

from datetime import datetime

test1 = 1494691162
datetime.fromtimestamp(test1)  # 待定
print("该时间之前的数据为训练样本，改该时间之后的数据为测试样本： ", datetime.fromtimestamp(test1 - 24 * 60 * 60))

# 划分数据集
train_sample = raw_sample_df.filter(raw_sample_df.timestamp <= (test1 - 24 * 60 * 60))
print("训练样本个数：")
print(train_sample.count())

test_sample = raw_sample_df.filter(raw_sample_df.timestamp > (test1 - 24 * 60 * 60))
print("测试样本个数：")
print(test_sample.count())  # 1152053

# 从hdfs加载广告基本信息
df_ad = spark.read.csv("hdfs://master:9000/user/hadoop/user/icss/RecommendSystem/dataset/ad_feature.csv", header=True)
# 必须先把NULL类型处理掉，不然无法设置schema
df_ad = df_ad.replace("NULL", "-1")

ad_feature_df = df_ad \
    .withColumn("adgroup_id", df_ad.adgroup_id.cast(IntegerType())).withColumnRenamed("adgroup_id", "adgroupId"). \
    withColumn("cate_id", df_ad.cate_id.cast(IntegerType())).withColumnRenamed("cate_id", "cateId"). \
    withColumn("campaign_id", df_ad.campaign_id.cast(IntegerType())).withColumnRenamed("campaign_id", "campaignId"). \
    withColumn("customer", df_ad.customer.cast(IntegerType())).withColumnRenamed("customer", "customerId"). \
    withColumn("brand", df_ad.brand.cast(IntegerType())).withColumnRenamed("brand", "brandId"). \
    withColumn("price", df_ad.price.cast(FloatType()))
ad_feature_df.show(5)

# 广告特征分析
print("总广告数：", df_ad.count())  # 846811
print("cateId数值个数", ad_feature_df.groupby("cateId").count().count())  # 6769
print("campaignId数值个数", ad_feature_df.groupby("campaignId").count().count())  # 423436
print("customerId数值个数", ad_feature_df.groupby("customerId").count().count())  # 255875
print("brandId数值个数", ad_feature_df.groupby("brandId").count().count())  # 99815

# 价格分析
print("价格高于1w的条目个数", ad_feature_df.select("price").filter("price >  10000").count())  # 6527
print("价格低于1的条目个数", ad_feature_df.select("price").filter("price < 1").count())  # 5762

# 从hdfs加载广告基本信息
schema_user = StructType([
    StructField("userId", IntegerType()),
    StructField("cms_segid", IntegerType()),
    StructField("cms_group_id", IntegerType()),
    StructField("final_gender_code", IntegerType()),
    StructField("age_level", IntegerType()),
    StructField("pvalue_level", IntegerType()),
    StructField("shopping_level", IntegerType()),
    StructField("occupation", IntegerType()),
    StructField("new_user_class_level", IntegerType())
])
df_user = spark.read.csv("hdfs://master:9000/user/hadoop/user/icss/RecommendSystem/dataset/user_profile.csv",
                         header=True, schema=schema_user)
df_user.printSchema()
df_user.show(5)
print("用户特征值个数情况")
print("cms_segid: ", df_user.groupby("cms_segid").count().count())  # 97
print("cms_group_id: ", df_user.groupby("cms_group_id").count().count())  # 13
print("final_gender_code: ", df_user.groupby("final_gender_code").count().count())  # 2
print("age_level: ", df_user.groupby("age_level").count().count())  # 7
print("shopping_level: ", df_user.groupby("shopping_level").count().count())  # 3
print("occupation: ", df_user.groupby("occupation").count().count())  # 2

# 用户特征确实情况展示
df_user.groupby("pvalue_level").count().show()
# +------------+------+
# |pvalue_level| count|
# +------------+------+
# |           1|154436|
# |           3| 37759|
# |           2|293656|
# |        null|575917|
# +------------+------+

df_user.groupby("new_user_class_level").count().show()

# +--------------------+------+
# |new_user_class_level| count|
# +--------------------+------+
# |                   1| 80548|
# |                   3|173047|
# |                   4|138833|
# |                   2|324420|
# |                null|344920|
# +--------------------+------+


# 数据缺失情况查看
t_count = df_user.count()
pl_na_count = t_count - df_user.dropna(subset=['pvalue_level']).count()
print("pvalue_level的空值情况：", pl_na_count, "空值占比：%0.2f%%" % (pl_na_count / t_count * 100))  # 54.24%

nul_na_count = t_count - df_user.dropna(subset=['new_user_class_level']).count()
print("new_user_class_level的空值情况：", nul_na_count, "空值占比：%0.2f%%" % (nul_na_count / t_count * 100))  # 32.49%

# 使用随机森林对缺失值进行预测
import numpy as np
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.regression import LabeledPoint

# 将pvalue_level中空值所在的行数据剔除
train_pvalue_data = df_user.dropna(subset=['pvalue_level']).rdd.map(
    lambda r: LabeledPoint(r.pvalue_level - 1,
                           [r.cms_segid, r.cms_group_id, r.final_gender_code, r.age_level, r.shopping_level,
                            r.occupation]))
# args1: 训练的数据； args2：目标值的分类个数0.1,2; args3:特征中是否包含分类特征{2:2,3:7}表示在特征中第二个特征是分类的，有两个类：arg4: 随机森林中数的棵树
model_pvalue = RandomForest.trainClassifier(train_pvalue_data, 3, {}, 5)

# 筛选出缺失值条目
pl_na_df = df_user.na.fill(-1).where(("pvalue_level=-1"))
pl_na_df.show(5)


def row(r):
    return r.cms_segid, r.cms_group_id, r.final_gender_code, r.age_level, r.shopping_level, r.occupation


# 转换为普通的rdd类型
rdd_pvalue = pl_na_df.rdd.map(row)
# 预测全部p_value_level值
predicts_pvalue = model_pvalue.predict(rdd_pvalue)
# 查看前20条
print(predicts_pvalue.take(20))
# [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
print("预测值总数：", predicts_pvalue.count())  # 575917
temp_pvalue = predicts_pvalue.map(lambda x: int(x)).collect()
pdf_pvalue = pl_na_df.toPandas()
new_user_profile_df = df_user.dropna(subset=['pvalue_level']).unionAll(
    spark.createDataFrame(pdf_pvalue, schema=schema_user))
new_user_profile_df.show(5)

train_new_user_data = df_user.dropna(subset=['new_user_class_level']).rdd.map(
    lambda r: LabeledPoint(r.new_user_class_level - 1,
                           [r.cms_segid, r.cms_group_id, r.final_gender_code, r.age_level, r.shopping_level,
                            r.occupation]))

model_new_user = RandomForest.trainClassifier(data=train_new_user_data, numClasses=4, categoricalFeaturesInfo={},
                                              numTrees=5)
nul_na_df = df_user.na.fill(-1).where("new_user_class_level=-1")

rdd_new_value = nul_na_df.rdd.map(row)
predicts_new_user = model_new_user.predict(rdd_new_value)
temp_new_user = predicts_new_user.map(lambda x: int(x)).collect()
pdf_new_user = nul_na_df.toPandas()
pdf_new_user['new_user_class_level'] = np.array(temp_new_user) + 1  # 注意+1， 还原预测值
new_user_profile_df = df_user.dropna(subset=['new_user_class_level']).unionAll(spark.createDataFrame(pdf_new_user, schema=schema_user))
# 检查缺失值处理结果
new_user_profile_df.groupby("pvalue_level").count().collect()
new_user_profile_df.groupby("new_user_class_level").count().collect()

# 需要先将缺失值全部转换为数值，与原有特征一起处理
user_profile_df = df_user.na.fill(-1)
# 使用独热编码时，必须先将待处理字段转换为字符串类型才可处理
user_profile_df = user_profile_df.withColumn("pvalue_levle", user_profile_df.pvalue_level.cast(StringType())).\
    withColumn("new_user_class_level", user_profile_df.new_user_class_level.cast(StringType()))
user_profile_df.printSchema()

stringIndexer_pvalue = StringIndexer(inputCol='pvalue_level', outputCol='pl_onehot_feature')
encoder_pvalue = OneHotEncoder(dropLast=False, inputCol='pl_onehot_feature', outputCol='pl_onehot_value')
pipeline_pvalue = Pipeline(stages=[stringIndexer_pvalue, encoder_pvalue])
pipeline_model_pvalue = pipeline_pvalue.fit(user_profile_df)
user_profile_df2 = pipeline_model_pvalue.transform(user_profile_df)
user_profile_df2.printSchema()
user_profile_df2.show()

stringIndexer_new_user = StringIndexer(inputCol='new_user_class_level', outputCol='nul_onehot_feature')
encoder_new_user = OneHotEncoder(dropLast=False, inputCol='nul_onehot_feature', outputCol='nul_onehot_value')
pipeline_new_user = Pipeline(stages=[stringIndexer_new_user, encoder_new_user])
pipeline_model_new_user = pipeline_new_user.fit(user_profile_df2)
user_profile_df3 = pipeline_model_new_user.transform(user_profile_df2)
user_profile_df3.printSchema()
user_profile_df3.show()

# 保存数据
new_df_raw.toPandas().to_csv("new_raw_sample.csv", header=True, index=False)
ad_feature_df.toPandas().to_csv("new_ad_feature.csv", header=True, index=False)
user_profile_df3.toPandas().to_csv("new_user_profile.csv", header=True, index=False)
