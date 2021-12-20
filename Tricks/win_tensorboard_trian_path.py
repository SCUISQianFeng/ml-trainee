# 1、tensorboard创建训练过程中train的路径无法创建的问题
#
# 解决：
# https://github.com/ibab/tensorflow-wavenet/issues/255

# log_dir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# 改成
# log_dir = ".\\logs\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
# 即可


