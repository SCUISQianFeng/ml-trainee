# coding:utf-8

#########################################################################
# 1、问题描述
# tf在win环境下的路径谜之诡异
# logdir = r"./data/keras/model/" + datetime.datateim.now().strftime("%Y%m%d-%H%M%S")
# 报错内容“./data/titanic/model/profle\train


# 解决：
# tf生成路径在linux环境无异常，但在win环境下的路径分隔符与win冲突，且为win自动生成，无法设置路径分隔符
# 只能提前将模型保存地址设置为绝对路径
# logdir_abs = os.path.abspath("./data")
# logdir = os.path.join(logdir_abs, datetime.datateim.now().strftime("%Y%m%d-%H%M%S"))
#########################################################################

