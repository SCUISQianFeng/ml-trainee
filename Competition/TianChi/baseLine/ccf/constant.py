# -*- coding:utf-8 -*-

"""
    常量
"""


class _const:
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, key, value):
        if key in self.__dict__:
            raise self.ConstError("Can't change const.{}".format(key))
        if not key.isupper():
            raise self.ConstCaseError("const name {} is not all uppercase".format(key))
        self.__dict__[key] = value


import sys, os

sys.path.append(os.pardir)
# sys.path.append('..')

# sys.modules[__name__] == _const()
const = _const()

const.MY_CONSTANT = 1
const.MY_SECOND_CONSTANT = 2
const.STR_SEPARATOR = ':'
