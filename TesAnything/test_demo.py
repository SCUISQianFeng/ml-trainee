#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng
@license: Apache Licence 
@file: test_demo.py 
@time: 2022/03/07
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""

def showNnumber(numbers):
    for n in numbers:
      print(n)


def chanageInt(number2):
    number2 = number2 + 1

    print("changeInt: number2= ", number2)

class Hello():

    def __init__(self,name):

        self.name=name
    def showInfo(self):

        print(self.name)


def func(a,**kwargs):
    pass

def func(a,*args,b):
    pass
def mul(x):
    return x*x

if __name__ == '__main__':
    g = (x*x for x in range(10))
    L = [x * x for x in range(10)]
    # print(sum(g)==sum(L))

    import copy

    a = [1, 2, 3, 4, ['a', 'b']]  # 原始对象
    b = copy.copy(a)
    c = copy.deepcopy(a)
    print(id(a[1]) != id(b[1]))
    print(id(a[-1])==id(b[-1]))
    print(id(a[-1])==id(c[-1]))
    print('*' * 30)
    A1 = {'小明': 93, '小红': 88, '小刚': 100}
    A2 = {'小刚': 100, '小明': 93, '小红': 88}
    print(A1 == A2)
    print('*' * 30)
    a = [1,2,3]
    b = [4,5]
    a.append(b)
    print(len(a))
    print('*' * 30)
    import re
    a = "601988-中国银行"
    print(re.findall("(\d+)-([\u4e00-\u9fa5]+)", a))

    print('*' * 30)
    a = {"a":1, "b":2, "c":3}
    print(max(a))
    # print(max(a), key=lambda x: a[x])
    print('*' * 30)
    n = [1, 2, 3, 4, 5]
    res = list(map(mul, n))
    print(res)

    print('*' * 30)
    from collections import deque

    queue = deque(['Eric', 'John', 'Michael'])
    queue.append('Terry')
    queue.append('Graham')
    print(queue)
    queue.popleft()
    print(queue)

    print('*' * 30)
    print(0 or False and 1)
    
    print('*' * 30)
    from collections import defaultdict, OrderedDict
    import Thread
    import time

    a = defaultdict(OrderedDict)


    def func(code, i):
        a[code][i] = i


    def timer():
        while 300000 != sum(map(lambda code: len(a[code]), a.keys())):
            print(sum(map(lambda code: len(a[code]), a.keys())))
            time.sleep(1)
        print("ending")


    t0 = Thread(target=timer)
    t0.start()

    ts = []
    for code in ["a", "b", "c"]:
        for i in range(100000):
            t = Thread(target=func, args=(code, i,))
            ts.append(t)
            t.start()

    for t in ts:
        t.join()

    print(sum([set(a[k].keys()) != set(a[k].values()) for k in a]))









