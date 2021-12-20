import numpy as np

from sklearn.linear_model import LinearRegression


class LinearRegression:
    def __init__(self, learning_rate=0.01, total_iterations=1000, print_cost=False):
        """
        初始化
        :param learning_rate: 学习率
        :param total_iterations: 总迭代次数
        :param print_cost: 是否打印出损失值
        """
        self.learning_rate = learning_rate
        self.total_iterations = total_iterations
        self.print_cost = print_cost

    def y_hat(self, x, w):
        return np.dot(w.T, x)  # [1,2] · [2, 5000] => [1, 500]

    def cost(self, yhat, y):
        return 2 * np.sum(np.power(yhat - y, 2)) / self.m

    def gradient_descent(self, w, x, y, yhat):
        dCdw = 2 * np.dot(x, (yhat - y).T) / self.m  # [2,500] · [500, 1] => [500, 1]
        w = w - self.learning_rate * dCdw
        return w

    def main(self, x, y):
        # x.shape : [1, 500], 需要扩充成[2, 500]
        one = np.ones(shape=(1, x.shape[1]))  # [1, 500]
        x = np.append(one, x, axis=0)  # x.shape:[2, 500]

        # 实际进行计算时，x的shape应该是从[2, 500] 转变成[500, 2]
        # 但是y的shape还是[1, 500]
        self.m = x.shape[1]
        self.n = x.shape[0]

        w = np.zeros(shape=(self.n, 1))  # w.shape:[2, 1]， 计算yhat就是的w.T · x =》 [1, 2] * [2, 500]
        for it in range(self.total_iterations + 1):
            yhat = self.y_hat(x, w)
            cost_value = self.cost(yhat, y)
            if it % 2000 == 0 and self.print_cost:
                print(f'Cost at iteration {it} is {cost_value}')
            w = self.gradient_descent(w, x, y, yhat)
        return w


if __name__ == "__main__":
    x = np.random.randn(1, 500)
    y = 3 * x + 5 + np.random.randn(1, 500) * 0.1
    regression = LinearRegression()
    w = regression.main(x, y)
    print(w)
