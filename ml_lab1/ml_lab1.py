import pandas as pd
import numpy as np
from numpy import linalg as la


# import matplotlib.pyplot as plt
# import matplotlib.markers as marks

# from scipy import

def estimate1(a, y):
    mean = np.mean(y)
    return np.sum((a - mean) ** 2) / np.sum((y - mean) ** 2)


def estimate2(a, y):
    # a_tmp = [int(v > 0.5) for v in a[1:10]]
    # y_tmp = list(y[1:10].transpose()[0])
    # print("a: " + str(a_tmp))
    # print("y: " + str(y_tmp))
    return sum([int(a[i] > 0.5) == y[i] for i in range(0, len(a))]) / len(a)


def a1(x, w):
    return np.dot(x, w)


def grad1(x, y, w):
    return np.dot(x.transpose(), a1(x, w) - y) / y.size


def a2(x, w):
    return 1 / (1 + np.exp(-1 * np.dot(x, w)))


def grad2(x, y, w):
    a = 1 / (1 + np.exp(-1 * np.dot(x, w)))  # 791x1
    diff = a - y  # 791x1 - 791x1
    return np.dot(x.transpose(), diff)  # 15x791 * 791x1


class LinearModel:

    def __init__(self, grad_func, estimate, resolve, t=0.000005, k=1., n=1000000):
        self.grad_func = grad_func
        self.estimate = estimate
        self.resolve = resolve
        self.w = None
        self.t = t
        self.k = k
        self.n = n

    def train(self, x, y, x_valid=None, y_valid=None):
        if self.w is None:
            # self.w = np.array(np.random.randn(x.shape[1], 1))
            self.w = np.array(np.random.rand(x.shape[1], 1))

        print("x-shape: " + str(x.shape))
        print("y-shape: " + str(y.shape))
        print("w-shape: " + str(self.w.shape))
        steps = 100

        for i in range(1, self.n):

            grad = self.grad_func(x, y, self.w)
            self.w = self.w - self.t * grad

            if i % (self.n / steps) == 0:
                p = np.random.permutation(len(y))
                y = y[p]
                x = x[p]
                self.t = self.t / self.k
                r_train = self.validation(x, y)
                g = np.max(np.abs(grad))
                if x_valid is not None and y_valid is not None:
                    r_valid = self.validation(x_valid, y_valid)
                    print("Idx[%d/%d]\ttrain=%3d%%\tvalid=%3d%%\tt=%10f\tg=%f" % (
                        int(i / (self.n / steps)),
                        steps,
                        int(r_train * 100),
                        int(r_valid * 100),
                        self.t,
                        g))
                else:
                    print("Idx[%d/%d]\ttrain=%3d%%\tt=%10f\tg=%f" % (
                        int(i / (self.n / steps)),
                        steps,
                        int(r_train * 100),
                        self.t,
                        g))

    def validation(self, x, y):
        if x is None or y is None:
            return -0.01
        a = self.resolve(x, self.w)
        return self.estimate(a, y)

    def test(self, x):
        return self.resolve(x, self.w)


def LoadY(y_path, test_len=0):
    frame_y = pd.read_csv(y_path, header=None)
    train_len = len(frame_y) - test_len
    y = np.array(frame_y.ix[:train_len - 1])
    y_valid = np.array(frame_y.ix[train_len:]) if test_len is not 0 else None
    return y, y_valid


def LoadX(x_path, test_len=0, do_norm=False):
    frame_x = pd.read_csv(x_path, header=None)
    # for i in range(0, len(frame_y)):
    #    if frame_y.ix[i, 0] == 0:
    #        frame_y.ix[i, 0] = -1
    if do_norm:
        for i in range(0, frame_x.shape[1]):
            tmp = frame_x.ix[:, i]
            max = np.max(tmp)
            frame_x.ix[:, i] = tmp / max
    train_len = frame_x.shape[0] - test_len
    x = np.append(frame_x.ix[:train_len - 1, :], np.ones((train_len, 1), dtype=float), axis=1)
    x_valid = np.append(frame_x.ix[train_len:, :], np.ones((test_len, 1), dtype=float),
                       axis=1) if test_len is not 0 else None
    return x, x_valid


def lr1_1():
    print("# ========= Start part 1 =========")
    x, x_valid = LoadX('dataset/t1_linreg/t1_linreg_x_train.csv', test_len=0)
    y, y_valid = LoadY('dataset/t1_linreg/t1_linreg_y_train.csv', test_len=0)

    print("# ========= Train =========")
    model = LinearModel(grad1, estimate1, a1, t=0.000008, n=10000000)
    model.train(x, y, x_valid, y_valid)

    print("# ========= Test ==========")
    val = model.validation(x_valid, y_valid)

    print("# =========================")
    print("# ====== Result: %3d%% =====" % int(val * 100))
    print("# =========================")

    x_test, _ = LoadX('dataset/t1_linreg/t1_linreg_x_test.csv')
    a_test = model.test(x_test)
    print(a_test)

    file = open("lr1_1_test.csv", 'w')
    file.writelines([str(it[0]).__add__('\n') for it in a_test])
    file.close()

def lr1_2():
    print("# ========= Start part 2 =========")
    x, x_valid = LoadX('dataset/t1_logreg/t1_logreg_x_train.csv', test_len=0, do_norm=True)
    y, y_valid = LoadY('dataset/t1_logreg/t1_logreg_y_train.csv', test_len=0)

    print("# ========= Train =========")
    model = LinearModel(grad2, estimate2, a2, t=0.0005, n=1000000)
    model.train(x, y, x_valid, y_valid)

    print("# ========= Test ==========")
    val = model.validation(x_valid, y_valid)

    print("# =========================")
    print("# ====== Result: %3d%% =====" % int(val * 100))
    print("# =========================")

    x_test, _ = LoadX('dataset/t1_logreg/t1_logreg_x_test.csv', do_norm=True)
    a_test = [int(v > 0.5) for v in model.test(x_test)]
    print(a_test)

    file = open("lr1_2_test.csv", 'w')
    file.writelines([str(it).__add__('\n') for it in a_test])
    file.close()



if __name__ == "__main__":
    lr1_1()
    lr1_2()
