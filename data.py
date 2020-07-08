import numpy as np
import random

AMIN = 0.1
AMAX = 5
PMIN = 0.
PMAX = np.pi
XMIN = -5.0
XMAX = 5.0

class Gen(object):
    def __init__(self, task_num=10, ks=10, kq=100, filename=None):
        self.xs = []
        self.ys = []
        self.xq = []
        self.yq = []
        self.task = []
        if filename is None:
            self.generate(task_num, ks, kq)
        else:
            with np.load(filename) as data:
                self.task = data['task']
                self.xs = data['xs']
                self.ys = data['ys']
                self.xq = data['xq']
                self.yq = data['yq']

    def generate(self, task_num, ks, kq):
        for i in range(task_num):
            a = random.uniform(AMIN, AMAX)
            p = random.uniform(PMIN, PMAX)
            self.task.append((a, p))
            xs = []
            ys = []
            for j in range(ks):
                x = random.uniform(XMIN, XMAX)
                y = self.sin(a, p, [x])
                xs.append([x])
                ys.append(y)
            xq = np.linspace(XMIN, XMAX, kq)[:, np.newaxis].tolist()
            yq = [self.sin(a, p, x) for x in xq]
            self.xs.append(xs)
            self.xq.append(xq)
            self.ys.append(ys)
            self.yq.append(yq)
        self.xs = np.array(self.xs, dtype=np.float32)
        self.ys = np.array(self.ys, dtype=np.float32)
        self.xq = np.array(self.xq, dtype=np.float32)
        self.yq = np.array(self.yq, dtype=np.float32)

    def sin(self,a, p, x):
        return [a*np.sin(x[0] + p)]

    def save(self, filename):
        np.savez(filename, task=self.task, xs=self.xs, ys=self.ys, xq=self.xq, yq=self.yq)

if __name__ == '__main__':
    g = Gen(3, 5, 100)
    import matplotlib.pyplot as plt
    plt.plot(g.xq[0], g.yq[0])
    plt.plot(g.xq[1], g.yq[1])
    plt.show()