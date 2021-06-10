import numpy as np
import matplotlib.pyplot as plt     #导入画图包

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = 1.0                             #初始权重猜测
a = []
b = []


def forward(x):         #权值计算函数
    return x * w


def cost(xs, ys):           #损失计算
    cost = 0
    for x, y in zip(xs, ys):     #打包
        y_pred = forward(x)
        cost += (y_pred - y) ** 2       #平方
    return cost / len(xs)               #除以样本数量，求均值


def gradient(xs, ys):           #梯度计算
    grad = 0                    #初始化梯度=0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)         #梯度公式
    return grad / len(xs)


print("Predict(before training)", 4, forward(4))
for epoch in range(100):
    cost_val = cost(x_data, y_data)                             #当前损失
    grad_val = gradient(x_data, y_data)               #当前梯度
    w -= 0.01 * grad_val
    print('Epoch:', epoch, 'w=', w, 'loss=', cost_val)
    a.append(w)
    b.append([cost_val])
print("Predict(after training)", 4, forward(4))

plt.plot(a, b)
plt.ylabel('cost')
plt.xlabel('w')
plt.show()
