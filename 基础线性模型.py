import torch
from torch import nn  # 帮助创建和训练神经网络的库
from torch.nn import functional as F  # nn的子库
from torch import optim  # 优化函数的库
import matplotlib.pyplot as plt
import torchvision

batch_size = 100  # 每次输入的图片张数


# 深度学习训练四个步骤：1、处理和加载数据；2、建立模型；3、确定损失函数和优化器；4、训练
def show_loss(x_label, loss):
    plt.plot(x_label, loss, color='blue')  #
    # plt.legend(['value'], loc='upper right')  #
    # plt.xlabel('step')
    # plt.ylabel('value')
    plt.show()


train_loader = torch.utils.data.DataLoader(  # 加载训练集
    torchvision.datasets.MNIST('mnist_data', train=True, download=False,  # 创建名为'mnist_data'的文件存放数据,已经下载过所以dwnlad=flase
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)


# 建立模型步骤：1、init初始化；2、forward前馈
class Net(nn.Module):  # 继承Module
    def __init__(self):
        super(Net, self).__init__()  # 继承父类的初始化函数
        self.fc1 = nn.Linear(1 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # sigmoid,tanh,ReLU等非线性函数，作为激活函数，添加非线性的元素
        x = F.relu(self.fc2(x))  # sigmoid,tanh,ReLU等非线性函数，作为激活函数，添加非线性的元素
        x = self.fc3(x)
        return x


class Net_Linear(nn.Module):  # 继承Module
    def __init__(self):
        super(Net_Linear, self).__init__()  # 继承父类的初始化函数
        self.fc1 = nn.Linear(1 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # sigmoid,tanh,ReLU等非线性函数，作为激活函数，添加非线性的元素
        # x = F.relu(self.fc2(x))  # sigmoid,tanh,ReLU等非线性函数，作为激活函数，添加非线性的元素
        x = self.fc2(x)
        return x


class NetCNN(nn.Module):
    def __init__(self):
        super(NetCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 1和10分别是输入和输出通道，卷积核kernel
        self.conv2 = nn.Conv2d(10, 30, kernel_size=5)
        self.pooling = nn.MaxPool2d(2)  # 最大池化，就是选里面最大的重新组成一个张量
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


class InceptionANetCNN(nn.Module):
    def __init__(self, in_channels):
        super(InceptionANetCNN, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)  # 一层卷积核的作用是改变通道数
        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=1, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)


class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = InceptionANetCNN(in_channels=10)
        self.incep2 = InceptionANetCNN(in_channels=20)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


net = Net()  # 实例化
net_linear = Net_Linear()
netcnn = NetCNN()

loss = nn.CrossEntropyLoss()  # 选择一个损失函数,交叉熵
loss2 = nn.L1Loss  # 均方差损失

optimizer = optim.SGD(net.parameters(), lr=0.01)  # 优化器，SGD指随机梯度下降

for epoch in range(1):  # 震哥哥数据集迭代一遍
    for a, (x, y) in enumerate(train_loader):
        print(a)  # x指的是图片当前的张数
        print(x.shape)  # x是图片，用net处理
        print(y.shape)  # y是label 0-9
        y_hat = net(x.view(batch_size, 28 * 28))  # 改变张量y的大小，改成一个[batch_size,28*28]大小的张量
        y_hat_Net_Linear = net_linear(x.view(batch_size, 28 * 28))  # 改变张量y的大小，改成一个[batch_size,28*28]大小的张量

        print(y_hat.shape)
        loss1 = loss(y_hat, y)
        loss_net_linear = loss(y_hat_Net_Linear, y)
        #   print(loss1)
        optimizer.zero_grad()  # 梯度清零，否则生成计算图会消耗资源
        loss1.backward()  # 反向传播
        optimizer.step()  # 根据梯度更新网络参数
        print('epoch= ：', epoch, 'loss1 = : ', loss1.item())
        print('epoch= ：', epoch, 'loss_net_linear = : ', loss_net_linear.item())
        x_label = []
        x_label.append(a)
    show_loss(x_label, loss)
# 可视化输出loss变化????
