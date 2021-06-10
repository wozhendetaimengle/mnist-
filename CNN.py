import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision
from utils import plot_curve, plot_image, one_hot

batch_size = 1

train_loader = torch.utils.data.DataLoader(  # 加载训练集
    torchvision.datasets.MNIST('mnist_data', train=False, download=False,  # 创建名为'mnist_data'的文件存放数据
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 30, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(480, 10)

    def forward(self, x):
        # Flatten data from (n, 1，28,28) to (n,784)
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)  # flatten
        x = self.fc(x)
        return x


model = Net()

loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 梯度下降的优化器

for epoch in range(1):  # 对整个数据集迭代3遍
    for _, (x, y) in enumerate(train_loader):  # 对整个数据集迭代一次
        # y_hat = model(x)
        # optimizer.zero_grad()
        # l = loss(y_hat, y)
        # # print(l.item())
        # l.backward()
        # optimizer.step()
        # print(y_hat)
        print(x.size())
        break
