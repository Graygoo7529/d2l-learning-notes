import sys; sys.path.insert(0, sys.path[0]+"\\..\\_my_envs")
import _my_pkgs
from common_pkgs import *


batch_size = 256
train_iter, test_iter = _my_pkgs.data.load_data_fashion_mnist(batch_size)

from torch import nn

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none')

updater = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10 # 在每个迭代周期（epoch）中，遍历整个训练数据集，并将训练数据集所有样本使用一次
_my_pkgs.train.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)