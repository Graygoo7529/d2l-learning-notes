import sys; sys.path.insert(0, sys.path[0]+"\\..\\_my_envs")
import _my_pkgs
from common_pkgs import *

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return _my_pkgs.models.basic.corr2d(x, self.weight) + self.bias
    

# 目标图像边缘检测
X = torch.ones((6, 8))
X[:, 2:6] = 0
print(X)

k = torch.tensor([[1.0, -1.0]]) # 二维张量, 1 行 2 列
Y = _my_pkgs.models.basic.corr2d(X, k)
print(Y)

if 0:
    conv2d = Conv2D((1,2))
    lr = 3e-2

    for i in range(10):
        Y_hat = conv2d(X)
        loss = (Y_hat - Y) ** 2
        conv2d.zero_grad()
        loss.sum().backward()
        conv2d.weight.data[:] -= lr * conv2d.weight.grad
        if (i + 1) % 2 == 0:
            print(f'epoch {i+1}, loss {loss.sum():.3f}')

    print(conv2d.weight.data.reshape((1, 2)))

X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
print(X)

pool2d = nn.MaxPool2d(3)
print(pool2d(X))

# import torch.nn.functional as F
# X_padded = F.pad(X, (1, 1, 1, 1), mode='constant', value=0).squeeze(0).squeeze(0)
# print(_my_pkgs.models.basic.pool2d(X_padded, (3,3)))



