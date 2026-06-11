# 训练神经网络时出现的实机挑战
# 1. 数据预处理的方式会对最终的结果产生巨大的影响（eg. 标准化输入特征）
# 2. 中间层的变量，可能具有更广阔的变化范围（eg. 一个层的可变值是另一个层的 100 倍，猜想：这种变量分布的偏移可能会妨碍网络的收敛）
# 3. 深层次网络的过拟合

# 方法：基于批量统计的标准化
# 做法：在每次训练迭代中，规范化输入，通过减去其（基于当前小批量处理的）均值并除以标准差。

import sys; sys.path.insert(0, sys.path[0]+"\\..\\_my_envs")
import _my_pkgs
from common_pkgs import *

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not torch.is_grad_enabled(): # 预测模式
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            mean = X.mean(dim = 0) # 全连接层
            var = ((X-mean)**2).mean(dim = 0)
        else:
            mean = X.mean(dim=(0, 2, 3), keepdim = True) # 二维卷积层，计算通道维上的均值和方差
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim = True)

        X_hat = (X - mean) / torch.sqrt(var + eps) # 训练模式
        # 更新 moving_mean & moving_var
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta # 拉伸和移位
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    # num_features: 完全连接层的输出数量，或卷积层的输出通道数
    # num_dims: 2，完全连接层；4，卷积层；
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9
        )
        return Y
    
# torch 标准实现 —————— nn.BatchNorm2d(num_features)
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), 
    nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10)
)



lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = _my_pkgs.data.load_data_fashion_mnist(batch_size)
_my_pkgs.train.train_ch6(net, train_iter, test_iter, num_epochs, lr, _my_pkgs.tools.gpu.try_gpu())
