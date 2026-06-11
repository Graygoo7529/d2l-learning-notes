import sys; sys.path.insert(0, sys.path[0]+"\\..\\_my_envs")
import _my_pkgs
from common_pkgs import *

batch_size = 256
train_iter, test_iter = _my_pkgs.data.load_data_fashion_mnist(batch_size)

# 初始化模型参数
# 使用正态分布初始化我们的权重W，偏置初始化为0。
num_inputs = 784
num_outputs = 10
# f: WX+b, X:m x num_inputs(784) W: num_inputs(784) x num_outputs(10) -> m x num_outputs(10)
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# 定义模型和损失
if 0:
    def net(X):
        return _my_pkgs.norm.softmax(_my_pkgs.models.basic.affine(X.reshape(-1, W.shape[0]), W, b))
    loss = _my_pkgs.loss.cross_entropy
else:
    def net(X):
        return _my_pkgs.models.basic.affine(X.reshape(-1, W.shape[0]), W, b)
    loss = _my_pkgs.loss.softmax_cross_entropy

# # 训练
lr = 0.1
num_epochs = 10 # 在每个迭代周期（epoch）中，遍历整个训练数据集，并将训练数据集所有样本使用一次

def updater(batch_size):
    return _my_pkgs.optim.sgd([W, b], lr, batch_size)
_my_pkgs.train.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)