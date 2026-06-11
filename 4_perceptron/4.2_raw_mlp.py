import sys; sys.path.insert(0, sys.path[0]+"\\..\\_my_envs")
import _my_pkgs
from common_pkgs import *

batch_size = 256
train_iter, test_iter = _my_pkgs.data.load_data_fashion_mnist(batch_size)

# input: mxn(784) -> hidden: 784x256 -> out: 256xc(10)
num_inputs, num_outputs, num_hiddens = 784, 10, 256

# 初始化参数
# 手动实现一个简单的多层感知机是很容易的。
# 然而如果有大量的层，从零开始实现多层感知机会变得很麻烦（例如，要命名和记录模型的参数）

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01) # torch.randn是生成以0为均值1为方差的随机数
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True)) # *0.01 取方差为 0.01
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

# 定义模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = _my_pkgs.activate.relu(_my_pkgs.models.basic.affine(X, W1, b1))
    return _my_pkgs.models.basic.affine(H, W2, b2)

# 损失
loss = _my_pkgs.loss.softmax_cross_entropy

# 训练
num_epochs, lr = 10, 0.1
def updater(batch_size):
    return _my_pkgs.optim.sgd(params, lr, batch_size)
_my_pkgs.train.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)