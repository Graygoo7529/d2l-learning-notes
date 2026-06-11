import sys; sys.path.insert(0, sys.path[0]+"\\..\\_my_envs")
import _my_pkgs
from common_pkgs import *


# _my_tools.hello.hello()

# 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = _my_pkgs.data.synthetic_linear_data(true_w, true_b, 1000)

# 读取数据集
data_iter = _my_pkgs.data.load_array((features,labels),batch_size=10)
# print(next(iter(data_iter)))

# 定义模型
net = torch.nn.Sequential(torch.nn.Linear(2, 1)) #第一个指定输入特征形状，即2，第二个指定输出特征形状，即1，
# 相比与自己的简单实现，torch 将参数都定义在模型里了，而不是在模型外面构建参数再输入进来
# 在训练时，从把模型的参数交给损失计算和训练优化器
# 此外，torch 的模型(net)、损失(loss)、优化(optimize) 都实现为类而不是方法。

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 训练
lr = 0.03
num_epochs = 3
# net = torch.nn.Sequential(torch.nn.Linear(2, 1))
loss = torch.nn.MSELoss() # 在这里 reduction=‘mean’，即 loss 结果是标量、且已经标准化（除以批量大小）
# 3.2 raw_linear_regression 手动使用 reduction=‘sum’ 降维为标量，但没有标准化，因此需要在 optimize 时对梯度进行标准化
# 在损失函数中降维和标准化，可以使 optimize 与批量大小解耦，只考虑学习率；这可能也是 torch 将它们实现为类而非方法的原因。
optimize = torch.optim.SGD(net.parameters(), lr=lr)

for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        optimize.zero_grad()
        l.backward()
        optimize.step()
    train_l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {float(train_l):f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)









