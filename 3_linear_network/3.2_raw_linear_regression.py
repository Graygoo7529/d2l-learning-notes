import sys; sys.path.insert(0, sys.path[0]+"\\..\\_my_envs")
import _my_pkgs
from common_pkgs import *


def raw_data_iter(batch_size, features, labels):
    # raw_data_iter 它的执行效率很低，可能会在实际问题上陷入麻烦。 
    #例如，它要求我们将所有数据加载到内存中，并执行大量的随机内存访问。
    #在深度学习框架中实现的内置迭代器效率要高得多， 它可以处理存储在文件中的数据和数据流提供的数据。
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# # 生成数据
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = _my_pkgs.data.synthetic_linear_data(true_w, true_b, 1000)
print('features:', features[0],'\nlabel:', labels[0])

_my_pkgs.plot.set_figsize()
# [:,(1)]样本切片，: 表示选择所有样本（行），而 1 表示选择第二个特征（列）
plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
plt.show()

# # 初始化模型参数
#w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
w = torch.zeros(2, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# # 训练
lr = 0.03
num_epochs = 3 # 在每个迭代周期（epoch）中，我们使用data_iter函数遍历整个数据集， 并将训练数据集中所有样本都使用一次
net = _my_pkgs.models.basic.linreg
loss = _my_pkgs.loss.squared_loss
optimize = _my_pkgs.optim.sgd
batch_size = 10

for epoch in range(num_epochs):
    for X, y in raw_data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        optimize([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')