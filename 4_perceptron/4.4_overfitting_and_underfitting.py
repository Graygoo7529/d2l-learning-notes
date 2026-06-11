import sys; sys.path.insert(0, sys.path[0]+"\\..\\_my_envs")
import _my_pkgs
from common_pkgs import *
import math

# 给定 x，使用以下三阶多项式来生成训练和测试数据的标签

max_degree = 20  # 多项式的最大阶数
n_train, n_test = 100, 100  # 训练和测试数据集大小
true_w = np.zeros(max_degree)  # 分配空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))  #生成x（所具有的特征）
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!
# labels的维度:(n_train+n_test,)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape) #噪声项服从均值为 0 且标准差为 0.1 的正态分布

# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
    torch.float32) for x in [true_w, features, poly_features, labels]]

#print(features[:2], poly_features[:2, :], labels[:2])


def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = _my_pkgs.data.load_array((train_features, train_labels.reshape(-1,1)),
                                batch_size)
    test_iter = _my_pkgs.data.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = _my_pkgs.tools.plot.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        _my_pkgs.train.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (_my_pkgs.scoring.base.evaluate_loss(net, train_iter, loss),
                                     _my_pkgs.scoring.base.evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())



# 三阶多项式函数拟合(正常)
if 0:
    train(poly_features[:n_train, :4], poly_features[n_train:, :4],labels[:n_train], labels[n_train:])

# 线性函数拟合(欠拟合)
if 0:
    train(poly_features[:n_train, :2], poly_features[n_train:, :2],labels[:n_train], labels[n_train:])

# 从多项式特征中选取所有维度
if 1:
    train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:], num_epochs=1500)