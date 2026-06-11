import sys; sys.path.insert(0, sys.path[0]+"\\..\\_my_envs")
import _my_pkgs
from common_pkgs import *

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)



if 0:
    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'out shape: \t', X.shape)

batch_size = 256
train_iter, test_iter = _my_pkgs.data.load_data_fashion_mnist(batch_size=batch_size)



lr, num_epochs = 0.9, 10
_my_pkgs.train.train_ch6(net, train_iter, test_iter, 
                           num_epochs, lr, _my_pkgs.tools.gpu.try_gpu())
