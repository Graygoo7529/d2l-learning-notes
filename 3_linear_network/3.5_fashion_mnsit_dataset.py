import sys; sys.path.insert(0, sys.path[0]+"\\..\\_my_envs")
import _my_pkgs
from common_pkgs import *

batch_size = 32
train_iter, test_iter = _my_pkgs.data.load_data_fashion_mnist(batch_size, resize=64)

X, y = next(iter(train_iter))
print(X.shape, X.dtype, y.shape, y.dtype)
# X, y = next(iter(torch.utils.data.DataLoader(mnist_train, batch_size=18)))
_my_pkgs.tools.plot.show_images(X.reshape(batch_size, 64, 64), 2, 9, titles=_my_pkgs.data.get_fashion_mnist_labels(y))

timer = _my_pkgs.tools.timer.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')

