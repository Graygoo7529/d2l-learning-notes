import sys; sys.path.insert(0, sys.path[0]+"\\..\\_my_envs")
import _my_pkgs
from common_pkgs import *

# NiN块以一个普通卷积层开始，后面是两个 1 x 1 卷积层。这两个卷积层充当带有 ReLU 激活函数的逐像素全连接层。

#卷积层的输入和输出由四维张量组成，张量的每个轴分别对应样本、通道、高度和宽度。
#全连接层的输入和输出通常是分别对应于样本和特征的二维张量。
#NiN的想法是在每个像素位置（针对每个高度和宽度）应用一个全连接层。
#如果我们将权重连接到每个空间位置，我们可以将其视为 1 x 1 的卷积层，或作为在每个像素位置上独立作用的全连接层。 
# 从另一个角度看，即将空间维度中的每个像素视为单个样本，将通道维度视为不同特征（feature）




def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
    )

net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(p=0.5),
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)

X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = _my_pkgs.data.load_data_fashion_mnist(batch_size, resize=224)
_my_pkgs.train.train_ch6(net, train_iter, test_iter, num_epochs, lr, _my_pkgs.tools.gpu.try_gpu())

