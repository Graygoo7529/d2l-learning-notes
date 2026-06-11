import sys; sys.path.insert(0, sys.path[0]+"\\..\\_my_envs")
import _my_pkgs
from common_pkgs import *


# 梯度消失示例
if 0:
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    #y = torch.sigmoid(x) # sigmoid 梯度消失问题
    y = torch.relu(x) # relu 更稳定（梯度不会直接消失）
    y.backward(torch.ones_like(x))

    _my_pkgs.tools.plot.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
            legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
    

# 梯度爆炸示例
if 0:
    M = torch.normal(0, 1, size=(4,4)) # M 数值为：均值为 0，方差为 1
    print('一个矩阵 \n',M)
    for i in range(100):
        M = torch.mm(M,torch.normal(0, 1, size=(4, 4)))

    print('乘以100个矩阵后\n', M)