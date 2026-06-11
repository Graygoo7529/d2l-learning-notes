import sys; sys.path.insert(0, sys.path[0]+"\\..\\_my_envs")
import _my_pkgs
from common_pkgs import *

print(torch.__version__)
print(torch.cuda.is_available())

print(_my_pkgs.tools.gpu.try_gpu(), 
      _my_pkgs.tools.gpu.try_gpu(10), 
      _my_pkgs.tools.gpu.try_all_gpus())


X = torch.ones(2, 3, device=_my_pkgs.tools.gpu.try_gpu())
Y = torch.rand(2, 3, device=_my_pkgs.tools.gpu.try_gpu(1))



#Z = X.cuda(1) # 在第 2 块 gpu 上创建 X 的副本

# 将模型参数放在GPU上
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=_my_pkgs.tools.gpu.try_gpu())
net(X)
print(net[0].weight.data.device)

