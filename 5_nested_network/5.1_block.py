import sys; sys.path.insert(0, sys.path[0]+"\\..\\_my_envs")
import _my_pkgs
from common_pkgs import *

if 0:
    net = nn.Sequential(nn.Linear(20,256), nn.ReLU(), nn.Linear(256,10))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)
    
    def forward(self, X):
        return self.out(_my_pkgs.activate.relu(self.hidden(X)))
    
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X



#net = MLP()
net = MySequential(nn.Linear(20,256), nn.ReLU(), nn.Linear(256,10))
X = torch.rand(2,20)
print(net(X))