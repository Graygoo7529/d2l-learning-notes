import sys; sys.path.insert(0, sys.path[0]+"\\..\\_my_envs")
import _my_pkgs
from common_pkgs import *

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
if 0:
    y = _my_pkgs.activate.relu(x)
    _my_pkgs.tools.plot.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
    y.backward(torch.ones_like(x), retain_graph=True)
    _my_pkgs.tools.plot.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))

if 0:
    y = _my_pkgs.activate.sigmoid(x)
    _my_pkgs.tools.plot.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
    #x.grad.data.zero_()
    y.backward(torch.ones_like(x),retain_graph=True)
    _my_pkgs.tools.plot.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))

if 0:
    y = _my_pkgs.activate.tanh(x)
    _my_pkgs.tools.plot.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
    #x.grad.data.zero_()
    y.backward(torch.ones_like(x),retain_graph=True)
    _my_pkgs.tools.plot.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))

if 1:
    y = _my_pkgs.activate.p_relu(x,0.25)
    _my_pkgs.tools.plot.plot(x.detach(), y.detach(), 'x', 'p_relu(x)', figsize=(5, 2.5))
    #x.grad.data.zero_()
    y.backward(torch.ones_like(x),retain_graph=True)
    _my_pkgs.tools.plot.plot(x.detach(), x.grad, 'x', 'grad of p_relu', figsize=(5, 2.5))