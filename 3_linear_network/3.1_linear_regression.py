import sys; sys.path.insert(0, sys.path[0]+"\\..\\_my_envs")
import _my_pkgs
from common_pkgs import *

# # 3.1.1 比较循环运算和矢量化运算
if 0:

    n = 600000
    a = torch.ones([n])
    b = torch.ones([n])

    c = torch.zeros(n)
    timer = _my_pkgs.timer.Timer()
    for i in range(n):
        c[i] = a[i] + b[i]  #循环相加
    print(f'{timer.stop():.5f} sec')

    timer.start()
    d = a + b               #重载的矢量相加
    print(f'{timer.stop():.5f} sec')

# # 3.1.2 正态分布
if 1:

    x = np.arange(-7, 7, 0.01)

    # 均值和标准差对
    params = [(0, 1), (0, 2), (3, 1)]
    _my_pkgs.plot.plot(x, [_my_pkgs.maths.dist.normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
            ylabel='p(x)', figsize=(4.5, 2.5),
            legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
    plt.show()