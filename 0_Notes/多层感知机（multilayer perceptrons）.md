## 多层感知机基础
### 隐藏层
回顾仿射变换： $\hat{y} = w_{1}x_{1} + ... + w_{d} x_{d} +b.$
线性回归通过单个仿射变换将输入直接映射到输出，这意味着每个特征对模型输出的影响是线性的、可加的。在一些问题中，可以通过特征预处理，使变换后的特征与输出之间更接近线性关系；但对图像、语音、文本等复杂数据，这种线性假设过于受限。

例如在猫狗图像分类中，某个位置像素强度的增加，并不总是稳定地提高或降低图像属于某一类别的可能性。单个像素的意义依赖周围像素的上下文，因此仅依赖原始像素的线性组合难以刻画图像中的复杂结构。

为了表示更一般的函数关系，可以在网络中加入隐藏层以克服线性模型的限制。**多层感知机**（multilayer perceptron，MLP）将多个全连接层堆叠起来，前面的层学习特征表示，最后一层在该表示上进行线性预测。

假设矩阵 $\mathbf{X} \in \mathbb{R}^{n \times d}$ 表示 $n$ 个样本的小批量，且每个样本具有 $d$ 维度原始输入特征，对于具有 $h$ 个隐藏单元的单隐藏层多层感知机，将隐藏表示（hidden representations）或隐藏层变量（hidden variable）定义为 $\mathbf{H} \in \mathbb{R}^{n \times h}$ 为隐藏层的输出。隐藏层和输出层采用全连接，因此有隐藏层权重 $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$ 、隐藏层偏置 $\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$、输出层权重 $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$ 和输出层偏置 $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$，由此定义输出 $\mathbf{O} \in \mathbb{R}^{n \times q}$ 为：
$$
\begin{aligned}
    \mathbf{H} & = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.
\end{aligned}
$$
然而，上述模型不足以表述非线性关系，因为只需合并隐藏层，便可将模型退化为一个具有新的权重和偏置的单层线性模型。为了发挥多层架构的潜力，需要在隐藏层仿射变换之后对每个隐藏单元应用非线性的**激活函数**（activation function），使用激活函数后的单隐藏层感知机定义为：
$$
\begin{aligned}
    \mathbf{H} & = \sigma (\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.
\end{aligned}
$$
**多层感知机是通用近似器**：通过隐藏单元和输入之间的相互作用，多层感知机具有近似拟合连续函数的能力；即使是网络只有一个隐藏层，给定足够的隐藏单元和正确的权重， 仍然能对函数建模(尽管实际很困难)，而使用更深的网络，可以更容易地逼近。

### 激活函数
#### ReLU 激活函数
给定元素 $x$，ReLU 被定义为该元素与 0 的最大值： 
$$
\operatorname{ReLU}(x) = \max(x,0).
$$
ReLU 仅保留正元素并丢弃负元素，激活函数是分段线性的；当输入为负，其导数为 0，当输入为正，其导数为 1，当输入为 0，ReLU 不可导（可默认使用左侧的导数）。因此，ReLU 的求导表现是，要么让参数通过，要么让参数消失（该单元的梯度被截断）。

```tikz
\begin{document}
\begin{tikzpicture}[domain=-4:4]
  \draw[very thin,color=gray!40] (-4.1,-0.5) grid (4.1,4.1);
  \draw[->] (-4.2,0) -- (4.3,0) node[right] {$x$};
  \draw[->] (0,-0.7) -- (0,4.3) node[above] {$f(x)$};

  % ReLU
  \draw[color=red, thick, domain=-4:0] plot (\x,{0});
  \draw[color=red, thick, domain=0:4] plot (\x,{\x})
    node[right] {$\mathrm{ReLU}(x)=\max(0,x)$};

  % derivative of ReLU
  \draw[color=blue, thick, domain=-4:0] plot (\x,{0});
  \draw[color=blue, thick, domain=0:4] plot (\x,{1})
    node[right] {$\mathrm{ReLU}'(x)$};

  % note at x=0
  \draw[dashed,color=blue] (0,0) -- (0,1);
  \node[below right] at (0,0) {$0$};
\end{tikzpicture}
\end{document}


```

训练神经网络依赖反向传播（参数更新使用梯度：损失函数对于参数的导数）。反向传播本质上是在多层之间不断相乘各层导数（依据梯度计算的链式法则）。若激活函数导数长期小于 1，深层网络中的梯度会逐层衰减，导致前面层更新缓慢，即**梯度消失**。ReLU 在正半轴导数为 1，能让激活神经元的梯度更直接地传播，从而缓解该问题；但若神经元长期处于负区间，其梯度为 0，可能一直得不到更新。

ReLU 函数的变体：例如，参数化 ReLU 添加了一个线性项，因此即使参数是负的，某些信息仍然可以通过：

$$\operatorname{pReLU}(x) = \max(0, x) + \alpha \min(0, x).$$

对于一个只由仿射变换 + ReLU / pReLU 组成的 MLP，整体函数是一个连续的分段线性函数。这是因为，（1）仿射变换是连续分段线性，（2）ReLU / pReLU 也是连续分段线性的，（3）连续分段线性函数的复合仍是连续分段线性的。换言之，在、固定激活区域内，整个网络退化为局部仿射函数；不同区域对应不同的激活模式，因此有不同的仿射表达式。


#### sigmoid 激活函数
对于一个定义域在 $\mathbb{R}$ 中的输入，sigmoid 函数将输入变换为区间(0, 1)上的输出。sigmoid 源于对生物神经元“激发”或“不激发”进行建模，它是一个平滑的、可微的阈值单元近似。
$$\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$
$$\frac{d}{dx} \operatorname{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right).$$

sigmoid 导数如公式和图像所示。当输入为 0，sigmoid函数的导数达到最大值 0.25；当输入在任一方向上越远离 0 点时，导数越接近 0。

```tikz
\begin{document}
\begin{tikzpicture}[domain=-6:6, xscale=0.75, yscale=3.0]
  \draw[very thin,color=gray!45] (-6.1,-0.1) grid (6.1,1.15);
  \draw[->] (-6.3,0) -- (6.4,0) node[right] {$x$};
  \draw[->] (0,-0.15) -- (0,1.25) node[above] {$f(x)$};

  % reference lines
  \draw[dashed,color=gray] (-6,1) -- (6,1);
  \draw[dashed,color=gray] (-6,0.5) -- (6,0.5);
  \draw[dashed,color=gray] (-6,0.25) -- (6,0.25);

  % sigmoid
  \draw[color=red, thick, domain=-6:6, samples=100]
    plot (\x,{1/(1+exp(0-\x))});

  % derivative of sigmoid
  \draw[color=blue, thick, domain=-6:6, samples=100]
    plot (\x,{(1/(1+exp(0-\x)))*(1-(1/(1+exp(0-\x))))});

  % labels
  \node[color=red,right] at (3.4,0.88) {$\mathrm{sigmoid}(x)$};
  \node[color=blue,right] at (1.0,0.25) {$\mathrm{sigmoid}'(x)$};

  % key values
  \node[left] at (0,1) {$1$};
  \node[left] at (0,0.5) {$0.5$};
  \node[left] at (0,0.25) {$0.25$};
\end{tikzpicture}
\end{document}


```

#### tanh 激活函数
tanh(双曲正切)函数将输入变换为区间(-1, 1)上的输出，函数的形状类似于sigmoid函数，但 tanh 函数关于坐标系原点中心对称；当输入在 0 附近时，tanh 函数接近线性变换。
$$\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$
$$\frac{d}{dx} \operatorname{tanh}(x) = 1 - \operatorname{tanh}^2(x).$$
tanh 导数如公式和图像所示。当输入为 0，tanh 函数的导数达到最大值 1；当输入在任一方向上越远离 0 点时，导数越接近 0。

```tikz
\begin{document}
\begin{tikzpicture}[domain=-4:4, xscale=0.9, yscale=2.0]
  \draw[very thin,color=gray!45] (-4.1,-1.15) grid (4.1,1.15);
  \draw[->] (-4.3,0) -- (4.4,0) node[right] {$x$};
  \draw[->] (0,-1.25) -- (0,1.3) node[above] {$f(x)$};

  \draw[dashed,color=gray] (-4,1) -- (4,1);
  \draw[dashed,color=gray] (-4,-1) -- (4,-1);

  % tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
  \draw[color=red, thick, domain=-4:4, samples=100]
    plot (\x,{(1-exp(0-2*\x))/(1+exp(0-2*\x))});

  % tanh'(x) = 1 - tanh(x)^2
  \draw[color=blue, thick, domain=-4:4, samples=100]
    plot (\x,{1-((1-exp(0-2*\x))/(1+exp(0-2*\x)))*((1-exp(0-2*\x))/(1+exp(0-2*\x)))});

  \node[color=red,right] at (2.3,0.95) {$\mathrm{tanh}(x)$};
  \node[color=blue,right] at (1.1,0.45) {$\mathrm{tanh}'(x)$};

  \node[left] at (0,1) {$1$};
  \node[left] at (0,-1) {$-1$};
  \node[below right] at (0,0) {$0$};
\end{tikzpicture}
\end{document}

```

tanh 函数和 sigmoid 存在关联 $\operatorname{tanh}(x) + 1 = 2 \operatorname{sigmoid}(2x).$

$$
\operatorname{tanh}(x)+1
=
\frac{e^{2x}-1}{e^{2x}+1}+1
=
\frac{2e^{2x}}{e^{2x}+1}

$$
$$
\operatorname{sigmoid}(2x)
=
\frac{1}{1+e^{-2x}}
=
\frac{e^{2x}}{e^{2x}+1}

$$

### 多层感知机超参数搜索与联合优化
多层感知机的超参数有：学习率、权重衰减、隐藏层数、每层的隐藏单元数等，构建多个超参数的搜索与联合优化策略具有挑战性。

这是因为超参数之间并不是相互独立的。例如，学习率影响参数更新幅度；批量大小影响梯度估计的噪声；隐藏层数和隐藏单元数决定模型容量；训练轮数过少可能欠拟合，训练轮数过多又可能过拟合。改变其中一个超参数，往往会改变其他超参数的合适取值。

若使用网格搜索，假设每个超参数只尝试 5 个候选值，若有 4 个超参数，则组合数为：
$$
5^4 = 625.
$$
随着超参数数量增加，搜索空间会快速膨胀。因此，简单枚举所有组合通常代价很高。

实际中更常用的策略是**随机搜索**。相比固定网格，随机搜索可以在连续空间中探索更多不同取值，尤其适合学习率、权重衰减等跨数量级变化的超参数。例如：
$$
\eta \sim \operatorname{loguniform}(10^{-4}, 10^{-1}),
$$
$$
\lambda \sim \operatorname{loguniform}(10^{-6}, 10^{-2}).
$$
其中 $\eta$ 表示学习率，$\lambda$ 表示权重衰减系数。

为了降低搜索成本，可以使用“先粗筛，再精训”的思想：
1. 随机采样多组超参数
2. 每组只训练少量轮数，快速淘汰表现较差的配置
3. 保留验证集表现较好的配置，分配更多训练轮数
4. 对最终候选配置进行完整训练

这种思想对应于 Successive Halving、Hyperband、ASHA 等多保真超参数搜索方法。其核心是：不要让明显较差的配置消耗完整训练资源，而是将更多计算分配给更有希望的配置。

对于多层感知机，可以采用如下搜索顺序：
- 先搜索学习率和权重衰减，因为它们直接影响优化过程
- 再搜索隐藏层数和隐藏单元数，因为它们决定模型容量
- 若出现过拟合，再考虑 dropout、权重衰减和早停
- 最后固定较优配置，在验证集上选择表现最好的模型

一个简单的搜索空间可以写成：
$$
\eta \in [10^{-4}, 10^{-1}],
$$
$$
\lambda \in [10^{-6}, 10^{-2}],
$$
$$
h \in \{128, 256, 512, 1024\},
$$
$$
L \in \{1,2,3\}.
$$
其中 $h$ 表示每层隐藏单元数，$L$ 表示隐藏层数。

> 总结：多个超参数搜索的难点在于组合空间大、单次训练代价高、超参数之间存在相互作用。实践中通常先用随机搜索和少量训练轮数进行粗筛，再对较优配置增加训练资源；对于更大规模任务，可以使用 Hyperband、贝叶斯优化或基于早停的自动搜索方法。

