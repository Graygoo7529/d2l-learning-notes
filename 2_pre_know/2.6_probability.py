# import torch
# from torch.distributions import multinomial
# from d2l import torch as d2l
# import matplotlib.pyplot as plt


# #2.6.1 投骰子
# if 1:
#     #为了抽取一个样本，即掷骰子，我们只需传入一个概率向量。 
#     #输出是另一个相同长度的向量：它在索引处的值是采样结果中出现的次数。

#     fair_probs = torch.ones([6]) / 6
#     sample_num = 1000
#     counts = multinomial.Multinomial(sample_num, fair_probs).sample() #每个数字被投中了多少次
#     counts / sample_num  #每个数字被投中的相对频率作为概率的估计值
#     print(counts / sample_num) # each ~= 1/6

#     #绘图看概率的估计值如何随着时间的推移收敛到真实概率。 
#     #让我们进行500组实验，每组抽取10个样本。
#     counts = multinomial.Multinomial(10, fair_probs).sample((500,))
#     cum_counts = counts.cumsum(dim=0)
#     estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

#     d2l.set_figsize((6, 4.5))
#     for i in range(6):
#         d2l.plt.plot(estimates[:, i].numpy(),
#                     label=("P(die=" + str(i + 1) + ")"))
#     d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
#     d2l.plt.gca().set_xlabel('Groups of experiments')
#     d2l.plt.gca().set_ylabel('Estimated probability')
#     d2l.plt.legend()
#     d2l.plt.show()


