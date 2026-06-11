# #深度学习框架通过自动计算导数，即自动微分（automatic differentiation）来加快求导。 
# #实际中，根据设计好的模型，系统会构建一个计算图（computational graph）， 来跟踪计算是哪些数据通过哪些操作组合起来产生输出。
# #自动微分使系统能够随后反向传播梯度。 这里，反向传播（backpropagate）意味着跟踪整个计算图，填充关于每个参数的偏导数。

# import torch

# #2.5.1 y 是标量
# if 0:
#     x = torch.arange(4.0, requires_grad=True) #requires_grad=True 需要计算关于x的梯度
#     x.grad  # x.grad 需要存储关于x每个分量的偏导数
#     #在我们计算关于的梯度之前，需要一个地方来存储梯度。 重要的是，我们不会在每次对一个参数求导时都分配新的内存。 
#     #因为我们经常会成千上万次地更新相同的参数，每次都分配新的内存可能很快就会将内存耗尽。 
#     #注意，一个标量函数关于向量的梯度是向量；并且与具有相同的形状。

#     y = 2 * torch.dot(x, x)

#     y.backward() #调用反向传播函数来自动计算y关于x每个分量的梯度
#     print(x.grad) #tensor([ 0.,  4.,  8., 12.]) == 4 * x

#     # 计算另一个函数关于x的梯度；在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
#     x.grad.zero_()
#     y = x.sum()
#     y.backward()
#     x.grad

# #2.5.2 y非标量
# if 0:
#     #当y不是标量时，向量y关于向量x的导数的最自然解释是一个矩阵。
#     #当调用向量的反向计算时，我们通常会试图计算一批训练样本中每个组成部分的损失函数的导数。 这里，我们的目的不是计算微分矩阵，而是单独计算批量中每个样本的偏导数之和。

#     x = torch.arange(4.0, requires_grad=True) #requires_grad=True 需要计算关于x的梯度
#     y = x * x

#     # 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
#     # 本例只想求偏导数的和，所以传递一个1的梯度是合适的
#     y.sum().backward() # 等价于y.backward(torch.ones(len(x)))
#     print(x.grad)


# #2.5.3 分离计算：希望将某些计算移动到记录的计算图之外
# #例如，假设y是作为x的函数计算的，而z则是作为y和x的函数计算的。 想象一下，我们想计算z关于x的梯度，但由于某种原因，希望将y视为一个常数， 并且只考虑到x在y被计算后发挥的作用。
# #这里可以分离y来返回一个新变量u，该变量与y具有相同的值， 但丢弃计算图中如何计算y的任何信息。 换句话说，梯度不会向后流经u到x。 因此，下面的反向传播函数计算z=u*x关于x的偏导数，同时将u作为常数处理， 而不是z=x*x*x关于x的偏导数。
# if 0:
#     x = torch.arange(4.0, requires_grad=True) #requires_grad=True 需要计算关于x的梯度
#     y = x * x
#     u = y.detach() #从计算中分离
#     z = u * x

#     z.sum().backward()
#     print(x.grad == u)

#     #由于记录了y的计算结果，我们可以随后在y上调用反向传播， 得到y=x*x关于的x的导数，即2*x。
#     x.grad.zero_()
#     y.sum().backward()
#     print(x.grad == 2 * x)


# #2.5.4 控制流的梯度计算(分段计算)
# if 1:
#     def f(a):
#         b = a * 2
#         while b.norm() < 1000:
#             b = b * 2
#         if b.sum() > 0:
#             c = b
#         else:
#             c = 100 * b
#         return c

#     #即使构建函数的计算图需要通过Python控制流（例如，条件、循环或任意函数调用），我们仍然可以计算得到的变量的梯度。
#     #现在可以分析上面定义的f函数。 请注意，它在其输入a中是分段线性的。 换言之，对于任何a，存在某个常量标量k，使得f(a)=k*a，其中k的值取决于输入a，因此可以用d/a验证梯度是否正确。

#     a = torch.randn(size=(), requires_grad=True)
#     d = f(a)
#     d.backward()
#     print(a.grad == d / a)

