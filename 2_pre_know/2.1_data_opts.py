# import torch

# print("hello")
# x = torch.arange(12)
# print(x)
# print(x.shape)          #形状
# print(x.numel())        #元素总数
# X_3x4 = x.reshape(3, 4)
# print(X_3x4)
# zero_2x3x4 = torch.zeros((2, 3, 4))
# print(zero_2x3x4)
# x_3x4_randn = torch.randn(3, 4)  #从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样


#张量连结（concatenate）
# X = torch.arange(12, dtype=torch.float32).reshape((3,4))
# Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# print(torch.cat((X, Y), dim=0)) #只会改变操作的维 dim，其他维不变
# print(torch.cat((X, Y), dim=1))


#索引和切片
#可以用[-1]选择最后一个元素，可以用[1:3]选择第二个和第三个元素
# X[-1] = 9
# X[1:3] = 9
# #[0:2, :]选择第1行和第2行的全部元素（与列无关）
# X[0:2, :] = 12


#节约内存
# X = torch.arange(12, dtype=torch.float32).reshape((3,4))
# Y = torch.arange(12, dtype=torch.float32).reshape((3,4))
# before = id(Y)
# Y = Y + X   #为结果分配新的内存，并将Y重新指向
# print(id(Y) == before)
# before = id(X)
# X += Y      #不分配新的内存
# print(id(X) == before)
# before = id(X)
# X[:] = X + Y     #不分配新的内存
# print(id(X) == before)


#与numpy转换
# X = torch.arange(12, dtype=torch.float32).reshape((3,4))
# A = X.numpy()
# B = torch.tensor(A)
# print(type(A), type(B))
# #与标量转换
# a = torch.tensor([3.5])
# print(a, a.item(), float(a), int(a))
