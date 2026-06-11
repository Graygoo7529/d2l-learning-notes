# import torch
#按照维数聚合
# A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# A_sum_axis0 = A.sum(axis=0)
# print(A_sum_axis0)
# print(A.sum(axis=0) / A.shape[0]) #按行求均值，等效于 A.mean(axis=0)

#保持轴数
# A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# sum_A_nokeep = A.sum(axis=1)
# sum_A_keep = A.sum(axis=1, keepdims=True)
# print(sum_A_nokeep, sum_A_nokeep.shape)
# print(sum_A_keep, sum_A_keep.shape)
# print(A / sum_A_keep)
# print(A.cumsum(axis=0))

#点积（Dot Product）
# x = torch.arange(4, dtype=torch.float32)
# y = torch.ones(4, dtype = torch.float32)
# torch.dot(x, y)

#矩阵-向量积
# A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# x = torch.arange(4, dtype=torch.float32)
# print(torch.mv(A, x))

#矩阵-矩阵乘法
# A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# B = torch.ones(4, 3)
# print(torch.mm(A, B))

#范数
# u = torch.tensor([3.0, -4.0])
# torch.norm(u) #L2
# torch.abs(u).sum() #L1
# torch.norm(torch.ones((4, 9))) #矩阵L2(Frobenius norm)