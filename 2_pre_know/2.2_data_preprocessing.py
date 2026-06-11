#读取数据集
# import os

# os.makedirs(os.path.join('.', 'data'), exist_ok=True)
# data_file = os.path.join('.', 'data', 'house_tiny.csv')
# with open(data_file, 'w') as f:
#     f.write('NumRooms,Alley,Price\n')  # 列名
#     f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
#     f.write('2,NA,106000\n')
#     f.write('4,NA,178100\n')
#     f.write('NA,NA,140000\n')

# import pandas as pd
# data = pd.read_csv(data_file)
# print(data)

#处理缺失值
# inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# inputs = inputs.fillna(inputs.mean(numeric_only=True)) #使用 fillna 插值, 仅在数据类型为数值的列进行平均值插值
# print(inputs)
# inputs = pd.get_dummies(inputs, dummy_na=True) #Pave 2 “Alley_Pave”和“Alley_nan" #dummy_na=True把NA作为离散值来处理
# print(inputs)


#转换为张量格式
# import torch

# X = torch.tensor(inputs.to_numpy(dtype=float))
# y = torch.tensor(outputs.to_numpy(dtype=float))
# print(X, y)