# _*_ coding : utf-8 _*_
# @Time : 2026/3/10 20:49
# @Author : phy013x
# @File : rope.py

import torch

# torch版本的三元运算符
# x = torch.tensor([1, 2, 3, 4, 5, 6])
# y = torch.tensor([0, 10, 20, 30, 40, 50])
# condition = torch.logical_and(x > 3, y > 3)
# print(torch.where(condition, x, y))

# 枚举 begin end step
# t = torch.arange(0, 10, 2)
# print(t)

# 外积
# v1 = torch.tensor([1, 2, 3])
# v2 = torch.tensor([4, 5, 6])
# print(torch.outer(v1, v2))

# 拼接 cat

# 在第dim维前扩展1维度
t1 = torch.tensor([1, 2, 3])
print(t1.shape)
print(t1.unsqueeze(0).shape)





