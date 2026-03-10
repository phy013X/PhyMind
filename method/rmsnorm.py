# _*_ coding : utf-8 _*_
# @Time : 2026/3/10 19:11
# @Author : phy013x
# @File : rmsnorm

import torch

# 张量开方取倒数
t1 = torch.rsqrt(torch.tensor([1.0, 4.0]))
print(t1)

# 全1张量
t2 = torch.ones((1, 3))
print(t2)



