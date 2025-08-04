import numpy as np
import torch
import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
#张量相乘，对应数相乘
z=A*B
print(A)
print(z)
print(z.shape)
print(z.sum())
y=torch.ones(4)

print(x+y)
A_sum_axis0 = A.sum(axis=0)
print(torch.dot(A_sum_axis0,y))
print(A_sum_axis0)
print(A_sum_axis0.shape)
print(A.mean())
print(A.mean(axis=0))