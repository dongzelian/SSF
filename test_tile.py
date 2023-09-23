import time
import torch
import numpy as np
device=torch.device("cuda:0")
a=torch.rand((10000,10000)).to(device)
b=torch.rand((10000,10000)).to(device)
loop=200
start=time.time()
for i in range(loop):
    c=torch.matmul(a,b)
cost_time=time.time()-start
# print("c=a*b: a:{}, b:{}, c:{}".format(a.size(),b.size(),c.size()))
# print("cost time:{}".format(cost_time))
# print("cost time per loop:{}".format(cost_time/loop))
rep=10
cost_list=np.zeros((rep,))
for j in range(rep):
    start=time.time()
    for i in range(loop):
        c=torch.matmul(a,b)
    cost_time=time.time()-start
    cost_list[j]=cost_time
# print("c=a*b: a:{}, b:{}, c:{}".format(a.size(),b.size(),c.size()))
# print num with 3 lefted
print("cost time ave:{:.3f}\tdiv:{:.3f}.".format(cost_list.mean()*1000,cost_list.std()*1000))
# print("cost time per loop:{}".format(cost_time/loop))

# random choose 50% cols and rows in a and b
ind=torch.randperm(10000)[:5000].to(device)
a[ind,:]=0
# a[:,ind]=0
b[ind,:]=0
# b[:,ind]=0
cost_list=np.zeros((rep,))
for j in range(rep):
    start=time.time()
    for i in range(loop):
        c=torch.matmul(a,b)
    cost_time=time.time()-start
    cost_list[j]=cost_time
# print("c=a*b: a:{}, b:{}, c:{}".format(a.size(),b.size(),c.size()))
print("cost time ave:{:.3f}\tdiv:{:.3f}.".format(cost_list.mean()*1000,cost_list.std()*1000))
# print("cost time per loop:{}".format(cost_time/loop))