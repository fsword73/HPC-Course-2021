import torch
import matplotlib.pyplot as plt
import numpy as np

x = torch.unsqueeze(torch.linspace(-3.1415926, 3.1415926, 200), dim=1)  
# x data (tensor), shape=(100, 1)
y = torch.sin(x) + 0.15*torch.rand(x.size())              
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()


import torch.nn.functional as F
class Net(torch.nn.Module): 
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__() 
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    def forward(self, x): 
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数
        x = self.predict(x)             # 输出值
        return x
my_neural_net = Net(n_feature=1, n_hidden=10, n_output=1)
print(my_neural_net)  # net 的结构

# optimizer 是训练的工具
optimizer = torch.optim.SGD(my_neural_net.parameters(), lr=0.2)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)

optimizer2 = torch.optim.SGD(my_neural_net.parameters(), lr=0.02) 
optimizer1 = torch.optim.SGD(my_neural_net.parameters(), lr=0.1) 

for t in range(300):
    prediction = my_neural_net(x)     # 喂给 net 训练数据 x, 输出预测值

    loss = loss_func(prediction, y)     # 计算两者的误差

    if t >=200:
        optimizer2.zero_grad()
    elif t >100 :
        optimizer1.zero_grad()   # 清空上一步的残余更新参数值
    else:
        optimizer.zero_grad()
    loss.backward()         # 误差反向传播, 计算参数更新值
    if t >=200:
        optimizer2.step()        # 将参数更新值施加到 net 的 parameters 上
    elif t>100:
        optimizer1.step()
    else:
        optimizer.step()
    if t % 1 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.title('step %d' % t)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
    if t > 0 and  t % 299  == 0:
        plt.show()
