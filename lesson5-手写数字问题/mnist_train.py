import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision
import matplotlib.pyplot as plt
from util import plot_curve,plot_image,one_hot
"""
MNIST手写数字识别数据：
each number owns 7000 images
train/test spliting:60k vs 10k
28*28 = 784
x = [1,2,3,4,....784]

"""
batchsize = 512
# 1.step1 加载并处理数据(Dataloader)
# 1.1 定义数据预处理方式
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,),(0.3081,))]
)
"""
transforms.Compose()是把多种数据处理的方法集合在一起
transforms.Totensor()是将需要被处理的数据转换为Tensor类型
transforms.Normalize((0.1307),(0.3081))
是将数据被转化为Tensor类型后，对其减均值(0.1307)和除方差(0.3081)以
实现数据的正则化
"""
# 1.2定义Dataset和Dataloader
# Data set
train_dataset = torchvision.datasets.MNIST('minist_data',train=True,transform=transform,download=False)
test_dataset = torchvision.datasets.MNIST('minist_data/',train=False,transform=transform,download=False)
# Dataloader
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = batchsize,shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = batchsize,shuffle = False)

x,y = next(iter(train_loader))
print(x.shape,y.shape,x.min(),x.max())
plot_image(x,y,'image sample')

# step2.定义网络结构的相关代码
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
#         xw+b
        self.fc1 = nn.Linear(28*28,256)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64,10)
    def forward(self, x):
#         x:[b,1,28,28]
#         h1 = relu(xw+b)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        output = self.fc3(h2)
        return output
# 对网络进行实例化
net = Net()


train_loss = []
#定义损失函数和优化器
# [w1,b1,w2,b2,w3,b3,...]net.parameters()
optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9)
#3.training
for epoch in range(3):
    for batch_idx,(x,y) in enumerate(train_loader):
# x:torch.Size([512, 1, 28, 28]) y:torch.Size([512])
#         print(x.shape,y.shape)
# [b,1,28,28]->[b,feature]
        x = x.view(x.size(0),28*28)
        # [b,28*28->b,10]

        """
        Forward pass
        """
        out = net(x)
        y_one_hot = one_hot(y)
        # loss = mse(out,y_one_hot)
        loss = F.mse_loss(out,y_one_hot)
        train_loss.append(loss.item())
        # Backward and optimizer
        # 1.优化器保存先前的梯度信息
        optimizer.zero_grad()
        # 2.计算梯度
        loss.backward()
#         3.梯度更新 w' = w - lr*grads
        optimizer.step()
        if batch_idx%10 == 0:
            print(epoch,batch_idx,loss.item())
plot_curve(train_loss)

total_correct = 0
with torch.no_grad():
    for x,y in test_loader:
        x = x.view(x.size(0),28*28)
        out = net(x)
        # out:[b,10]->pred:[b]
        preds = out.argmax(dim = 1)
        correct = preds.eq(y).sum().float().item()
        total_correct+=correct

    total_num = len(test_loader.dataset)
    acc = total_correct/float(total_num)
    print('Accuracy of the network on the 10000 test images: {} %'.format(100*acc))