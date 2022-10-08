import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader

batch_size = 64
train_data = r'F:\mymmsegmentation\DIBCO-BIN'
# 训练集(以CIFAR-10数据集为例)  transforms.Compose([transforms.Resize((224,224))]),transform=transforms.ToTensor()
transform = (transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
# train_dataset = datasets.ImageFolder(root='F:\mymmsegmentation\DIBCO-BIN',transform=transforms.ToTensor())
train_dataset = datasets.ImageFolder(train_data,transform)
train_loader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size)



def get_mean_std_value(loader):
    '''
    求数据集的均值和标准差
    :param loader:
    :return:
    '''
    data_sum,data_squared_sum,num_batches = 0,0,0

    for data,_ in loader:
        # data: [batch_size,channels,height,width]
        # 计算dim=0,2,3维度的均值和，dim=1为通道数量，不用参与计算
        data_sum += torch.mean(data,dim=[0,2,3])    # [batch_size,channels,height,width]
        # 计算dim=0,2,3维度的平方均值和，dim=1为通道数量，不用参与计算
        data_squared_sum += torch.mean(data**2,dim=[0,2,3])  # [batch_size,channels,height,width]
        # 统计batch的数量
        num_batches += 1
    # 计算均值
    mean = data_sum/num_batches
    # 计算标准差
    std = (data_squared_sum/num_batches - mean**2)**0.5
    return mean,std

mean,std = get_mean_std_value(train_loader)
print('mean = {},std = {}'.format(mean,std))
