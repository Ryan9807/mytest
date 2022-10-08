import torch
from torch.serialization import load
import torchvision.datasets as datasets
import torchvision.transforms as tansformes
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
train_data = r'F:\mymmsegmentation\DIBCO-BIN'
transform = (transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
train_dataset = datasets.ImageFolder(train_data,transform)
train_loader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)

def get_mean_std(loader):
    # Var[x] = E[X**2]-E[X]**2
    channels_sum,channels_squared_sum,num_batches = 0,0,0
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    print(num_batches)
    print(channels_sum)
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2) **0.5

    return mean,std

mean,std = get_mean_std(train_loader)

print(mean)
print(std)
