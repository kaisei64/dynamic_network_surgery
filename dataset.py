import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = 'cuda'
dtype = torch.float
criterion = nn.CrossEntropyLoss()

# データの前処理
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# データの読み込み
train_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                             train=True,
                                             transform=transform_train,
                                             download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                            train=False,
                                            transform=transform_test,
                                            download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=256,
                                           shuffle=True,
                                           num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=256,
                                          shuffle=False,
                                          num_workers=2)
