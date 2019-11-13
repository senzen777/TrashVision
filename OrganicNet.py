import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import cv2

# HYPERPARAMETERS
train_batch_size = 64
test_batch_size = 64
train_validation_split = 4/5
log_interval = 20
lr = 0.01
momentum = 0.9
num_epochs = 3

def prepare_data():
    training_path = 'data/Kaggle_data/TRAIN/'
    # use the RandomCrop to make all images the same size
    # this gave much better results than transforms.Resize()
    trainval_set = torchvision.datasets.ImageFolder(root=training_path, transform=transforms.Compose([transforms.RandomCrop([256,256],pad_if_needed=True), transforms.ToTensor()]))
    trainval_size = len(trainval_set)
    train_size = int(trainval_size * train_validation_split)

    train_set, val_set = torch.utils.data.random_split(trainval_set, [train_size, trainval_size-train_size])

    train_loader = torch.utils.data.DataLoader(train_set,batch_size=train_batch_size,num_workers=0,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size=test_batch_size,num_workers=0,shuffle=True)

    testing_path = 'data/Kaggle_data/TEST/'
    test_set = torchvision.datasets.ImageFolder(root=testing_path,transform=torchvision.transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=test_batch_size,num_workers=0,shuffle=True)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5) # read in color image
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc3 = nn.Linear(61*61*16,10)

        self._initialize_weights()

    def _initialize_weights(self):
        pass

    def forward(self, x):
        # start with 256 x 256 x 3 image
        x = F.relu(self.conv1(x))
        # have 252 x 252 x 6 data
        x = F.max_pool2d(x, 2, stride=2)
        # have 126 x 126 x 6 data
        x = F.relu(self.conv2(x))
        # have 122 x 122 x 16 data
        x = F.max_pool2d(x, 2, stride=2)
        # have 61 x 61 x 16 data
        x = torch.flatten(x, start_dim=1)
        # have 59536 x 1 data
        x = self.fc3(x)
        return x

def train(model, criterion, train_loader, optimizer, device):
    model.train() # set the mode of the model to training
    total_loss = 0
    for i, data in enumerate(train_loader):
        imgs, lbls = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i+1) % log_interval == 0:
            mean_loss = total_loss / log_interval
            print('\tbatch {:4d}: loss={:.3f}'.format(i+1, mean_loss))
            total_loss = 0.

def test(model, test_loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for data in test_loader:
            imgs, lbls = data[0].to(device), data[1].to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs.data, 1)
            total += lbls.shape[0]
            correct += (preds == lbls).sum().item()

    acc = correct / total
    print('\tacc={:.3f}'.format(acc))

def main():
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    print('Using {}'.format(device))

    train_set, val_set, test_set, train_loader, val_loader, test_loader = prepare_data()
    print('Training set size: {}'.format(len(train_set)))
    print('Validation set size: {}'.format(len(val_set)))
    print('Test set size: {}'.format(len(test_set)))

    model = Net()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr = lr,
        momentum = momentum
    )

    for e in range(num_epochs):
        print('Training epoch {}'.format(e))
        train(model, criterion, train_loader, optimizer, device)
        print('Testing on validation set')
        test(model, val_loader, device)



if __name__ == '__main__':
    main()
