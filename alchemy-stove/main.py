import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, rb=ResBlock, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(rb, 64, 1, stride=1)
        # self.layer2 = self.make_layer(rb, 128, 2, stride=2)
        # self.layer3 = self.make_layer(rb, 256, 2, stride=2)
        self.layer4 = self.make_layer(rb, 32, 1, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channels, stride))
            self.in_channel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.func = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.func(x)


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.flat1 = nn.Flatten(1, 2)
        self.flat2 = nn.Flatten()
        self.lstm = nn.LSTM(32, 32, batch_first=True)
        self.l1 = nn.Linear(3072, 64)
        self.l2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flat1(x)
        x, _ = self.lstm(x)
        x = self.flat2(x)
        x = self.relu(self.l1(x))
        x = self.l2(x)
        return x

# optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
class DataClass:
    def __init__(self):
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                     transform=self.transform_train)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=128, shuffle=True, num_workers=2)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                    transform=self.transform_test)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=100, shuffle=False, num_workers=2)


data = DataClass()


class TrainClass:
    def __init__(self, mc: type):
        self.clses = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.criterion = nn.CrossEntropyLoss()
        self.m = mc()
        self.optimizer = torch.optim.Adam(self.m.parameters())
        self.bs = 128

    def train(self) -> nn.Module:
        global data
        for epoch in range(0, 10):
            print('\nEpoch: %d' % (epoch + 1))
            self.m.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            for i, d in enumerate(data.trainloader, 0):
                # prepare dataset
                length = len(data.trainloader)
                inputs, labels = d
                self.optimizer.zero_grad()

                # forward & backward
                outputs = self.m(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print ac & loss in each batch
                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()
                if i % 10 == 0:
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))

            # get the ac with testdataset in each epoch
            print('Waiting Test...')
            with torch.no_grad():
                correct = 0
                total = 0
                for i in data.testloader:
                    self.m.eval()
                    images, labels = i
                    outputs = self.m(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                print('Test\'s ac is: %.3f%%' % (100 * correct / total))

        return self.m


def main():
    print("FC")
    rs = TrainClass(FC)
    rs.train()
    print("CNN: resnet18")
    rs = TrainClass(ResNet18)
    rs.train()
    print("LSTM")
    rs = TrainClass(LSTM)
    rs.train()



if __name__ == "__main__":
    main()
