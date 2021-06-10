import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

DATA_ROOT = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\data'

transform = transforms.Compose(
    [transforms.ToTensor(),
     # transforms.Resize([784]),
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
     transforms.Normalize((0.5), (0.5))])

batch_size = 4096

trainset = torchvision.datasets.MNIST(root=DATA_ROOT, train=True,
                                      download=True, transform=transform)

x, y = trainset.data, trainset.train_labels
# x = x.reshape([-1, 784])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

# testset = torchvision.datasets.MNIST(root=DATA_ROOT, train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

classes = [str(i) for i in range(10)]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        self.fc1 = nn.Linear(28 * 28, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        # x = F.softmax(x)
        return x


net = Net()

import torch.optim as optim

# criterion = nn.CrossEntropyLoss()
# criterion = nn.NLLLoss(reduction='mean')
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(net.parameters(), lr=1)

# Train
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    pbar = tqdm(enumerate(trainloader, 0))
    for i, data in pbar:
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.reshape(-1, 784)
        labels = F.one_hot(labels)
        labels = labels.float()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(input=outputs, target=labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        a_labels = torch.argmax(labels, axis=1)
        accuracy = torch.mean(
            torch.as_tensor(a_labels == torch.argmax(outputs, axis=1), dtype=torch.float)).numpy()

        pbar.desc = '[%d, %5d] loss: %.3f,accuracy %.f' % (epoch + 1, i + 1, loss.item(), accuracy * 100)

        running_loss = 0.0

    full_labels = trainset.train_labels

    # full_x = trainset.train_data
    # full_x = torch.as_tensor(full_x.reshape([60000, 784]),dtype=torch.float)
    #
    #
    # full_output = net(full_x)

print('Finished Training')
