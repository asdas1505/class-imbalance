import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dset
from torchvision import datasets, transforms

from utils.models import Net
from utils.data_loader import load_dataset
from utils.functions import train_function
from utils.loss import loss_function

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

batch_size = 128
data_root = './datasets'
dataset = 'mnist'


train_loader, test_loader, class_weights = load_dataset(data_root, y_index=0, class_ratio=0.1, dataset=dataset, batch_size=batch_size)

net = Net().to(device)
print(net)

epochs = 10
lr = 2e-3

criterion = loss_function(loss_fn='nll')

optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)

train_function(train_loader, net, optimizer, criterion, epochs, device)

# Test
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.view(inputs.shape[0], -1)

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100*correct / total))

class_correct = [0 for i in range(10)]
class_total = [0 for i in range(10)]

with torch.no_grad():
    for data in test_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs = inputs.view(inputs.shape[0], -1)

        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(10):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %d: %3f' % (i, (class_correct[i]/class_total[i])))