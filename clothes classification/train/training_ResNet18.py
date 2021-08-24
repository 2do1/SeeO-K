'''
< training_ResNet18>
labels = ['가방', '긴 겉옷', '긴팔 셔츠', '긴팔 원피스', '긴팔 티셔츠', '긴팔 후드티', '나시', '나시 원피스', '반바지',
          '반팔 셔츠', '반팔 원피스', '반팔 티셔츠', '샌들', '스니커즈', '슬림핏바지', '앵클부츠', '일자핏바지', '짧은 겉옷', '치마']
- batch size:100
- epoch:30
- lr:0.001
'''

from torch.autograd import Variable
import time
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torchvision.transforms as transforms

PATH = './'

transf = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5), (0.5))])

train_data = torchvision.datasets.ImageFolder(root='./data', transform=transf)
test_data = torchvision.datasets.ImageFolder(root='./test_data', transform=transf)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

print('train data samples 수 : ', len(train_data))
print('각 sample의 길이 : ', len(train_data[0]))     # 각 sample은 (data, label) 꼴의 tuple로 구성되어 있다.
print('각 sample의 shape와 dtype : ', train_data[0][0].shape, train_data[0][0].dtype)

print('test data samples 수 : ', len(test_data))
print('각 sample의 길이 : ', len(test_data[0]))     # 각 sample은 (data, label) 꼴의 tuple로 구성되어 있다.
print('각 sample의 shape와 dtype : ', test_data[0][0].shape, test_data[0][0].dtype)


num_classes = 19
epochs = 30

batch_images, batch_labels = next(iter(train_loader))

print(batch_images.shape, type(batch_images), batch_images[0].dtype)
print(batch_labels.shape, type(batch_labels), batch_labels[0].dtype)

model = models.resnet18(pretrained=True)

# 출력이 클래스 개수인 linear layer 추가
num_ftrs = model.fc.in_features
model.fx = nn.Linear(num_ftrs, num_classes)

critertion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

count = 0

loss_list = []
iteration_list = []
accuracy_list = []

prediction_list = []

print('[ResNet18] class:19, batch size:100, lr:0.001, epoch:30')
for epoch in range(epochs):
    start = time.time()
    avg_loss = 0
    total_batch = len(train_loader)

    for i, (images, labels) in enumerate(train_loader):
        images2 = torch.cat([images,images], dim=1)
        images3 = torch.cat([images, images2], dim=1)
        #print(images3.shape)

        train = Variable(images3)
        labels = Variable(labels)

        # Forward pass
        out = model(train)
        loss = critertion(out, labels)

        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss / total_batch
        total = labels.size(0)
        preds = torch.max(out.data, 1)[1]
        correct = (preds == labels).sum().item()

    print("[Epoch: {:>4}] \t loss = {:.4f} \t time = {:.4f}".format(
        epoch + 1, avg_loss.data, time.time() - start))

### test accuracy
model.eval()

input_list = []
predict_output_list = []
labels_list = []

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        labels_list.append(labels)

        images2 = torch.cat([images,images], dim=1)
        images3 = torch.cat([images, images2], dim=1)

        out = model(images3)

        input_list.append(images)
        predict_output_list.append(out)

        preds = torch.max(out.data, 1)[1]

        total += len(labels)
        correct += (preds == labels).sum().item()

    print('Accuracy: ', 100. * correct / total)

torch.save(model.state_dict(), PATH + 'ResNet18.pt')