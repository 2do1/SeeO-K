import matplotlib.pyplot as plt
import ipywidgets as widgets
from torch.autograd import Variable
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

transf = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5), (0.5))])

train_data = torchvision.datasets.ImageFolder(root='./data', train=True, transform=transf)
test_data = torchvision.datasets.ImageFolder(root='./test_data', train=False, transform=transf)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

print('train data samples 수 : ', len(train_data))
print('각 sample의 길이 : ', len(train_data[0]))     # 각 sample은 (data, label) 꼴의 tuple로 구성되어 있다.
print('각 sample의 shape와 dtype : ', train_data[0][0].shape, train_data[0][0].dtype)

print('test data samples 수 : ', len(test_data))
print('각 sample의 길이 : ', len(test_data[0]))     # 각 sample은 (data, label) 꼴의 tuple로 구성되어 있다.
print('각 sample의 shape와 dtype : ', test_data[0][0].shape, test_data[0][0].dtype)



num_classes = 19
epochs = 20

batch_images, batch_labels = next(iter(train_loader))

print(batch_images.shape, type(batch_images), batch_images[0].dtype)
print(batch_labels.shape, type(batch_labels), batch_labels[0].dtype)

# convolution neural network
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.dropout = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=19)
        self.fc3 = nn.Linear(in_features=120, out_features=19)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # x = self.fc3(x)
        # x = self.softmax(x)

        return x


# model training
model = ConvNet()

critertion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

count = 0

loss_list = []
iteration_list = []
accuracy_list = []

prediction_list = []

for epoch in range(epochs):
    start = time.time()
    avg_loss = 0
    total_batch = len(train_loader)

    for i, (images, labels) in enumerate(train_loader):

        train = Variable(images.view(-1, 1, 28, 28))
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

        out = model(images)

        input_list.append(images)
        predict_output_list.append(out)

        preds = torch.max(out.data, 1)[1]

        total += len(labels)
        correct += (preds == labels).sum().item()

    print('Accuracy: ', 100. * correct / total)

### test UI
labels = ['가방', '긴 겉옷', '긴팔 셔츠', '긴팔 원피스', '긴팔 티셔츠', '긴팔 후드티', '나시', '나시 원피스', '반바지',
          '반팔 셔츠', '반팔 원피스', '반팔 티셔츠', '샌들', '스니커즈', '슬림핏바지', '앵클부츠', '일자핏바지', '짧은 겉옷', '치마']


def io_imshow(batch_idx, idx):
  batch_label = labels_list[batch_idx]
  print('real label:', labels[batch_label[idx]])
  plt.subplot(121)
  batch_image = input_list[batch_idx]
  plt.imshow(batch_image[idx].reshape(28,28).cpu(), cmap='gray')
  batch_output = predict_output_list[batch_idx]
  predicted_output = batch_output[idx].tolist()
  predicted_idx = predicted_output.index(max(predicted_output))
  predicted_label = labels[predicted_idx]
  print('predicted output:', predicted_label)
  plt.show()

widgets.interact(io_imshow, batch_idx=widgets.IntSlider(min=0, max=len(labels_list)-1, continuous_update=False), idx=widgets.IntSlider(min=0, max=len(labels_list[0])-1, continuous_update=False))


# save model
PATH = './'

torch.save(model, PATH + 'model.pt')