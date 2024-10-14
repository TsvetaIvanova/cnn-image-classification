# -*- coding: utf-8 -*
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, transforms
import time
from PIL import Image
from scipy.io import loadmat
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torchsummary import summary
from torchvision.transforms import v2
from collections import Counter

""" just confirming the torch version as the requirement is abover 2.2.2 """
print(torch.__version__)

"""where cuda gpu is available, run cuda"""
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"The neural network is currently training on a {device} device")

"""data augmentation and normalization for the training set""" 
data_train_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(50),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

"""transforms for the validation set, without augmentation to simulate testing period"""
data_val_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

"""transforms for the test set"""
data_test_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

"""loading the dataset by torchvision.datatasets """
train_set = datasets.Flowers102(root='data', split='train', download=True, transform=data_train_transform)
test_set = datasets.Flowers102(root='data', split='test', download=True, transform=data_test_transform)
val_set = datasets.Flowers102(root='data', split='val', download=True, transform=data_val_transform)


"""data loaders needed with PyTorch, defining the batch sizes, bigger batch sizes are better"""
train_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
val_loader = DataLoader(dataset=val_set, batch_size=16, shuffle=False)

"""checking the images after the transforms in the train set containing the augmentation"""
def show_image(dataset, index=0):
    image, class_flower = dataset[index]
    image = image.clamp(0, 1)
    plt.imshow(image.permute(1, 2, 0))
    plt.title(f'Flower class: {class_flower}')
    plt.show()
show_image(train_set)


"""creating the convolutional neural network __init__ and forward function"""
class ConvNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 14 * 14, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 102)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        # every adjustment to the tensors needs an adjustment in the fc1 input and the flatten
        #print(x.shape)
        x = x.view(-1, 512 * 14 * 14)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

""" initializing lists to track the accuracies and losses in a table"""
training_accuracies = []
training_losses = []
validation_accuracies = []
validation_losses = []
test_accuracies = []

"""the train function"""
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == y).type(torch.float).sum().item()
        total += y.size(0)

    average_loss = total_loss / len(dataloader)
    accuracy = correct / total * 100
    training_losses.append(average_loss)
    training_accuracies.append(accuracy)
    print(f"Training Loss: {average_loss:.4f}, Training Accuracy: {accuracy:.2f}%")

"""the test, which is my validation function"""
def test(dataloader, model, loss_fn, label="Validation"):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = loss_fn(outputs, y)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == y).type(torch.float).sum().item()
            total += y.size(0)

    average_loss = total_loss / len(dataloader)
    accuracy = correct / total * 100
    if label == "Validation":
        validation_losses.append(average_loss)
        validation_accuracies.append(accuracy)
    print(f"{label} Loss: {average_loss:.4f}, {label} Accuracy: {accuracy:.2f}%")
    return average_loss

"""initializing the model and outputting its shape"""
model = ConvNeuralNetwork().to(device)
summary(model, (3, 224, 224))
print(model)

"""defining the loss function, optimizer, and learning rate scheduler"""
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


""" creating a table to track accuracies and loss"""
def create_and_save_dataframe():
    if len(training_accuracies) == len(validation_accuracies):
        df_training = pd.DataFrame({
            'Epoch': range(1, epochs+1),
            'Training Accuracy': training_accuracies,
            'Training Loss': training_losses,
            'Validation Accuracy': validation_accuracies,
            'Validation Loss': validation_losses
        })
        df_training.to_csv('training_validation_metrics.csv', index=False)
        print(df_training)
    else:
        print("maybe I have some differences in column lengths?")

""" Training and evaluation loop"""
epochs = 1000
for t in range(epochs):
    print(f"Epoch {t+1},\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    val_loss = test(val_loader, model, loss_fn)
    scheduler.step(val_loss)

    # Checkpointing every 100 epochs
    if t % 100 == 0 and t != 0:
        torch.save(model.state_dict(), f'model_epoch_{t}.pth')
        print(f'Checkpoint epoch {t}')

print("This marks the end of training the model, now commences the eval stage on the test data")
create_and_save_dataframe()

# saving the final trained model
torch.save(model.state_dict(), "new_trained_model.pth")
print("Saved PyTorch Model State to new_trained_model.pth")

# load and evaluate the model on the test data
model = ConvNeuralNetwork().to(device)
model.load_state_dict(torch.load("new_trained_model.pth", map_location=torch.device(device)))
model.to(device)

# Set the model to evaluation mode
model.eval()

""" Evaluation loop"""
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct / total} %')
