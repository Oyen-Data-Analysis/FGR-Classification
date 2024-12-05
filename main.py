from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from config import *
from dataset import PlacentaDataset
import torch.optim as optim
from model import Net
from utils.util_losses import *

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def scrape_dir():
    data = []
    labels = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".mha"):
                projected_segmented_path = os.path.join(OUTPUT_DIR, file.replace(".mha", "_segmented.jpg"))
                if os.path.exists(projected_segmented_path):
                    data.append(projected_segmented_path)
                    if "Control" in root:
                        labels.append(0)
                    elif "FGR" in root:
                        labels.append(1)
    return data, labels

raw_data, labels = scrape_dir()
dataset = PlacentaDataset(raw_data, labels)
train_size = int(0.8*dataset.__len__())
test_size = dataset.__len__() - train_size
print(train_size, test_size)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Convert inputs to float
        inputs = inputs.float()
        inputs = inputs.unsqueeze(1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 20 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
            running_loss = 0.0

print('Finished Training')

torch.save(net.state_dict(), MODEL_SAVE_PATH + '/cel.pth')

correct = 0
total = 0
all_labels = []
all_predictions = []
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.float()
        images = images.unsqueeze(1)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels)
        all_predictions.extend(predicted)
        for prediction, label in zip(predicted, labels):
            print(f'Prediction: {prediction}, Label: {label}')
    TN, FP, FN, TP= confusion_matrix(all_labels, all_predictions).ravel() # Only works for binary classification!!!
    print(f'TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}')

print(f'Accuracy of the network on {total} test images: {100 * correct / total}%')