# from common.public import public # type: ignore

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import csv
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from efficient_kan import KAN

# Load data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

data = pd.read_csv('./data/inputs.csv', header=None)
targets = pd.read_csv('./data/targets.csv', header=None)
data_arr = np.array(data)
data_arr = np.float32(data_arr)
# print(data_arr.dtype)
targets_arr = np.array(targets)
targets_arr = np.float32(targets_arr)
# print(data.shape)
# print(targets.shape)
train_data, val_data, train_targets, val_targets = train_test_split(data_arr, targets_arr,
                                                                    test_size=0.2,
                                                                    random_state=42)
train_data_tensor = torch.tensor(train_data)
train_targets_tensor = torch.tensor(train_targets)
val_data_tensor = torch.tensor(val_data)
val_targets_tensor = torch.tensor(val_targets)

# print(train_data.shape, val_data.shape, train_targets.shape, val_targets.shape)
train_dataset = torch.utils.data.TensorDataset(train_data_tensor, train_targets_tensor)
val_dataset = torch.utils.data.TensorDataset(val_data_tensor, val_targets_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define model
model = KAN([51, 64, 4])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Define optimizer
# optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_func = nn.MSELoss()
train_loss_all = []
val_loss_all = []
# Define learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

# Define loss
criterion = nn.CrossEntropyLoss()
losses = []
for epoch in range(20):
    # Train
    train_loss = 0
    train_num = 0
    model.train()
    with tqdm(train_loader) as pbar:
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(pbar):
            inputs = inputs.view(-1, 51).to(device)
            # print(images.dtype)
            # print(labels.shape)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(output.shape)
            # print(labels.shape)
            # loss = mae(output, labels.to(device))
            # loss.backward()
            # optimizer.step()
            # accuracy = (output == labels.to(device)).float().mean()
            # pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

            loss = loss_func(outputs, targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*inputs.size(0)
            train_num += inputs.size(0)


            # print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))
            pbar.set_postfix(lr=optimizer.param_groups[0]['lr'])
    train_loss_all.append(train_loss/train_num)
    pbar.set_postfix(loss=train_loss/train_num)

    # Validation
    model.eval()
    val_loss = 0
    val_accuracy = 0
    val_num = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.view(-1, 51).to(device)
            outputs = model(inputs)
            val_loss += loss_func(outputs, labels.to(device)).item()*inputs.size(0)
            val_num += inputs.size(0)

            # val_accuracy += (
            #     (output == labels.to(device)).float().mean().item()
            # )
    val_loss_all.append(val_loss/val_num)
    # val_accuracy /= len(valloader)

    # Update learning rate
    scheduler.step()

    print(
        f"Epoch {epoch + 1}, Val Loss: {val_loss/val_num}"
    )

# Save the trained model
torch.save(model.state_dict(), "./model/kan.pth")

# Plot the loss values against the number of epochs
fig, ax = plt.subplots()
ax.plot(range(1, epoch + 2), train_loss_all, label='Train Loss')
ax.plot(range(1, epoch + 2), val_loss_all, label='Val Loss')
ax.set_title('Loss Curves')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend()
plt.show()
