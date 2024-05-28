# from common.public import public # type: ignore

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from efficient_kan.mlp import MLP

# Load data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input = pd.read_csv('./data/train_input.csv', header=None)
target = pd.read_csv('./data/train_target.csv', header=None)
input_arr = np.float32(np.array(input))
target_arr = np.float32(np.array(target))
train_input, val_input, train_target, val_target = train_test_split(input_arr, target_arr,
                                                                    test_size=0.2,
                                                                    random_state=42)
train_input_tensor = torch.tensor(train_input)
train_target_tensor = torch.tensor(train_target)
val_input_tensor = torch.tensor(val_input)
val_target_tensor = torch.tensor(val_target)

# print(train_data.shape, val_data.shape, train_targets.shape, val_targets.shape)
train_dataset = torch.utils.data.TensorDataset(train_input_tensor, train_target_tensor)
val_dataset = torch.utils.data.TensorDataset(val_input_tensor, val_target_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define model
model = MLP(51, 100, 4)
model.to(device)
# Define optimizer
# optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# Define learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

# Define loss
# loss_func = nn.CrossEntropyLoss()
loss_func = nn.MSELoss()
train_loss_all = []
val_loss_all = []
losses = []
for epoch in range(30):
    # Train
    train_loss = 0
    train_num = 0
    model.train()
    with tqdm(train_loader) as pbar:
        running_loss = 0.0
        for i, (input, target) in enumerate(pbar):
            input = input.view(-1, 51).to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = loss_func(output, target.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*input.size(0)
            train_num += input.size(0)
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
        for input, target in val_loader:
            input = input.view(-1, 51).to(device)
            output = model(input)
            val_loss += loss_func(output, target.to(device)).item()*input.size(0)
            val_num += input.size(0)
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
