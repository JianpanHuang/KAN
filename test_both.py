import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from efficient_kan import KAN
from efficient_kan.mlp import MLP

# Load data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input = pd.read_csv('./data/test_input.csv', header=None)
target = pd.read_csv('./data/test_target.csv', header=None)
x = pd.read_csv('./data/x.csv', header=None)
input_tensor = torch.Tensor(np.float32(np.array(input)))
target_tensor = torch.Tensor(np.float32(np.array(target)))
x = np.float32(np.array(x))
# print(input_arr)
# print(target_arr)

# Load MLP model
model_mlp = MLP(51, 100, 4)
model_mlp.to(device)
model_mlp.load_state_dict(torch.load('./model/mlp.pth'))
output_mlp = model_mlp(input_tensor)
output_mlp_np = output_mlp.detach().numpy()
# print(output_kan)

# Load KAN model
model_kan = KAN([51, 100, 4])
model_kan.to(device)
model_kan.load_state_dict(torch.load('./model/kan.pth'))
output_kan = model_kan(input_tensor)
output_kan_np = output_kan.detach().numpy()
print(output_kan_np)


# Plot
y_target = target_tensor[0][0]*np.sin(target_tensor[0][1]*x)+target_tensor[0][2]*np.cos(target_tensor[0][3]*x)
y_mlp = output_mlp_np[0][0]*np.sin(output_mlp_np[0][1]*x)+output_mlp_np[0][2]*np.cos(output_mlp_np[0][3]*x)
y_kan = output_kan_np[0][0]*np.sin(output_kan_np[0][1]*x)+output_kan_np[0][2]*np.cos(output_kan_np[0][3]*x)
plt.plot(x,y_target,'o-',color='b',label="GT")
plt.plot(x,y_mlp,'o-',color='g',label="MLP")
plt.plot(x,y_kan,'o-',color='r',label="KAN")
plt.xlabel("x",fontsize=16)
plt.ylabel("y",fontsize=16)
plt.title("y = a*sin(b*x)+c*cos(d*x)")
plt.show()


