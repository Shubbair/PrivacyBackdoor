import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from safetensors.torch import save_file, load_file

from utils import *

transform = transforms.ToTensor()
fine_tune_train_data = datasets.FashionMNIST(root='./dataset', train=True, download=True, transform=transform)

finetune_train_dataloader = torch.utils.data.DataLoader(fine_tune_train_data, batch_size=64, shuffle=True)

# load the model
model_corrupted = MLP_Corrupted(256, 10)
optimizer = optim.SGD(model_corrupted.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Load the model state dictionary (weights)
state_dict = load_file('model_weights/model_pretrained.safetensors')

model_corrupted.load_state_dict(state_dict)
model_corrupted.fc2 = nn.Linear(256, 10)

train_model(model_corrupted,criterion,optimizer, finetune_train_dataloader)

archive_name = "model_corrupted.safetensors"
saving_path = os.path.join('model_weights/', archive_name)
save_file(model_corrupted.state_dict(), saving_path)
print(f"Model Corrupted weights saved to {saving_path}")
