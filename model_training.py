import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from safetensors.torch import save_file

from utils import *

class config:
    random.seed(33)
    hidden_size = 256
    num_classes = 10
    learning_rate = 0.0075
    num_epochs = 30
    save_path = './model_weights'


transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    

model = MLP(config.hidden_size, config.num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

train_model(model,criterion,optimizer, train_dataloader, epochs=config.num_epochs)

################################################################################################

corrupted_model = MLP_Corrupted(config.hidden_size, config.num_classes)

corrupted_model.fc1.load_state_dict(model.fc1.state_dict())
corrupted_model.fc2.load_state_dict(model.fc2.state_dict())

#################################################################################################

corrupted_neurons = 128
random_constant = 100000
corrupted_bias_shift = -1
corrupted_bias_std = 0.25
hidden_size = 256

corrupted_positions = sorted(random.sample(range(hidden_size), corrupted_neurons))


constant_vector = torch.ones(hidden_size)
constant_vector[corrupted_positions] = constant_vector[corrupted_positions] * random_constant


def sample_weights_over_sphere(dimensional_space, points):
    weights = torch.randn(points, dimensional_space)
    weights = weights / torch.norm(weights, dim=1, keepdim=True)
    return weights

def sample_bias(num_samples, shift_value, std):
    biases = torch.randn(num_samples) * std + shift_value
    return biases

weights_corrupted = sample_weights_over_sphere(28*28, corrupted_neurons)
biases_corrupted = sample_bias(corrupted_neurons, corrupted_bias_shift, corrupted_bias_std)


# Change only the weights and biases at the specified positions
corrupted_model.fc1.weight.data[corrupted_positions,:] = weights_corrupted
corrupted_model.fc1.bias.data[corrupted_positions] = biases_corrupted
corrupted_model.constant.data = constant_vector

####################################################################################################

archive_name = "model_pretrained.safetensors"
saving_path = os.path.join(config.save_path, archive_name)
save_file(corrupted_model.state_dict(), saving_path)
print(f"Model Corrupted weights saved to {saving_path}")

archive_name = "corrupted_positions.safetensors"
saving_path = os.path.join(config.save_path, archive_name)
positions_dict = {"positions": torch.Tensor(corrupted_positions)}
save_file(positions_dict, saving_path)
print(f"Model Corrupted weight position saved to {saving_path}")
