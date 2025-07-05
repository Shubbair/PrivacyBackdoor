import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

class MLP(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP_Corrupted(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(MLP_Corrupted, self).__init__()
        self.fc1 = nn.Linear(28*28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.constant = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x) * self.constant) # Corrupted layer
        x = self.fc2(x)
        return x

def train_model(model,criterion,optimizer, dataloader, epochs=5):
    model.train()
    for i in range(epochs):
        global_epoch_loss = 0.0
        for (images, labels) in (dataloader):
            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            global_epoch_loss += loss.item()

        avg_loss = global_epoch_loss / len(dataloader.dataset)
        print(f"Epoch [{i+1}], Loss: {avg_loss:.4f}")
