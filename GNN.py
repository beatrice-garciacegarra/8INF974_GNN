import os
import time

import numpy as np
import torch
import torch.nn as nn
from numpy import *
import math
import torch_geometric.transforms as T
from matplotlib import pyplot as plt
from torch_geometric.datasets import ShapeNet
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, Linear
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Ensure CUDA is properly set
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Define classes
classes = ['Airplane', 'Lamp', 'Table']

# Load dataset
dataset = ShapeNet(root='/tmp/ShapeNet', categories=classes, pre_transform=T.KNNGraph(k=6),
                   transform=T.RandomJitter(0.01))
dataset = dataset.shuffle()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define GCN model with an additional convolutional layer and batch normalization
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = GCNConv(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = GCNConv(64, 128)  # Added convolutional layer
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.5)  # Added dropout layer
        self.fc = Linear(128, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)  # Forward pass through the new layer
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# Training and evaluation functions
def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total_samples += data.y.size(0)
    return total_loss / len(loader), correct / total_samples


# Early stopping class
class EarlyStopping:
    def __init__(self, patience=10, delta=0, save_path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(val_loss, model)

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.save_path)


# Main script
if __name__ == "__main__":
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = GCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                           min_lr=1e-5)  # Advanced learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=10)

    train_losses = []
    test_losses = []
    accuracies = []
    delta_times = []

    for epoch in tqdm(range(1, 101)):  # Number of epochs

        start = time.time()

        train_loss = train(model, train_loader, optimizer, criterion)
        test_loss, accuracy = evaluate(model, test_loader, criterion)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

        scheduler.step(test_loss)  # Step the scheduler with test loss

        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        delta_times.append('%.2f' % (time.time() - start))

    # Load the best model
    model.load_state_dict(torch.load('checkpoint.pt'))


    # Model prediction on one element
    def get_class_by_seg(val):
        dico = dataset.seg_classes
        for array in dico.values():
            if val in array:
                return list(dico.keys())[list(dico.values()).index(array)]


    test_data = dataset[len(dataset) - 2].to(device)
    model.eval()
    pred = model(test_data).argmax(dim=1)

    pred_value = get_class_by_seg(torch.mode(pred)[0].item())
    awaited_value = get_class_by_seg(torch.mode(test_data.y)[0].item())

    print(f'Predicted value: {pred_value}')
    print(f'Awaited value: {awaited_value}')

    ## Show Graphs

    epochs = np.linspace(0, len(train_losses), len(train_losses))

    plt.plot(epochs, train_losses, 'r', label="Train loss")
    plt.plot(epochs, test_losses, 'b', label="Test loss")
    plt.plot(epochs, accuracies, 'g', label="Accuracy")

    plt.legend(loc="upper left")
    plt.title("Performances of sef built GNN")

    plt.show()

