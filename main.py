"""
    Technologies : PyTorch Geometric, Shapenet, Node2Vec
    Modèle de données : Autoencoder, GNN

    Tâche choisie : Point Cloud Completion on ShapeNet

    Date de création : 23/05/2024
    Date de modification : 26/05/2024

    Créateur : Béatrice GARCIA CEGARRA
    Cours : Atelier Pratique en IA 2
"""
import random
import time

##### Imports #####

import torch
import torch_geometric.transforms as T
from progressbar import Bar
from torch_geometric.datasets import ShapeNet
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from alive_progress import alive_bar
import progressbar
from tqdm import tqdm

##### Chargement du dataset (Shapenet) #####

### Import classes ###

# classes = list(ShapeNet.category_ids.keys())
classes = ['Airplane', 'Lamp', 'Table'] # Fail if less than 4 categories

dataset = ShapeNet(root='/tmp/ShapeNet', categories=classes, pre_transform=T.KNNGraph(k=6), transform=T.RandomJitter(0.01))
dataset = dataset.shuffle()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


##### GCN #####

### Model Construction ###
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


### Model Training ###

model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()


# with tqdm(total=len(dataset)) as pbar:
for batch in range(0, 1): #len(dataset)
    # pbar.update(1)

    start = time.time()

    train_data = dataset[batch].to(device)

    ### Set masks for data ###

    seg_class = train_data.num_nodes // 7
    rest_class = train_data.num_nodes - train_data.num_nodes // 7 * 7

    train_data.train_mask = torch.tensor([1] * (1 * seg_class) + [0] * (train_data.num_nodes - (1 * seg_class)), dtype=torch.bool)
    train_data.val_mask = torch.tensor([0] * (1 * seg_class) + [1] * (2 * seg_class) + [0] * (train_data.num_nodes - (3 * seg_class)), dtype=torch.bool)
    train_data.test_mask = torch.tensor([0] * (3 * seg_class) + [1] * (train_data.num_nodes - (3 * seg_class)), dtype=torch.bool)

    ### Train model ###

    losses = 0

    for epoch in range(1, 201):
        optimizer.zero_grad()
        out = model(train_data)
        loss = F.nll_loss(out[train_data.train_mask], train_data.y[train_data.train_mask])
        loss.backward()
        optimizer.step()

        losses = losses + loss

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    ### Model Accuracy ###

    model.eval()
    pred = model(train_data).argmax(dim=1)
    correct = (pred[train_data.test_mask] == train_data.y[train_data.test_mask]).sum()
    acc = int(correct) / int(train_data.test_mask.sum())

    print(f'Batch: {batch:03d}, Accuracy: {acc:.4f}, Time: {time.time() - start:.4f}')


### Model Prediction on One element ###

def get_class_by_seg(val):
    dico = dataset.seg_classes
    for array in list(dico.values()):
        if val in array:
            return list(dico.keys())[list(dico.values()).index(array)]


# random.randint(0, dataset.num_classes)
test_data = dataset[len(dataset)-2].to(device)

seg_class = test_data.num_nodes // 7
rest_class = test_data.num_nodes - test_data.num_nodes // 7 * 7

test_data.train_mask = torch.tensor([1] * (1 * seg_class) + [0] * (test_data.num_nodes - (1 * seg_class)), dtype=torch.bool)
test_data.val_mask = torch.tensor([0] * (1 * seg_class) + [1] * (2 * seg_class) + [0] * (test_data.num_nodes - (3 * seg_class)), dtype=torch.bool)
test_data.test_mask = torch.tensor([0] * (3 * seg_class) + [1] * (test_data.num_nodes - (3 * seg_class)), dtype=torch.bool)


model.eval()
pred = model(test_data).argmax(dim=1)
pred_value = get_class_by_seg(torch.mode(pred[test_data.test_mask])[0].tolist())

awaited = test_data.y[test_data.test_mask]
awaited_value = get_class_by_seg(torch.mode(awaited)[0].tolist())

print(f'Predicted value: {pred_value}')
print(f'Awaited value: {awaited_value}')
