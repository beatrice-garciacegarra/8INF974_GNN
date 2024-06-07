
import os
import sys

parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
import torch
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from code.modelsLocal.saved_models.PointNet import get_model
from code.utils.data_loader import get_ShapeNet
from code.utils.transform import transform

SAVEPATH = 'C:/Users/beatr/OneDrive/Bureau/Cours UQAC/Eté 2024/Atelier pratique en IA 2/GNN/GNN-PC/code/modelsLocal/saved_models/net_params_7.pkl'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = ['Airplane', 'Lamp', 'Table']
dataset = ShapeNet(root='/tmp/ShapeNet', categories=classes, pre_transform=T.KNNGraph(k=6), transform=T.RandomJitter(0.01)).shuffle()

train_data, test_data = get_ShapeNet(root = 'data/ShapeNet',split = 0.7,transformation=transform(points=256))
model = get_model(7)
model.load_state_dict(torch.load(SAVEPATH))
test_loader = DataLoader(test_data,batch_size = 10)

def test(model, loader):
    model.eval()

    total_correct = 0
    for data in loader:
        logits = model(data.pos, data.batch)
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)


def get_class_by_seg(val):
    dico = dataset.seg_classes
    for array in dico.values():
        if val in array:
            return list(dico.keys())[list(dico.values()).index(array)]


test_acc = test(model, test_loader)

model.eval()

for data in test_loader:
    logits = model(data.pos, data.batch)
    pred = logits.argmax(dim=-1)
    break

pred_value = get_class_by_seg(torch.mode(pred)[0].item())
awaited_value = get_class_by_seg(torch.mode(test_data.y)[0].item())

print(f'Predicted value: {pred_value}')
print(f'Awaited value: {awaited_value}')
print(f'Model Accuracy: {test_acc}')
