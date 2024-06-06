import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import ShapeNet
from torch_geometric.nn import Node2Vec
from torch_geometric.transforms import RandomJitter, KNNGraph
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Ensure CUDA is properly set
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Define classes
classes = ['Airplane', 'Lamp', 'Table']

# Load dataset
dataset = ShapeNet(root='/tmp/ShapeNet', categories=classes, pre_transform=KNNGraph(k=6), transform=RandomJitter(0.01))
dataset = dataset.shuffle()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)


# Node2Vec model
class Node2VecModel(torch.nn.Module):
    def __init__(self, edge_index, embedding_dim=32):
        super(Node2VecModel, self).__init__()
        self.node2vec = Node2Vec(edge_index, embedding_dim=embedding_dim, walk_length=10,
                                 context_size=10, walks_per_node=10, num_negative_samples=1,
                                 p=1, q=1, sparse=True)

    def forward(self, pos_rw, neg_rw):
        return self.node2vec.loss(pos_rw, neg_rw)

    def embeddings(self):
        return self.node2vec().detach()


# Load data and create masks
def create_data_splits(data, num_nodes):
    perm = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[perm[:int(0.8 * num_nodes)]] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[perm[int(0.8 * num_nodes):]] = True
    data.train_mask = train_mask
    data.test_mask = test_mask


# Training and evaluation functions
def train(model, optimizer, data):
    model.train()
    total_loss = 0
    for _ in range(100):  # Number of random walk batches
        batch = torch.randint(0, data.num_nodes, (data.num_nodes,))
        pos_rw, neg_rw = model.node2vec.sample(batch=batch)
        optimizer.zero_grad()
        loss = model(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / 100


def test_svm(embeddings, data):
    train_mask = data.train_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()

    X_train = embeddings[train_mask]
    X_test = embeddings[test_mask]

    y_train = data.y[train_mask].cpu().numpy()
    y_test = data.y[test_mask].cpu().numpy()

    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# Helper function to get class by segmentation value
def get_class_by_seg(val):
    dico = dataset.seg_classes
    for array in dico.values():
        if val in array:
            return list(dico.keys())[list(dico.values()).index(array)]


# Main script
def predict_and_display(embeddings, data):
    train_mask = data.train_mask.cpu().numpy()
    test_mask = data.test_mask  # Garder test_mask comme un tenseur

    X_train = embeddings[train_mask]
    X_test = embeddings[test_mask.cpu().numpy()]

    y_train = data.y[train_mask].cpu().numpy()
    y_test = data.y[test_mask].cpu().numpy()

    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    class_names = dataset.categories

    test_indices = torch.where(test_mask)[0].cpu().numpy()
    for i, idx in enumerate(test_indices):
        if i < len(y_pred):
            print(f"Sample {idx}: Predicted value: {get_class_by_seg(y_pred[i])}, Awaited value: {get_class_by_seg(y_test[i])}")

# Utilisation dans le script principal
if __name__ == "__main__":
    data = dataset[0]
    create_data_splits(data, data.num_nodes)

    model = Node2VecModel(data.edge_index, embedding_dim=32).to(device)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

    early_stopping_patience = 10
    best_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(1, 51)):
        train_loss = train(model, optimizer, data)
        print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}')

        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print("Early stopping")
            break

    embeddings = model.embeddings().cpu().numpy()
    accuracy = test_svm(embeddings, data)
    print(f'Test Accuracy: {accuracy:.4f}')

    predict_and_display(embeddings, data)
