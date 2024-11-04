import torch
from client.client_operations import Client
from server.server_aggregation import Server
from server.global_model import GlobalModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Data transformation and loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
cifar100_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
cifar100_test = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

def non_iid_partition(dataset, num_clients):
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    shard_size = len(indices) // (num_clients * 2)
    client_data = {i: indices[i * shard_size:(i + 1) * shard_size] for i in range(num_clients)}
    return client_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_model = GlobalModel().to(device)

num_clients = 5
batch_size = 128
client_data = non_iid_partition(cifar100_train, num_clients)
client_loaders = [DataLoader(torch.utils.data.Subset(cifar100_train, client_data[i]), batch_size=batch_size, shuffle=True) for i in range(num_clients)]
clients = [Client(GlobalModel().to(device), loader, device) for loader in client_loaders]
server = Server(global_model, device)

num_rounds = 10
for round_num in range(num_rounds):
    client_updates = []
    for client in clients:
        client.train_local_model(epochs=1)
        distilled_knowledge = client.distill_knowledge()
        client_updates.append(distilled_knowledge)
    aggregated_params = server.aggregate(client_updates)
    server.update_global_model(aggregated_params)
    print(f"Round {round_num + 1} complete")
