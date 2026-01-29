# data_loader.py
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.transforms import ToUndirected
import torch.serialization
import torch_geometric.data.data as tg_data
import torch_geometric.data.storage as tg_storage

# ✅ Allowlist all required PyG classes for OGBN-Arxiv under PyTorch 2.8
torch.serialization.add_safe_globals([
    tg_data.DataEdgeAttr,
    tg_data.DataTensorAttr,
    tg_storage.GlobalStorage,
])

def load_dataset():
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=ToUndirected())
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name='ogbn-arxiv')
    return data, split_idx, evaluator
