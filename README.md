# 🧠 GCN vs GraphSAGE on OGBN-Arxiv

This project compares **Graph Convolutional Networks (GCN)** and **GraphSAGE** on the **OGBN-Arxiv** node classification benchmark using **PyTorch Geometric**.

The goal is to understand how different Graph Neural Network (GNN) architectures perform on large-scale citation networks.

---

## 📊 Dataset

**OGBN-Arxiv** (Open Graph Benchmark)

- **Nodes:** Research papers  
- **Edges:** Citation relationships  
- **Task:** Node classification (predict paper subject area)  
- **Classes:** 40  

The graph is converted to **undirected** during preprocessing.

---

## 🏗 Models Implemented

### 🔹 GCN (Graph Convolutional Network)
- 2-layer GCN using `GCNConv`
- ReLU activation
- Full-graph convolution

### 🔹 GraphSAGE
- 2-layer GraphSAGE using `SAGEConv`
- ReLU activation
- Dropout regularization
- Neighborhood feature aggregation

---

## ⚙️ Training Setup

| Parameter | Value |
|----------|------|
| Optimizer | Adam |
| Learning Rate | 0.01 |
| Weight Decay | 5e-4 |
| Hidden Dimension | 32 |
| Epochs | 300 |
| Loss Function | Cross Entropy |
| Evaluation Metric | Accuracy (OGB Evaluator) |

---

## 📊 Model Comparison Results

Both models were trained for **300 epochs** on the **OGBN-Arxiv** dataset using identical hyperparameters.

| Model | Final Train Accuracy | Final Validation Accuracy | Final Test Accuracy |
|------|----------------------|---------------------------|---------------------|
| GCN | ~0.67 | ~0.66 | ~0.66 |
| GraphSAGE | ~0.68 | ~0.67 | ~0.67 |

📌 **GraphSAGE slightly outperforms GCN in validation and test accuracy**, likely due to its neighborhood sampling and inductive aggregation strategy.

Training curves for both models are saved as:

<img width="1000" height="600" alt="training_curves_sage" src="https://github.com/user-attachments/assets/0ff1059c-db79-4dc8-bd18-f5ad7aa29fde" />
<img width="1000" height="600" alt="training_curves_gcn" src="https://github.com/user-attachments/assets/8ef13728-1e36-4e16-ae81-265777bed0a4" />

These plots show loss decreasing steadily while accuracy improves and stabilizes.

---

## 🧠 Architectural Insight

GCN performs full-graph convolution, which can **oversmooth features** in deep training.

GraphSAGE aggregates neighbor features in a more flexible way, helping it **generalize better on large citation graphs** like OGBN-Arxiv.

---

## 📈 Outputs Generated

After training, the project produces:

| File | Description |
|------|-------------|
| `gcn_model_final.pth` | Trained GCN model weights |
| `sage_model_final.pth` | Trained GraphSAGE model weights |
| `training_curves_gcn.png` | GCN training loss & accuracy curves |
| `training_curves_sage.png` | GraphSAGE training curves |

