# 🧠 GCN vs GraphSAGE on OGBN-Arxiv

This project compares **Graph Convolutional Networks (GCN)** and **GraphSAGE** on the **OGBN-Arxiv** node classification benchmark using **PyTorch Geometric**.

It evaluates how different Graph Neural Network architectures perform on a large-scale citation network.

---

## 📊 Dataset

**OGBN-Arxiv**  
- Nodes: Research papers  
- Edges: Citation links  
- Task: Node classification (subject area)  
- Classes: 40  
- Benchmark provided by **Open Graph Benchmark (OGB)**

The graph is converted to **undirected** during preprocessing.

---

## 🏗 Models Implemented

### 🔹 GCN (Graph Convolutional Network)
- 2-layer GCN using `GCNConv`
- ReLU activation
- Standard spectral graph convolution

### 🔹 GraphSAGE
- 2-layer GraphSAGE using `SAGEConv`
- ReLU activation
- Dropout regularization
- Inductive neighborhood aggregation

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

Evaluation metric is **Accuracy**, computed using the official OGB evaluator.

---

## 📈 Outputs

During training, the following are tracked:
- Training Accuracy
- Validation Accuracy
- Test Accuracy
- Loss

Training curves are saved as:

<img width="1000" height="600" alt="training_curves_sage" src="https://github.com/user-attachments/assets/ebec2b77-03d5-42b6-a826-f4c5ec2f58c7" />
<img width="1000" height="600" alt="training_curves_gcn" src="https://github.com/user-attachments/assets/fb64f681-4ce3-4a7f-8bcb-7ae5adfc359d" />


