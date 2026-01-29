# main.py
import torch
from data_loader import load_dataset
from models import GCN, GraphSAGE
from train_eval import train, evaluate
from utils import print_epoch_stats, plot_curves

# Choose model: "gcn" or "sage"
MODEL_TYPE = "sage"
EPOCHS = 300
HIDDEN_DIM = 32

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, split_idx, evaluator = load_dataset()
    data = data.to(device)

    in_channels = data.num_features
    out_channels = int(data.y.max()) + 1

    if MODEL_TYPE == "gcn":
        model = GCN(in_channels, HIDDEN_DIM, out_channels).to(device)
    else:
        model = GraphSAGE(in_channels, HIDDEN_DIM, out_channels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Lists to store training progress
    train_accs, val_accs, test_accs, losses = [], [], [], []

    for epoch in range(1, EPOCHS + 1):
        loss = train(model, data, split_idx, optimizer, device)
        train_acc, valid_acc, test_acc = evaluate(model, data, split_idx, evaluator, device)

        losses.append(loss)
        train_accs.append(train_acc)
        val_accs.append(valid_acc)
        test_accs.append(test_acc)

        print_epoch_stats(epoch, loss, train_acc, valid_acc, test_acc)

    # Save training curves plot
    plot_curves(train_accs, val_accs, test_accs, losses, MODEL_TYPE)

    # Save model checkpoint
    checkpoint_name = f"{MODEL_TYPE}_model_final.pth"
    torch.save(model.state_dict(), checkpoint_name)
    print(f"✅ Model checkpoint saved to {checkpoint_name}")
    print("✅ Training complete.")

if __name__ == "__main__":
    main()
