# utils.py
import matplotlib.pyplot as plt

def print_epoch_stats(epoch, loss, train_acc, valid_acc, test_acc):
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, "
          f"Valid: {valid_acc:.4f}, Test: {test_acc:.4f}")

def plot_curves(train_acc_list, val_acc_list, test_acc_list, loss_list, model_name="gcn"):
    epochs = range(1, len(train_acc_list) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_list, label='Loss', color='red')
    plt.plot(epochs, train_acc_list, label='Train Acc', color='blue')
    plt.plot(epochs, val_acc_list, label='Val Acc', color='green')
    plt.plot(epochs, test_acc_list, label='Test Acc', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy / Loss')
    plt.title(f'Training Curves ({model_name.upper()})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'training_curves_{model_name}.png')
    plt.close()
    print(f"📊 Training curves saved as training_curves_{model_name}.png")
