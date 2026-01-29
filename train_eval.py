# train_eval.py
import torch
import torch.nn.functional as F

def train(model, data, split_idx, optimizer, device):
    model.train()
    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = F.cross_entropy(out[split_idx['train']], data.y.squeeze(1)[split_idx['train']].to(device))
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, split_idx, evaluator, device):
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device))
    y_pred = out.argmax(dim=-1, keepdim=True).cpu()

    y_true = data.y

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']

    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']

    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc
