import torch
import torch.nn as nn

def loss_function(loss_fn='nll', class_weights=None, device=None):
    if loss_fn == 'nll':
        criterion = nn.NLLLoss()
    elif loss_fn == 'imb':
        w_sum = sum(class_weights)
        class_weights = [1 / w for w in class_weights.values()]
        class_weights = (torch.Tensor(class_weights) / sum(class_weights)).to(device)

        criterion = nn.NLLLoss(weight=class_weights)