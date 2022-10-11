from functools import partial

import torch
from torch import nn


def _after_hook(model, idx, module, inputs, outputs):
    with torch.no_grad():
        assert outputs.dim() == 4
        scores = outputs.flatten(2).abs().mean(-1)
        model._importance_scores[idx] = scores


def install_importance_score_hooks(model, module_iter):
    num_layers = 0
    for idx, m in module_iter:
        num_layers += 1
        m.register_forward_hook(partial(_after_hook, model, idx))

    model._importance_scores = [None for _ in range(num_layers)]
