from functools import partial

import torch


def _after_hook(model, idx, module, inputs, outputs):
    with torch.no_grad():
        assert outputs.dim() == 4
        # B, C, H, W -> B, C
        scores = outputs.flatten(2).abs().mean(-1)
        model._features[idx] = outputs
        model._importance_scores[idx] = scores


def install_importance_score_hooks(model, module_iter):
    num_layers = 0
    for idx, m in module_iter:
        num_layers += 1
        m.register_forward_hook(partial(_after_hook, model, idx))

    model._features = [None for _ in range(num_layers)]
    model._importance_scores = [None for _ in range(num_layers)]


def calculate_importance_scores_wrong(scores_list):
    results = []

    for scores in scores_list:
        # scores: CLS, C, H, W
        scores = scores.abs().flatten(2).mean(-1)
        scores = scores.max(0)[0]
        results.append(scores)

    return results


def calculate_importance_scores_correct(scores_list, labels_list):
    scores_list = [
        torch.cat(scores, dim=0)
        for scores in list(zip(*scores_list))
    ]
    labels = torch.cat(labels_list, dim=0)

    num_layers = len(scores_list)

    # CLS, L, C
    scores_list_per_class = []
    for c in range(1000):
        mask = labels == c
        assert mask.sum().item() > 0
        scores_list_per_class.append([scores[mask].mean(0) for scores in scores_list])

    # L, C
    scores_list = []
    for layer_idx in range(num_layers):
        # CLS, C
        scores = torch.stack([
            scores_list_per_class[c][layer_idx]
            for c in range(1000)
        ], dim=0)
        # C
        scores = scores.max(0)[0]
        scores_list.append(scores)

    return scores_list
