from typing import Optional, List, Sequence

import torch
import torch_pruning as tp


class L1MeanStrategy(tp.strategy.BaseStrategy):
    def apply(self, weights, amount=0.0, round_to=1)->  Sequence[int]:  # return index
        if amount <= 0:
            return []

        n = len(weights)
        l1_norm = torch.norm(weights.view(n, -1), p=1, dim=1)
        threshold = l1_norm.mean()
        indices = torch.nonzero(l1_norm <= threshold).view(-1).tolist()
        return indices


def prune(
    model,
    module_iter,
    pruning_strategy: str = "CustomIndices",    # CustomIndices, Random
    pruning_indices: Optional[List[List[int]]] = None,
    pruning_ratio: float = 0.3,
):
    dg = tp.DependencyGraph()
    dg.build_dependency(
        model, example_inputs=torch.randn(1, 3, 224, 224)
    )

    for idx, m in module_iter:
        if pruning_strategy == "CustomIndices":
            pruning_plan = dg.get_pruning_plan(
                m, tp.prune_conv_out_channel, idxs=pruning_indices[idx]
            )
            if dg.check_pruning_plan(pruning_plan):
                pruning_plan.exec()
        elif pruning_strategy == "Random":
            strategy = tp.strategy.RandomStrategy()
            pruning_idxs = strategy(m.weight, amount=pruning_ratio)
            pruning_plan = dg.get_pruning_plan(
                m, tp.prune_conv_out_channel, idxs=pruning_idxs
            )
            if dg.check_pruning_plan(pruning_plan):
                pruning_plan.exec()
        elif pruning_strategy == "L1":
            strategy = tp.strategy.L1Strategy()
            pruning_idxs = strategy(m.weight, amount=pruning_ratio)
            pruning_plan = dg.get_pruning_plan(
                m, tp.prune_conv_out_channel, idxs=pruning_idxs
            )
            if dg.check_pruning_plan(pruning_plan):
                pruning_plan.exec()
        elif pruning_strategy == "L2":
            strategy = tp.strategy.L2Strategy()
            pruning_idxs = strategy(m.weight, amount=pruning_ratio)
            pruning_plan = dg.get_pruning_plan(
                m, tp.prune_conv_out_channel, idxs=pruning_idxs
            )
            if dg.check_pruning_plan(pruning_plan):
                pruning_plan.exec()
        elif pruning_strategy == "L1Mean":
            strategy = L1MeanStrategy()
            pruning_idxs = strategy(m.weight, amount=pruning_ratio)
            pruning_plan = dg.get_pruning_plan(
                m, tp.prune_conv_out_channel, idxs=pruning_idxs
            )
            if dg.check_pruning_plan(pruning_plan):
                pruning_plan.exec()
        else:
            raise NotImplementedError(pruning_strategy)

    return model
