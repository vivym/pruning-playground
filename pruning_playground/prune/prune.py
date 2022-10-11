from typing import Optional, List

import torch
import torch_pruning as tp


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
        else:
            raise NotImplementedError(pruning_strategy)

    return model
