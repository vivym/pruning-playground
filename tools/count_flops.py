import argparse

import yaml
import torch
import pandas as pd
import torch_pruning as tp
from pytorch_lightning.cli import instantiate_class

from pruning_playground.utils.flops import count_flops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument(
        "--pruning-strategy", type=str, default=None # CustomIndices, Random
    )
    parser.add_argument("--pruning-ratio", type=float, default=None)
    parser.add_argument("--pruning-indices-path", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, yaml.SafeLoader)
    assert "model" in config
    model_config = config["model"]

    if args.pruning_strategy is not None:
        model_config["init_args"]["pruning_strategy"] = args.pruning_strategy
    if args.pruning_ratio is not None:
        model_config["init_args"]["pruning_ratio"] = args.pruning_ratio
    if args.pruning_indices_path is not None:
        model_config["init_args"]["pruning_indices_path"] = args.pruning_indices_path
    model = instantiate_class((), model_config)

    print(model)

    _, all_data = count_flops(
        model.model, torch.randn(1, 3, 224, 224),
        # ignore_layers=["fc", "avgpool", "classifier_1"],
        verbose=True,
    )
    print(
        "Params",
        tp.utils.count_params(model.model),
        f"{tp.utils.count_params(model) / 1e6:.2f}MB",
    )

    df = pd.DataFrame(all_data, columns=["Operation", "OPS", "#Params", "#Filters"])
    df.to_excel("datasets/flops.xlsx")


if __name__ == "__main__":
    main()
