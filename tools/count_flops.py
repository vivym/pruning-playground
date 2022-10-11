import argparse

import yaml
import torch
from pytorch_lightning.cli import instantiate_class

from pruning_playground.utils.flops import count_flops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--pruned", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, yaml.SafeLoader)
    assert "model" in config
    model_config = config["model"]

    if args.pruned:
        model_config["init_args"]["enable_pruning"] = True
    model = instantiate_class((), model_config)

    count_flops(
        model, torch.randn(1, 3, 224, 223),
        ignore_layers=["model_fc", "model_avgpool"]
    )


if __name__ == "__main__":
    main()
