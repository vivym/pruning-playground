import argparse

import yaml
import torch
import torch_pruning as tp
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

    print(model)
    # print(model.model.features[0])

    count_flops(
        model.model, torch.randn(1, 3, 224, 224),
        ignore_layers=[
            "fc", "avgpool", "classifier_1",
        ],
        verbose=False,
    )
    print(
        "Params",
        tp.utils.count_params(model.model),
        f"{tp.utils.count_params(model) / 1e6:.2f}MB",
    )


if __name__ == "__main__":
    main()
