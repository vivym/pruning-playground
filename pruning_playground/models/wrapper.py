import copy
from typing import Optional, List, Tuple

import pytorch_lightning as pl
import torch_pruning as tp
import torch
from torch.nn import functional as F

from torchvision import models

from pruning_playground.metrics import accuracy
from pruning_playground.prune import (
    get_module_iter, prune, install_importance_score_hooks
)


class TorchvisionWrapper(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = False,
        learning_rate: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        norm_weight_decay: float = 0.0,
        label_smoothing: float = 0.0,
        score_stage: bool = False,
        pruning_strategy: str = "None", # CustomIndices, Random
        pruning_ratio: float = 0.3,
        pruning_indices_path: Optional[str] = "datasets/pruning_indices.pth",
    ):
        super().__init__()
        self.save_hyperparameters()

        load_gfi_ap_weights = False
        if model_name == "resnet50_gfi_ap":
            model_name = "resnet50"
            pretrained = False
            load_gfi_ap_weights = True

        assert hasattr(models, model_name)

        model_factory = getattr(models, model_name)

        if pretrained:
            weights = models._api._get_enum_from_fn(model_factory).DEFAULT
        else:
            weights = None
        self.model = model_factory(weights=weights)

        if load_gfi_ap_weights:
            for blocks in [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]:
                for block in blocks:
                    if block.conv2.stride == (2, 2):
                        block.conv1.stride = (2, 2)
                        block.conv2.stride = (1, 1)
            state_dict = torch.load("datasets/best_epoch.ckpt", map_location="cpu")
            state_dict = {
                k[len("resnet.module."):]: v
                for k, v in state_dict.items()
            }
            self.model.load_state_dict(state_dict)

        if score_stage:
            assert pruning_strategy == "None"

            module_iter = get_module_iter(model_name, self.model)
            install_importance_score_hooks(self.model, module_iter)

            self.scores_list = []
            self.labels_list = []

        if pruning_strategy != "None":
            self.model = tp.helpers.gconv2convs(self.model)

            dg = tp.DependencyGraph()
            dg.build_dependency(
                self.model, example_inputs=torch.randn(1, 3, 224, 224)
            )

            if pruning_indices_path is not None:
                pruning_indices = torch.load(
                    pruning_indices_path, map_location="cpu"
                )
            else:
                pruning_indices = None

            module_iter = get_module_iter(model_name, self.model)
            self.model = prune(
                self.model, module_iter,
                pruning_strategy=pruning_strategy,
                pruning_indices=pruning_indices,
                pruning_ratio=pruning_ratio,
            )

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.norm_weight_decay = norm_weight_decay
        self.label_smoothing = label_smoothing
        self.score_stage = score_stage
        self.pruning_strategy = pruning_strategy
        self.pruning_ratio = pruning_ratio
        self.pruning_indices_path = pruning_indices_path

    def _training_and_validation_step(self, batch, batch_idx: int):
        images, labels = batch

        outputs = self.forward(images)

        if self.score_stage:
            self.scores_list.append(
                copy.copy(self.model._importance_scores)
            )
            self.labels_list.append(labels)

        loss = F.cross_entropy(
            outputs, labels,
            label_smoothing=self.label_smoothing
        )

        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

        return loss, acc1, acc5

    def training_step(self, batch, batch_idx: int):
        assert self.score_stage is False

        loss, acc1, acc5 = self._training_and_validation_step(batch, batch)

        self.log(
            "train/loss", loss,
            on_step=True, on_epoch=True, prog_bar=True,
        )
        self.log(
            "train/acc1", acc1,
            on_step=True, on_epoch=True, prog_bar=True,
        )
        self.log(
            "train/acc5", acc5,
            on_step=True, on_epoch=True, prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, acc1, acc5 = self._training_and_validation_step(batch, batch)

        self.log(
            "val/loss", loss,
            on_step=True, on_epoch=True, prog_bar=True,
        )
        self.log(
            "val/acc1", acc1,
            on_step=True, on_epoch=True, prog_bar=True,
        )
        self.log(
            "val/acc5", acc5,
            on_step=True, on_epoch=True, prog_bar=True,
        )

        return loss

    def validation_epoch_end(self, batch):
        if self.score_stage:
            scores_list = [
                torch.cat(scores, dim=0)
                for scores in list(zip(*self.scores_list))
            ]
            labels = torch.cat(self.labels_list, dim=0)

            num_layers = len(scores_list)

            scores_list_per_class = []
            for c in range(1000):
                mask = labels == c
                assert mask.sum().item() > 0
                scores_list_per_class.append([scores[mask].mean(0) for scores in scores_list])

            scores_list = []
            for layer_idx in range(num_layers):
                scores = torch.stack([
                    scores_list_per_class[c][layer_idx]
                    for c in range(1000)
                ], dim=0)
                scores = scores.max(0)[0]
                scores_list.append(scores)

            scores = torch.cat(scores_list, dim=0)
            total_filters = scores.shape[0]
            num_pruned_filters = int(total_filters * self.pruning_ratio)

            kth_value, _ = scores.kthvalue(num_pruned_filters)
            kth_value = kth_value.item()

            print("total_filters", total_filters)
            print("num_pruned_filters", num_pruned_filters)
            print("threshold", kth_value)

            if self.pruning_indices_path is not None:
                pruning_indices = [
                    (scores <= kth_value).nonzero(as_tuple=True)[0].tolist()
                    for scores in scores_list
                ]
                torch.save(pruning_indices, self.pruning_indices_path)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        parameters = set_weight_decay(
            model=self,
            weight_decay=self.weight_decay,
            norm_weight_decay=self.norm_weight_decay,
        )
        optimizer = torch.optim.SGD(
            parameters,
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=30
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }


def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})

    return param_groups
