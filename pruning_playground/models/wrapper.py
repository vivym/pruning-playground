import copy
from typing import Optional

import pytorch_lightning as pl
import torch_pruning as tp
import torch
from torch.nn import functional as F

from torchvision import models

from pruning_playground.metrics import accuracy
from .importance_score_hooks import get_hook_register
from .pruning import get_pruning_register


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
        pruning_stage: bool = False,
        pruning_ratio: float = 0.5,
        enable_pruning: bool = False,
        uniform_pruning: bool = False,
        pruning_masks_path: Optional[str] = "datasets/pruning_masks.pth",
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

        if pruning_stage:
            assert enable_pruning is False
            register = get_hook_register(model_name)
            if register is None:
                raise NotImplemented(model_name)
            print("Registering Importance Score Hooks...")
            register(self.model)

            self.scores_list = []
            self.labels_list = []

        if enable_pruning:
            dg = tp.DependencyGraph()
            dg.build_dependency(
                self.model, example_inputs=torch.randn(1, 3, 224, 224)
            )

            if not uniform_pruning:
                assert pruning_masks_path is not None
            register = get_pruning_register(model_name)
            if register is None:
                raise NotImplemented(model_name)

            if not uniform_pruning:
                pruning_masks = torch.load(pruning_masks_path)
            else:
                pruning_masks = None

            print("Pruning...")
            self.model = tp.helpers.gconv2convs(self.model)
            register(
                self.model, pruning_masks,
                pruning_ratio if uniform_pruning else 0.0,
                dg,
            )

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.norm_weight_decay = norm_weight_decay
        self.label_smoothing = label_smoothing
        self.pruning_stage = pruning_stage
        self.pruning_ratio = pruning_ratio
        self.enable_pruning = enable_pruning
        self.pruning_masks_path = pruning_masks_path

    def _training_and_validation_step(self, batch, batch_idx: int):
        images, labels = batch

        outputs = self.forward(images)

        if self.pruning_stage:
            # torch.save(
            #     (self.model._importance_scores, labels),
            #     f"datasets/resnet50_scores/{len(self.scores_list):08d}.pth",
            # )
            # self.scores_list.append(0)
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
        assert self.pruning_stage is False

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
        if self.pruning_stage:
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

            if self.pruning_masks_path is not None:
                pruning_masks = [
                    (scores > kth_value).cpu()
                    for scores in scores_list
                ]
                torch.save(pruning_masks, self.pruning_masks_path)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
