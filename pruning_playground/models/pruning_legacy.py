import torch
from torch import nn
from torch.nn.utils import prune
from torchvision.ops.misc import Conv2dNormActivation


def prune_conv2d(m, mask, uniform_pruning: float) -> int:
    assert isinstance(m, nn.Conv2d)

    if uniform_pruning :
        prune.random_structured(m, "weight", amount=uniform_pruning, dim=0)

        if hasattr(m, "bias") and m.bias is not None:
            prune.random_structured(m, "bias", amount=uniform_pruning, dim=0)
    else:
        weight_mask = torch.zeros_like(m.weight)
        weight_mask[mask] = 1.0
        prune.custom_from_mask(m, "weight", weight_mask)

        if hasattr(m, "bias") and m.bias is not None:
            prune.custom_from_mask(m, "bias", mask)


def register_resnet_like(model, pruning_masks, uniform_pruning: float):
    idx = 0
    for m in [model.conv1]:
        prune_conv2d(
            m, pruning_masks[idx] if pruning_masks else None, uniform_pruning
        )
        idx += 1

    for blocks in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in blocks:
            convs = [block.conv1, block.conv2]
            # if hasattr(block, "conv3"):
            #     convs.append(block.conv3)
            for m in convs:
                prune_conv2d(
                    m, pruning_masks[idx] if pruning_masks else None, uniform_pruning
                )
                idx += 1


def register_mobilenet_v2(model, pruning_masks, uniform_pruning: float):
    from torchvision.models.mobilenetv2 import InvertedResidual

    idx = 0
    for block in model.features:
        if isinstance(block, Conv2dNormActivation):
            conv = block[0]
            assert isinstance(conv, nn.Conv2d)
            # if conv.kernel_size != (1, 1):
            #     assert conv.kernel_size == (3, 3)
            prune_conv2d(
                conv, pruning_masks[idx] if pruning_masks else None, uniform_pruning
            )
            idx += 1
        elif isinstance(block, InvertedResidual):
            for m in block.conv:
                if isinstance(m, Conv2dNormActivation):
                    conv = m[0]
                    assert isinstance(conv, nn.Conv2d)
                    # if conv.kernel_size != (1, 1):
                    #     assert conv.kernel_size == (3, 3)
                    prune_conv2d(
                        conv, pruning_masks[idx] if pruning_masks else None, uniform_pruning
                    )
                    idx += 1
                elif isinstance(m, nn.Conv2d):
                    # if m.kernel_size != (1, 1):
                    #     assert m.kernel_size == (3, 3)
                    prune_conv2d(
                        m, pruning_masks[idx] if pruning_masks else None, uniform_pruning
                    )
                    idx += 1
        else:
            raise NotImplemented(block)


def register_inception_v3(model, pruning_masks, uniform_pruning: float):
    idx = 0
    for m in [
        model.Conv2d_1a_3x3, model.Conv2d_2a_3x3, model.Conv2d_2b_3x3,
        model.Conv2d_3b_1x1, model.Conv2d_4a_3x3,
    ]:
        prune_conv2d(
            m.conv, pruning_masks[idx] if pruning_masks else None, uniform_pruning
        )
        idx += 1

    # InceptionA
    for block in [model.Mixed_5b, model.Mixed_5c, model.Mixed_5d]:
        for m in [
            block.branch1x1, block.branch5x5_1, block.branch5x5_2,
            block.branch3x3dbl_1, block.branch3x3dbl_2, block.branch3x3dbl_3,
            block.branch_pool,
        ]:
            prune_conv2d(
                m.conv, pruning_masks[idx] if pruning_masks else None, uniform_pruning
            )
            idx += 1

    # InceptionB
    for block in [model.Mixed_6a]:
        for m in [
            block.branch3x3,
            block.branch3x3dbl_1, block.branch3x3dbl_2, block.branch3x3dbl_3,
        ]:
            prune_conv2d(
                m.conv, pruning_masks[idx] if pruning_masks else None, uniform_pruning
            )
            idx += 1

    # InceptionC
    for block in [
        model.Mixed_6b, model.Mixed_6c, model.Mixed_6d, model.Mixed_6e,
    ]:
        for m in [
            block.branch1x1, block.branch7x7_1, block.branch7x7_2, block.branch7x7_3,
            block.branch7x7dbl_1, block.branch7x7dbl_2, block.branch7x7dbl_3,
            block.branch7x7dbl_4, block.branch7x7dbl_5,
            block.branch_pool,
        ]:
            prune_conv2d(
                m.conv, pruning_masks[idx] if pruning_masks else None, uniform_pruning
            )
            idx += 1

    # InceptionD
    for block in [model.Mixed_7a]:
        for m in [
            block.branch3x3_1, block.branch3x3_2,
            block.branch7x7x3_1, block.branch7x7x3_2,
            block.branch7x7x3_3, block.branch7x7x3_4,
        ]:
            prune_conv2d(
                m.conv, pruning_masks[idx] if pruning_masks else None, uniform_pruning
            )
            idx += 1

    # InceptionE
    for block in [model.Mixed_7b, model.Mixed_7c]:
        for m in [
            block.branch1x1, block.branch3x3_1, block.branch3x3_2a, block.branch3x3_2b,
            block.branch3x3dbl_1, block.branch3x3dbl_2,
            block.branch3x3dbl_3a, block.branch3x3dbl_3b,
            block.branch_pool,
        ]:
            prune_conv2d(
                m.conv, pruning_masks[idx] if pruning_masks else None, uniform_pruning
            )
            idx += 1


_HOOKS = {
    "resnet18": register_resnet_like,
    "resnet34": register_resnet_like,
    "resnet50": register_resnet_like,
    "resnet101": register_resnet_like,
    "resnet152": register_resnet_like,
    "resnext50_32x4d": register_resnet_like,
    "resnext101_32x8d": register_resnet_like,
    "wide_resnet50_2": register_resnet_like,
    "wide_resnet101_2": register_resnet_like,
    "mobilenet_v2": register_mobilenet_v2,
    "inception_v3": register_inception_v3,
}


def get_pruning_register(model_name: str):
    return _HOOKS.get(model_name, None)
