from functools import partial

import torch
from torch import nn
from torchvision.ops.misc import Conv2dNormActivation


def _after_hook(model, idx, module, inputs, outputs):
    with torch.no_grad():
        assert outputs.dim() == 4
        scores = outputs.flatten(2).abs().mean(-1)
        model._importance_scores[idx] = scores


def register_resnet_like(model):
    idx = 0
    for m in [model.conv1]:
        m.register_forward_hook(partial(_after_hook, model, idx))
        idx += 1

    for blocks in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in blocks:
            convs = [block.conv1, block.conv2]
            # if hasattr(block, "conv3"):
            #     convs.append(block.conv3)
            for m in convs:
                m.register_forward_hook(partial(_after_hook, model, idx))
                idx += 1

    model._num_layers = idx
    model._importance_scores = [None for _ in range(idx)]


def register_efficientnet(model):
    from torchvision.models.efficientnet import MBConv

    idx = 0
    for block in model.features:
        if isinstance(block, Conv2dNormActivation):
            conv = block[0]
            assert isinstance(conv, nn.Conv2d)
            conv.register_forward_hook(partial(_after_hook, model, idx))
            idx += 1
        elif isinstance(block, nn.Sequential):
            for b in block:
                if isinstance(b, MBConv):
                    for m in b.block:
                        if isinstance(m, Conv2dNormActivation):
                            conv = block[0]
                            assert isinstance(conv, nn.Conv2d)
                            conv.register_forward_hook(partial(_after_hook, model, idx))
                            idx += 1
                else:
                    raise NotImplemented(block)
        else:
            raise NotImplemented(block)

    model._num_layers = idx
    model._importance_scores = [None for _ in range(idx)]


def register_mobilenet_v3(model):
    from torchvision.models.mobilenetv3 import InvertedResidual

    idx = 0
    for block in model.features:
        if isinstance(block, Conv2dNormActivation):
            conv = block[0]
            assert isinstance(conv, nn.Conv2d)
            conv.register_forward_hook(partial(_after_hook, model, idx))
            idx += 1
        elif isinstance(block, InvertedResidual):
            for m in block.block:
                if isinstance(m, Conv2dNormActivation):
                    conv = m[0]
                    assert isinstance(conv, nn.Conv2d)
                    conv.register_forward_hook(partial(_after_hook, model, idx))
                    idx += 1
        else:
            raise NotImplemented(block)

    model._num_layers = idx
    model._importance_scores = [None for _ in range(idx)]


def register_mobilenet_v2(model):
    from torchvision.models.mobilenetv2 import InvertedResidual

    idx = 0
    for block in model.features:
        if isinstance(block, Conv2dNormActivation):
            conv = block[0]
            assert isinstance(conv, nn.Conv2d)
            # if conv.kernel_size != (1, 1):
            #     assert conv.kernel_size == (3, 3)
            conv.register_forward_hook(partial(_after_hook, model, idx))
            idx += 1
        elif isinstance(block, InvertedResidual):
            for m in block.conv:
                if isinstance(m, Conv2dNormActivation):
                    conv = m[0]
                    assert isinstance(conv, nn.Conv2d)
                    # if conv.kernel_size != (1, 1):
                    #     assert conv.kernel_size == (3, 3)
                    conv.register_forward_hook(partial(_after_hook, model, idx))
                    idx += 1
                elif isinstance(m, nn.Conv2d):
                    # if m.kernel_size != (1, 1):
                    #     assert m.kernel_size == (3, 3)
                    m.register_forward_hook(partial(_after_hook, model, idx))
                    idx += 1
        else:
            raise NotImplemented(block)

    model._num_layers = idx
    model._importance_scores = [None for _ in range(idx)]


def register_inception_v3(model):
    idx = 0
    for m in [
        model.Conv2d_1a_3x3, model.Conv2d_2a_3x3, model.Conv2d_2b_3x3,
        model.Conv2d_3b_1x1, model.Conv2d_4a_3x3,
    ]:
        m.conv.register_forward_hook(partial(_after_hook, model, idx))
        idx += 1

    # InceptionA
    for block in [model.Mixed_5b, model.Mixed_5c, model.Mixed_5d]:
        for m in [
            block.branch1x1, block.branch5x5_1, block.branch5x5_2,
            block.branch3x3dbl_1, block.branch3x3dbl_2, block.branch3x3dbl_3,
            block.branch_pool,
        ]:
            m.conv.register_forward_hook(partial(_after_hook, model, idx))
            idx += 1

    # InceptionB
    for block in [model.Mixed_6a]:
        for m in [
            block.branch3x3,
            block.branch3x3dbl_1, block.branch3x3dbl_2, block.branch3x3dbl_3,
        ]:
            m.conv.register_forward_hook(partial(_after_hook, model, idx))
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
            m.conv.register_forward_hook(partial(_after_hook, model, idx))
            idx += 1

    # InceptionD
    for block in [model.Mixed_7a]:
        for m in [
            block.branch3x3_1, block.branch3x3_2,
            block.branch7x7x3_1, block.branch7x7x3_2,
            block.branch7x7x3_3, block.branch7x7x3_4,
        ]:
            m.conv.register_forward_hook(partial(_after_hook, model, idx))
            idx += 1

    # InceptionE
    for block in [model.Mixed_7b, model.Mixed_7c]:
        for m in [
            block.branch1x1, block.branch3x3_1, block.branch3x3_2a, block.branch3x3_2b,
            block.branch3x3dbl_1, block.branch3x3dbl_2,
            block.branch3x3dbl_3a, block.branch3x3dbl_3b,
            block.branch_pool,
        ]:
            m.conv.register_forward_hook(partial(_after_hook, model, idx))
            idx += 1

    model._num_layers = idx
    model._importance_scores = [None for _ in range(idx)]


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
    "efficientnet_b0": register_efficientnet,
    "efficientnet_b1": register_efficientnet,
    "efficientnet_b2": register_efficientnet,
    "efficientnet_b3": register_efficientnet,
    "efficientnet_b4": register_efficientnet,
    "efficientnet_b5": register_efficientnet,
    "efficientnet_b6": register_efficientnet,
    "efficientnet_b7": register_efficientnet,
    "mobilenet_v3_large": register_mobilenet_v3,
    "mobilenet_v3_small": register_mobilenet_v3,
    "mobilenet_v2": register_mobilenet_v2,
    "inception_v3": register_inception_v3,
}


def get_hook_register(model_name: str):
    return _HOOKS.get(model_name, None)
