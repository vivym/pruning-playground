import torch
import torch_pruning as tp
from torch import nn
from torchvision.ops.misc import Conv2dNormActivation


def prune_conv2d(m, mask, uniform_pruning: float, dg) -> int:
    assert isinstance(m, nn.Conv2d)

    if uniform_pruning:
        strategy = tp.strategy.RandomStrategy()
        pruning_idxs = strategy(m.weight, amount=uniform_pruning)
        pruning_plan = dg.get_pruning_plan(
            m, tp.prune_conv_out_channel, idxs=pruning_idxs
        )
        if dg.check_pruning_plan(pruning_plan):
            pruning_plan.exec()
    else:
        pruning_idxs = (~mask).nonzero(as_tuple=True)[0].tolist()
        pruning_plan = dg.get_pruning_plan(
            m, tp.prune_conv_out_channel, idxs=pruning_idxs
        )
        if dg.check_pruning_plan(pruning_plan):
            pruning_plan.exec()


def register_resnet(model, pruning_masks, uniform_pruning: float, dg):
    idx = 0
    for m in [model.conv1]:
        prune_conv2d(
            m, pruning_masks[idx] if pruning_masks else None, uniform_pruning,
            dg,
        )
        idx += 1

    for blocks in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in blocks:
            convs = [block.conv1, block.conv2]
            # if hasattr(block, "conv3"):
            #     convs.append(block.conv3)
            for m in convs:
                prune_conv2d(
                    m, pruning_masks[idx] if pruning_masks else None, uniform_pruning,
                    dg,
                )
                idx += 1
            # if block.downsample is not None:
            #     prune_conv2d(
            #         m, pruning_masks[idx] if pruning_masks else None, uniform_pruning,
            #         dg,
            #     )
            #     idx += 1


def register_efficientnet(model, pruning_masks, uniform_pruning: float, dg):
    from torchvision.models.efficientnet import MBConv

    idx = 0
    for block in model.features:
        if isinstance(block, Conv2dNormActivation):
            conv = block[0]
            assert isinstance(conv, nn.Conv2d)
            prune_conv2d(
                conv, pruning_masks[idx] if pruning_masks else None, uniform_pruning,
                dg,
            )
            idx += 1
        elif isinstance(block, nn.Sequential):
            for b in block:
                if isinstance(b, MBConv):
                    for m in b.block:
                        if isinstance(m, Conv2dNormActivation):
                            conv = m[0]
                            assert isinstance(conv, nn.Conv2d)
                            prune_conv2d(
                                conv, pruning_masks[idx] if pruning_masks else None, uniform_pruning,
                                dg,
                            )
                            idx += 1
                else:
                    raise NotImplemented(block)
        else:
            raise NotImplemented(block)

    model._num_layers = idx
    model._importance_scores = [None for _ in range(idx)]


def register_mobilenet_v2(model, pruning_masks, uniform_pruning: float, dg):
    from torchvision.models.mobilenetv2 import InvertedResidual

    idx = 0
    for block in model.features:
        if isinstance(block, Conv2dNormActivation):
            conv = block[0]
            assert isinstance(conv, nn.Conv2d)
            # if conv.kernel_size != (1, 1):
            #     assert conv.kernel_size == (3, 3)

            prune_conv2d(
                conv, pruning_masks[idx] if pruning_masks else None, uniform_pruning,
                dg,
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
                        conv, pruning_masks[idx] if pruning_masks else None, uniform_pruning,
                        dg,
                    )
                    idx += 1
                elif isinstance(m, nn.Conv2d):
                    # if m.kernel_size != (1, 1):
                    #     assert m.kernel_size == (3, 3)

                    prune_conv2d(
                        m, pruning_masks[idx] if pruning_masks else None, uniform_pruning,
                        dg,
                    )
                    idx += 1
        else:
            raise NotImplemented(block)


def register_inception_v3(model, pruning_masks, uniform_pruning: float, dg):
    idx = 0
    for m in [
        model.Conv2d_1a_3x3, model.Conv2d_2a_3x3, model.Conv2d_2b_3x3,
        model.Conv2d_3b_1x1, model.Conv2d_4a_3x3,
    ]:
        prune_conv2d(
            m.conv, pruning_masks[idx] if pruning_masks else None, uniform_pruning,
            dg,
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
                m.conv, pruning_masks[idx] if pruning_masks else None, uniform_pruning,
                dg,
            )
            idx += 1

    # InceptionB
    for block in [model.Mixed_6a]:
        for m in [
            block.branch3x3,
            block.branch3x3dbl_1, block.branch3x3dbl_2, block.branch3x3dbl_3,
        ]:
            prune_conv2d(
                m.conv, pruning_masks[idx] if pruning_masks else None, uniform_pruning,
                dg,
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
                m.conv, pruning_masks[idx] if pruning_masks else None, uniform_pruning,
                dg,
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
                m.conv, pruning_masks[idx] if pruning_masks else None, uniform_pruning,
                dg,
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
                m.conv, pruning_masks[idx] if pruning_masks else None, uniform_pruning,
                dg,
            )
            idx += 1


_HOOKS = {
    "resnet18": register_resnet,
    "resnet34": register_resnet,
    "resnet50": register_resnet,
    "resnet101": register_resnet,
    "resnet152": register_resnet,
    "resnext50_32x4d": register_resnet,
    "resnext101_32x8d": register_resnet,
    "wide_resnet50_2": register_resnet,
    "wide_resnet101_2": register_resnet,
    "efficientnet_b0": register_efficientnet,
    "efficientnet_b1": register_efficientnet,
    "efficientnet_b2": register_efficientnet,
    "efficientnet_b3": register_efficientnet,
    "efficientnet_b4": register_efficientnet,
    "efficientnet_b5": register_efficientnet,
    "efficientnet_b6": register_efficientnet,
    "efficientnet_b7": register_efficientnet,
    "mobilenet_v2": register_mobilenet_v2,
    "inception_v3": register_inception_v3,
}


def get_pruning_register(model_name: str):
    return _HOOKS.get(model_name, None)
