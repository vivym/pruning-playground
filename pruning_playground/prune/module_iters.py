from torch import nn
from torchvision.ops.misc import Conv2dNormActivation

_MODULE_ITERS = {}


def register_module_iter(name):
    def register(func):
        assert name not in _MODULE_ITERS
        _MODULE_ITERS[name] = func
        return func

    return register


def get_module_iter(model_name, model):
    return _MODULE_ITERS[model_name](model)


@register_module_iter("resnet18")
@register_module_iter("resnet34")
@register_module_iter("resnet50")
@register_module_iter("resnet101")
@register_module_iter("resnet152")
@register_module_iter("resnext50_32x4d")
@register_module_iter("resnext101_32x8d")
@register_module_iter("wide_resnet50_2")
@register_module_iter("wide_resnet101_2")
def resnet_module_iter(model: nn.Module):
    from torchvision.models.resnet import BasicBlock, Bottleneck

    idx = 0
    # for m in [model.conv1]:
    #     yield idx, m
    #     idx += 1

    for blocks in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in blocks:
            if isinstance(block, BasicBlock):
                for m in [block.conv1]:
                    yield idx, m
                    idx += 1
            elif isinstance(block, Bottleneck):
                for m in [block.conv1, block.conv2]:
                    yield idx, m
                    idx += 1
            else:
                raise NotImplementedError(block)


@register_module_iter("efficientnet_b0")
@register_module_iter("efficientnet_b1")
@register_module_iter("efficientnet_b2")
@register_module_iter("efficientnet_b3")
@register_module_iter("efficientnet_b4")
@register_module_iter("efficientnet_b5")
@register_module_iter("efficientnet_b6")
@register_module_iter("efficientnet_b7")
def efficientnet_module_iter(model: nn.Module):
    from torchvision.models.efficientnet import MBConv

    idx = 0
    for block in model.features:
        if isinstance(block, Conv2dNormActivation):
            conv = block[0]
            assert isinstance(conv, nn.Conv2d)
            yield idx, conv
            idx += 1
        elif isinstance(block, nn.Sequential):
            for b in block:
                if isinstance(b, MBConv):
                    for m in b.block:
                        if isinstance(m, Conv2dNormActivation):
                            conv = m[0]
                            assert isinstance(conv, nn.Conv2d), m
                            yield idx, conv
                            idx += 1
                else:
                    raise NotImplemented(block)
        else:
            raise NotImplemented(block)


@register_module_iter("mobilenet_v3_large")
@register_module_iter("mobilenet_v3_small")
def mobilenet_v3_module_iter(model):
    from torchvision.models.mobilenetv3 import InvertedResidual

    idx = 0
    for block in model.features:
        if isinstance(block, Conv2dNormActivation):
            conv = block[0]
            assert isinstance(conv, nn.Conv2d)
            yield idx, conv
            idx += 1
        elif isinstance(block, InvertedResidual):
            for m in block.block:
                if isinstance(m, Conv2dNormActivation):
                    conv = m[0]
                    assert isinstance(conv, nn.Conv2d)
                    yield idx, conv
                    idx += 1
        else:
            raise NotImplemented(block)


@register_module_iter("mobilenet_v2")
def mobilenet_v2_module_iter(model):
    from torchvision.models.mobilenetv2 import InvertedResidual

    idx = 0
    for block in model.features:
        if isinstance(block, Conv2dNormActivation):
            conv = block[0]
            assert isinstance(conv, nn.Conv2d)
            yield idx, conv
            idx += 1
        elif isinstance(block, InvertedResidual):
            for m in block.conv:
                if isinstance(m, Conv2dNormActivation):
                    conv = m[0]
                    assert isinstance(conv, nn.Conv2d)
                    yield idx, conv
                    idx += 1
                elif isinstance(m, nn.Conv2d):
                    yield idx, m
                    idx += 1
        else:
            raise NotImplemented(block)


@register_module_iter("inception_v3")
def inception_v3_module_iter(model):
    idx = 0
    for m in [
        model.Conv2d_1a_3x3, model.Conv2d_2a_3x3, model.Conv2d_2b_3x3,
        model.Conv2d_3b_1x1, model.Conv2d_4a_3x3,
    ]:
        yield idx, m.conv
        idx += 1

    # InceptionA
    for block in [model.Mixed_5b, model.Mixed_5c, model.Mixed_5d]:
        for m in [
            block.branch1x1, block.branch5x5_1, block.branch5x5_2,
            block.branch3x3dbl_1, block.branch3x3dbl_2, block.branch3x3dbl_3,
            block.branch_pool,
        ]:
            yield idx, m.conv
            idx += 1

    # InceptionB
    for block in [model.Mixed_6a]:
        for m in [
            block.branch3x3,
            block.branch3x3dbl_1, block.branch3x3dbl_2, block.branch3x3dbl_3,
        ]:
            yield idx, m.conv
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
            yield idx, m.conv
            idx += 1

    # InceptionD
    for block in [model.Mixed_7a]:
        for m in [
            block.branch3x3_1, block.branch3x3_2,
            block.branch7x7x3_1, block.branch7x7x3_2,
            block.branch7x7x3_3, block.branch7x7x3_4,
        ]:
            yield idx, m.conv
            idx += 1

    # InceptionE
    for block in [model.Mixed_7b, model.Mixed_7c]:
        for m in [
            block.branch1x1, block.branch3x3_1, block.branch3x3_2a, block.branch3x3_2b,
            block.branch3x3dbl_1, block.branch3x3dbl_2,
            block.branch3x3dbl_3a, block.branch3x3dbl_3b,
            block.branch_pool,
        ]:
            yield idx, m.conv
            idx += 1


@register_module_iter("densenet121")
@register_module_iter("densenet169")
@register_module_iter("densenet201")
@register_module_iter("densenet161")
def register_densenet(model):
    idx = 0
    for m in [model.features.conv0]:
        yield idx, m
        idx += 1

    for blocks in [
        model.features.denseblock1, model.features.denseblock2,
        model.features.denseblock3, model.features.denseblock4,
    ]:
        for name, block in blocks.named_children():
            assert name.startswith("denselayer")
            for m in [block.conv1, block.conv2]:
                yield idx, m
                idx += 1
