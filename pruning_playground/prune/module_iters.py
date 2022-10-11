

def resnet_module_iter(model):
    idx = 0
    for m in [model.conv1]:
        yield idx, m
        idx += 1

    for blocks in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in blocks:
            for m in [block.conv1, block.conv2]:
                yield idx, m
                idx += 1
