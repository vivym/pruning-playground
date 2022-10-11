from functools import reduce
from typing import Any, Dict, List, Tuple, Iterable

import torch
import torch.fx
import torch.nn as nn


def print_table(rows, header=['Operation', 'OPS']):
    r"""Simple helper function to print a list of lists as a table
    :param rows: a :class:`list` of :class:`list` containing the data to be printed. Each entry in the list
    represents an individual row
    :param input: (optional) a :class:`list` containing the header of the table
    """
    if len(rows) == 0:
        return
    col_max = [max([len(str(val[i])) for val in rows]) + 3 for i in range(len(rows[0]))]
    row_format = ''.join(["{:<" + str(length) + "}" for length in col_max])

    if len(header) > 0:
        print(row_format.format(*header))
        print(row_format.format(*['-' * (val - 2) for val in col_max]))

    for row in rows:
        print(row_format.format(*row))
    print(row_format.format(*['-' * (val - 3) for val in col_max]))


def same_device(model, input):
    # Remove dataparallel wrapper if present
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    # Make sure that the input is on the same device as the model
    if len(list(model.parameters())):
        input_device = input.device if not isinstance(input, Iterable) else input[0].device
        if next(model.parameters()).device != input_device:
            if isinstance(input, Iterable):
                for inp in input:
                    inp.to(next(model.parameters()).device)
            else:
                input.to(next(model.parameters()).device)

    return model, input


def _count_convNd(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    r"""Estimates the number of FLOPs in conv layer
    .. warning::
        Currently it ignore the padding
    :param node_string: an onnx node defining a convolutional layer
    :return: number of FLOPs
    :rtype: `int`
    """
    kernel_size = list(module.kernel_size)
    in_channels = module.in_channels
    out_channels = module.out_channels

    filters_per_channel = out_channels // module.groups
    conv_per_position_flops = reduce(lambda x, y: x * y, kernel_size) * \
        in_channels * filters_per_channel

    active_elements_count = output.shape[0] * reduce(lambda x, y: x * y, output.shape[2:])

    assert module.weight.numel() == conv_per_position_flops
    if hasattr(module, "weight_mask"):
        conv_per_position_flops = int(module.weight_mask.sum().item())

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_ops = 0
    if module.bias is not None:
        if hasattr(module, "bias_mask"):
            bias_ops = int(module.bias_mask.sum().item()) * active_elements_count
        else:
            bias_ops = out_channels * active_elements_count

    total_ops = overall_conv_flops + bias_ops

    return total_ops


def _count_relu(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    r"""Estimates the number of FLOPs of a  ReLU activation.
    The function will count the comparison operation as a FLOP.
    :param node_string: an onnx node defining a ReLU op
    :return: number of FLOPs
    :rtype: `int`
    """
    total_ops = 2 * output.numel()  # also count the comparison
    return total_ops


def _count_avgpool(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    r"""Estimates the number of FLOPs of an Average Pooling layer.
    :param node_string: an onnx node defining an average pooling layer
    :return: number of FLOPs
    :rtype: `int`
    """
    out_ops = output.numel()

    kernel_size = [module.kernel_size] * \
        (output.dim() - 2) if isinstance(module.kernel_size, int) else module.kernel_size

    ops_add = reduce(lambda x, y: x * y, kernel_size) - 1
    ops_div = 1
    total_ops = (ops_add + ops_div) * out_ops
    return total_ops


def _count_globalavgpool(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    r"""Estimates the number of FLOPs of an Average Pooling layer.
    :param node_string: an onnx node defining an average pooling layer
    :return: number of FLOPs
    :rtype: `int`
    """
    inp = args[0]

    ops_add = reduce(lambda x, y: x * y, [inp.shape[-2], inp.shape[-1]]) - 1
    ops_div = 1
    total_ops = (ops_add + ops_div) * output.numel()
    return total_ops


def _count_maxpool(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    r"""Estimates the number of FLOPs of a Max Pooling layer.
    :param node_string: an onnx node defining a max pooling layer
    :return: number of FLOPs
    :rtype: `int`
    """
    out_ops = output.numel()

    kernel_size = [module.kernel_size] * \
        (output.dim() - 2) if isinstance(module.kernel_size, int) else module.kernel_size
    ops_add = reduce(lambda x, y: x * y, kernel_size) - 1
    total_ops = ops_add * out_ops
    return total_ops


def _count_bn(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    r"""Estimates the number of FLOPs of a Batch Normalisation operation.
    :param node_string: an onnx node defining a batch norm op
    :return: number of FLOPs
    :rtype: `int`
    """
    total_ops = output.numel() * 2
    return total_ops


def _count_linear(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    r"""Estimates the number of a GEMM or linear layer.
    :param node_string: an onnx node defining a GEMM or linear layer
    :return: number of FLOPs
    :rtype: `int`
    """
    bias_ops = 0
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            bias_ops = output.shape[-1]
    total_ops = args[0].numel() * output.shape[-1] + bias_ops
    return total_ops


def _count_add_mul(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    r"""Estimates the number of FLOPs of a summation op.
    :param node_string: an onnx node defining a summation op
    :return: number of FLOPs
    :rtype: `int`
    """
    return output.numel() * len(args)


def _undefined_op(module: Any, output: torch.Tensor, args: Tuple[Any], kwargs: Dict[str, Any]) -> int:
    r"""Default case for undefined or free (in terms of FLOPs) operations
    :param node_string: an onnx node
    :return: always 0
    :rtype: `int`
    """
    return 0


def count_operations(module: Any) -> Any:
    if isinstance(module, torch.nn.modules.conv._ConvNd):
        return _count_convNd
    elif isinstance(module, nn.ReLU):
        return _count_relu
    elif isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        return _count_bn
    elif isinstance(module, torch.nn.modules.pooling._MaxPoolNd):
        return _count_maxpool
    elif isinstance(module, torch.nn.modules.pooling._AvgPoolNd):
        return _count_avgpool
    elif isinstance(module, torch.nn.modules.pooling._AdaptiveAvgPoolNd):
        return _count_globalavgpool
    elif isinstance(module, torch.nn.Linear):
        return _count_linear
    elif 'add' == module or 'mul' == module:
        return _count_add_mul
    elif 'matmul' == module:
        return _count_linear
    else:
        return _undefined_op


class ProfilingInterpreter(torch.fx.Interpreter):
    def __init__(self, mod: torch.nn.Module, custom_ops: Dict[str, Any] = {}):
        gm = torch.fx.symbolic_trace(mod)
        super().__init__(gm)

        self.custom_ops = custom_ops

        self.flops: Dict[torch.fx.Node, float] = {}
        self.parameters: Dict[torch.fx.Node, float] = {}

    def run_node(self, n: torch.fx.Node) -> Any:
        return_val = super().run_node(n)
        if isinstance(return_val, Tuple):
            self.flops[n] = return_val[1]
            self.parameters[n] = return_val[2]
            return_val = return_val[0]

        return return_val

    def call_module(self, target, args: Tuple[Any], kwargs: Dict[str, Any]) -> Any:
        # Execute the method and return the result
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        output = submod(*args, **kwargs)

        if submod in self.custom_ops:
            count_ops_funct = self.custom_ops[submod]
        else:
            count_ops_funct = count_operations(submod)
        current_ops = count_ops_funct(submod, output, args, kwargs)
        current_params = sum(p.numel() for p in submod.parameters())

        return output, current_ops, current_params

    def call_function(self, target, args: Tuple[Any], kwargs: Dict[str, Any]) -> Any:
        assert not isinstance(target, str)

        # Execute the function and return the result
        output = target(*args, **kwargs)

        current_ops = count_operations(target.__name__)(target, output, args, kwargs)

        return output, current_ops, 0


def count_flops(model: torch.nn.Module,
                 input: torch.Tensor,
                 custom_ops: Dict[Any,
                                  Any] = {},
                 ignore_layers: List[str] = [],
                 print_readable: bool = True,
                 verbose: bool = True,
                 *args):
    r"""Estimates the number of FLOPs of an :class:`torch.nn.Module`
    :param model: the :class:`torch.nn.Module`
    :param input: a N-d :class:`torch.tensor` containing the input to the model
    :param custom_ops: :class:`dict` containing custom counting functions. The keys represent the name
    of the targeted aten op, while the value a lambda or callback to a function returning the number of ops.
    This can override the ops present in the package.
    :param ignore_layers: :class:`list` containing the name of the modules to be ignored.
    :param print_readable: boolean, if True will print the number of FLOPs. default is True
    :param verbose: boolean, if True will print all the non-zero OPS operations from the network
    :return: number of FLOPs
    :rtype: `int`
    """
    model, input = same_device(model, input)

    # Place the model in eval mode, required for some models
    model_status = model.training
    model.eval()

    tracer = ProfilingInterpreter(model, custom_ops=custom_ops)
    tracer.run(input)

    ops = 0
    all_data = []

    for name, current_ops in tracer.flops.items():
        model_status = model.training

        if any(name.name == ign_name for ign_name in ignore_layers):
            continue

        ops += current_ops

        if current_ops and verbose:
            all_data.append(['{}'.format(name), current_ops])

    if print_readable:
        if verbose:
            print_table(all_data)
        print("Input size: {0}".format(tuple(input.shape)))
        print("{:,} FLOPs or approx. {:,.2f} GFLOPs".format(ops, ops / 1e+9))

    if model_status:
        model.train()

    return ops, all_data
