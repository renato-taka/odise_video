# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for channels of '2D' spatial NCHW tensors"""

    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps
        ).permute(0, 3, 1, 2)

'''
featureeExtractor继承两个类，nn.Module和ABCMeta，ABCmeta是python中的元类metaclass，用于定义抽象基类，
抽象基类是一个包含抽象方法、抽象属性的类，他不能被直接实例化，而是用于作为其他类的基类，并要字类实现抽象方法和属性
在这里，featureExtractor就是个抽象基类，他的子类需要实现它定义的抽象方法和属性，不然会引发错误
这个抽象基类只能被继承
'''
class FeatureExtractor(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def ignored_state_dict(self, destination=None, prefix=""):
        #destination是一个状态字典，用于存储状态字典
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        for name, module in self._modules.items():
            '''
            在 for name, module in self._modules.items(): 
            这个循环中，它遍历了 FeatureExtractor 类（或其子类）中的所有子模块，
            并使用 name 来访问子模块的名称，module 来访问子模块本身。这样可以对每个子模块执行相应的操作，
            比如在之前的代码中，检查子模块是否有特定的方法或属性。
            '''
            if module is not None and hasattr(module, "ignored_state_dict"):
                module.ignored_state_dict(destination, prefix + name + ".")
        return super().state_dict(destination=destination, prefix=prefix)

    # don't save DDPM model
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return OrderedDict()

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze()
        return self

    def _freeze(self):
        super().train(mode=False)
        for p in self.parameters():
            p.requires_grad = False

    @property
    @abstractmethod
    def feature_dims(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def feature_size(self) -> int:
        pass

    @property
    @abstractmethod
    def num_groups(self) -> int:
        pass

    @property
    @abstractmethod
    def grouped_indices(self, features):
        pass


def ensemble_logits_with_labels(
    logits: torch.Tensor, labels: List[List[str]], ensemble_method: str = "max"
):
    """Ensemble logits.
    Args:
        logits (torch.Tensor): logits of each model. The last dim is probability.
        labels (list[list[str]]): list of list of labels.
        ensemble_method (str): ensemble method. Options are 'mean' and 'max'.
    Returns:
        torch.Tensor: logits of ensemble model.
    """
    len_list = [len(l) for l in labels]
    assert logits.shape[-1] == sum(len_list), f"{logits.shape[-1]} != {sum(len_list)}"
    assert ensemble_method in ["mean", "max"]
    ensemble_logits = torch.zeros(
        *logits.shape[:-1], len(labels), dtype=logits.dtype, device=logits.device
    )
    if ensemble_method == "max":
        for i in range(len(labels)):
            ensemble_logits[..., i] = (
                logits[..., sum(len_list[:i]) : sum(len_list[: i + 1])].max(dim=-1).values
            )
    elif ensemble_method == "mean":
        for i in range(len(labels)):
            ensemble_logits[..., i] = logits[..., sum(len_list[:i]) : sum(len_list[: i + 1])].mean(
                dim=-1
            )
    else:
        raise ValueError(f"Unknown ensemble method: {ensemble_method}")

    return ensemble_logits


# Ref:https://stackoverflow.com/questions/27049998/convert-a-mixed-nested-list-to-a-nested-tuple
def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)
