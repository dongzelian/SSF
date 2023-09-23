from numbers import Number
from typing import Callable
from torch_pruning import metapruner
from torch_pruning.importance import MagnitudeImportance
from torch_pruning.pruner.algorithms.scheduler import linear_scheduler
import torch
import torch.nn as nn

class BNScaleImportance(MagnitudeImportance):
    """Learning Efficient Convolutional Networks through Network Slimming, 
    https://arxiv.org/abs/1708.06519
    """

    def __init__(self, group_reduction='mean', normalizer='mean',module2name={},param2name={}):
        super().__init__(p=1, group_reduction=group_reduction, normalizer=normalizer)
        self.module2name = module2name
        self.parameter2name = param2name
    def __call__(self, group, ch_groups=1):
        group_imp = []
        has_ssf=False
        for dep, _ in group:
            module = dep.target.module
            name=module.__class__.__name__
            if self.module2name.__contains__(module):
                name = self.module2name[module]
            elif self.parameter2name.__contains__(module):
                name = self.parameter2name[module]
            if "ssf_scale" in name:
                local_imp = torch.abs(module.data)
                group_imp.append(local_imp)
        if len(group_imp) == 0:
            print("WARNING: no ssf_scale in this group")
            return None
        group_imp = torch.stack(group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp


class SSFScalePruner(metapruner.MetaPruner):
    def __init__(
        self,
        model,
        example_inputs,
        importance,
        reg=1e-5,
        iterative_steps=1,
        iterative_sparsity_scheduler: Callable = linear_scheduler,
        ch_sparsity=0.5,
        ch_sparsity_dict=None,
        global_pruning=False,
        max_ch_sparsity=1.0,
        round_to=None,
        ignored_layers=None,
        customized_pruners=None,
        unwrapped_parameters=None,
        output_transform=None,
    ):
        self.module2name={module:name for name,module in model.named_modules()}
        self.parameter2name={parameter:name for name,parameter in model.named_parameters()}
        importance=BNScaleImportance(module2name=self.module2name,param2name=self.parameter2name)
        super(SSFScalePruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            iterative_steps=iterative_steps,
            iterative_sparsity_scheduler=iterative_sparsity_scheduler,
            ch_sparsity=ch_sparsity,
            ch_sparsity_dict=ch_sparsity_dict,
            global_pruning=global_pruning,
            max_ch_sparsity=max_ch_sparsity,
            round_to=round_to,
            ignored_layers=ignored_layers,
            customized_pruners=customized_pruners,
            unwrapped_parameters=unwrapped_parameters,
            output_transform=output_transform,
        )
        self.reg = reg
    def regularize(self, model):
        # for m in model.modules():
        #     if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine==True:
        #         m.weight.grad.data.add_(self.reg*torch.sign(m.weight.data))
        for j,(name,parameters) in enumerate(model.named_parameters()):
            key=["ssf_scale","ssf_shift"]
            if any([k in name for k in key]):
                parameters.grad.data.add_(self.reg*torch.sign(parameters.data))
                # QUESTION: 这种正则化的叠加是否和模型结构、回传梯度有关系, 直接从 torch-pruning 中拿过来的