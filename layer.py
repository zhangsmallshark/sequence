# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# The file has been adapted from the following Megatron-LM file:
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/mpu/mappings.py
# Git commit hash: 9dc3c42a84aa656f583703cf8b6b4f79f712b796
# We retain the following copyright from the original files:

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


def _all2all(input_,group, scatter_dim, gather_dim):
    """All2All for sequence parallel"""

    shapes = list(input_.size())
    seq_world_size = len(torch.distributed.get_process_group_ranks(group))
    #if torch.distributed.get_rank() == 0:
    #    print('FREEZE seq grp size ', seq_world_size)
    shapes[scatter_dim] = shapes[scatter_dim] // seq_world_size

    input_list = [t.contiguous() for t in torch.tensor_split(input_, seq_world_size, scatter_dim)]
    output_list = [torch.empty(*shapes, dtype=input_.dtype, device=input_.device) 
                   for _ in range(seq_world_size)]
    torch.distributed.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()

class _SequenceAll2All(torch.autograd.Function):
    """All2All."""

    @staticmethod
    def forward(ctx,input_,group,scatter_idx,gather_idx):
        ctx.group = group #process group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        return _all2all(input_,group,scatter_idx,gather_idx)

    @staticmethod
    def backward(ctx, grad_output):
        return _all2all(grad_output,ctx.group, ctx.gather_idx,ctx.scatter_idx),None,None,None


def sequence_all2all(input_,group, scatter_idx,gather_idx):
    return _SequenceAll2All.apply(input_,group,scatter_idx,gather_idx)

