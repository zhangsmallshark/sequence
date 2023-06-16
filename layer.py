# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.utils import log_dist

from .mappings import sequence_all2all
from deepspeed.runtime.utils import see_memory_usage


class DistributedAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
        TODO
        (1) Create process group internal to deepspeed if none
    """

    def __init__(self,
                 local_attention,
                 sequence_process_group,
                 scatter_idx=2,
                 gather_idx=0,
                 ):

        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group 
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        
    def forward(self, query, key, value):
        """ forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer

        Returns:
            * output (Tensor): context output in the format sq/p, b, h
        """
        #timers('Attn-preproc').stop()
        #see_memory_usage(f"Before ALL2ALL QKV", force=True)
        #log_dist('FREEZE  q k v PRE shape ', query.shape, key.shape, value.shape)
        
        #in shape : [s/p:h:]
        query_layer = sequence_all2all(query,self.spg,self.scatter_idx,self.gather_idx)
        key_layer = sequence_all2all(key,self.spg,self.scatter_idx,self.gather_idx)
        value_layer = sequence_all2all(value,self.spg,self.scatter_idx,self.gather_idx)
        
        #out shape : [s:h/p:]
        context_layer = self.local_attn(query_layer, key_layer, value_layer)

    
        #timers('Attn-ALL2ALLOUT').start()
        
        #in [s::h/p]
        output = sequence_all2all(context_layer,self.spg, self.gather_idx,self.scatter_idx)
        #out [s/p::h]
        #log_dist('ALL2ALL out post ', context_layer.shape)
        #timers('Attn-ALL2ALLOUT').stop()
        #see_memory_usage(f"After ALL2ALL Context Layer ", force=True)
        #print(f'context layer AFTER ALL {} '.format(context_layer.shape))
        return output
        
