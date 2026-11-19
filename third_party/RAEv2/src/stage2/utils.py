"""Tensor condition helper retained for the stock RAEv2 transport."""
import torch

def apply_cfg_dropout(model_conds, model_conds_null, cfg_dropout_prob=0.1):
    mask = torch.rand(model_conds['context'].shape[0], device=model_conds['context'].device) < cfg_dropout_prob
    return {
        k: torch.where(mask.view(-1, *([1]*(v.ndim-1))), model_conds_null[k], v) if v is not None else None
        for k, v in model_conds.items()
    }, mask
