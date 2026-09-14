"""int8 weight quantisation for the dynamics model.

Dynamic quantisation: weights are stored int8 and activations are quantised per
batch at runtime. It only touches ``nn.Linear``, which in this model is qkv,
the attention output projection and both MLP layers -- i.e. essentially all of
the FLOPs. Embeddings, LayerNorms and the softmax stay float.

CPU only. PyTorch's dynamic path lowers to fbgemm/qnnpack kernels that have no
CUDA equivalent; on GPU the equivalent win comes from a different route
(bitsandbytes, torchao, or TensorRT), which this repo does not ship.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def quantize_int8(model: nn.Module) -> nn.Module:
    if next(model.parameters()).device.type != "cpu":
        raise RuntimeError("int8 dynamic quantisation is CPU-only; run with --device cpu")
    import torch.ao.quantization as tq

    return tq.quantize_dynamic(model.eval(), {nn.Linear}, dtype=torch.qint8)


def weight_bytes(model: nn.Module) -> int:
    """Bytes held by parameters, counting quantised Linear layers correctly.

    A dynamically quantised Linear keeps its weight and bias in packed params,
    not in ``parameters()``, and its class is plain ``Linear``: only
    ``_get_name()`` says ``DynamicQuantizedLinear``. Matching on the class name
    silently left every int8 weight out of the total.
    """
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    for m in model.modules():
        name = m._get_name() if hasattr(m, "_get_name") else m.__class__.__name__
        if name.startswith("DynamicQuantizedLinear"):
            w = m.weight()
            total += w.numel() * w.element_size()
            b = m.bias()
            if b is not None:
                total += b.numel() * b.element_size()
    return total
