import torch

import triton
import triton.language as tl

@torch.jit.script
def navive_softmax(x):
    """Compute row-wise softmax of X using native pytorch
    
    We subtract the maximum element in order to avoid overflows. Softmax is invariant to this shift.
    """
    # - read MN elements ; write M elements
    x_max = x.max(dim=1)[0]

    # - read MN + M elements ; write MN elements
    z = x - x_max[:, None]

    # - read MN elements ; write MN elements
    numerator = torch.exp(z)

    # - read MN elements; write M elemeents
    denominator = numerator.sum(dim = 1)

    # - readd MN + M elements; write MN elements
    ret = numerator / denominator[:, None]

    # - in total: read 5 MN + 2M elements ; wrote 3MN + 2M elements
    return ret