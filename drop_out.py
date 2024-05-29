import tabulate
import torch

import triton
import triton.language as tl

@triton.jit
def _dropout(
    x_ptr,
    x_keep_ptr,
    output_ptr,
    n_elements,
    p,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # - load data
    # - x is the input vector and x_keep is the mask vector to tell which element is masked in the x vector
    x = tl.load(x_ptr + offsets, mask = mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask = mask)

    # - tl.where(condition, x, y): return a tensor of elements from either x or y, depending on condition
    # - mask x with 0 based on x_keep
    # - x/(1 - p) guarantes consistent norms no matter what p is
    output = tl.where(x_keep, x / (1 - p), 0.0)

    # - Write-back output
    tl.store(output_ptr + offsets, output, mask = mask)


def dropout(x, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output

# -----------------------------------------------------------
### - Seeded Dropout
# - previous dropout implementation needs to store the dropout mask for backpropagation
# - (1) has a smaller memory footprint; 
# - (2) requires less data movement; 
# - (3) simplifies the management of persisting randomness across multiple invocations of the kernel.

@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # - compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # - load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # - randomly prune it
    # - triton.language.rand function which generates a block of uniformly distributed float32 values in [0, 1)
    random = tl.rand(seed, offsets)
    x_keep = random > p

    # - write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask = mask)

def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE = 1024)
    return output
    

if __name__ == "__main__":
    ### - vanilla dropout
    x = torch.randn(size = (10, )).cuda()
    p = 0.5
    x_keep = (torch.rand(size = (10, )) > p).to(torch.int32).cuda()
    output = dropout(x, x_keep = x_keep, p = p)
    print(tabulate.tabulate([
        ["input"] + x.tolist(),
        ["keep mask"] + x_keep.tolist(),
        ["output"] + output.tolist(),
    ]))
    ### - seeded dropout
    x = torch.randn(size = (10, )).cuda()
    # - Compare this to the baseline - dropout mask is never instantiated!
    output = seeded_dropout(x, p = 0.5, seed = 123)
    output2 = seeded_dropout(x, p = 0.5, seed = 123)
    output3 = seeded_dropout(x, p = 0.5, seed = 512)

    print(
        tabulate.tabulate([
            ["input"] + x.tolist(),
            ["output (seed = 123)"] + output.tolist(),
            ["output (seed = 123)"] + output2.tolist(),
            ["output (seed = 512)"] + output3.tolist(),
        ]))
