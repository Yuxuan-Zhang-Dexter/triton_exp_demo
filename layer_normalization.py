import torch
import triton
import triton.language as tl

try:
    import apex
    HAS_APEX = True

except ModuleNotFoundError:
    HAS_APEX = False

# -----------------------------------------------------------
### - Forward Pass
@triton.jit
def _layer_norm_fwd_fused(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # - compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype = tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask = cols < N, other = 0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis = 0) / N
    # - compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype = tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask = cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * X
    var = tl.sum(_var, axis = 0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # - Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask = mask)
        b = tl.load(B + cols, mask = mask)
        x = tl.load(X + cols, mask = mask, other = 0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + B

        # - Write output
        tl.store(Y + cols, y, mask = mask)

# -----------------------------------------------------------
### - Backward Pass
@triton.jit
def _layer_norm_bwd_dx_fused(DX,
                             DY,
                             DW,
                             DB,
                             X,
                             W,
                             B,
                             Mean,
                             Rstd,
                             Lock,
                             stride,
                             N,
                             eps,
                             GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride

    # - Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols

    # - Load data to SRAM
    x = tl.load(X + cols, mask = mask, other = 0).to(tl.float32)
    dy = tl.load(DY + cols, mask = mask, other = 0.).to(tl.float32)
    w = tl.load(W + cols, mask = mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)

    # - Compute dx
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    c1 = tl.sum(xhat * wdy, axis = 0) / N
    c2 = tl.sum(wdy, axis = 0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd

    # - Write dx
    tl.store(DX + cols, dx, mask=mask)

    # - Accumulate partial sums for dw/db
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)

    # - First store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)

    # - Release the lock
    tl.atomic_xchg(Lock, 0)

@triton.jit
def