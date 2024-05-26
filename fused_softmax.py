import torch

import triton
import triton.language as tl

### - Vanallia Implementation
@torch.jit.script
def naive_softmax(x):
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
    # - we want to only read and write MN once so the speed up is ~4x (8MN + 4M) / 2MN
    return ret

### - Compiled Implementation
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # - parallelize rows of the softmax, which are independent
    row_idx = tl.program_id(0)

    # - row stride from one row pointer to another row pointer
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # - The block size is the next power of two greater than n_cols
    # - col_offsets help point to row elements in the block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    # - load the row into SRAM and mask elements with float -inf beyond n_cols
    row = tl.load(input_ptrs, mask = col_offsets < n_cols, other = -float('inf'))

    # - subtrac maximum for numerical stability (only deal with one row, reduce dimension)
    row_minus_max = row - tl.max(row, axis = 0)

    # - triton exp is fast but approximate
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis = 0)
    softmax_output = numerator / denominator
    
    # - write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask = col_offsets < n_cols)

def softmax(x):
    n_rows, n_cols = x.shape
    # - the block size is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # - Another trick we can use is to ask the compiler to use more threads per row by
    # - increasing the number of warps (`num_warps`) over which each row is distributed.

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    
    # - allocate output
    y = torch.empty_like(x)
    
    # - Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row o
    # - f the input matri

    softmax_kernel[(n_rows, )](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps = num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y

### - Benchmark Test
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals = [128 * i for i in range(2, 100)],
        line_arg='provider',
        line_vals=[
            'triton',
            'torch-native',
            'torch-jit',
        ],
        line_names=[
            "Triton",
            "Torch (native)",
            "Torch (jit)",
        ],
        styles = [('blue', '-'), ('green', '-'), ('green', '--')],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={'M': 4096},
    )
)

def benchmark(M, N, provider):
    x = torch.randn(M, N, device = 'cuda', dtype = torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, axis = -1), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles = quantiles)
    if provider == 'torch-jit':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_softmax(x), quantiles=quantiles)
    # - reading and writing is 2, nelement is the number of elements, element size is 4 bytes
    # - gbps = gigabytes / seconds
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)



if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device='cuda')
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis = 1)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
    benchmark.run(show_plots=True, print_data=True, save_path='./fig/')