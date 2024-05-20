import torch

import triton
import triton.language as tl


### - Add Compute Kernel - automatical loop process
@triton.jit
def add_kernel(x_ptr, y_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr, # - Number of elements each program should process
               ):
    # - launch grid so axis is 0
    # if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256]

    # program_id: returns the id of the current program instance along the given axis
    # blocks ids at axis = 0: pid = 1
    pid = tl.program_id(axis = 0)

    # - offsets is a list of pointers
    # thread pointers? block_start: 1 * 64 = 64; offsets = [64, 65, .. 127]
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # - Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements

    # - load x and y from DRAM
    x = tl.load(x_ptr + offsets, mask = mask)
    y = tl.load(y_ptr + offsets, mask = mask)
    output = x + y

    # - write x + y back to DRAM
    tl.store(output_ptr + offsets, output, mask=mask)
    

### - Add Compute Program
def add(x: torch.Tensor, y: torch.Tensor):
    # - preallocate the output
    # is_cuda: check whether it is stored in gpu
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda
    n_elements = output.numel()

    # - # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # the returned value is the number of blocks
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.

    return output


### - Benchmark Test
@triton.testing.perf_report(
        triton.testing.Benchmark(
                    x_names = ['size'], # Argument names to use as an x-axis for the plot
        x_vals = [2**i for i in range(12, 28, 1)], # Different possible values for `x_name`.
        x_log = True, # x axis is Logarithmic
        line_arg = 'provider', # Argument name whose value corresponds to a different line in the plot.
        line_vals = ['triton', 'torch'],
        line_names = ['Triton', 'Torch'],
        styles = [('blue', '-'), ('green', '-')],
        ylabel = 'GB/s',
        plot_name='vector-add-performance',
        args={}, # Values for function arguments not in `x_names` and `y_name`.
        )
)
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    # - calculate bandwidth gigabyte per second for data troughput: two read x, y, one write output
    gbps = lambda ms: 12 * size
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device = 'cuda')
    y = torch.rand(size, device = 'cuda')
    output_torch = x + y
    output_triton = add(x, y)
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
    benchmark.run(print_data=True, show_plots=True)