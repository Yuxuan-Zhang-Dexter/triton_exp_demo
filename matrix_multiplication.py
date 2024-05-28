import torch

import triton
import triton.language as tl


def is_cuda():
    return torch.cuda.is_available()

def is_hip_mi200():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            # Check if the device name matches an AMD MI200 series
            if "MI200" in device_name:  # Example check; replace with actual device name substring
                return True
    return False


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]

def get_hip_autotune_config():
    return [
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=4, num_stages=0),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2},
            num_warps=8, num_stages=0),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=0),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3},
            num_warps=4, num_stages=0),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 8},
            num_warps=4, num_stages=0),
    ]

def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    elif is_hip_mi200():
        return get_hip_autotune_config()
    else:
        raise RuntimeError("No compatible GPU found for autotuning.")


# - Build matmul_kernel
# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs = get_autotune_config(),
    key = ['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # - Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    # -----------------------------------------------------------
    ### - L2 Cache Optimizations
    # - triton is block-level, pid is ids of blocks. 
    # - triton compiler automatically define inside of the block
    # - we will map pid one dimension into two dimensions (pid_m, pid_n) like an array to a coordinate
    pid = tl.program_id(axis = 0)

    # - On M dimension, the number of blocks M/BLOCK_SIZE_M
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)

    # - On N dimension, the number of blocks N/BLOCK_SIZE_N
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # - GROUP_SIZE_M is the number of blocks is grouped on the M dimension
    # - num_pid_in_group is the total number of our grouped blocks
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # - group_id is to determine which group pid block belongs to
    group_id = pid // num_pid_in_group

    # - check the first pid of each group at the M dimention
    first_pid_m = group_id * GROUP_SIZE_M

    # - some group sizes are smaller than the GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    # - *Within groups*, programs are ordered in a column-major order
    # - pid_m and pid_n are like coordinates of blocks in output matrix
    # - Row-id of the program in the *launch grid*
    pid_m = first_pid_m + (pid % group_size_m)
    # - Col-id of the program in the *launch grid*
    pid_n = (pid % num_pid_in_group) // group_size_m

     # ----------------------------------------------------------
    # - Create pointers for the first blocks of A and B.
    # - We will advance this pointer as we move in the K direction
    # - and accumulate
    # - `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # - `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers

    # - at one group, offset points to M element on M dimension
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    # - at one group, offset points to N element on N dimension
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    # - similar, on k dimension
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # - calcualte a b pointers position based on stride 
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # - Iterate to compute a block of the C matrix.
    # - We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # - of fp32 values for higher accuracy.
    # - `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0
        a = tl.load(a_ptrs, mask = offs_k[None, :] < K - k * BLOCK_SIZE_K, other = 0.0)
        b = tl.load(b_ptrs, mask = offs_k[:, None] < K - k * BLOCK_SIZE_K, other = 0.0)

        # - We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)

        # - BLOCK_SIZE_M * BLOCK_SIZE_K x BLOCK_SIZE_K * BLOCK_SIZE_N = BLOCK_SIZE_M * BLOCK_SIZE_N
        # - Aggregate BLOCK_SIZE_M * BLOCK_SIZE_N on K dimension K / BLOCK_SIZE_K times
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # - Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask = c_mask)

@triton.jit
def leaky_relu(x):
    x = x +1
    return tl.where(x >= 0, x, 0.01 * x)

# -----------------------------------------------------------
### - define matmul function
def matmul(a, b, activation =""):
    # - Check constratins
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device = a.device, dtype = torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,
        M, N, K, 
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION = activation
    )

    return c

# -----------------------------------------------------------
### - define benchmark functions
TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

configs = []
for fp8_inputs in [False, True]:
    if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
        continue
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["triton"] if fp8_inputs else [ref_lib.lower(), "triton"],  # Label name for the lines
            line_names=["Triton"] if fp8_inputs else [ref_lib, "Triton"],  # Line styles
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-" +
            ("fp16" if not fp8_inputs else "fp8"),  # Name for the plot, used also as a file name for saving the plot.
            args={"fp8_inputs": fp8_inputs},
        ))


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    if TORCH_HAS_FP8 and fp8_inputs:
        a = a.to(torch.float8_e5m2)
        b = b.T
        b = b.to(torch.float8_e5m2)
    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    ### - Unit Test
    torch.manual_seed(0)
    a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    print(f"triton_output_with_fp16_inputs={triton_output}")
    print(f"torch_output_with_fp16_inputs={torch_output}")
    # - Bigger tolerance for AMD MI200 devices.
    # - MI200 devices use reduced precision fp16 and bf16 and flush input and
    # - output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    rtol = 1e-2 if is_hip_mi200() else 0
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
        
        # Compute absolute and relative differences
        abs_diff = torch.abs(triton_output - torch_output)
        rel_diff = abs_diff / torch.abs(torch_output)
        
        # Get the maximum differences
        max_abs_diff = torch.max(abs_diff).item()
        max_rel_diff = torch.max(rel_diff).item()

        # Find mismatched elements
        mismatched = torch.nonzero(abs_diff > (1e-2 + rtol * torch.abs(torch_output)), as_tuple=True)
        mismatched_elements = mismatched[0].tolist()

        # Collect mismatched elements into arrays
        triton_mismatch = triton_output.flatten()[mismatched_elements].cpu().numpy()
        torch_mismatch = torch_output.flatten()[mismatched_elements].cpu().numpy()

        # Calculate the percentage of mismatched elements
        total_elements = torch_output.numel()
        mismatch_percentage = (len(mismatched_elements) / total_elements) * 100

        # Print mismatched elements details
        print(f"Mismatched elements: {len(mismatched_elements)}")
        print(f"Percentage of mismatched elements: {mismatch_percentage:.2f}%")
        print(f"Max absolute difference: {max_abs_diff}")
        print(f"Max relative difference: {max_rel_diff}")
        print("Triton mismatched elements:", triton_mismatch)
        print("Torch mismatched elements:", torch_mismatch)

    TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
    if TORCH_HAS_FP8 and is_cuda():
        torch.manual_seed(0)
        a = torch.randn((512, 512), device="cuda", dtype=torch.float16)
        b = torch.randn((512, 512), device="cuda", dtype=torch.float16)
        a = a.to(torch.float8_e5m2)
        # pre-transpose b for efficiency.
        b = b.T
        b = b.to(torch.float8_e5m2)
        triton_output = matmul(a, b)
        torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
        print(f"triton_output_with_fp8_inputs={triton_output}")
        print(f"torch_output_with_fp8_inputs={torch_output}")
        if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
            print("✅ Triton and Torch match")
        else:
            print("❌ Triton and Torch differ")
            # Compute absolute and relative differences
            abs_diff = torch.abs(triton_output - torch_output)
            rel_diff = abs_diff / torch.abs(torch_output)

            # Get the maximum differences
            max_abs_diff = torch.max(abs_diff).item()
            max_rel_diff = torch.max(rel_diff).item()

            # Find mismatched elements
            mismatched = torch.nonzero(abs_diff > (1e-2 + rtol * torch.abs(torch_output)), as_tuple=True)
            mismatched_elements = mismatched[0].tolist()

            # Collect mismatched elements into arrays
            triton_mismatch = triton_output.flatten()[mismatched_elements].cpu().numpy()
            torch_mismatch = torch_output.flatten()[mismatched_elements].cpu().numpy()

            # Calculate the percentage of mismatched elements
            total_elements = torch_output.numel()
            mismatch_percentage = (len(mismatched_elements) / total_elements) * 100

            # Print mismatched elements details
            print(f"Mismatched elements: {len(mismatched_elements)}")
            print(f"Percentage of mismatched elements: {mismatch_percentage:.2f}%")
            print(f"Max absolute difference: {max_abs_diff}")
            print(f"Max relative difference: {max_rel_diff}")
            print("Triton mismatched elements:", triton_mismatch)
            print("Torch mismatched elements:", torch_mismatch)

    




        
        


