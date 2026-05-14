"""CUDA MCP server — provides nvidia-smi, VRAM, and CUDA version tools for the LLM.

Exposes tools the LLM can call directly:
- get_gpu_status — full nvidia-smi query (GPU name, driver, temp, VRAM, power)
- get_vram_usage — memory snapshot with training/quantization budget analysis
- get_cuda_version — CUDA toolkit + driver version + Blackwell/NVFP4 check
- get_gpu_clock — SM clock and memory clock (for performance profiling)
- check_nvcc — find and verify CUDA compiler (critical for Stage 2 + vLLM builds)
- check_nvfp4_support — verify Blackwell NVFP4 Tensor Core hardware support
- check_cuda_env — audit CUDA environment (PATH, toolkits, cuDNN)
- list_cuda_processes — show GPU-attached processes and their VRAM usage

Usage (configured automatically in opencode.json):
    uv run python scripts/cuda_mcp_server.py
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)

_GPU_INDEX = 0


def _run(args: list[str], timeout: int = 10) -> str:
    """Run a command and return stdout + stderr combined."""
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return (result.stdout.strip() + "\n" + result.stderr.strip()).strip()
    except FileNotFoundError:
        return f"ERROR: {args[0]} not found"
    except subprocess.TimeoutExpired:
        return f"ERROR: {args[0]} timed out after {timeout}s"


def _run_nvidia_smi(args: list[str]) -> str:
    return _run(["nvidia-smi", *args])


def get_gpu_status() -> str:
    """Full GPU status: name, driver, temperature, VRAM, power, utilization.

    Returns formatted nvidia-smi output for the primary GPU.
    Call this before starting any training or quantization run.
    """
    output = _run_nvidia_smi(
        [
            f"--id={_GPU_INDEX}",
            "--query-gpu=name,driver_version,utilization.gpu,utilization.memory,temperature.gpu,memory.used,memory.total,power.draw,power.limit,clocks.sm",
            "--format=csv,noheader",
        ]
    )

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        dev_name = torch.cuda.get_device_name(_GPU_INDEX) if cuda_ok else "N/A"
        free, total = torch.cuda.mem_get_info(_GPU_INDEX) if cuda_ok else (0, 0)
        free_gb = free / 1e9
        total_gb = total / 1e9
        used_gb = (total - free) / 1e9 if cuda_ok else 0

        lines = [
            "=== GPU STATUS ===",
            f"nvidia-smi: {output}",
            f"PyTorch CUDA: {'available' if cuda_ok else 'NOT AVAILABLE'}",
            f"Device:    {dev_name}",
            f"VRAM free: {free_gb:.2f} GB",
            f"VRAM used: {used_gb:.2f} GB",
            f"VRAM total:{total_gb:.2f} GB",
        ]
        return "\n".join(lines)
    except ImportError:
        return f"=== GPU STATUS ===\nnvidia-smi: {output}\nPyTorch CUDA: not importable"


def _parse_nvidia_smi_memory() -> tuple[float, float, float]:
    """Extract used, total, free VRAM (GB) from nvidia-smi."""
    output = _run_nvidia_smi(
        [
            f"--id={_GPU_INDEX}",
            "--query-gpu=memory.used,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    try:
        parts = [float(x.strip()) for x in output.split(",")]
        return parts[0] / 1024, parts[1] / 1024, parts[2] / 1024  # MiB -> GiB
    except (ValueError, IndexError):
        return 0.0, 16.0, 16.0


def get_vram_usage() -> str:
    """Detailed VRAM usage: allocated, reserved, free. Suitable for OOM debugging.

    Uses torch.cuda when available, falls back to nvidia-smi.
    """
    try:
        import torch  # noqa: F811
    except ImportError:
        torch = None

    has_torch_cuda = torch is not None and torch.cuda.is_available()

    if has_torch_cuda:
        free_bytes, total_bytes = torch.cuda.mem_get_info(_GPU_INDEX)
        allocated = torch.cuda.memory_allocated(_GPU_INDEX)
        reserved = torch.cuda.memory_reserved(_GPU_INDEX)
        free_gb = free_bytes / 1e9
        total_gb = total_bytes / 1e9
        alloc_gb = allocated / 1e9
        reserved_gb = reserved / 1e9
        used_gb = (total_bytes - free_bytes) / 1e9
        source = "torch.cuda"
    else:
        used_gb, total_gb, free_gb = _parse_nvidia_smi_memory()
        alloc_gb = 0.0
        reserved_gb = 0.0
        source = "nvidia-smi"

    lines = [
        "=== VRAM BUDGET ===",
        f"Source:     {source}",
        f"Total:      {total_gb:.2f} GB",
        f"Free:       {free_gb:.2f} GB",
        f"Used:       {used_gb:.2f} GB",
    ]

    if has_torch_cuda:
        lines += [
            f"Allocated:  {alloc_gb:.2f} GB (torch tensors)",
            f"Reserved:   {reserved_gb:.2f} GB (torch caching allocator)",
            f"Overhead:   {used_gb - alloc_gb:.2f} GB (fragmentation + context)",
        ]

    lines += ["", "=== BUDGET ANALYSIS ==="]

    if free_gb < 2:
        lines.append("CRITICAL: <2 GB free — OOM imminent")
    elif free_gb < 4:
        lines.append(f"WARNING: only {free_gb:.1f} GB free — tight for quantization")
    elif free_gb < 8:
        lines.append(f"OK: {free_gb:.1f} GB free — sufficient for inference, tight for training")
    elif free_gb < 14:
        lines.append(f"GOOD: {free_gb:.1f} GB free — can run training dry-run")
    else:
        lines.append(f"EXCELLENT: {free_gb:.1f} GB free — full training budget")

    lines += [
        "",
        "=== TRAINING BUDGET (QLoRA ZAYA1-8B) ===",
        "NF4 model:    ~7.2 GB",
        f"Available:    {free_gb:.1f} GB",
    ]
    room = free_gb - 7.2
    lines.append(f"Headroom:     {room:.1f} GB")
    if room < 2:
        lines.append("RESULT: INSUFFICIENT — need 9.5-11.5 GB for QLoRA training")
    elif room < 4:
        lines.append("RESULT: TIGHT — may OOM with large batch/seq length")
    else:
        lines.append(f"RESULT: OK — {room:.1f} GB headroom for training")

    lines += [
        "",
        "=== QUANTIZATION BUDGET (NVFP4 Stage 1) ===",
        "CT quantize:  ~6.2 GB peak",
        f"Available:    {free_gb:.1f} GB",
    ]
    if free_gb < 7:
        lines.append("RESULT: INSUFFICIENT — need 7+ GB for CT quantization")
    else:
        lines.append("RESULT: OK — sufficient for Stage 1 quantization")

    return "\n".join(lines)


def get_cuda_version() -> str:
    """CUDA toolkit and driver version, plus compute capability.

    Returns CUDA versions and sm_120 Blackwell capability check.
    """
    driver = _run_nvidia_smi(["--query-gpu=driver_version", "--format=csv,noheader"])

    try:
        import torch

        cuda_ver = torch.version.cuda or "N/A"
        cap = torch.cuda.get_device_capability(_GPU_INDEX) if torch.cuda.is_available() else (0, 0)
        sm_ver = f"sm_{cap[0]}{cap[1]}"
        is_blackwell = cap[0] >= 12

        lines = [
            "=== CUDA VERSIONS ===",
            f"NVIDIA driver:  {driver}",
            f"CUDA toolkit:   {cuda_ver}",
            f"Compute cap:    {cap[0]}.{cap[1]} ({sm_ver})",
            f"Blackwell:      {'YES (sm_120)' if is_blackwell else f'No (requires sm_120, got {sm_ver})'}",
            f"NVFP4 Tensor:   {'SUPPORTED' if is_blackwell else 'NOT SUPPORTED — requires Blackwell'}",
            f"FP8 native:     {'SUPPORTED' if cap[0] >= 9 else 'NOT SUPPORTED'}",
        ]
        return "\n".join(lines)
    except ImportError:
        return f"=== CUDA VERSIONS ===\nNVIDIA driver: {driver}\nCUDA toolkit: N/A (torch not importable)"


def get_gpu_clock() -> str:
    """SM clock and memory clock — useful for performance profiling."""
    fields = "clocks.current.sm,clocks.current.memory,clocks.max.sm,clocks.max.memory"
    sm = _run_nvidia_smi(
        [
            f"--id={_GPU_INDEX}",
            "--query-gpu=" + fields,
            "--format=csv,noheader",
        ]
    )
    return f"=== GPU CLOCKS (MHz) ===\ncurrent_sm,current_mem,max_sm,max_mem\n{sm}"


def check_nvcc() -> str:
    """Find and verify the CUDA compiler (nvcc). Critical for Stage 2 kernel
    development and vLLM CUDA builds. Checks PATH, WSL /usr/local/cuda, and
    Program Files.
    """
    lines = ["=== NVCC CHECK ==="]

    # 1. Check PATH
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        lines.append(f"nvcc in PATH: {nvcc_path}")
        ver = _run(["nvcc", "--version"], timeout=5)
        for line in ver.split("\n"):
            if "release" in line or "Cuda compilation" in line:
                lines.append(f"  {line.strip()}")
    else:
        lines.append("nvcc NOT in PATH")

    # 2. Check WSL /usr/local/cuda
    if shutil.which("wsl"):
        result = _run(
            [
                "wsl",
                "-d",
                "Ubuntu",
                "--",
                "bash",
                "-c",
                "ls /usr/local/cuda/bin/nvcc 2>/dev/null && "
                "/usr/local/cuda/bin/nvcc --version 2>/dev/null | grep 'release' "
                "|| echo 'NOT FOUND in WSL /usr/local/cuda'",
            ],
            timeout=10,
        )
        lines.append(f"\nWSL2 /usr/local/cuda: {result.strip()}")

    # 3. Check common Windows locations
    candidates = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit",
    ]
    for base in candidates:
        if os.path.isdir(base):
            versions = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and d.startswith("v")]
            if versions:
                lines.append(f"\nInstalled CUDA toolkits in {base}:")
                for v in sorted(versions):
                    nvcc = os.path.join(base, v, "bin", "nvcc.exe")
                    found = " (has nvcc)" if os.path.isfile(nvcc) else ""
                    lines.append(f"  {v}{found}")

    # 4. Summary
    lines.append("")
    if nvcc_path:
        lines.append("RESULT: nvcc FOUND — CUDA compilation available")
    else:
        lines.append("RESULT: nvcc NOT FOUND — cannot compile CUDA kernels")
        lines.append("  Fix: export PATH=/usr/local/cuda/bin:$PATH (WSL)")
        lines.append("  Or: install CUDA Toolkit from NVIDIA developer site")

    return "\n".join(lines)


def check_nvfp4_support() -> str:
    """Verify Blackwell NVFP4 Tensor Core hardware support.

    Checks GPU architecture, driver version, and reported NVFP4 support flags.
    """
    lines = ["=== NVFP4 SUPPORT CHECK ==="]

    # GPU architecture from nvidia-smi
    arch = _run_nvidia_smi(
        [
            f"--id={_GPU_INDEX}",
            "--query-gpu=name,compute_cap,gpu_bus_id",
            "--format=csv,noheader",
        ]
    )
    lines.append(f"GPU: {arch}")

    # Detailed compute capability
    cc = _run_nvidia_smi(
        [
            f"--id={_GPU_INDEX}",
            "--query-gpu=compute_cap",
            "--format=csv,noheader",
        ]
    )
    try:
        cap_major = int(cc.split(".")[0]) if "." in cc else 0
        lines.append(f"Compute capability: {cc}")
        lines.append(f"Major version: {cap_major}")

        if cap_major >= 12:
            lines.append("ARCHITECTURE: Blackwell (sm_120)")
            lines.append("NVFP4 Tensor Cores: SUPPORTED")
            lines.append("MMA instructions: fp4 MMA available on sm_120")
            lines.append("")
            lines.append("Stage 2 custom kernel: VIABLE")
            lines.append("  Target PTX ISA: sm_120 (Blackwell NVFP4 MMA)")
            lines.append("  Weight format: uint8 packed (2x 4-bit per byte)")
            lines.append("  Scale format: FP8_E4M3 per group of 16")
        elif cap_major >= 9:
            lines.append(f"ARCHITECTURE: Hopper (sm_{cc.replace('.', '')})")
            lines.append("NVFP4 Tensor Cores: NOT SUPPORTED")
            lines.append("  Requires Blackwell (sm_120+). Only FP8 native on Hopper.")
        elif cap_major >= 8:
            lines.append(f"ARCHITECTURE: Ampere (sm_{cc})")
            lines.append("NVFP4 Tensor Cores: NOT SUPPORTED")
        else:
            lines.append(f"ARCHITECTURE: sm_{cc} (pre-Ampere)")
            lines.append("NVFP4 Tensor Cores: NOT SUPPORTED")
    except (ValueError, IndexError):
        lines.append("Compute capability: UNKNOWN")

    # Driver check — NVFP4 requires driver >= 570
    driver = _run_nvidia_smi(
        [
            f"--id={_GPU_INDEX}",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ]
    )
    lines.append(f"\nDriver: {driver}")
    try:
        driver_major = int(driver.split(".")[0]) if driver else 0
        if driver_major >= 570:
            lines.append("Driver OK: >= 570 required for NVFP4")
        else:
            lines.append("WARNING: Driver < 570 — may lack NVFP4 support")
    except ValueError:
        pass

    return "\n".join(lines)


def check_cuda_env() -> str:
    """Audit CUDA environment: PATH, CUDA_HOME, toolkit installations.

    Useful for debugging "CUDA not found" or "nvcc not in PATH" issues.
    """
    lines = ["=== CUDA ENVIRONMENT AUDIT ==="]

    # Key environment variables
    for var in ["CUDA_HOME", "CUDA_PATH", "PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF"]:
        val = os.environ.get(var, "(not set)")
        if var == "PATH" and len(val) > 500:
            # Show only CUDA-relevant paths
            cuda_paths = [p for p in val.split(os.pathsep) if "cuda" in p.lower()]
            if cuda_paths:
                lines.append("CUDA in PATH:")
                for p in cuda_paths:
                    lines.append(f"  {p}")
            else:
                lines.append(f"{var}: (no CUDA paths found)")
        else:
            display = val if len(val) < 200 else val[:200] + "..."
            lines.append(f"{var}: {display}")

    # Check nvcc in PATH
    nvcc = shutil.which("nvcc")
    lines.append(f"\nnvcc in PATH: {nvcc or 'NOT FOUND'}")

    # Check common CUDA toolkit locations
    lines.append("")
    lines.append("CUDA Toolkit installations:")
    found = False
    for base in [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        "/usr/local/cuda",
        "/usr/local/cuda-13",
        "/usr/local/cuda-12",
    ]:
        if os.path.isdir(base):
            found = True
            lines.append(f"  {base}")
            for entry in sorted(os.listdir(base)):
                lines.append(f"    {entry}")

    if not found:
        lines.append("  (none found in common locations)")

    # Check CUDA Python availability
    lines.append("")
    try:
        import torch

        torch_cuda = torch.version.cuda or "N/A"
        lines.append(f"PyTorch CUDA version: {torch_cuda}")
        lines.append(f"torch available: {torch.cuda.is_available()}")
    except ImportError:
        lines.append("PyTorch CUDA: not importable in current env")

    return "\n".join(lines)


def list_cuda_processes() -> str:
    """Show GPU-attached processes and their VRAM usage.

    Identifies what's consuming GPU memory — critical for debugging OOMs.
    """
    output = _run_nvidia_smi(
        [
            f"--id={_GPU_INDEX}",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    lines = ["=== CUDA PROCESSES ==="]
    if not output or "ERROR" in output:
        lines.append("No processes found or nvidia-smi unavailable")
        return "\n".join(lines)

    total_mb = 0
    for proc in output.split("\n"):
        proc = proc.strip()
        if not proc:
            continue
        parts = [p.strip() for p in proc.split(",")]
        if len(parts) >= 3:
            pid, name, mem = parts[0], parts[1], parts[2]
            mem_mb = float(mem) if mem.replace(".", "").isdigit() else 0
            total_mb += mem_mb
            mem_gb = mem_mb / 1024
            lines.append(f"PID {pid}  {name}  {mem_gb:.2f} GB")

    lines.append(f"\nTotal used by processes: {total_mb / 1024:.2f} GB")

    # Add zombie check
    if total_mb > 2048:  # > 2 GB
        lines.append("WARNING: >2 GB used by other processes. Check for zombie training/inference processes.")

    return "\n".join(lines)


def _serve() -> None:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    server = Server("cuda-monitor")

    @server.tool()
    def get_gpu_status_tool() -> str:
        """GPU status: name, driver, temp, VRAM, power, utilization.

        Call before starting training or quantization.
        """
        return get_gpu_status()

    @server.tool()
    def get_vram_usage_tool() -> str:
        """VRAM usage with training/quantization budget analysis.

        Call when debugging OOM or sizing a run.
        """
        return get_vram_usage()

    @server.tool()
    def get_cuda_version_tool() -> str:
        """Get CUDA toolkit version, driver version, compute capability, and Blackwell/NVFP4 support check."""
        return get_cuda_version()

    @server.tool()
    def get_gpu_clock_tool() -> str:
        """Get current SM and memory clock speeds. Call for performance profiling."""
        return get_gpu_clock()

    @server.tool()
    def check_nvcc_tool() -> str:
        """Find and verify the CUDA compiler. Call before attempting any CUDA
        kernel compilation (Stage 2, vLLM builds).
        """
        return check_nvcc()

    @server.tool()
    def check_nvfp4_support_tool() -> str:
        """Verify Blackwell NVFP4 Tensor Core support on this GPU.
        Call before starting NVFP4 quantization or kernel work.
        """
        return check_nvfp4_support()

    @server.tool()
    def check_cuda_env_tool() -> str:
        """Full CUDA environment audit: PATH, toolkits, environment variables.
        Call when debugging 'CUDA not found' or 'nvcc not in PATH'.
        """
        return check_cuda_env()

    @server.tool()
    def list_cuda_processes_tool() -> str:
        """List GPU-attached processes and their VRAM usage.
        Call when debugging OOM to find what's consuming GPU memory.
        """
        return list_cuda_processes()

    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    import asyncio

    asyncio.run(run())


if __name__ == "__main__":
    _serve()
