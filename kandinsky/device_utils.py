"""Device detection and management utilities for multi-backend support (CUDA, CPU)."""

import torch
from typing import Union, Literal
from functools import wraps


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """Check if Metal Performance Shaders (MPS) is available on macOS."""
    try:
        return torch.backends.mps.is_available()
    except AttributeError:
        # MPS not available on non-Apple platforms
        return False


def is_mps_device(device: Union[torch.device, str]) -> bool:
    """Check if device is MPS (Metal Performance Shaders)."""
    if isinstance(device, str):
        return device.lower() == "mps"
    return device.type == "mps"


def is_cpu_device(device: Union[torch.device, str]) -> bool:
    """Check if device is CPU."""
    if isinstance(device, str):
        return device.lower() == "cpu"
    return device.type == "cpu"


def get_device_type(device: Union[torch.device, str]) -> Literal["cuda", "mps", "cpu"]:
    """
    Extract device type string for use in torch.autocast and other APIs.

    Args:
        device: torch.device or string representation

    Returns:
        "cuda", "mps", or "cpu"
    """
    if isinstance(device, str):
        device_lower = device.lower()
        if "cuda" in device_lower:
            return "cuda"
        elif "mps" in device_lower:
            return "mps"
        else:
            return "cpu"
    return device.type


def get_default_device() -> torch.device:
    """
    Auto-detect best available device.

    Priority: CUDA > MPS > CPU

    Returns:
        torch.device instance
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_from_string(device_str: str) -> torch.device:
    """
    Parse device string and return torch.device.

    Handles:
    - "cuda" -> cuda:0
    - "cuda:1" -> cuda:1
    - "mps" -> mps
    - "cpu" -> cpu
    - None or empty -> auto-detect

    Args:
        device_str: Device specification string

    Returns:
        torch.device instance
    """
    if not device_str or device_str.lower() == "auto":
        return get_default_device()

    device_lower = device_str.lower()

    if device_lower == "cpu":
        return torch.device("cpu")

    if device_lower == "mps":
        if not is_mps_available():
            raise ValueError(
                "MPS device requested but Metal Performance Shaders is not available. "
                "This requires macOS with Apple Silicon GPU and PyTorch 1.12+."
            )
        return torch.device("mps")

    if "cuda" in device_lower:
        return torch.device(device_str)

    return torch.device("cpu")


def empty_cache(device: Union[torch.device, str] = None) -> None:
    """
    Device-agnostic cache clearing.

    Clears GPU cache for CUDA/MPS. No-op for CPU.

    Args:
        device: torch.device or string. If None, uses default device.
    """
    if device is None:
        device = get_default_device()

    if isinstance(device, str):
        device = get_device_from_string(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        try:
            torch.mps.empty_cache()
        except AttributeError:
            # torch.mps.empty_cache may not be available in older PyTorch versions
            pass


def get_dtype_for_device(device: Union[torch.device, str],
                         prefer_bfloat16: bool = True) -> torch.dtype:
    """
    Get recommended dtype for device.

    CUDA: bfloat16 (if available and preferred) or float16
    MPS: float16 (MPS doesn't support bfloat16)
    CPU: float32 (safer, better support)

    Args:
        device: torch.device or string
        prefer_bfloat16: Try to use bfloat16 on CUDA if available

    Returns:
        torch.dtype
    """
    if isinstance(device, str):
        device = get_device_from_string(device)

    if device.type == "cuda":
        if prefer_bfloat16 and torch.cuda.is_available():
            # Check if device supports bfloat16
            props = torch.cuda.get_device_properties(device)
            if props.major >= 8:  # Ampere or newer supports bfloat16
                return torch.bfloat16
        return torch.float16

    if device.type == "mps":
        # MPS doesn't support bfloat16, use float16 for better performance
        return torch.float16

    # CPU benefits from float32
    return torch.float32


def validate_device_availability(device: Union[torch.device, str]) -> None:
    """
    Validate that specified device is available.

    Raises ValueError if device is not available.

    Args:
        device: torch.device or string

    Raises:
        ValueError: If device is unavailable
    """
    if isinstance(device, str):
        device = get_device_from_string(device)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                f"CUDA device requested ({device}) but CUDA is not available. "
                "Install CUDA or use CPU instead."
            )
        # For specific GPU index, check if it exists
        if device.index is not None:
            if device.index >= torch.cuda.device_count():
                raise ValueError(
                    f"CUDA device {device.index} requested but only "
                    f"{torch.cuda.device_count()} devices available."
                )

    if device.type == "mps":
        if not is_mps_available():
            raise ValueError(
                f"MPS device requested ({device}) but Metal Performance Shaders is not available. "
                "This requires macOS with Apple Silicon GPU and PyTorch 1.12+."
            )


def log_device_info(device: Union[torch.device, str]) -> None:
    """
    Log device information and capabilities for debugging.

    Args:
        device: torch.device or string
    """
    if isinstance(device, str):
        device = get_device_from_string(device)

    print(f"\n{'='*60}")
    print(f"Device Configuration")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Device Type: {get_device_type(device)}")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name}")
        print(f"Compute Capability: {props.major}.{props.minor}")
        total_memory = props.total_memory / 1e9
        print(f"Total Memory: {total_memory:.1f} GB")

    elif device.type == "mps":
        print(f"GPU: Apple Metal Performance Shaders")
        print(f"Available: Yes")

    elif device.type == "cpu":
        print(f"Device: CPU")

    dtype = get_dtype_for_device(device)
    print(f"Data Type: {dtype}")
    print(f"{'='*60}\n")


def should_compile_for_device(device: Union[torch.device, str]) -> bool:
    """
    Determine if torch.compile should be enabled for a device.

    torch.compile on MPS is immature and adds significant overhead.
    CPU compilation is generally not beneficial.

    Args:
        device: torch.device or string

    Returns:
        True if compilation is recommended, False otherwise
    """
    if isinstance(device, str):
        device = get_device_from_string(device)

    # Only enable torch.compile for CUDA
    return device.type == "cuda"


def get_compile_mode_for_device(
    device: Union[torch.device, str],
    default_mode: str = "default"
) -> str:
    """
    Get recommended torch.compile mode for a device.

    Available modes:
    - "default": Standard compilation
    - "reduce-overhead": Optimized for smaller models/frequent calls
    - "max-autotune": Aggressive optimization (slower compilation)

    Args:
        device: torch.device or string
        default_mode: Mode to use when compilation is enabled

    Returns:
        Compile mode string if compilation recommended, or "default" for disabled
    """
    if isinstance(device, str):
        device = get_device_from_string(device)

    if not should_compile_for_device(device):
        # Return "default" for no-op when compilation disabled
        return "default"

    return default_mode


def compile_if_cuda(mode="default", **kwargs):
    """
    Conditionally apply torch.compile based on device.

    Only compiles for CUDA devices. MPS and CPU skip compilation to avoid overhead.

    Args:
        mode: torch.compile mode ("default", "reduce-overhead", "max-autotune", etc.)
        **kwargs: Additional arguments to torch.compile (dynamic, fullgraph, etc.)

    Returns:
        Decorator that applies torch.compile on CUDA, no-op otherwise

    Example:
        @compile_if_cuda()
        def my_function(x):
            return x * 2

        @compile_if_cuda(mode="max-autotune", dynamic=True)
        def optimized_function(x):
            return x + 1
    """
    def decorator(func):
        device = get_default_device()
        if should_compile_for_device(device):
            # CUDA: apply torch.compile
            return torch.compile(func, mode=mode, **kwargs)
        else:
            # MPS/CPU: skip compilation
            return func
    return decorator
