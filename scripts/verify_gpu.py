"""GPU sanity gate — proves Blackwell sm_120 kernels actually execute.

Not just `torch.cuda.is_available()`: it runs a real matmul and synchronises, which is what
surfaces a "no kernel image is available for execution" arch mismatch. Run before any training.

    uv run python scripts/verify_gpu.py
"""

from __future__ import annotations


def main() -> None:
    import torch

    print("torch:", torch.__version__, "| compiled CUDA:", torch.version.cuda)
    assert torch.cuda.is_available(), "CUDA not available — driver missing or nouveau bound"

    arch = torch.cuda.get_arch_list()
    print("arch_list:", arch)
    assert "sm_120" in arch, f"wheel lacks sm_120 Blackwell kernels: {arch}"

    cap = torch.cuda.get_device_capability(0)
    assert cap == (12, 0), f"expected Blackwell (12, 0), got {cap}"

    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    c = a @ b
    torch.cuda.synchronize()  # force the kernel to actually run — a lazy matmul can hide arch errors

    free, total = torch.cuda.mem_get_info()
    print("device:", torch.cuda.get_device_name(0))
    print("matmul checksum:", float(c.sum()))
    print(f"VRAM: {(total - free) / 1e9:.2f} GB used / {total / 1e9:.2f} GB total")
    print("ALL CHECKS PASSED — sm_120 kernels execute.")


if __name__ == "__main__":
    main()
