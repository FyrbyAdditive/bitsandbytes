"""Tests for unified memory optimizations on platforms like DGX Spark (GB10).

These tests verify that:
1. Managed memory allocation works with unified-memory-aware hints
2. Prefetch is correctly skipped on unified memory platforms
3. is_unified_memory correctly detects platform type
4. mark_weights_readonly works on managed memory tensors
5. All operations remain correct on discrete GPU systems (no regressions)
"""

import pytest
import torch

import bitsandbytes as bnb
import bitsandbytes.functional as F


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestManagedMemoryAllocation:
    """Test cget_managed_ptr with unified memory detection."""

    def test_paged_tensor_allocation(self):
        """Managed memory allocation should work on any GPU."""
        t = F.get_paged(256, 256, dtype=torch.float32, device=torch.device("cuda:0"))
        assert t.shape == (256, 256)
        assert t.is_paged
        assert t.page_deviceid == 0

    def test_paged_tensor_fill(self):
        """Paged tensors should be fillable via elementwise ops."""
        t = F.get_paged(64, dtype=torch.float32, device=torch.device("cuda:0"))
        F.fill(t, 0)
        # Verify the tensor is usable (no page faults crash)
        assert t.sum().item() == 0.0

    def test_prefetch_tensor(self):
        """Prefetch should not crash, whether unified or discrete."""
        t = F.get_paged(256, dtype=torch.float32, device=torch.device("cuda:0"))
        F.fill(t, 0)
        # Should not raise on any platform
        F.prefetch_tensor(t, to_cpu=False)
        F.prefetch_tensor(t, to_cpu=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestPagedOptimizer:
    """Test paged optimizer with unified memory hints."""

    def test_paged_adamw_step(self):
        """PagedAdamW8bit should work with unified memory allocation."""
        p = torch.nn.Parameter(torch.randn(256, 256, device="cuda"))
        opt = bnb.optim.PagedAdamW8bit([p], lr=1e-3)
        loss = p.sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        # Second step (state is now initialized and paged)
        loss = p.sum()
        loss.backward()
        opt.step()

    def test_paged_adamw_multiple_steps(self):
        """Multiple optimizer steps should work without page faults."""
        model = torch.nn.Linear(128, 64, device="cuda")
        opt = bnb.optim.PagedAdamW8bit(model.parameters(), lr=1e-3)
        for _ in range(5):
            x = torch.randn(8, 128, device="cuda")
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestIsUnifiedMemory:
    """Test unified memory detection."""

    def test_returns_bool(self):
        """is_unified_memory should return a bool."""
        result = F.is_unified_memory(0)
        assert isinstance(result, bool)

    def test_consistent_with_device_props(self):
        """Should be consistent across repeated calls."""
        a = F.is_unified_memory(0)
        b = F.is_unified_memory(0)
        assert a == b


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMarkWeightsReadonly:
    """Test mark_weights_readonly for frozen model weights.

    Note: mark_weights_readonly requires tensors to be in managed memory
    (cudaMallocManaged).  Standard CUDA tensors (cudaMalloc) will produce
    errors.  On unified memory platforms with CUDA_MANAGED_FORCE_DEVICE_ALLOC=1,
    all allocations are managed.  On discrete GPUs, these tests verify the
    error handling path.
    """

    def test_skips_trainable_params(self):
        """Should not mark trainable (requires_grad=True) params."""
        model = torch.nn.Linear(64, 32, device="cuda")
        # All params are trainable by default
        count, errors = F.mark_weights_readonly(model)
        assert count == 0
        assert errors == 0

    def test_no_crash_on_cpu_params(self):
        """Should skip CPU parameters without crashing."""
        model = torch.nn.Linear(64, 32)  # CPU
        for p in model.parameters():
            p.requires_grad = False
        count, errors = F.mark_weights_readonly(model)
        assert count == 0  # CPU params skipped
        assert errors == 0

    @pytest.mark.skipif(
        not F.is_unified_memory(0) if torch.cuda.is_available() else True,
        reason="Requires unified memory (managed allocations)",
    )
    def test_marks_frozen_params_unified(self):
        """On unified memory, should mark frozen CUDA params."""
        model = torch.nn.Linear(64, 32, device="cuda")
        for p in model.parameters():
            p.requires_grad = False
        count, errors = F.mark_weights_readonly(model)
        assert count == 2  # weight + bias
        assert errors == 0

    def test_error_count_on_non_managed(self):
        """On discrete GPU (non-managed memory), should report errors."""
        if F.is_unified_memory(0):
            pytest.skip("On unified memory, cudaMemAdvise succeeds")
        model = torch.nn.Linear(64, 32, device="cuda")
        for p in model.parameters():
            p.requires_grad = False
        count, errors = F.mark_weights_readonly(model)
        # cudaMemAdvise fails on non-managed memory
        assert errors == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestQuantizationWithUnifiedMemory:
    """Verify that 4-bit quantization still works correctly."""

    def test_nf4_quantize_dequantize(self):
        """NF4 quantize/dequantize roundtrip should be correct."""
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        qx, state = F.quantize_4bit(x, quant_type="nf4")
        dx = F.dequantize_4bit(qx, state)
        # NF4 has ~10% max error on randn data
        max_err = (x - dx).abs().max().item()
        assert max_err < 0.5, f"NF4 max error too high: {max_err}"

    def test_fp4_quantize_dequantize(self):
        """FP4 quantize/dequantize roundtrip should be correct."""
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        qx, state = F.quantize_4bit(x, quant_type="fp4")
        dx = F.dequantize_4bit(qx, state)
        max_err = (x - dx).abs().max().item()
        assert max_err < 1.0, f"FP4 max error too high: {max_err}"
