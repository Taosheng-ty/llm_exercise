"""
Tests for Exercise 05: FLOPs Counter
"""

import importlib.util
import os

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

linear_flops = _mod.linear_flops
attention_flops = _mod.attention_flops
ffn_flops = _mod.ffn_flops
transformer_block_flops = _mod.transformer_block_flops
total_training_flops = _mod.total_training_flops


class TestLinearFlops:
    def test_basic(self):
        # 2 * 1 * 4 * 8 * 16 = 1024
        assert linear_flops(1, 4, 8, 16) == 1024

    def test_scales_with_batch(self):
        f1 = linear_flops(1, 4, 8, 16)
        f2 = linear_flops(2, 4, 8, 16)
        assert f2 == 2 * f1

    def test_scales_with_seq_len(self):
        f1 = linear_flops(1, 4, 8, 16)
        f2 = linear_flops(1, 8, 8, 16)
        assert f2 == 2 * f1

    def test_square_matrix(self):
        # 2 * 1 * 1 * N * N for a single token
        N = 64
        assert linear_flops(1, 1, N, N) == 2 * N * N


class TestAttentionFlops:
    def test_basic(self):
        # B=1, H=2, S=4, D=8
        # QK^T (causal): 2*H*S*S*D//2 = 2*2*4*4*8//2 = 256
        # AV: H*S*S*D = 2*4*4*8 = 256
        # total: (256 + 256) * 1 = 512
        result = attention_flops(1, 2, 4, 8)
        assert result == 512

    def test_scales_with_batch(self):
        f1 = attention_flops(1, 4, 8, 16)
        f2 = attention_flops(3, 4, 8, 16)
        assert f2 == 3 * f1

    def test_quadratic_in_seq_len(self):
        """Attention FLOPs should scale quadratically with seq_len."""
        f1 = attention_flops(1, 1, 4, 8)
        f2 = attention_flops(1, 1, 8, 8)
        # Should be roughly 4x (exactly 4x since both terms are S^2)
        assert f2 == 4 * f1


class TestFFNFlops:
    def test_basic(self):
        # 2 * 1 * 4 * 32 * 64 * 3 = 49152
        assert ffn_flops(1, 4, 32, 64) == 49152

    def test_three_projections(self):
        """SwiGLU has 3 projections, should be 3x a single linear."""
        single_linear = linear_flops(1, 4, 32, 64)
        ffn = ffn_flops(1, 4, 32, 64)
        assert ffn == 3 * single_linear


class TestTransformerBlockFlops:
    def test_positive(self):
        flops = transformer_block_flops(1, 16, 64, 4, 128)
        assert flops > 0

    def test_includes_all_components(self):
        B, S, H, NH, FFN = 1, 8, 32, 4, 64
        D = H // NH

        qkv = 3 * linear_flops(B, S, H, H)
        attn = attention_flops(B, NH, S, D)
        out_proj = linear_flops(B, S, H, H)
        ffn_f = ffn_flops(B, S, H, FFN)

        total = transformer_block_flops(B, S, H, NH, FFN)
        assert total == qkv + attn + out_proj + ffn_f

    def test_scales_with_batch(self):
        f1 = transformer_block_flops(1, 8, 32, 4, 64)
        f2 = transformer_block_flops(2, 8, 32, 4, 64)
        assert f2 == 2 * f1


class TestTotalTrainingFlops:
    def test_is_3x_forward(self):
        """Training FLOPs should be 3x forward FLOPs."""
        B, S, H, NH, FFN, NL, V = 1, 8, 32, 4, 64, 2, 100
        train = total_training_flops(B, S, H, NH, FFN, NL, V)

        block = transformer_block_flops(B, S, H, NH, FFN)
        lm_head = linear_flops(B, S, H, V)
        fwd = NL * block + lm_head

        assert train == 3 * fwd

    def test_scales_with_layers(self):
        kwargs = dict(batch_size=1, seq_len=8, hidden_dim=32, num_heads=4,
                      ffn_hidden_dim=64, vocab_size=100)
        f1 = total_training_flops(num_layers=1, **kwargs)
        f2 = total_training_flops(num_layers=2, **kwargs)
        # Difference should be 3 * one_block
        block = transformer_block_flops(1, 8, 32, 4, 64)
        assert f2 - f1 == 3 * block

    def test_realistic_model(self):
        """Test with a realistic small model config (GPT-2 small-ish)."""
        flops = total_training_flops(
            batch_size=4,
            seq_len=512,
            hidden_dim=768,
            num_heads=12,
            ffn_hidden_dim=3072,
            num_layers=12,
            vocab_size=50257,
        )
        # Should be a large number
        assert flops > 1e12  # > 1 TFLOP
