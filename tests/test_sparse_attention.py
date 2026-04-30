import unittest
from types import SimpleNamespace

import torch

from candidates.autoregressive.architecture import CausalSelfAttention
from candidates.registry import build_model


class SparseAttentionTest(unittest.TestCase):
    def test_local_global_mask_keeps_causal_window_and_global_source_tokens(self) -> None:
        attn = CausalSelfAttention(
            dim=8,
            num_heads=2,
            num_kv_heads=1,
            rope_base=10000.0,
            qk_gain_init=1.0,
            attn_norm_mode="baseline",
            attn_norm_eps=1e-6,
            sparse_attn_mode="local_global",
            sparse_attn_window=3,
            sparse_attn_global_tokens=2,
        )

        mask = attn._local_global_mask(6, torch.device("cpu"))[0, 0].tolist()

        self.assertEqual(mask[0], [True, False, False, False, False, False])
        self.assertEqual(mask[1], [True, True, False, False, False, False])
        self.assertEqual(mask[2], [True, True, True, False, False, False])
        self.assertEqual(mask[3], [True, True, True, True, False, False])
        self.assertEqual(mask[4], [True, True, True, True, True, False])
        self.assertEqual(mask[5], [True, True, False, True, True, True])

    def test_build_model_accepts_sparse_attention_env_controls(self) -> None:
        args = SimpleNamespace(
            vocab_size=32,
            num_layers=2,
            model_dim=16,
            num_heads=2,
            num_kv_heads=1,
            mlp_mult=2,
            tie_embeddings=True,
            tied_embed_init_std=0.005,
            logit_softcap=30.0,
            rope_base=10000.0,
            qk_gain_init=1.0,
            attn_norm_mode="baseline",
            attn_norm_eps=1e-6,
            sparse_attn_mode="local_global",
            sparse_attn_window=4,
            sparse_attn_global_tokens=1,
            activation_mode="relu2",
            swiglu_clamp_enabled=False,
            swiglu_linear_clamp_min=-10.0,
            swiglu_linear_clamp_max=10.0,
            swiglu_gate_clamp_max=10.0,
            parallel_residual=False,
            encoder_layer_order="",
            decoder_layer_order="",
        )

        model = build_model(args)
        input_ids = torch.randint(0, args.vocab_size, (2, 8))
        target_ids = torch.randint(0, args.vocab_size, (2, 8))

        loss = model(input_ids, target_ids)

        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
