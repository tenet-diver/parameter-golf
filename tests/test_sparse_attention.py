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
            sparse_attn_block_size=2,
        )

        mask = attn._local_global_mask(6, torch.device("cpu"))[0, 0].tolist()

        self.assertEqual(mask[0], [True, False, False, False, False, False])
        self.assertEqual(mask[1], [True, True, False, False, False, False])
        self.assertEqual(mask[2], [True, True, True, False, False, False])
        self.assertEqual(mask[3], [True, True, True, True, False, False])
        self.assertEqual(mask[4], [True, True, True, True, True, False])
        self.assertEqual(mask[5], [True, True, False, True, True, True])

    def test_block_local_global_mask_uses_block_aligned_causal_windows(self) -> None:
        attn = CausalSelfAttention(
            dim=8,
            num_heads=2,
            num_kv_heads=1,
            rope_base=10000.0,
            qk_gain_init=1.0,
            attn_norm_mode="baseline",
            attn_norm_eps=1e-6,
            sparse_attn_mode="block_local_global",
            sparse_attn_window=4,
            sparse_attn_global_tokens=1,
            sparse_attn_block_size=2,
        )

        mask = attn._block_local_global_mask(8, torch.device("cpu"))[0, 0].tolist()

        self.assertEqual(mask[0], [True, False, False, False, False, False, False, False])
        self.assertEqual(mask[3], [True, True, True, True, False, False, False, False])
        self.assertEqual(mask[4], [True, False, True, True, True, False, False, False])
        self.assertEqual(mask[7], [True, False, False, False, True, True, True, True])

    def test_rotating_block_mask_adds_dilated_remote_blocks(self) -> None:
        attn = CausalSelfAttention(
            dim=8,
            num_heads=2,
            num_kv_heads=1,
            rope_base=10000.0,
            qk_gain_init=1.0,
            attn_norm_mode="baseline",
            attn_norm_eps=1e-6,
            sparse_attn_mode="rotating_block_local_global",
            sparse_attn_window=2,
            sparse_attn_global_tokens=0,
            sparse_attn_block_size=2,
        )

        mask = attn._block_local_global_mask(
            12,
            torch.device("cpu"),
            pattern_step=2,
            rotating=True,
        )[0, 0].tolist()

        self.assertEqual(mask[10][2], True)
        self.assertEqual(mask[10][4], False)

    def test_head_swarm_mask_uses_different_remote_blocks_by_head(self) -> None:
        attn = CausalSelfAttention(
            dim=16,
            num_heads=4,
            num_kv_heads=1,
            rope_base=10000.0,
            qk_gain_init=1.0,
            attn_norm_mode="baseline",
            attn_norm_eps=1e-6,
            sparse_attn_mode="head_swarm_block",
            sparse_attn_window=2,
            sparse_attn_global_tokens=0,
            sparse_attn_block_size=2,
        )

        mask = attn._head_swarm_block_mask(16, torch.device("cpu"), pattern_step=0)[0]

        self.assertNotEqual(mask[0, 14].tolist(), mask[3, 14].tolist())

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
            sparse_attn_block_size=2,
            activation_mode="relu2",
            swiglu_clamp_enabled=False,
            swiglu_linear_clamp_min=-10.0,
            swiglu_linear_clamp_max=10.0,
            swiglu_gate_clamp_max=10.0,
            parallel_residual=False,
            encoder_layer_order="",
            decoder_layer_order="",
            mlp_block_groups=1,
            mtp_num_tokens=1,
            mtp_loss_weight=0.0,
        )

        model = build_model(args)
        input_ids = torch.randint(0, args.vocab_size, (2, 8))
        target_ids = torch.randint(0, args.vocab_size, (2, 8))

        loss = model(input_ids, target_ids)

        self.assertTrue(torch.isfinite(loss))

    def test_build_model_accepts_block_sparse_mlp_and_mtp_training_loss(self) -> None:
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
            sparse_attn_mode="dense",
            sparse_attn_window=4,
            sparse_attn_global_tokens=1,
            sparse_attn_block_size=2,
            activation_mode="relu2",
            swiglu_clamp_enabled=False,
            swiglu_linear_clamp_min=-10.0,
            swiglu_linear_clamp_max=10.0,
            swiglu_gate_clamp_max=10.0,
            parallel_residual=True,
            encoder_layer_order="",
            decoder_layer_order="",
            mlp_block_groups=4,
            mtp_num_tokens=3,
            mtp_loss_weight=0.2,
        )

        model = build_model(args)
        model.train()
        input_ids = torch.randint(0, args.vocab_size, (2, 8))
        target_ids = torch.randint(0, args.vocab_size, (2, 8))

        loss = model(input_ids, target_ids)

        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
