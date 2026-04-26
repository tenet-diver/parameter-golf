import unittest

import torch

from candidates.autoregressive.model import MLP


class TrainGptSwiGluClampTest(unittest.TestCase):
    def test_swiglu_clamp_caps_linear_and_gate_outliers(self) -> None:
        clamped = MLP(
            dim=1,
            mlp_mult=1,
            activation_mode="swiglu",
            swiglu_clamp_enabled=True,
            swiglu_linear_clamp_min=-10.0,
            swiglu_linear_clamp_max=10.0,
            swiglu_gate_clamp_max=10.0,
        )
        unclamped = MLP(
            dim=1,
            mlp_mult=1,
            activation_mode="swiglu",
            swiglu_clamp_enabled=False,
            swiglu_linear_clamp_min=-10.0,
            swiglu_linear_clamp_max=10.0,
            swiglu_gate_clamp_max=10.0,
        )

        with torch.no_grad():
            clamped.fc.weight.copy_(torch.tensor([[100.0], [100.0]], dtype=torch.float32))
            unclamped.fc.weight.copy_(torch.tensor([[100.0], [100.0]], dtype=torch.float32))
            clamped.proj.weight.fill_(1.0)
            unclamped.proj.weight.fill_(1.0)

        x = torch.ones((1, 1, 1), dtype=torch.float32)
        clamped_out = clamped(x).item()
        unclamped_out = unclamped(x).item()

        self.assertLess(clamped_out, 200.0)
        self.assertGreater(clamped_out, 90.0)
        self.assertGreater(unclamped_out, clamped_out * 50.0)


if __name__ == "__main__":
    unittest.main()
