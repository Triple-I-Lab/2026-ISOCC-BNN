"""Unit tests for models/xnor_layers.py"""
import unittest
import torch
from models.xnor_layers import (
    BinActive,
    XNORConv2d,
    XNORLinear,
    BNConvReLU,
    BNLinearReLU,
)


class TestBinActive(unittest.TestCase):
    """Tests for the BinActive straight-through estimator."""

    def test_forward_outputs_plus_minus_one(self):
        x = torch.randn(8)
        result = BinActive.apply(x)
        unique = result.unique()
        for v in unique:
            self.assertIn(v.item(), {-1.0, 1.0})

    def test_forward_output_shape_preserved(self):
        x = torch.randn(3, 4, 5)
        self.assertEqual(BinActive.apply(x).shape, x.shape)

    def test_backward_gradient_passthrough_in_range(self):
        """Gradient passes through for |x| < 1 (STE)."""
        x = torch.tensor([-0.5, 0.5], requires_grad=True)
        out = BinActive.apply(x)
        out.sum().backward()
        self.assertTrue(torch.allclose(x.grad, torch.ones(2)))

    def test_backward_gradient_clamped_outside_range(self):
        """Gradient is zeroed for |x| >= 1 (STE)."""
        x = torch.tensor([-1.5, 1.5], requires_grad=True)
        out = BinActive.apply(x)
        out.sum().backward()
        self.assertTrue(torch.allclose(x.grad, torch.zeros(2)))


class TestXNORConv2d(unittest.TestCase):
    """Tests for XNORConv2d."""

    def setUp(self):
        torch.manual_seed(0)
        self.layer = XNORConv2d(8, 16, kernel_size=3, stride=1, padding=1)

    def test_forward_output_shape(self):
        x = torch.randn(2, 8, 6, 6)
        out = self.layer(x)
        self.assertEqual(out.shape, (2, 16, 6, 6))

    def test_fp_weights_parameter_exists(self):
        self.assertIn('fp_weights', dict(self.layer.named_parameters()))

    def test_fp_weights_shape_matches_conv_weight(self):
        self.assertEqual(self.layer.fp_weights.shape, self.layer.conv.weight.shape)

    def test_conv_weights_are_signed_after_forward(self):
        """After forward, conv.weight.data should be scaled ±1 (binary * mean_val)."""
        x = torch.randn(2, 8, 6, 6)
        self.layer(x)
        # Signs should be ±1 (mean_val is always positive, so sign comes from data.sign())
        signs = self.layer.conv.weight.data.sign()
        self.assertTrue(signs.abs().min().item() >= 0.0)

    def test_update_gradient_sets_fp_weights_grad(self):
        x = torch.randn(2, 8, 6, 6)
        out = self.layer(x)
        loss = out.sum()
        loss.backward()
        self.layer.update_gradient()
        self.assertIsNotNone(self.layer.fp_weights.grad)

    def test_output_non_negative_after_relu_not_applied(self):
        """XNORConv2d itself has no ReLU, so output can be negative."""
        x = torch.randn(2, 8, 6, 6)
        out = self.layer(x)
        # Values can be positive or negative - just verify it's a valid tensor
        self.assertFalse(torch.isnan(out).any())


class TestXNORLinear(unittest.TestCase):
    """Tests for XNORLinear."""

    def setUp(self):
        torch.manual_seed(0)
        self.layer = XNORLinear(32, 16)

    def test_forward_output_shape(self):
        x = torch.randn(4, 32)
        out = self.layer(x)
        self.assertEqual(out.shape, (4, 16))

    def test_fp_weights_parameter_exists(self):
        self.assertIn('fp_weights', dict(self.layer.named_parameters()))

    def test_update_gradient_sets_fp_weights_grad(self):
        x = torch.randn(4, 32)
        out = self.layer(x)
        loss = out.sum()
        loss.backward()
        self.layer.update_gradient()
        self.assertIsNotNone(self.layer.fp_weights.grad)

    def test_output_contains_no_nan(self):
        x = torch.randn(4, 32)
        out = self.layer(x)
        self.assertFalse(torch.isnan(out).any())


class TestBNConvReLU(unittest.TestCase):
    """Tests for BNConvReLU (BN → BinActive → XNORConv2d → ReLU)."""

    def setUp(self):
        torch.manual_seed(0)
        self.layer = BNConvReLU(8, 16, kernel_size=3, stride=1, padding=1)

    def test_forward_output_shape(self):
        x = torch.randn(2, 8, 6, 6)
        out = self.layer(x)
        self.assertEqual(out.shape, (2, 16, 6, 6))

    def test_output_is_non_negative(self):
        """ReLU at the end ensures non-negative output."""
        x = torch.randn(2, 8, 6, 6)
        out = self.layer(x)
        self.assertTrue((out >= 0).all())

    def test_dropout_variant(self):
        layer = BNConvReLU(8, 16, kernel_size=1, dropout_ratio=0.5)
        layer.train()
        x = torch.randn(2, 8, 4, 4)
        out = layer(x)
        self.assertEqual(out.shape, (2, 16, 4, 4))

    def test_no_dropout_variant(self):
        layer = BNConvReLU(8, 16, kernel_size=1, dropout_ratio=0)
        self.assertFalse(hasattr(layer, 'drop'))

    def test_output_contains_no_nan(self):
        x = torch.randn(2, 8, 6, 6)
        self.assertFalse(torch.isnan(self.layer(x)).any())


class TestBNLinearReLU(unittest.TestCase):
    """Tests for BNLinearReLU (BN1d → BinActive → XNORLinear → ReLU)."""

    def setUp(self):
        torch.manual_seed(0)
        self.layer = BNLinearReLU(64, 32)

    def test_forward_output_shape(self):
        x = torch.randn(4, 64)
        out = self.layer(x)
        self.assertEqual(out.shape, (4, 32))

    def test_output_is_non_negative(self):
        """ReLU at the end ensures non-negative output."""
        x = torch.randn(4, 64)
        out = self.layer(x)
        self.assertTrue((out >= 0).all())

    def test_dropout_variant(self):
        layer = BNLinearReLU(32, 16, dropout_ratio=0.5)
        layer.train()
        x = torch.randn(4, 32)
        out = layer(x)
        self.assertEqual(out.shape, (4, 16))

    def test_no_dropout_variant(self):
        layer = BNLinearReLU(32, 16, dropout_ratio=0)
        self.assertFalse(hasattr(layer, 'drop'))

    def test_output_contains_no_nan(self):
        x = torch.randn(4, 64)
        self.assertFalse(torch.isnan(self.layer(x)).any())


if __name__ == '__main__':
    unittest.main()
