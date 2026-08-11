"""Unit tests for models/dorefa_layers.py"""
import unittest
import torch
from models.dorefa_layers import (
    ScaleSigner,
    scale_sign,
    Quantizer,
    quantize,
    dorefa_w,
    dorefa_a,
    DOREFAConv2d,
    DOREFALinear,
)


class TestScaleSigner(unittest.TestCase):
    """Tests for ScaleSigner: output = sign(x) * E(|x|)."""

    def test_output_sign_matches_input_sign(self):
        x = torch.tensor([-3.0, -1.0, 1.0, 3.0])
        out = scale_sign(x)
        self.assertTrue(torch.equal(out.sign(), x.sign()))

    def test_output_magnitude_equals_mean_abs(self):
        x = torch.tensor([-2.0, 2.0, 4.0])
        out = scale_sign(x)
        expected_mag = torch.mean(torch.abs(x))
        self.assertTrue(torch.allclose(out.abs(), expected_mag * torch.ones_like(out)))

    def test_backward_is_identity(self):
        x = torch.randn(4, requires_grad=True)
        out = scale_sign(x)
        out.sum().backward()
        self.assertTrue(torch.allclose(x.grad, torch.ones(4)))

    def test_output_shape_preserved(self):
        x = torch.randn(3, 4)
        self.assertEqual(scale_sign(x).shape, x.shape)


class TestQuantizer(unittest.TestCase):
    """Tests for Quantizer: round(x * (2^nbit - 1)) / (2^nbit - 1)."""

    def test_1bit_quantization_values(self):
        """With nbit=1, scale=1 so values stay in {0, 1} after rounding."""
        x = torch.tensor([0.0, 0.3, 0.7, 1.0])
        out = quantize(x, nbit=1)
        unique = out.unique()
        for v in unique:
            self.assertIn(round(v.item()), {0, 1})

    def test_2bit_quantization_values(self):
        """With nbit=2, scale=3, quantized values are multiples of 1/3."""
        x = torch.tensor([0.0, 0.5, 1.0])
        out = quantize(x, nbit=2)
        residual = (out * 3).round() - (out * 3)
        self.assertTrue(torch.allclose(residual, torch.zeros_like(residual), atol=1e-5))

    def test_output_shape_preserved(self):
        x = torch.randn(3, 4).clamp(0, 1)
        self.assertEqual(quantize(x, nbit=4).shape, x.shape)

    def test_backward_passes_gradient_through(self):
        x = torch.randn(4, requires_grad=True)
        out = quantize(x.clamp(0, 1), nbit=4)
        out.sum().backward()
        self.assertIsNotNone(x.grad)


class TestDorefaHelpers(unittest.TestCase):
    """Tests for dorefa_w and dorefa_a helper functions."""

    def test_dorefa_w_1bit_output_matches_scale_sign(self):
        w = torch.randn(8, 4)
        result = dorefa_w(w, nbit_w=1)
        expected = scale_sign(w)
        self.assertTrue(torch.allclose(result, expected))

    def test_dorefa_w_multibit_output_in_minus1_to_1(self):
        w = torch.randn(8, 4)
        result = dorefa_w(w, nbit_w=4)
        self.assertTrue((result >= -1.0).all() and (result <= 1.0).all())

    def test_dorefa_a_output_in_0_to_1(self):
        x = torch.randn(8, 4) * 5
        result = dorefa_a(x, nbit_a=4)
        self.assertTrue((result >= 0.0).all() and (result <= 1.0).all())

    def test_dorefa_a_output_shape_preserved(self):
        x = torch.randn(3, 4, 5)
        self.assertEqual(dorefa_a(x, nbit_a=4).shape, x.shape)


class TestDOREFAConv2d(unittest.TestCase):
    """Tests for DOREFAConv2d."""

    def setUp(self):
        torch.manual_seed(0)

    def test_forward_output_shape_1bit(self):
        layer = DOREFAConv2d(8, 16, kernel_size=3, padding=1, nbit_w=1, nbit_a=1)
        x = torch.randn(2, 8, 6, 6)
        out = layer(x)
        self.assertEqual(out.shape, (2, 16, 6, 6))

    def test_forward_output_shape_multibit(self):
        layer = DOREFAConv2d(8, 16, kernel_size=3, padding=1, nbit_w=4, nbit_a=4)
        x = torch.randn(2, 8, 6, 6)
        out = layer(x)
        self.assertEqual(out.shape, (2, 16, 6, 6))

    def test_forward_fullprecision_path(self):
        """nbit_w=32 and nbit_a=32 skips quantization."""
        layer = DOREFAConv2d(4, 8, kernel_size=1, nbit_w=32, nbit_a=32)
        x = torch.randn(2, 4, 4, 4)
        out = layer(x)
        self.assertEqual(out.shape, (2, 8, 4, 4))

    def test_backward_does_not_raise(self):
        layer = DOREFAConv2d(4, 8, kernel_size=1, nbit_w=1, nbit_a=1)
        x = torch.randn(2, 4, 4, 4)
        out = layer(x)
        out.sum().backward()

    def test_output_contains_no_nan(self):
        layer = DOREFAConv2d(4, 8, kernel_size=3, padding=1, nbit_w=2, nbit_a=2)
        x = torch.randn(2, 4, 4, 4)
        self.assertFalse(torch.isnan(layer(x)).any())


class TestDOREFALinear(unittest.TestCase):
    """Tests for DOREFALinear."""

    def setUp(self):
        torch.manual_seed(0)

    def test_forward_output_shape_1bit(self):
        layer = DOREFALinear(32, 16, nbit_w=1, nbit_a=1)
        x = torch.randn(4, 32)
        out = layer(x)
        self.assertEqual(out.shape, (4, 16))

    def test_forward_output_shape_multibit(self):
        layer = DOREFALinear(32, 16, nbit_w=4, nbit_a=4)
        x = torch.randn(4, 32)
        out = layer(x)
        self.assertEqual(out.shape, (4, 16))

    def test_forward_fullprecision_path(self):
        layer = DOREFALinear(16, 8, nbit_w=32, nbit_a=32)
        x = torch.randn(2, 16)
        out = layer(x)
        self.assertEqual(out.shape, (2, 8))

    def test_backward_does_not_raise(self):
        layer = DOREFALinear(16, 8, nbit_w=1, nbit_a=1)
        x = torch.randn(4, 16)
        out = layer(x)
        out.sum().backward()

    def test_output_contains_no_nan(self):
        layer = DOREFALinear(16, 8, nbit_w=2, nbit_a=2)
        x = torch.randn(4, 16)
        self.assertFalse(torch.isnan(layer(x)).any())


if __name__ == '__main__':
    unittest.main()
