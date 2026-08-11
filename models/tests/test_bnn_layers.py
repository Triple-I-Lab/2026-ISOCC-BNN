"""Unit tests for models/bnn_layers.py"""
import unittest
import torch
from models.bnn_layers import Binarize, BNNLinear, BNNConv2d


class TestBinarize(unittest.TestCase):
    """Tests for the Binarize quantization function."""

    def test_det_mode_outputs_sign(self):
        t = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        result = Binarize(t, quant_mode='det')
        expected = torch.sign(t)
        self.assertTrue(torch.equal(result, expected))

    def test_det_mode_values_are_minus_one_zero_or_one(self):
        t = torch.randn(8, 8)
        result = Binarize(t, quant_mode='det')
        self.assertTrue(result.abs().max().item() <= 1.0)

    def test_bin_mode_outputs_plus_minus_one(self):
        t = torch.randn(16)
        result = Binarize(t, quant_mode='bin')
        unique = result.unique()
        for v in unique:
            self.assertIn(v.item(), {-1.0, 1.0})

    def test_bin_mode_preserves_dtype_float32(self):
        t = torch.randn(4, 4)
        result = Binarize(t, quant_mode='bin')
        self.assertEqual(result.dtype, torch.float32)

    def test_bin_mode_preserves_dtype_float64(self):
        t = torch.randn(4, 4).double()
        result = Binarize(t, quant_mode='bin')
        self.assertEqual(result.dtype, torch.float64)

    def test_stochastic_mode_outputs_plus_minus_one(self):
        t = torch.randn(32)
        result = Binarize(t)  # default mode falls through to stochastic
        unique = result.unique()
        for v in unique:
            self.assertIn(v.item(), {-1.0, 1.0})

    def test_output_shape_is_preserved(self):
        t = torch.randn(3, 4, 5)
        for mode in ('det', 'bin'):
            with self.subTest(mode=mode):
                self.assertEqual(Binarize(t, mode).shape, t.shape)


class TestBNNLinear(unittest.TestCase):
    """Tests for BNNLinear."""

    def setUp(self):
        torch.manual_seed(0)
        self.layer = BNNLinear(64, 32)

    def test_weight_org_buffer_registered(self):
        self.assertIn('weight_org', dict(self.layer.named_buffers()))

    def test_weight_org_equals_initial_weight(self):
        layer = BNNLinear(16, 8)
        self.assertTrue(torch.equal(layer.weight_org, layer.weight.data))

    def test_forward_output_shape(self):
        x = torch.randn(4, 64)
        out = self.layer(x)
        self.assertEqual(out.shape, (4, 32))

    def test_forward_first_layer_skips_input_binarization(self):
        """Input with size 784 (MNIST) should not be binarized."""
        layer = BNNLinear(784, 512)
        x = torch.randn(2, 784)
        # Should not raise; values are kept in full precision
        out = layer(x)
        self.assertEqual(out.shape, (2, 512))

    def test_weights_are_binary_after_forward(self):
        """After forward, weight.data should be ±1."""
        x = torch.randn(4, 64)
        self.layer(x)
        unique = self.layer.weight.data.unique()
        for v in unique:
            self.assertIn(round(v.item()), {-1, 0, 1})

    def test_no_bias_variant(self):
        layer = BNNLinear(16, 8, bias=False)
        out = layer(torch.randn(2, 16))
        self.assertEqual(out.shape, (2, 8))


class TestBNNConv2d(unittest.TestCase):
    """Tests for BNNConv2d."""

    def setUp(self):
        torch.manual_seed(0)
        self.layer = BNNConv2d(16, 32, kernel_size=3, padding=1, bias=False)

    def test_weight_org_buffer_registered(self):
        self.assertIn('weight_org', dict(self.layer.named_buffers()))

    def test_forward_output_shape(self):
        x = torch.randn(2, 16, 8, 8)
        out = self.layer(x)
        self.assertEqual(out.shape, (2, 32, 8, 8))

    def test_first_conv_skips_input_binarization(self):
        """RGB input (3 channels) should not be binarized."""
        layer = BNNConv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=False)
        x = torch.randn(2, 3, 32, 32)
        out = layer(x)
        self.assertEqual(out.shape, (2, 32, 32, 32))

    def test_weights_are_binary_after_forward(self):
        x = torch.randn(2, 16, 8, 8)
        self.layer(x)
        unique = self.layer.weight.data.unique()
        for v in unique:
            self.assertIn(round(v.item()), {-1, 0, 1})

    def test_with_bias(self):
        layer = BNNConv2d(16, 32, kernel_size=3, padding=1, bias=True)
        out = layer(torch.randn(2, 16, 8, 8))
        self.assertEqual(out.shape, (2, 32, 8, 8))


if __name__ == '__main__':
    unittest.main()
