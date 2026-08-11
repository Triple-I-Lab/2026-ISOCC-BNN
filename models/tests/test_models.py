"""Unit tests for model architectures in models/."""
import unittest
import torch
import torch.nn as nn
from models.bnn_caffenet import bnn_caffenet, BNNCaffenet
from models.xnor_lenet import lenet5, LeNet5
from models.xnor_mlp import mlp, MLP
from models.xnor_nin import nin, NIN
from models.dorefa_resnet import dorefa_resnet18, ResNet


class TestBNNCaffenet(unittest.TestCase):
    """Tests for BNN CaffeNet (CIFAR-10 model, 3×32×32 input)."""

    def setUp(self):
        torch.manual_seed(0)
        self.model = bnn_caffenet(num_classes=10)
        self.model.eval()

    def test_factory_returns_correct_type(self):
        self.assertIsInstance(self.model, BNNCaffenet)

    def test_forward_output_shape(self):
        x = torch.randn(2, 3, 32, 32)
        out = self.model(x)
        self.assertEqual(out.shape, (2, 10))

    def test_custom_num_classes(self):
        model = bnn_caffenet(num_classes=5)
        out = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(out.shape, (2, 5))

    def test_init_w_does_not_raise(self):
        self.model.init_w()

    def test_output_contains_no_nan(self):
        x = torch.randn(2, 3, 32, 32)
        self.assertFalse(torch.isnan(self.model(x)).any())

    def test_output_log_probabilities_sum_to_one(self):
        """LogSoftmax output: exp sums to 1 per sample."""
        x = torch.randn(2, 3, 32, 32)
        out = self.model(x)
        prob_sum = out.exp().sum(dim=1)
        self.assertTrue(torch.allclose(prob_sum, torch.ones(2), atol=1e-5))


class TestLeNet5(unittest.TestCase):
    """Tests for XNOR LeNet-5 (MNIST model, 1×28×28 input)."""

    def setUp(self):
        torch.manual_seed(0)
        self.model = lenet5(out_classes=10)
        self.model.eval()

    def test_factory_returns_correct_type(self):
        self.assertIsInstance(self.model, LeNet5)

    def test_forward_output_shape(self):
        x = torch.randn(2, 1, 28, 28)
        out = self.model(x)
        self.assertEqual(out.shape, (2, 10))

    def test_custom_num_classes(self):
        model = lenet5(out_classes=5)
        out = model(torch.randn(2, 1, 28, 28))
        self.assertEqual(out.shape, (2, 5))

    def test_init_w_does_not_raise(self):
        self.model.init_w()

    def test_output_contains_no_nan(self):
        x = torch.randn(2, 1, 28, 28)
        self.assertFalse(torch.isnan(self.model(x)).any())


class TestMLP(unittest.TestCase):
    """Tests for XNOR MLP (MNIST model, 1×28×28 input)."""

    def setUp(self):
        torch.manual_seed(0)
        self.model = mlp(out_classes=10)
        self.model.eval()

    def test_factory_returns_correct_type(self):
        self.assertIsInstance(self.model, MLP)

    def test_forward_output_shape(self):
        x = torch.randn(4, 1, 28, 28)
        out = self.model(x)
        self.assertEqual(out.shape, (4, 10))

    def test_custom_num_classes(self):
        model = mlp(out_classes=5)
        out = model(torch.randn(2, 1, 28, 28))
        self.assertEqual(out.shape, (2, 5))

    def test_init_w_does_not_raise(self):
        self.model.init_w()

    def test_output_contains_no_nan(self):
        x = torch.randn(4, 1, 28, 28)
        self.assertFalse(torch.isnan(self.model(x)).any())


class TestNIN(unittest.TestCase):
    """Tests for XNOR Network-in-Network (CIFAR-10 model, 3×32×32 input)."""

    def setUp(self):
        torch.manual_seed(0)
        self.model = nin(out_classes=10)
        self.model.eval()

    def test_factory_returns_correct_type(self):
        self.assertIsInstance(self.model, NIN)

    def test_forward_output_shape(self):
        x = torch.randn(2, 3, 32, 32)
        out = self.model(x)
        self.assertEqual(out.shape, (2, 10))

    def test_custom_num_classes(self):
        model = nin(out_classes=5)
        out = model(torch.randn(1, 3, 32, 32))
        self.assertEqual(out.shape, (1, 5))

    def test_init_w_does_not_raise(self):
        self.model.init_w()

    def test_output_contains_no_nan(self):
        x = torch.randn(2, 3, 32, 32)
        self.assertFalse(torch.isnan(self.model(x)).any())


class TestDorefaResNet18(unittest.TestCase):
    """Tests for DoReFa ResNet-18 (CIFAR-10 model, 3×32×32 input)."""

    def setUp(self):
        torch.manual_seed(0)
        self.model = dorefa_resnet18(wbit=1, abit=1)
        self.model.eval()

    def test_factory_returns_correct_type(self):
        self.assertIsInstance(self.model, ResNet)

    def test_forward_output_shape(self):
        x = torch.randn(2, 3, 32, 32)
        out = self.model(x)
        self.assertEqual(out.shape, (2, 10))

    def test_multibit_variant(self):
        model = dorefa_resnet18(wbit=4, abit=4)
        out = model(torch.randn(1, 3, 32, 32))
        self.assertEqual(out.shape, (1, 10))

    def test_init_w_does_not_raise(self):
        self.model.init_w()

    def test_output_contains_no_nan(self):
        x = torch.randn(2, 3, 32, 32)
        self.assertFalse(torch.isnan(self.model(x)).any())

    def test_train_mode_forward(self):
        self.model.train()
        x = torch.randn(2, 3, 32, 32)
        out = self.model(x)
        self.assertEqual(out.shape, (2, 10))


if __name__ == '__main__':
    unittest.main()
