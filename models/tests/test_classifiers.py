"""Unit tests for classifiers (BnnClassifier, XnorClassifier, DorefaClassifier)."""
import os
import tempfile
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from classifiers.bnn_classifier import BnnClassifier
from classifiers.xnor_classifier import XnorClassifier
from classifiers.dorefa_classifier import DorefaClassifier
from models.bnn_caffenet import bnn_caffenet
from models.xnor_nin import nin
from models.dorefa_resnet import dorefa_resnet18


def _make_loader(n_samples=8, n_classes=10, image_shape=(3, 32, 32), batch_size=4):
    """Return a DataLoader backed by random in-memory tensors."""
    images = torch.randn(n_samples, *image_shape)
    labels = torch.randint(0, n_classes, (n_samples,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


class TestBnnClassifier(unittest.TestCase):
    """Tests for BnnClassifier."""

    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device('cpu')
        self.model = bnn_caffenet(num_classes=10)
        self.model.to(self.device)
        self.train_loader = _make_loader()
        self.test_loader = _make_loader()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.clf = BnnClassifier(
            self.model, self.train_loader, self.test_loader, self.device
        )

    def test_init_stores_components(self):
        self.assertIs(self.clf.model, self.model)
        self.assertIs(self.clf.train_loader, self.train_loader)
        self.assertIs(self.clf.test_loader, self.test_loader)
        self.assertIs(self.clf.device, self.device)

    def test_train_step_returns_list_of_floats(self):
        losses = self.clf.train_step(self.criterion, self.optimizer)
        self.assertIsInstance(losses, list)
        self.assertTrue(len(losses) > 0)
        for l in losses:
            self.assertIsInstance(l, float)

    def test_test_returns_accuracy_in_range(self):
        acc = self.clf.test(self.criterion)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 100.0)

    def test_save_checkpoint_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = os.path.join(tmpdir, 'run', 'ckpt')
            state = {'epoch': 1, 'state_dict': self.model.state_dict()}
            BnnClassifier.save_checkpoint(state, is_best=True, checkpoint=checkpoint)
            head, tail = os.path.split(checkpoint)
            self.assertTrue(os.path.exists(os.path.join(head, f'{tail}_checkpoint.pth.tar')))
            self.assertTrue(os.path.exists(os.path.join(head, f'{tail}_best.pth.tar')))

    def test_save_checkpoint_no_best_file_when_not_best(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = os.path.join(tmpdir, 'run', 'ckpt')
            state = {'epoch': 1, 'state_dict': self.model.state_dict()}
            BnnClassifier.save_checkpoint(state, is_best=False, checkpoint=checkpoint)
            head, tail = os.path.split(checkpoint)
            self.assertTrue(os.path.exists(os.path.join(head, f'{tail}_checkpoint.pth.tar')))
            self.assertFalse(os.path.exists(os.path.join(head, f'{tail}_best.pth.tar')))

    def test_train_requires_checkpoint(self):
        with self.assertRaises(ValueError):
            self.clf.train(self.criterion, self.optimizer, epochs=1, scheduler=None,
                           checkpoint=None)

    def test_train_one_epoch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = os.path.join(tmpdir, 'run', 'ckpt')
            self.clf.train(self.criterion, self.optimizer, epochs=1, scheduler=None,
                           checkpoint=checkpoint)


class TestXnorClassifier(unittest.TestCase):
    """Tests for XnorClassifier."""

    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device('cpu')
        self.model = nin(out_classes=10)
        self.model.to(self.device)
        self.train_loader = _make_loader()
        self.test_loader = _make_loader()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.clf = XnorClassifier(
            self.model, self.train_loader, self.test_loader, self.device
        )

    def test_init_stores_components(self):
        self.assertIs(self.clf.model, self.model)

    def test_train_step_returns_list_of_floats(self):
        losses = self.clf.train_step(self.criterion, self.optimizer)
        self.assertIsInstance(losses, list)
        for l in losses:
            self.assertIsInstance(l, float)

    def test_test_returns_accuracy_in_range(self):
        acc = self.clf.test(self.criterion)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 100.0)

    def test_save_checkpoint_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = os.path.join(tmpdir, 'run', 'ckpt')
            state = {'epoch': 1, 'state_dict': self.model.state_dict()}
            XnorClassifier.save_checkpoint(state, is_best=True, checkpoint=checkpoint)
            head, tail = os.path.split(checkpoint)
            self.assertTrue(os.path.exists(os.path.join(head, f'{tail}_checkpoint.pth.tar')))

    def test_train_requires_checkpoint(self):
        with self.assertRaises(ValueError):
            self.clf.train(self.criterion, self.optimizer, epochs=1, scheduler=None,
                           checkpoint=None)

    def test_train_one_epoch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = os.path.join(tmpdir, 'run', 'ckpt')
            self.clf.train(self.criterion, self.optimizer, epochs=1, scheduler=None,
                           checkpoint=checkpoint)


class TestDorefaClassifier(unittest.TestCase):
    """Tests for DorefaClassifier."""

    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device('cpu')
        self.model = dorefa_resnet18(wbit=1, abit=1)
        self.model.to(self.device)
        self.train_loader = _make_loader()
        self.test_loader = _make_loader()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.clf = DorefaClassifier(
            self.model, self.train_loader, self.test_loader, self.device
        )

    def test_init_stores_components(self):
        self.assertIs(self.clf.model, self.model)

    def test_train_step_returns_list_of_floats(self):
        losses = self.clf.train_step(self.criterion, self.optimizer)
        self.assertIsInstance(losses, list)
        for l in losses:
            self.assertIsInstance(l, float)

    def test_test_returns_accuracy_in_range(self):
        acc = self.clf.test(self.criterion)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 100.0)

    def test_save_checkpoint_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = os.path.join(tmpdir, 'run', 'ckpt')
            state = {'epoch': 1, 'state_dict': self.model.state_dict()}
            DorefaClassifier.save_checkpoint(state, is_best=True, checkpoint=checkpoint)
            head, tail = os.path.split(checkpoint)
            self.assertTrue(os.path.exists(os.path.join(head, f'{tail}_checkpoint.pth.tar')))

    def test_train_requires_checkpoint(self):
        with self.assertRaises(ValueError):
            self.clf.train(self.criterion, self.optimizer, epochs=1, scheduler=None,
                           checkpoint=None)

    def test_train_one_epoch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = os.path.join(tmpdir, 'run', 'ckpt')
            self.clf.train(self.criterion, self.optimizer, epochs=1, scheduler=None,
                           checkpoint=checkpoint)


if __name__ == '__main__':
    unittest.main()
