import torch.nn as nn
from .bnn_layers import *


__all__ = ['bnn_mnist']


    
class BNNMnist(nn.Module):

    def __init__(self, num_classes=10):
        super(BNNMnist, self).__init__()
 
        self.classifier = nn.Sequential(
                nn.Flatten(),
                BNNLinear(28*28, 256, bias=False),
                nn.Hardtanh(inplace=True),
                BNNLinear(256, num_classes, bias=False),
        )


    def forward(self, x):
        return self.classifier(x)


    def init_w(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.zeros_(m.bias)
        return

def bnn_mnist(num_classes=10):
    return BNNMnist(num_classes)