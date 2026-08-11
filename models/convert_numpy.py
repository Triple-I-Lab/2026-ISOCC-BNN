import numpy as np
from dataloader.mnist import load_test_data
import torch

# test data -> numpy
test_loader = load_test_data()
data_iter = iter(test_loader)
test_data = np.zeros((len(data_iter)*1000, 28*28)).astype(np.uint8)
test_labels = np.zeros((len(data_iter)*1000, 1))
i = 0
for data, label in data_iter:
  for x, y in zip(data, label):
    test_data[i] = x.flatten().to(torch.uint8)
    print(test_data[i])
    test_labels[i] = y
    i += 1
print(test_data.shape, test_labels.shape)
np.savetxt("test_data.txt", test_data, "%d", delimiter='')
np.savetxt("test_labels.txt", test_labels, "%d", delimiter='')

# weights -> numpy
model = torch.load('./results/bnn_mnist_1_hidden_0_batch_acc_90_37_sgd_lr_001.pth.tar', weights_only=False)
w1 = model['state_dict']['classifier.1.weight']
w2 = model['state_dict']['classifier.3.weight']

# binarize: (-1, 1) -> (0, 1)
binarize = lambda x: x.clone().apply_(lambda val: float(1) if val == 1 else float(0)).to(torch.uint8)
w1 = binarize(w1)
w2 = binarize(w2)

w1 = np.array(w1).astype(np.uint8)
w2 = np.array(w2).astype(np.uint8)
print(w1.shape, w2.shape)
np.savetxt("weights_layer1.txt", w1, "%d", delimiter='')
np.savetxt("weights_layer2.txt", w2, "%d", delimiter='')
