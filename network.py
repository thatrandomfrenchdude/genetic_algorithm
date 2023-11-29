import argparse

import numpy as np
import random
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from mnist import MnistNet, train, test

def mnist_main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = MnistNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

def init_networks(population):
    return [NetworkParams() for _ in range(population)]

class NetworkParams():
    layers = [1, 5, 10, 25, 50, 100, 250, 500, 1000]
    shapes = []

    def __init__(self):
        self.epochs = np.random.randint(1, 15)
        self.dense_layers = self.layers[np.random.randint(1, range(len(self.layers)))]
        self.layer_shape = self.shapes[np.random.randint(1, range(len(self.shapes)))]
        self.activations = [random.choice(['sigmoid', 'relu', 'softmax', 'tanh', 'elu', 'selu', 'linear']) for _ in range(self.dense_layers)]
        self.dropout = np.random.uniform(0, 1)
        self.loss_function = random.choice([
            'categorical_crossentropy',
            'binary_crossentropy',
            'mean_squared_error',
            'mean_absolute_error',
            'sparse_categorical_crossentropy'
        ])
        self.optimizer = random.choice(['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'])
        self.accuracy = 0

    def init_hyperparams(self):
        hyperparams = {
            'epochs': self.epochs,
            'dense_layers': self.dense_layers,
            'layer_shape': self.layer_shape,
            'activations': self.activations,
            'dropout': self.dropout,
            'loss_function': self.loss_function,
            'optimizer': self.optimizer
        }
        return hyperparams

# class Network():
#     def __init__(self):
#         self._epochs = np.random.randint(1, 15)

#         self._units1 = np.random.randint(1, 500)
#         self._units2 = np.random.randint(1, 500)

#         self._act1 = random.choice(['sigmoid', 'relu', 'softmax', 'tanh', 'elu', 'selu', 'linear'])
#         self._act2 = random.choice(['sigmoid', 'relu', 'softmax', 'tanh', 'elu', 'selu', 'linear'])
#         self._act3 = random.choice(['sigmoid', 'relu', 'softmax', 'tanh', 'elu', 'selu', 'linear'])

#         self._loss = random.choice([
#             'categorical_crossentropy',
#             'binary_crossentropy',
#             'mean_squared_error',
#             'mean_absolute_error',
#             'sparse_categorical_crossentropy'
#         ])
#         self._opt = random.choice(['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'])

#         self._accuracy = 0

#     def init_hyperparams(self):
#         hyperparams = {
#             'epochs': self._epochs,
#             'units1': self._units1,
#             'act1': self._act1,
#             'units2': self._units2,
#             'act2': self._act2,
#             'act3': self._act3,
#             'loss': self._loss,
#             'optimizer': self._opt
#         }
#         return hyperparams