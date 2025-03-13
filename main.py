import argparse
import torch
from sources.train_test import train, test
import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a simple MLP on MNIST')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs to train (default: 3)')
    args = parser.parse_args()
    model_name = ["simplemlp", "fromscratch", "simplecnn", "cnnmodel_c2f1"]
    print(device)

    for model in model_name:
        print("Training: ", model)
        # train(model, args.epochs, device)
        print("Testing: ")
        test(model, device)