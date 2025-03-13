import argparse
import torch
from torchvision import datasets, transforms

def parse_args():
    """
    파싱 인자 설정
    """
    parser = argparse.ArgumentParser(description="Model Training & Evaluation")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    return parser.parse_args()

def get_mnist_data_loader(batch_size=64, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root='./data/mnist', train=train, transform=transform, download=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader 

if __name__ == "__main__":
    args = parse_args()
    train_loader = get_mnist_data_loader(args.batch_size, train=True)
    test_loader = get_mnist_data_loader(args.batch_size, train=False)
    print("Data loaders are successfully created!")