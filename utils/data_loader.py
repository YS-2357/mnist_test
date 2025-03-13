import torch
from torchvision import datasets, transforms

def get_mnist_data_loader(batch_size=64, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root='./data/mnist', train=train, transform=transform, download=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
<<<<<<< HEAD
    return data_loader 
=======
    return data_loader 

train = get_mnist_data_loader()
>>>>>>> b530b1be7230ca70c976b6042dffda27ef834e4b
