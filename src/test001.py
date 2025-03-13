import argparse
import os
import torch
from utils import get_mnist_data_loader


def test(model, data_loader, device, criterion=torch.nn.CrossEntropyLoss()):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss = criterion(output, target)
    accuracy = 100. * correct / len(data_loader.dataset)
    print(f'Accuracy: {correct}/{len(data_loader.dataset)} ({accuracy:.0f}%)', end='\t')
    print(f'loss: {loss:.4f}')

if __name__() == '__main__':  ## 모델위치를 받아서 테스트
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model_path', type=str, default='models/model_jeonghun.pth', help='model path')
    args = parser.parse_args()

    model = torch.load(args.model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = get_mnist_data_loader(train=False)

    test(model, data_loader, device)



