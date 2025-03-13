import argparse
import os
import torch
import importlib

import torch.nn as nn
import sys
# sys.path.append('../utils')
from utils.data_loader import get_mnist_data_loader
from models.CNNmodel_c2f1 import CNNmodel_c2f1


def test(model, data_loader, device, criterion=torch.nn.CrossEntropyLoss()):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader)

    print(f'Accuracy: {correct}/{len(data_loader.dataset)} ({accuracy:.0f}%)', end='\t')
    print(f'loss: {loss:.4f}')

    return accuracy, avg_loss


# if __name__ == '__main__':  ## 모델위치를 받아서 테스트
#     parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
#     # parser.add_argument('--model_path', type=str, default='models/model_jeonghun.py', help='model_path')
#     parser.add_argument('--weights', type=str, default='models/model_jeonghun.pth', help='model with state_dict')
#     parser.add_argument('--model', type=str, default=CNNmodel_c2f1(), help='model')

    
#     args = parser.parse_args()

#     model = torch.load(args.model)
#     weights = torch.load(args.weights)
#     model.load_state_dict(weights)
#     data_loader = get_mnist_data_loader(train=False)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)

#     test(model, data_loader, device)

##  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model_def', type=str, default='models.CNNmodel_c2f1', help='model definition file')
    parser.add_argument('--weights', type=str, default='models/CNNmodel_c2f1.pth', help='model with state_dict')
    args = parser.parse_args()

    # 모델 정의 임포트
    module_name = args.model_def
    try:
        module = importlib.import_module(module_name)
        model_class = getattr(module, 'CNNmodel_c2f1') # 클래스 이름으로 가져옴
    except ImportError:
        print(f"Error: Could not import module '{module_name}'")
        exit()
    except AttributeError:
        print(f"Error: Could not find class 'CNNmodel_c2f1' in module '{module_name}'")
        exit()

    # 모델 객체 생성 및 가중치 로드
    model = model_class()
    model.load_state_dict(torch.load(args.weights))

    data_loader = get_mnist_data_loader(train=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    test(model, data_loader, device, criterion)



