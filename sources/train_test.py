import torch
import torch.nn as nn
import torch.optim as optim
from models.From_Scratch import FromScratch
from models.fromscratch import SimpleCNN
from models.mlp import SimpleMLP
from models.model_jeonghun import CNNmodel_c2f1
from utils.data_loader import get_mnist_data_loader

def train(model_name, epochs, device):
    # 데이터 로더
    train_loader = get_mnist_data_loader(train=True)
    
    if model_name == "fromscratch":
        model = FromScratch()
    elif model_name == "simplecnn":
        model = SimpleCNN()
    elif model_name == "simplemlp":
        model = SimpleMLP()
    elif model_name == "cnnmodel_c2f1":
        model = CNNmodel_c2f1()
    else:
        print("Model not found!")
        return

    # 모델, 손실 함수, 옵티마이저
    model = SimpleMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 학습 루프
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            if isinstance(model, SimpleMLP):
                data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

    # 모델 가중치 저장
    if isinstance(model, FromScratch):
        torch.save(model.state_dict(), "fromscratch.pth")
    elif isinstance(model, SimpleCNN):
        torch.save(model.state_dict(), "simplecnn.pth")
    elif isinstance(model, SimpleMLP):
        torch.save(model.state_dict(), "simplemlp.pth")
    elif isinstance(model, CNNmodel_c2f1):
        torch.save(model.state_dict(), "cnnmodel_c2f1.pth")
    else:
        print("Model not found!")
        return
    print("Model saved")
    return model

def test(model, device, model_path=None):
    # 테스트 루프
    test_loader = get_mnist_data_loader(train=False)
    if model_path:
        model.load_state_dict(torch.load(model_path))
        print("model loaded")
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if isinstance(model, SimpleMLP):
                data = data.view(data.size(0), -1)

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f'Test Accuracy: {100. * correct / len(test_loader.dataset)}%')

model = SimpleMLP()
test(model, "cpu", "simplemlp.pth")
    