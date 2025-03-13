import torch
from models.From_Scratch import FromScratch
from models.fromscratch import SimpleCNN
from models.mlp import SimpleMLP
from models.model_jeonghun import CNNmodel_c2f1

def test(model, test_loader):
    # 테스트 루프
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            
            if isinstance(model, SimpleMLP):
                data = data.view(data.size(0), -1)

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f'Test Accuracy: {100. * correct / len(test_loader.dataset)}%')