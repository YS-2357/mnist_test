import torch
from mnist_test.models.SimpleMLP import SimpleMLP
from utils.data_loader import get_mnist_data_loader
from src.utils import load_model

# 테스트 데이터 로더
batch_size = 64
test_loader = get_mnist_data_loader(batch_size=batch_size, train=False)

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMLP().to(device)
load_model(model, "./models/weights/mlp_mnist.pth")
model.eval()

# 테스트 실행
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(-1, 28*28).to(device)  # Flatten MNIST images
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
