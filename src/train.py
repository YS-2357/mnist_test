import torch
import torch.nn as nn
import torch.optim as optim
from mnist_test.models.SimpleMLP import SimpleMLP
from utils.data_loader import get_mnist_data_loader
from src.utils import save_model

# 하이퍼파라미터 설정
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# 데이터 로더
train_loader = get_mnist_data_loader(batch_size=batch_size, train=True)

# 모델, 손실 함수, 옵티마이저 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 모델 학습
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.view(-1, 28*28).to(device)  # Flatten MNIST images
        labels = labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 통계
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")

# 학습 완료 후 모델 저장
save_model(model, "./models/weights/mlp_mnist.pth")
print("Training complete. Model saved!")
