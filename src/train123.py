import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.data_loader import get_mnist_data_loader
from models.From_Scratch import FromScratch
from models.mlp import SimpleMLP
from models.model_jeonghun import CNNmodel_c2f1

def train(model, num_epochs=10, lr=1e-3):
    train_loader = get_mnist_data_loader(batch_size=64, train=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 학습 루프 시작
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        # 훈련 진행 상황 출력
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=False)
        for images, labels in train_progress:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            train_progress.set_postfix(loss=loss.item())
        
        train_loss /= len(train_loader.dataset)
        train_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    # 모델 저장
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", f"{model.__class__.__name__}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="fromscratch", help="model name")
    parser.add_argument("--num_epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    args = parser.parse_args()

    if args.model == "fromscratch":
        model = FromScratch()
    elif args.model == "mlp":
        model = SimpleMLP()
    elif args.model == "model_jeonghun":
        model = CNNmodel_c2f1()
    else:
        raise NotImplementedError(f"{args.model} is not implemented")

    train(model, num_epochs=args.num_epochs, lr=args.lr)