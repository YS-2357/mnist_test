import torch
import os

def save_model(model, path):
    """ 모델 저장 함수 """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    """ 모델 불러오기 함수 """
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
    else:
        print(f"Model file not found: {path}")
