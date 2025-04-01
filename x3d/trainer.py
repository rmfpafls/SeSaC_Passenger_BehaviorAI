import os
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.io import read_video
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from pytorchvideo.transforms import Normalize
from torch.amp import autocast
from dataset import mapping_label

# 학습 및 검증 클래스
class VideoTrainer:
    def __init__(self, model, device, criterion, optimizer, scaler, checkpoint_dir):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_model(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for videos, targets in train_loader:
            videos, targets = videos.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = self.model(videos)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_predictions += (preds == targets).sum().item()
            total_samples += targets.size(0)

        avg_loss = running_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        return avg_loss, accuracy

    def evaluate_model(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for videos, targets in dataloader:
                videos, targets = videos.to(self.device), targets.to(self.device)

                with autocast(device_type='cuda'):
                    outputs = self.model(videos)
                    loss = self.criterion(outputs, targets)

                total_loss += loss.item() * videos.size(0)
                _, preds = torch.max(outputs, 1)
                total_correct += (preds == targets).sum().item()
                total_samples += targets.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy
    
    def test_model(self, model, test_loader, criterion, index_to_action):
        """
        테스트 데이터셋을 사용하여 모델의 성능 평가 및 예측 결과 출력

        Args:
            model (torch.nn.Module): 학습된 모델
            test_loader (DataLoader): 테스트 데이터 로더
            criterion: 손실 함수
            index_to_action (dict): 인덱스와 행동 이름 매핑 딕셔너리

        Returns:
            list of tuples: [(video_path, true_label, predicted_label), ...]
        """
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        results = []

        with torch.no_grad():
            for videos, targets in test_loader:
                videos, targets = videos.to(self.device), targets.to(self.device)

                # 예측 수행
                with autocast(device_type='cuda'):
                    outputs = model(videos)
                    loss = criterion(outputs, targets)

                # 손실 누적
                total_loss += loss.item() * videos.size(0)

                # 예측 레이블 계산
                _, preds = torch.max(outputs, 1)
                

                # 결과 저장
                for idx in range(len(videos)):
                    true_label = index_to_action[targets[idx].item()]
                    predicted_label = index_to_action[preds[idx].item()]
                    results.append((true_label, predicted_label))

                # 정확도 계산
                total_correct += (preds == targets).sum().item()
                total_samples += targets.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
        return results

    def predict_action(self, video_path, label_map_csv_path):
        """
        Predicts the action label for a given video clip while returning a random frame image.

        Args:
            video_path (str): Path to the video file.
            label_map_csv_path (str): Path to the CSV file containing label_index and activity.

        Returns:
            PIL.Image.Image: Randomly captured image from the video.
            str: Predicted action label.
        """
        # Load the label map from the CSV file
        label_map = mapping_label(label_map_csv_path)

        self.model.to(self.device)
        self.model.eval()

        # 비디오에서 전체 프레임 로드
        video_frames, _, _ = read_video(video_path, start_pts=0, end_pts=None, pts_unit="sec")

        # 랜덤한 프레임 선택
        num_frames = video_frames.size(0)  # 전체 프레임 수
        random_frame_idx = random.randint(0, num_frames - 1)  # 랜덤 인덱스 선택
        random_frame = video_frames[random_frame_idx]  # 랜덤 프레임 선택
        pil_image = Image.fromarray(random_frame.numpy())  # PIL 이미지로 변환

        transforms = Compose([
            Resize((160, 160)), 
            CenterCrop((160, 160)), 
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])


        video_frames = video_frames.permute(3, 0, 1, 2).float() / 255.0  # (T, H, W, C) -> (C, T, H, W)
        video_tensor = transforms(video_frames)
        video_tensor = video_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            with autocast(device_type='cuda'):
                outputs = self.model(video_tensor) # 전체 비디오 텐서를 모델에 입력
            predicted_index = torch.argmax(outputs, dim=1).item()
            predicted_label = label_map.get(predicted_index, "Unknown")

        return pil_image, predicted_label

    def save_checkpoint(self, epoch, valid_loss, valid_accuracy):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "loss": valid_loss,
            "accuracy": valid_accuracy
        }, os.path.join(self.checkpoint_dir, "best_model.pth"))
        print(f"Best model saved at epoch {epoch + 1} with accuracy: {valid_accuracy:.4f}")
        
        
# 체크포인트 로드 함수
def load_checkpoint(checkpoint_path, model, optimizer, scaler, DEVICE):
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1  # 다음 에폭부터 시작

        # 손실 및 정확도 리스트 복원
        train_loss_values = checkpoint.get("train_loss_values", [])
        train_accuracy_values = checkpoint.get("train_accuracy_values", [])
        valid_loss_values = checkpoint.get("valid_loss_values", [])
        valid_accuracy_values = checkpoint.get("valid_accuracy_values", [])
        print(f"Checkpoint loaded. Starting from epoch {start_epoch}.")
        # return start_epoch, train_loss_values, train_accuracy_values, valid_loss_values, valid_accuracy_values
        return model
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting training from scratch.")
        return 0, [], [], [], []
        
        
# 손실 및 정확도 그래프 출력 함수
def plot_training_validation_metrics(train_loss, train_accuracy, valid_loss, valid_accuracy, epochs):
    plt.figure(figsize=(12, 8))

    # Loss 그래프
    plt.subplot(2, 1, 1)
    plt.plot(range(1, epochs + 1), train_loss, label='Train Loss', marker='o')
    plt.plot(range(1, epochs + 1), valid_loss, label='Valid Loss', marker='o')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    # Accuracy 그래프
    plt.subplot(2, 1, 2)
    plt.plot(range(1, epochs + 1), train_accuracy, label='Train Accuracy', marker='o')
    plt.plot(range(1, epochs + 1), valid_accuracy, label='Valid Accuracy', marker='o')
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()