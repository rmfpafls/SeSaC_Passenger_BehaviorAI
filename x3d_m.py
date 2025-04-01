import pandas as pd
import torch
from pytorchvideo.models.hub import x3d_xs, x3d_m
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, CenterCrop
from pytorchvideo.transforms import Normalize
import torch.nn as nn
import torch.optim as optim
import random
from torchvision.io import read_video
import matplotlib.pyplot as plt
import os
from torch.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split

# 행동 매핑 파일 경로
label_map_path = r'C:\final_project\label_maps\label_map_midlevel.csv'

# 매핑 파일 읽기
label_map = pd.read_csv(label_map_path)

# 매핑 딕셔너리 생성
index_to_action = dict(zip(label_map['index'], label_map['midlevel_activity']))

# 사용자 정의 데이터셋 클래스
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # 비디오 읽기
        video, _, _ = read_video(video_path, pts_unit="sec")
        video = video.permute(3, 0, 1, 2).float()  # (T, H, W, C) -> (C, T, H, W)

        # 필요한 프레임 수로 샘플링
        if video.shape[1] > self.num_frames:
            start = random.randint(0, video.shape[1] - self.num_frames)
            video = video[:, start:start + self.num_frames, :, :]
        else:
            video = torch.nn.functional.pad(video, (0, 0, 0, 0, 0, self.num_frames - video.shape[1]))

        if self.transform:
            video = self.transform(video)

        return video, label

# 데이터 전처리 함수
def get_transforms():
    return Compose([
        Resize((160, 160)),
        CenterCrop((256, 256)),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


def plot_training_validation_metrics(
    train_loss, train_accuracy, valid_loss, valid_accuracy, epochs, start_epoch
):
    # x축 범위 설정
    x_range = range(start_epoch + 1, start_epoch + len(train_loss) + 1)
    
    plt.figure(figsize=(12, 8))

    # Loss 그래프
    plt.subplot(2, 1, 1)
    plt.plot(x_range, train_loss, label='Train Loss')
    plt.plot(x_range, valid_loss, label='Valid Loss')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    # Accuracy 그래프
    plt.subplot(2, 1, 2)
    plt.plot(x_range, train_accuracy, label='Train Accuracy')
    plt.plot(x_range, valid_accuracy, label='Valid Accuracy')
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# Validation 평가 함수
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for videos, targets in dataloader:
            videos, targets = videos.to(DEVICE), targets.to(DEVICE)

            with autocast(device_type='cuda'):
                outputs = model(videos)
                loss = criterion(outputs, targets)

            # 손실 계산
            total_loss += loss.item() * videos.size(0)

            # 정확도 계산
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


# 체크포인트 로드 함수
def load_checkpoint(checkpoint_path, model, optimizer, scaler):
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
        return start_epoch, train_loss_values, train_accuracy_values, valid_loss_values, valid_accuracy_values
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting training from scratch.")
        return 0, [], [], [], []
    
def test_model(model, test_loader, criterion, index_to_action):
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
            videos, targets = videos.to(DEVICE), targets.to(DEVICE)

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


if __name__ == '__main__':
    # CSV 파일 경로
    csv_path = r"C:\Final_project_Sesac\cut_videos\kinect_color\filetered_label.csv"

    # 데이터셋 읽기
    df = pd.read_csv(csv_path)

    # 데이터셋 분할
    train_ratio, valid_ratio, test_ratio = 0.7, 0.2, 0.1
    train_data, temp_data = train_test_split(df, test_size=(1 - train_ratio), random_state=42, stratify=df['label'])
    valid_data, test_data = train_test_split(temp_data, test_size=(test_ratio / (valid_ratio + test_ratio)), random_state=42, stratify=temp_data['label'])

    print(f"Train set size: {len(train_data)}")
    print(f"Valid set size: {len(valid_data)}")
    print(f"Test set size: {len(test_data)}")

    # 하이퍼파라미터 설정
    BATCH_SIZE = 32
    EPOCHS = 150
    LEARNING_RATE = 1e-4
    NUM_FRAMES = 16
    NUM_CLASSES = 39
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR = "./checkpoints_x3d_m"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 데이터 로더 생성
    transform = get_transforms()
    train_loader = DataLoader(VideoDataset(train_data['video_path'].tolist(), train_data['label'].tolist(), NUM_FRAMES, transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(VideoDataset(valid_data['video_path'].tolist(), valid_data['label'].tolist(), NUM_FRAMES, transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(VideoDataset(test_data['video_path'].tolist(), test_data['label'].tolist(), NUM_FRAMES, transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 모델 초기화 및 출력 계층 수정
    model = x3d_m(pretrained=True)
    model.blocks[5].proj = nn.Linear(model.blocks[5].proj.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-2
    )

    # Mixed Precision Training 설정
    scaler = GradScaler(device='cuda')
    
    # 체크포인트 로드
    # checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    # start_epoch, train_loss_values, train_accuracy_values, valid_loss_values, valid_accuracy_values = load_checkpoint(checkpoint_path, model, optimizer, scaler)
    
    train_loss_values = []
    train_accuracy_values = []
    valid_loss_values = []
    valid_accuracy_values = []
    
    # 학습 루프
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (videos, targets) in enumerate(train_loader):
            videos, targets = videos.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()

            # Mixed Precision 활성화
            with autocast(device_type='cuda'):
                outputs = model(videos)
                loss = criterion(outputs, targets)

            # Loss 스케일링
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 손실 및 정확도 계산
            running_loss += loss.item() * videos.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += (preds == targets).sum().item()
            total_samples += targets.size(0)

        # Train Loss 및 Accuracy 계산
        train_epoch_loss = running_loss / total_samples
        train_epoch_accuracy = correct_predictions / total_samples
        train_loss_values.append(train_epoch_loss)
        train_accuracy_values.append(train_epoch_accuracy)

        # Validation Loss 및 Accuracy 계산
        valid_epoch_loss, valid_epoch_accuracy = evaluate_model(model, valid_loader, criterion)
        valid_loss_values.append(valid_epoch_loss)
        valid_accuracy_values.append(valid_epoch_accuracy)

        print(f"Epoch [{epoch + 1}/{EPOCHS}], "
              f"Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {train_epoch_accuracy:.4f}, "
              f"Valid Loss: {valid_epoch_loss:.4f}, Valid Accuracy: {valid_epoch_accuracy:.4f}")

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'epoch_{epoch}.pth')
        # 최신 체크포인트 저장
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "loss": valid_epoch_loss,
            "accuracy": valid_epoch_accuracy,
            "train_loss_values": train_loss_values,
            "train_accuracy_values": train_accuracy_values,
            "valid_loss_values": valid_loss_values,
            "valid_accuracy_values": valid_accuracy_values,
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}.")

        # 베스트 모델 저장
        if valid_epoch_accuracy > max(valid_accuracy_values[:-1], default=0):
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "loss": valid_epoch_loss,
                "accuracy": valid_epoch_accuracy
            }, os.path.join(CHECKPOINT_DIR, f"best_model.pth"))
            print(f"Best model saved at epoch {epoch + 1} with accuracy: {valid_epoch_accuracy:.4f}")

        # GPU 캐시 정리
        torch.cuda.empty_cache()

    # 손실 및 정확도 그래프 출력
    plot_training_validation_metrics(
        train_loss_values, train_accuracy_values,
        valid_loss_values, valid_accuracy_values,
        EPOCHS, 0
    )

    print("학습 및 평가 완료!")
    
    result = test_model(model, test_loader, criterion, index_to_action)