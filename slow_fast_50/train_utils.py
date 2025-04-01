import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from pytorchvideo.transforms import ShortSideScale, Normalize
from pytorchvideo.models.hub import slowfast_r50
from pytorchvideo.models.head import ResNetBasicHead
from video_dataset import CustomVideoDataset
from dataset_utils import load_and_merge_csv, split_dataset
from lightning_module import VideoClassificationLightningModule


# 함수 정의
def get_datasets_and_loaders(directory_path, transform, batch_size=32):
    """데이터셋을 로드하고 DataLoader를 반환"""
    merged_df = load_and_merge_csv(directory_path)
    train_df, val_df, test_df = split_dataset(merged_df, test_size=0.1, val_size=0.2, stratify_column="label_idx")
    
    train_dataset = CustomVideoDataset(train_df, transform=transform)
    val_dataset = CustomVideoDataset(val_df, transform=transform)
    test_dataset = CustomVideoDataset(test_df, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    num_classes = len(train_dataset.annotations.iloc[:, 1].unique())
    return train_loader, val_loader, test_loader, num_classes


def create_model(num_classes, checkpoint_path=None):
    """모델 생성 및 가중치 로드"""
    model = slowfast_r50(pretrained=False)  # Pretrained=False, 사용자 가중치 로드 예정
    model.blocks[-1] = ResNetBasicHead(
        proj=torch.nn.Linear(model.blocks[-1].proj.in_features, num_classes),
        dropout=torch.nn.Dropout(0.5),
    )
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded model weights from: {checkpoint_path}")
    return model


def train_and_evaluate_model(model, args, train_loader, val_loader, test_loader, logger, max_epochs, save_path):
    """모델 학습, 검증, 테스트 및 저장"""
    # Lightning 모듈 생성
    video_model = VideoClassificationLightningModule(model, args)
    
    # Trainer 설정
    trainer = pl.Trainer(
        precision=16,
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=1,
        accumulate_grad_batches=8,
        logger=logger
    )
    
    # 학습
    trainer.fit(video_model, train_loader, val_loader)
    
    # 테스트
    trainer.test(video_model, test_loader)
    
    # 학습된 모델 저장
    torch.save(video_model.model.state_dict(), save_path)
    print(f"Model weights saved to: {save_path}")


def plot_metrics(log_dir):
    """학습 및 검증 손실과 정확도를 그래프로 시각화"""
    metrics_file = os.path.join(log_dir, "metrics.csv")
    metrics_df = pd.read_csv(metrics_file)

    # Loss 그래프
    plt.figure(figsize=(10, 6))
    if 'train_loss_epoch' in metrics_df:
        plt.plot(metrics_df['epoch'], metrics_df['train_loss_epoch'], label='Train Loss', color='blue')
    if 'val_loss_epoch' in metrics_df:
        plt.plot(metrics_df['epoch'], metrics_df['val_loss_epoch'], label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Accuracy 그래프
    plt.figure(figsize=(10, 6))
    if 'train_acc_epoch' in metrics_df:
        plt.plot(metrics_df['epoch'], metrics_df['train_acc_epoch'], label='Train Accuracy', color='green')
    if 'val_acc_epoch' in metrics_df:
        plt.plot(metrics_df['epoch'], metrics_df['val_acc_epoch'], label='Validation Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 최종 출력
    print("Final Metrics:")
    if 'train_acc_epoch' in metrics_df:
        print(f"Train Accuracy: {metrics_df['train_acc_epoch'].iloc[-1]:.2f}")
    if 'val_acc_epoch' in metrics_df:
        print(f"Validation Accuracy: {metrics_df['val_acc_epoch'].iloc[-1]:.2f}")
    if 'train_loss_epoch' in metrics_df:
        print(f"Train Loss: {metrics_df['train_loss_epoch'].iloc[-1]:.4f}")
    if 'val_loss_epoch' in metrics_df:
        print(f"Validation Loss: {metrics_df['val_loss_epoch'].iloc[-1]:.4f}")


# 메인 코드
if __name__ == "__main__":
    # 설정
    directory_path = r"C:\Users\user\Downloads\slow_fast_50\kinect_color"
    checkpoint_path = r"C:\Users\user\Downloads\slow_fast_50\bfb_slowfast_r50_1120_kinect_color.pth"
    batch_size = 32
    max_epochs = 10
    save_path = r".\slow_fast_50\bfb_slowfast_r50_1120_kinect_color.pth"

    # Transform 설정
    transform = Compose([
        ShortSideScale(224),
        Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    ])

    # Logger 설정
    csv_logger = CSVLogger("logs", name="slowfast_r50_resume_training")

    # 데이터 로드
    train_loader, val_loader, test_loader, num_classes = get_datasets_and_loaders(directory_path, transform, batch_size)
    
    print(len(train_loader))
    print(len(val_loader))
    print(len(test_loader))

    # 모델 생성 및 가중치 로드
    pretrained_model = create_model(num_classes, checkpoint_path)

    # 학습 및 검증 실행
    args = {"lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4, "max_epochs": max_epochs, "num_classes": num_classes}
    train_and_evaluate_model(pretrained_model, args, train_loader, val_loader, test_loader, csv_logger, max_epochs, save_path)

    # 학습 결과 시각화
    plot_metrics(csv_logger.log_dir)
