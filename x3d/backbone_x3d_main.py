import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorchvideo.models.hub import x3d_xs
from torch.optim import AdamW
from torch.amp import GradScaler
from torchvision.transforms import Compose, Resize, CenterCrop
from sklearn.model_selection import train_test_split
from dataset import VideoDataset, get_transforms, mapping_label
from trainer import VideoTrainer, plot_training_validation_metrics


# 메인 코드
if __name__ == '__main__':
    # CSV 파일 경로
    csv_path = r"C:\final_project\cut_videos\kinect_color\filetered_label.csv"
    label_map_path = r'C:\final_project\label_maps\label_map_midlevel.csv'
    index_to_action = mapping_label(label_map_path)

    # 데이터셋 읽기 및 분할
    df = pd.read_csv(csv_path)
    train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    valid_data, test_data = train_test_split(temp_data, test_size=0.33, random_state=42, stratify=temp_data['label'])

    # 하이퍼파라미터 설정
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    NUM_FRAMES = 16
    NUM_CLASSES = 39
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR = "./checkpoints_x3d"

    # 데이터 로더 생성
    transform = get_transforms()
    train_loader = DataLoader(VideoDataset(train_data['video_path'].tolist(), train_data['label'].tolist(), NUM_FRAMES, transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(VideoDataset(valid_data['video_path'].tolist(), valid_data['label'].tolist(), NUM_FRAMES, transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 모델 초기화 및 출력 계층 수정
    model = x3d_xs(pretrained=False)
    model.blocks[5].proj = nn.Linear(model.blocks[5].proj.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    # 손실 함수, 옵티마이저 및 스케일러
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scaler = GradScaler()

    # 트레이너 초기화
    trainer = VideoTrainer(model, DEVICE, criterion, optimizer, scaler, CHECKPOINT_DIR)

    # 학습 및 검증 루프
    train_loss_values, train_accuracy_values = [], []
    valid_loss_values, valid_accuracy_values = [], []

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = trainer.train_one_epoch(train_loader)
        valid_loss, valid_accuracy = trainer.evaluate(valid_loader)

        train_loss_values.append(train_loss)
        train_accuracy_values.append(train_accuracy)
        valid_loss_values.append(valid_loss)
        valid_accuracy_values.append(valid_accuracy)

        print(f"Epoch [{epoch + 1}/{EPOCHS}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}")

        if valid_accuracy > max(valid_accuracy_values[:-1], default=0):  # 첫 번째에는 항상 저장
            trainer.save_checkpoint(epoch, valid_loss, valid_accuracy)

    # 손실 및 정확도 그래프 출력
    plot_training_validation_metrics(train_loss_values, train_accuracy_values, valid_loss_values, valid_accuracy_values, EPOCHS)
    print("학습 및 평가 완료!")
