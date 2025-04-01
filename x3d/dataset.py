import random
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, CenterCrop
from pytorchvideo.transforms import Normalize
from torchvision.io import read_video
from sklearn.model_selection import train_test_split

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


# 매핑 관련 함수
def mapping_label(label_map_path, language = None):
    if language is not None:
        language.lower()
    
    label_map = pd.read_csv(label_map_path)
    
    if language == 'eng':
        index_to_action = dict(zip(label_map['index'], label_map['midlevel_activity']))
    elif language == 'kor':
        index_to_action = dict(zip(label_map['index'], label_map['행동']))
    
    return index_to_action

# 데이터 전처리 함수
def get_transforms():
    return Compose([
        Resize((160, 160)),
        CenterCrop((160, 160)),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
def split_dataset(csv_path):
    # CSV 파일 경로
    csv_path = csv_path

    # 데이터셋 읽기
    df = pd.read_csv(csv_path)

    # 데이터셋 분할 비율
    train_ratio = 0.7  # 학습 데이터 비율
    valid_ratio = 0.2  # 검증 데이터 비율
    test_ratio = 0.1   # 테스트 데이터 비율

    # 데이터셋 분할
    train_data, temp_data = train_test_split(df, test_size=(1 - train_ratio), random_state=42, stratify=df['label'])
    valid_data, test_data = train_test_split(temp_data, test_size=(test_ratio / (valid_ratio + test_ratio)), random_state=42, stratify=temp_data['label'])

    # 결과 확인
    print(f"Train set size: {len(train_data)}")
    print(f"Valid set size: {len(valid_data)}")
    print(f"Test set size: {len(test_data)}")

    # 분할된 데이터 저장
    train_data.to_csv(r"C:\final_project\train_data.csv", index=False)
    valid_data.to_csv(r"C:\final_project\valid_data.csv", index=False)
    test_data.to_csv(r"C:\final_project\test_data.csv", index=False)

    print("데이터셋이 성공적으로 분할되고 저장되었습니다!")