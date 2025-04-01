import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
from pytorchvideo.data.encoded_video import EncodedVideo
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose
from pytorchvideo.transforms import (
    UniformTemporalSubsample,
    ShortSideScale,
    Normalize,
)

'''
If you try to Jupyter environment, try this code.
import zipfile
zip_file_path='/content/drive/MyDrive/kinect_color.zip'
extract_path = "/content/kinect/"
with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)
'''

class CustomVideoDataset(Dataset):
    def __init__(self, annotations_df, transform=None, kinect_path=""):
        """
        Custom Video Dataset 초기화
        - annotations_df: 데이터프레임
        - transform: 데이터 변환 함수
        - kinetic_path: 비디오 파일 경로의 기본 디렉토리
        """
        self.annotations = annotations_df
        self.transform = transform
        self.kinect_path = kinect_path  # 비디오 파일의 기본 경로

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # 비디오 경로 및 라벨 가져오기
        try:
            # file_id로 비디오 경로 생성
            file_id = self.annotations.iloc[idx]["file_id"]
            video_path = str(self.kinect_path) + os.sep + f"{file_id}"

            # 라벨 가져오기
            label = self.annotations.iloc[idx]["label_idx"]
        except KeyError as e:
            raise KeyError(f"Missing required column in annotations: {e}")

        # 비디오 데이터 로드
        try:
            video = EncodedVideo.from_path(video_path, decoder='pyav')
        except Exception as e:
            raise FileNotFoundError(f"Video file not found or cannot be loaded: {video_path}") from e

        try:
            # 비디오 디코딩 및 텐서 변환
            video_container = video._container
            video_stream = video_container.streams.video[0]
            frames = video_container.decode(video_stream)

            video_frames = [torch.from_numpy(frame.to_rgb().to_ndarray()) for frame in frames]
            video_tensor = torch.stack(video_frames).permute(3, 0, 1, 2).float() /255.0

            # SlowFast 경로 생성
            slow_pathway = UniformTemporalSubsample(8)(video_tensor)
            fast_pathway = UniformTemporalSubsample(32)(video_tensor)

            # Transform 적용
            if self.transform:
                slow_pathway = self.transform(slow_pathway)
                fast_pathway = self.transform(fast_pathway)

        finally:
            del video  # 비디오 닫기

        return [slow_pathway, fast_pathway], label


def load_and_merge_csv(directory_path):
    # 하위 폴더까지 포함하여 모든 .csv 파일 탐색
    all_files = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(directory_path)
        for file in files if file.endswith('.csv')
    ]
    print("csv file loaded counts  ", len(all_files))
    # 모든 파일을 DataFrame으로 읽기
    dataframes = [pd.read_csv(file) for file in all_files]

    # DataFrame 병합
    merged_df = pd.concat(dataframes, ignore_index=True)
    return merged_df


def filter_minority_classes_and_limit(df, stratify_column, min_samples=10, max_samples=1000):
    """
    부족한 레이블을 확인하고 해당 데이터를 삭제하며, 너무 많은 레이블은 제한된 샘플 수로 줄이는 함수.

    Parameters:
    - df: pandas DataFrame
    - stratify_column: Stratify 기준이 되는 레이블 컬럼
    - min_samples: 각 클래스에 최소 필요한 샘플 개수
    - max_samples: 각 클래스에 최대 허용 샘플 개수

    Returns:
    - df: 부족한 레이블이 제거되고, 너무 많은 레이블이 제한된 데이터프레임
    """
    label_counts = df[stratify_column].value_counts()
    print("Before filtering:")
    print(label_counts)

    # 부족한 레이블 필터링
    insufficient_labels = label_counts[label_counts < min_samples].index
    print("\nLabels with insufficient samples (removed):")
    for label in insufficient_labels:
        print(f"Label: {label}, Count: {label_counts[label]}")

    # 부족한 레이블 삭제
    filtered_df = df[~df[stratify_column].isin(insufficient_labels)]

    # 너무 많은 레이블 제한
    limited_df = pd.concat(
        [
            filtered_df[filtered_df[stratify_column] == label].sample(n=min(count, max_samples), random_state=42)
            for label, count in filtered_df[stratify_column].value_counts().items()
        ]
    )

    print("\nAfter filtering and limiting:")
    print(limited_df[stratify_column].value_counts())
    return limited_df


def split_dataset(df, test_size=0.15, val_size=0.15, stratify_column="label_idx"):
    """
    데이터프레임을 train, validation, test로 분할.
    부족한 레이블은 필터링하고 너무 많은 레이블은 제한하여 Stratify 조건을 만족시킴.
    """
    if stratify_column in df:
        # 부족한 레이블 필터링 및 제한
        df = filter_minority_classes_and_limit(df, stratify_column=stratify_column, min_samples=100, max_samples=1000)
        stratify = df[stratify_column]
    else:
        stratify = None

    # Train/Test 비율 계산
    train_size = 1 - test_size - val_size

    # Train + Test로 분할
    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_size), stratify=stratify, random_state=42
    )

    # Validation/Test 비율 계산
    temp_stratify = temp_df[stratify_column] if stratify_column in temp_df else None
    val_df, test_df = train_test_split(
        temp_df, test_size=test_size / (test_size + val_size), stratify=temp_stratify, random_state=42
    )

    return train_df, val_df, test_df

def get_transform():
    # Transform and Dataset
    transform = Compose([
        ShortSideScale(224),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    return transform

# 매핑 관련 함수
def mapping_label(label_map_path):
    label_map = pd.read_csv(label_map_path)
    index_to_action = dict(zip(label_map['index'], label_map['midlevel_activity']))
    return index_to_action