import torch
import pandas as pd
from torch.utils.data import Dataset
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import UniformTemporalSubsample
import torch.nn.functional as F
import io
import av
from iopath.common.file_io import g_pathmgr
import os

kinetic_path='/home/elicer/kinect_color/'

class CustomVideoDataset(Dataset):
    def __init__(self, annotations_df, transform=None, kinetic_path=""):
        """
        Custom Video Dataset 초기화
        - annotations_df: 데이터프레임
        - transform: 데이터 변환 함수
        - kinetic_path: 비디오 파일 경로의 기본 디렉토리
        """
        self.annotations = annotations_df
        self.transform = transform
        self.kinetic_path = kinetic_path  # 비디오 파일의 기본 경로

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # 비디오 경로 및 라벨 가져오기
        try:
            # file_id로 비디오 경로 생성
            file_id = self.annotations.iloc[idx]["file_id"]
            video_path = str(kinetic_path)+ f"{file_id}"

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
